import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from workspace_agent.loop import SYSTEM_POLICY, AgentRunner
from workspace_agent.model import ModelProtocolError, OpenAICompatibleModel
from workspace_agent.safety import WorkspaceGuard
from workspace_agent.schemas import ModelReply, ToolCall, Usage
from workspace_agent.tools import TOOL_SCHEMAS, WorkspaceTools
from workspace_agent.trace import TraceStore, TraceWriter


def _completion(
    message: dict[str, Any],
    *,
    usage: dict[str, int | None] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"choices": [{"message": message}]}
    if usage is not None:
        payload["usage"] = usage
    return payload


def _injected_model(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    api_key: str = "test-key",
) -> tuple[OpenAICompatibleModel, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    model = OpenAICompatibleModel(
        "https://models.test/v1/",
        api_key,
        "test-model",
        4.5,
        client=client,
    )
    return model, client


@pytest.mark.asyncio
async def test_model_sends_exact_request_and_parses_final_reply() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["authorization"]
        captured["content_type"] = request.headers["content-type"]
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=_completion(
                {"role": "assistant", "content": "Finished."},
                usage={
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "total_tokens": 10,
                },
            ),
        )

    model, client = _injected_model(handler)
    messages = [{"role": "user", "content": "Inspect the workspace"}]
    try:
        reply = await model.complete(messages, TOOL_SCHEMAS)
    finally:
        await client.aclose()

    assert captured == {
        "url": "https://models.test/v1/chat/completions",
        "authorization": "Bearer test-key",
        "content_type": "application/json",
        "payload": {
            "model": "test-model",
            "messages": messages,
            "tools": TOOL_SCHEMAS,
            "tool_choice": "auto",
            "temperature": 0,
        },
    }
    assert reply.message == {"role": "assistant", "content": "Finished."}
    assert reply.tool_calls == ()
    assert reply.usage == Usage(
        prompt_tokens=7,
        completion_tokens=3,
        total_tokens=10,
    )


@pytest.mark.asyncio
async def test_model_parses_multiple_tool_calls_and_preserves_raw_message() -> None:
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path":"notes/资料.txt","limit":20}',
                },
            },
            {
                "id": "call-2",
                "type": "function",
                "function": {
                    "name": "stat_path",
                    "arguments": '{"path":"notes"}',
                },
            },
        ],
    }
    model, client = _injected_model(
        lambda request: httpx.Response(200, json=_completion(message))
    )
    try:
        reply = await model.complete([], TOOL_SCHEMAS)
    finally:
        await client.aclose()

    assert reply.message is message or reply.message == message
    assert reply.tool_calls == (
        ToolCall(
            id="call-1",
            name="read_file",
            arguments={"path": "notes/资料.txt", "limit": 20},
        ),
        ToolCall(
            id="call-2",
            name="stat_path",
            arguments={"path": "notes"},
        ),
    )
    assert reply.usage == Usage(
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
    )


@pytest.mark.asyncio
async def test_model_missing_usage_marks_every_component_unknown() -> None:
    model, client = _injected_model(
        lambda request: httpx.Response(
            200,
            json=_completion({"role": "assistant", "content": "Done"}),
        )
    )
    try:
        reply = await model.complete([], [])
    finally:
        await client.aclose()

    assert reply.usage.as_dict() == {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": []},
        {"choices": [{}]},
        {"choices": [{"message": "not-an-object"}]},
        _completion(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "read_file",
                            "arguments": "{bad json",
                        },
                    }
                ],
            }
        ),
        _completion(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"offset":NaN}',
                        },
                    }
                ],
            }
        ),
        _completion(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "read_file",
                            "arguments": '["notes.txt"]',
                        },
                    }
                ],
            }
        ),
        _completion(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "",
                        "function": {
                            "name": "read_file",
                            "arguments": "{}",
                        },
                    }
                ],
            }
        ),
        _completion(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "",
                            "arguments": "{}",
                        },
                    }
                ],
            }
        ),
        _completion(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "read_file",
                            "arguments": {},
                        },
                    }
                ],
            }
        ),
    ],
)
@pytest.mark.asyncio
async def test_model_rejects_malformed_response_shapes(
    payload: dict[str, Any],
) -> None:
    model, client = _injected_model(
        lambda request: httpx.Response(200, json=payload)
    )
    try:
        with pytest.raises(ModelProtocolError):
            await model.complete([], [])
    finally:
        await client.aclose()


@pytest.mark.parametrize("failure_kind", ["timeout", "network", "429", "500"])
@pytest.mark.asyncio
async def test_model_retries_transient_failures(
    failure_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    async def no_delay(seconds: float) -> None:
        assert 0 <= seconds <= 1

    monkeypatch.setattr("workspace_agent.model.asyncio.sleep", no_delay)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            if failure_kind == "timeout":
                raise httpx.ReadTimeout("timed out", request=request)
            if failure_kind == "network":
                raise httpx.ConnectError("offline", request=request)
            status = int(failure_kind)
            headers = {"Retry-After": "60"} if status == 429 else None
            return httpx.Response(status, headers=headers, text="private body")
        return httpx.Response(
            200,
            json=_completion({"role": "assistant", "content": "Recovered"}),
        )

    model, client = _injected_model(handler)
    try:
        reply = await model.complete([], [])
    finally:
        await client.aclose()

    assert attempts == 3
    assert reply.message["content"] == "Recovered"


@pytest.mark.parametrize("status", [400, 401, 403, 600])
@pytest.mark.asyncio
async def test_model_does_not_retry_nonretryable_client_errors(
    status: int,
) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(status, text="sensitive response body")

    model, client = _injected_model(handler, api_key="never-print-this-key")
    try:
        with pytest.raises(RuntimeError) as captured:
            await model.complete([], [])
    finally:
        await client.aclose()

    assert attempts == 1
    assert "never-print-this-key" not in str(captured.value)
    assert "sensitive response body" not in str(captured.value)


@pytest.mark.asyncio
async def test_model_bounds_retry_attempts_and_redacts_failure_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    async def no_delay(seconds: float) -> None:
        return None

    monkeypatch.setattr("workspace_agent.model.asyncio.sleep", no_delay)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(500, text="private upstream response")

    model, client = _injected_model(handler, api_key="top-secret-key")
    try:
        with pytest.raises(RuntimeError) as captured:
            await model.complete([], [])
    finally:
        await client.aclose()

    assert attempts == 3
    message = str(captured.value)
    assert "top-secret-key" not in message
    assert "private upstream response" not in message


def test_model_rejects_blank_api_key() -> None:
    with pytest.raises(ValueError, match="API key"):
        OpenAICompatibleModel("https://models.test/v1", "   ", "model", 5)


@pytest.mark.asyncio
async def test_model_closes_only_owned_client() -> None:
    injected = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json=_completion({"role": "assistant", "content": "ok"}),
            )
        )
    )
    external_model = OpenAICompatibleModel(
        "https://models.test/v1",
        "key",
        "model",
        5,
        client=injected,
    )
    await external_model.aclose()
    assert not injected.is_closed
    await injected.aclose()

    async with OpenAICompatibleModel(
        "https://models.test/v1",
        "key",
        "model",
        5,
    ) as owned_model:
        owned_client = owned_model._client
        assert not owned_client.is_closed
    assert owned_client.is_closed


class ScriptedModel:
    def __init__(self, *items: ModelReply | BaseException) -> None:
        self._items = list(items)
        self.calls: list[
            tuple[list[dict[str, Any]], list[dict[str, Any]]]
        ] = []

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelReply:
        self.calls.append((copy.deepcopy(messages), tools))
        if not self._items:
            raise AssertionError("scripted model exhausted")
        item = self._items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class RecordingTraceStore(TraceStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.writers: list[TraceWriter] = []

    def create(self, run_id: str) -> TraceWriter:
        writer = super().create(run_id)
        self.writers.append(writer)
        return writer


class CountingWorkspaceTools(WorkspaceTools):
    def __init__(
        self,
        guard: WorkspaceGuard,
        *,
        max_read_bytes: int,
        max_write_bytes: int,
    ) -> None:
        super().__init__(
            guard,
            max_read_bytes=max_read_bytes,
            max_write_bytes=max_write_bytes,
        )
        self.executed: list[tuple[str, dict[str, Any]]] = []

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
    ):
        self.executed.append((name, copy.deepcopy(arguments)))
        return super().execute(name, arguments)


def _workspace_tools(
    tmp_path: Path,
    tool_type: type[WorkspaceTools] = WorkspaceTools,
) -> WorkspaceTools:
    root = tmp_path / "workspace"
    root.mkdir(exist_ok=True)
    return tool_type(
        WorkspaceGuard(root),
        max_read_bytes=1024,
        max_write_bytes=1024,
    )


def _tool_reply(
    *calls: ToolCall,
    usage: Usage = Usage(),
) -> ModelReply:
    raw_calls = [
        {
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": json.dumps(
                    call.arguments,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        }
        for call in calls
    ]
    return ModelReply(
        message={
            "role": "assistant",
            "content": None,
            "tool_calls": raw_calls,
        },
        tool_calls=tuple(calls),
        usage=usage,
    )


def _final_reply(
    content: str | None,
    *,
    usage: Usage = Usage(),
) -> ModelReply:
    return ModelReply(
        message={"role": "assistant", "content": content},
        usage=usage,
    )


def _event_types(events: list[dict[str, Any]]) -> list[str]:
    return [event["type"] for event in events]


@pytest.mark.asyncio
async def test_agent_loop_executes_tool_then_final_and_closes_trace(
    tmp_path: Path,
) -> None:
    tools = _workspace_tools(tmp_path)
    (tools.guard.root / "note.txt").write_text("hello 资料", encoding="utf-8")
    call = ToolCall(
        id="call-1",
        name="read_file",
        arguments={"path": "note.txt", "limit": 100},
    )
    raw_tool_message = _tool_reply(
        call,
        usage=Usage(4, 2, 6),
    )
    model = ScriptedModel(
        raw_tool_message,
        _final_reply("The file says hello.", usage=Usage(3, 4, 7)),
    )
    traces = RecordingTraceStore(tmp_path / "traces")
    events: list[dict[str, Any]] = []
    runner = AgentRunner(model, tools, traces, max_model_calls=4)

    result = await runner.run("Read note.txt", events.append)

    assert result.status == "completed"
    assert result.message == "The file says hello."
    assert result.model_calls == 2
    assert result.usage == Usage(7, 6, 13)
    assert events == [
        {"type": "run_started", "run_id": result.run_id},
        {
            "type": "model_call_started",
            "run_id": result.run_id,
            "call": 1,
        },
        {
            "type": "usage_updated",
            "run_id": result.run_id,
            "model_calls": 1,
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 6,
            },
        },
        {
            "type": "tool_started",
            "run_id": result.run_id,
            "step": 1,
            "tool": "read_file",
            "args": {"path": "note.txt", "limit": 100},
        },
        {
            "type": "tool_finished",
            "run_id": result.run_id,
            "step": 1,
            "tool": "read_file",
            "ok": True,
            "result_summary": "Read 12 bytes from note.txt",
        },
        {
            "type": "model_call_started",
            "run_id": result.run_id,
            "call": 2,
        },
        {
            "type": "usage_updated",
            "run_id": result.run_id,
            "model_calls": 2,
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 6,
                "total_tokens": 13,
            },
        },
        {
            "type": "assistant_message",
            "run_id": result.run_id,
            "content": "The file says hello.",
        },
        {
            "type": "run_completed",
            "run_id": result.run_id,
            "status": "completed",
            "message": "The file says hello.",
            "model_calls": 2,
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 6,
                "total_tokens": 13,
            },
        },
    ]
    second_messages, sent_tools = model.calls[1]
    assert sent_tools is TOOL_SCHEMAS
    assert second_messages[:2] == [
        {"role": "system", "content": SYSTEM_POLICY},
        {"role": "user", "content": "Read note.txt"},
    ]
    assert second_messages[2] == raw_tool_message.message
    assert second_messages[3]["role"] == "tool"
    assert second_messages[3]["tool_call_id"] == "call-1"
    tool_payload = json.loads(second_messages[3]["content"])
    assert tool_payload["trust_boundary"] == "UNTRUSTED_WORKSPACE_DATA"
    assert tool_payload["ok"] is True
    assert tool_payload["data"]["content"] == "hello 资料"

    writer = traces.writers[0]
    records = [
        json.loads(line)
        for line in writer.path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["tool"] == "read_file"
    assert records[0]["args"] == {"path": "note.txt", "limit": 100}
    assert records[0]["status"] == "success"
    with pytest.raises(ValueError, match="closed"):
        writer.append(
            step=2,
            tool="read_file",
            args={},
            result_summary="should fail",
            status="ok",
        )


@pytest.mark.asyncio
async def test_agent_loop_runs_multiple_tools_sequentially_and_recovers(
    tmp_path: Path,
) -> None:
    tools = _workspace_tools(tmp_path)
    (tools.guard.root / "a.txt").write_text("alpha", encoding="utf-8")
    model = ScriptedModel(
        _tool_reply(
            ToolCall("call-1", "read_file", {"path": "a.txt", "limit": 20}),
            ToolCall("call-2", "stat_path", {"path": "missing.txt"}),
        ),
        _final_reply("I recovered from the missing path."),
    )
    traces = RecordingTraceStore(tmp_path / "traces")
    events: list[dict[str, Any]] = []

    result = await AgentRunner(
        model,
        tools,
        traces,
        max_model_calls=3,
    ).run("Inspect two paths", events.append)

    assert result.status == "completed"
    tool_messages = [
        message
        for message in model.calls[1][0]
        if message["role"] == "tool"
    ]
    assert [message["tool_call_id"] for message in tool_messages] == [
        "call-1",
        "call-2",
    ]
    first_payload, second_payload = [
        json.loads(message["content"]) for message in tool_messages
    ]
    assert first_payload["ok"] is True
    assert second_payload["ok"] is False
    assert second_payload["error_code"] == "NOT_FOUND"
    assert _event_types(events)[3:7] == [
        "tool_started",
        "tool_finished",
        "tool_started",
        "tool_finished",
    ]


@pytest.mark.asyncio
async def test_agent_loop_accumulates_usage_and_propagates_unknowns(
    tmp_path: Path,
) -> None:
    model = ScriptedModel(
        _tool_reply(
            ToolCall("call-1", "stat_path", {"path": "."}),
            usage=Usage(2, 3, 5),
        ),
        _final_reply(
            "Done",
            usage=Usage(4, None, None),
        ),
    )
    events: list[dict[str, Any]] = []

    result = await AgentRunner(
        model,
        _workspace_tools(tmp_path),
        RecordingTraceStore(tmp_path / "traces"),
        max_model_calls=3,
    ).run("Inspect the root", events.append)

    assert result.usage == Usage(6, None, None)
    usage_events = [
        event["usage"]
        for event in events
        if event["type"] == "usage_updated"
    ]
    assert usage_events == [
        {
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "total_tokens": 5,
        },
        {
            "prompt_tokens": 6,
            "completion_tokens": None,
            "total_tokens": None,
        },
    ]


@pytest.mark.asyncio
async def test_agent_loop_blocks_third_identical_consecutive_tool_execution(
    tmp_path: Path,
) -> None:
    repeated = [
        _tool_reply(
            ToolCall(
                f"call-{number}",
                "stat_path",
                {"path": "missing.txt"},
            )
        )
        for number in range(1, 4)
    ]
    model = ScriptedModel(*repeated, _final_reply("Stopped repeating."))
    tools = _workspace_tools(tmp_path, CountingWorkspaceTools)
    traces = RecordingTraceStore(tmp_path / "traces")
    events: list[dict[str, Any]] = []

    result = await AgentRunner(
        model,
        tools,
        traces,
        max_model_calls=5,
    ).run("Inspect one path", events.append)

    assert result.status == "failed"
    assert "repeated" in result.message.lower()
    assert isinstance(tools, CountingWorkspaceTools)
    assert len(tools.executed) == 2
    assert len(model.calls) == 3
    assert events[-1] == {
        "type": "run_failed",
        "run_id": result.run_id,
        "status": "failed",
        "message": result.message,
        "model_calls": 3,
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
    assert _event_types(events).count("tool_started") == 2
    assert _event_types(events).count("tool_finished") == 2
    records = [
        json.loads(line)
        for line in traces.writers[0].path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [record["status"] for record in records] == ["error", "error"]


@pytest.mark.asyncio
async def test_agent_loop_retries_two_empty_replies_then_succeeds(
    tmp_path: Path,
) -> None:
    model = ScriptedModel(
        _final_reply(""),
        _final_reply("   "),
        _final_reply("Useful final"),
    )

    result = await AgentRunner(
        model,
        _workspace_tools(tmp_path),
        RecordingTraceStore(tmp_path / "traces"),
        max_model_calls=4,
    ).run("Give a result", lambda event: None)

    assert result.status == "completed"
    assert result.model_calls == 3
    assert model.calls[1][0][-1]["role"] == "user"
    assert "non-empty" in model.calls[1][0][-1]["content"]
    assert model.calls[2][0][-1]["role"] == "user"
    assert "non-empty" in model.calls[2][0][-1]["content"]


@pytest.mark.asyncio
async def test_agent_loop_fails_after_three_empty_replies(
    tmp_path: Path,
) -> None:
    model = ScriptedModel(
        _final_reply(None),
        _final_reply(""),
        _final_reply(" \n "),
    )
    traces = RecordingTraceStore(tmp_path / "traces")
    events: list[dict[str, Any]] = []

    result = await AgentRunner(
        model,
        _workspace_tools(tmp_path),
        traces,
        max_model_calls=4,
    ).run("Give a result", events.append)

    assert result.status == "failed"
    assert result.model_calls == 3
    assert _event_types(events)[-1] == "run_failed"
    with pytest.raises(ValueError, match="closed"):
        traces.writers[0].append(
            step=1,
            tool="stat_path",
            args={},
            result_summary="closed",
            status="error",
        )


@pytest.mark.asyncio
async def test_agent_loop_fails_explicitly_at_model_call_limit(
    tmp_path: Path,
) -> None:
    model = ScriptedModel(
        _tool_reply(ToolCall("call-1", "stat_path", {"path": "."}))
    )
    events: list[dict[str, Any]] = []

    result = await AgentRunner(
        model,
        _workspace_tools(tmp_path),
        RecordingTraceStore(tmp_path / "traces"),
        max_model_calls=1,
    ).run("Inspect root", events.append)

    assert result.status == "failed"
    assert result.model_calls == 1
    assert "limit" in result.message.lower()
    assert _event_types(events)[-1] == "run_failed"


@pytest.mark.asyncio
async def test_agent_loop_sanitizes_model_exception(
    tmp_path: Path,
) -> None:
    model = ScriptedModel(
        RuntimeError(
            "key=sk-private root=C:/secret response=private-body"
        )
    )
    events: list[dict[str, Any]] = []

    result = await AgentRunner(
        model,
        _workspace_tools(tmp_path),
        RecordingTraceStore(tmp_path / "traces"),
        max_model_calls=2,
    ).run("Inspect root", events.append)

    assert result.status == "failed"
    assert result.message == "Model call failed: RuntimeError"
    assert events == [
        {"type": "run_started", "run_id": result.run_id},
        {
            "type": "model_call_started",
            "run_id": result.run_id,
            "call": 1,
        },
        {
            "type": "run_failed",
            "run_id": result.run_id,
            "status": "failed",
            "message": "Model call failed: RuntimeError",
            "model_calls": 1,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        },
    ]
    serialized = json.dumps(events)
    assert "sk-private" not in serialized
    assert "C:/secret" not in serialized
    assert "private-body" not in serialized


@pytest.mark.asyncio
async def test_agent_loop_accepts_async_event_sink(tmp_path: Path) -> None:
    events: list[dict[str, Any]] = []

    async def sink(event: dict[str, Any]) -> None:
        events.append(event)

    result = await AgentRunner(
        ScriptedModel(_final_reply("Done")),
        _workspace_tools(tmp_path),
        RecordingTraceStore(tmp_path / "traces"),
        max_model_calls=2,
    ).run("Finish", sink)

    assert result.status == "completed"
    assert _event_types(events) == [
        "run_started",
        "model_call_started",
        "usage_updated",
        "assistant_message",
        "run_completed",
    ]


@pytest.mark.asyncio
async def test_agent_loop_propagates_sink_failure_and_closes_trace(
    tmp_path: Path,
) -> None:
    class SinkDisconnected(Exception):
        pass

    events: list[dict[str, Any]] = []

    def sink(event: dict[str, Any]) -> None:
        events.append(event)
        if event["type"] == "model_call_started":
            raise SinkDisconnected("browser disconnected")

    traces = RecordingTraceStore(tmp_path / "traces")
    runner = AgentRunner(
        ScriptedModel(_final_reply("must not complete")),
        _workspace_tools(tmp_path),
        traces,
        max_model_calls=2,
    )

    with pytest.raises(SinkDisconnected):
        await runner.run("Finish", sink)

    assert "run_completed" not in _event_types(events)
    assert "run_failed" not in _event_types(events)
    with pytest.raises(ValueError, match="closed"):
        traces.writers[0].append(
            step=1,
            tool="stat_path",
            args={},
            result_summary="closed",
            status="error",
        )


def test_system_policy_is_generic_and_marks_injection_boundary() -> None:
    assert "only the user's task" in SYSTEM_POLICY
    assert "UNTRUSTED_WORKSPACE_DATA" in SYSTEM_POLICY
    assert "list_dir" in SYSTEM_POLICY
    assert "search_files" in SYSTEM_POLICY
    assert "read_file" in SYSTEM_POLICY
    assert "stat_path" in SYSTEM_POLICY
    assert "write_file" in SYSTEM_POLICY
    assert "move_file" in SYSTEM_POLICY
    assert "has_more" in SYSTEM_POLICY
    assert "cursor" in SYSTEM_POLICY
    assert "Falcon" not in SYSTEM_POLICY
    assert "take-home" not in SYSTEM_POLICY


@pytest.mark.asyncio
async def test_agent_loop_sanitizes_write_content_in_events_and_trace(
    tmp_path: Path,
) -> None:
    content = "private write 内容"
    call = ToolCall(
        "call-write",
        "write_file",
        {
            "path": "created.txt",
            "content": content,
            "overwrite": False,
        },
    )
    events: list[dict[str, Any]] = []
    traces = RecordingTraceStore(tmp_path / "traces")

    result = await AgentRunner(
        ScriptedModel(_tool_reply(call), _final_reply("Created.")),
        _workspace_tools(tmp_path),
        traces,
        max_model_calls=3,
    ).run("Create a file", events.append)

    assert result.status == "completed"
    started = next(
        event for event in events if event["type"] == "tool_started"
    )
    assert started["args"] == {
        "path": "created.txt",
        "overwrite": False,
        "content_bytes": len(content.encode("utf-8")),
        "content_sha256": hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest(),
    }
    trace_record = json.loads(
        traces.writers[0].path.read_text(encoding="utf-8")
    )
    assert trace_record["args"] == started["args"]
    assert content not in json.dumps(events, ensure_ascii=False)
    assert content not in traces.writers[0].path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("content", "content_type", "content_size", "secret"),
    [
        ({"secret": "dict-secret"}, "dict", 1, "dict-secret"),
        (["list-secret"], "list", 1, "list-secret"),
    ],
)
@pytest.mark.asyncio
async def test_agent_loop_recovers_from_safely_redacted_malformed_write_content(
    tmp_path: Path,
    content: object,
    content_type: str,
    content_size: int,
    secret: str,
) -> None:
    call = ToolCall(
        "call-write",
        "write_file",
        {
            "path": "created.txt",
            "content": content,
            "overwrite": False,
        },
    )
    events: list[dict[str, Any]] = []
    traces = RecordingTraceStore(tmp_path / "traces")
    model = ScriptedModel(
        _tool_reply(call),
        _final_reply("Recovered from invalid arguments."),
    )

    result = await AgentRunner(
        model,
        _workspace_tools(tmp_path),
        traces,
        max_model_calls=3,
    ).run("Create a file", events.append)

    assert result.status == "completed"
    assert len(model.calls) == 2
    assert next(
        event for event in events if event["type"] == "tool_started"
    ) == {
        "type": "tool_started",
        "run_id": result.run_id,
        "step": 1,
        "tool": "write_file",
        "args": {
            "path": "created.txt",
            "overwrite": False,
            "content_type": content_type,
            "content_size": content_size,
        },
    }
    assert next(
        event for event in events if event["type"] == "tool_finished"
    ) == {
        "type": "tool_finished",
        "run_id": result.run_id,
        "step": 1,
        "tool": "write_file",
        "ok": False,
        "result_summary": "write_file failed: content must be a string",
    }
    tool_payload = json.loads(model.calls[1][0][-1]["content"])
    assert tool_payload["ok"] is False
    assert tool_payload["error_code"] == "INVALID_INPUT"
    trace_text = traces.writers[0].path.read_text(encoding="utf-8")
    trace_record = json.loads(trace_text)
    assert trace_record["status"] == "error"
    assert trace_record["args"]["content_type"] == content_type
    assert trace_record["args"]["content_size"] == content_size
    assert secret not in json.dumps(events, ensure_ascii=False)
    assert secret not in trace_text


@pytest.mark.asyncio
async def test_agent_loop_rejects_blank_task_before_creating_trace(
    tmp_path: Path,
) -> None:
    traces = RecordingTraceStore(tmp_path / "traces")
    runner = AgentRunner(
        ScriptedModel(_final_reply("unused")),
        _workspace_tools(tmp_path),
        traces,
        max_model_calls=2,
    )

    with pytest.raises(ValueError, match="task"):
        await runner.run(" \n ", lambda event: None)

    assert traces.writers == []
