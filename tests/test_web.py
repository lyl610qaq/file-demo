from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect, WebSocketState

from workspace_agent.config import Settings
from workspace_agent.demo_data import materialize_demo_seed
from workspace_agent.schemas import ModelReply, ToolCall, ToolResult, Usage
from workspace_agent.web import SlidingWindowLimiter, create_app


class FinalOnlyModel:
    async def complete(self, messages, tools) -> ModelReply:
        return ModelReply(
            message={"role": "assistant", "content": "Finished"},
            usage=Usage(5, 1, 6),
        )


class ScriptedModel:
    def __init__(self, *replies: ModelReply) -> None:
        self.replies = deque(replies)

    async def complete(self, messages, tools) -> ModelReply:
        return self.replies.popleft()


class BlockingModel:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    async def complete(self, messages, tools) -> ModelReply:
        self.started.set()
        await asyncio.to_thread(self.release.wait)
        return ModelReply(message={"role": "assistant", "content": "Finished"})


class SlowModel:
    def __init__(self) -> None:
        self.cancelled = threading.Event()

    async def complete(self, messages, tools) -> ModelReply:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        raise AssertionError("slow model unexpectedly completed")


class ClosableModel(FinalOnlyModel):
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class FailingModel:
    async def complete(self, messages, tools) -> ModelReply:
        raise RuntimeError("api-key=server-secret path=C:/private")


class SimulatedCrash(BaseException):
    pass


def settings_for(tmp_path: Path, **overrides: Any) -> Settings:
    workspace = tmp_path / "workspace"
    seed = tmp_path / "seed"
    static = tmp_path / "static"
    workspace.mkdir(parents=True)
    seed.mkdir(parents=True)
    static.mkdir(parents=True)
    (workspace / "a.md").write_text("body", encoding="utf-8")
    (seed / "seed.md").write_text("seed", encoding="utf-8")
    (static / "index.html").write_text("<main>app</main>", encoding="utf-8")
    (static / "app.js").write_text("window.loaded = true", encoding="utf-8")
    values: dict[str, Any] = {
        "workspace_root": workspace,
        "seed_root": seed,
        "trace_root": tmp_path / "traces",
        "static_root": static,
        "allowed_origin": "http://testserver",
        "rate_limit_per_minute": 20,
        "max_concurrent_runs": 1,
    }
    values.update(overrides)
    return Settings(**values)


def empty_settings_for(tmp_path: Path, **overrides: Any) -> Settings:
    settings = settings_for(tmp_path, **overrides)
    (settings.workspace_root / "a.md").unlink()
    return settings


def tool_reply(call: ToolCall) -> ModelReply:
    return ModelReply(
        message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
            ],
        },
        tool_calls=(call,),
    )


def receive_until_terminal(socket) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    while True:
        event = socket.receive_json()
        events.append(event)
        if event["type"] in {"run_completed", "run_failed"}:
            return events


def post_reset(
    client: TestClient | AsyncClient,
    *,
    origin: str | None = "http://testserver",
):
    headers = {} if origin is None else {"origin": origin}
    return client.post("/api/reset", headers=headers)


def reset_artifacts(parent: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in parent.iterdir()
            if path.name.startswith(".workspace-reset-")
        ),
        key=lambda path: path.name,
    )


def websocket_route_endpoint(app):
    return next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/ws/agent"
    )


class RouteSocket:
    def __init__(
        self,
        first_frame: dict[str, Any],
        *,
        raise_after_send: bool = False,
    ) -> None:
        self.headers = {"origin": "http://testserver"}
        self.client = SimpleNamespace(host="127.0.0.1")
        self.application_state = WebSocketState.CONNECTING
        self.first_frame = first_frame
        self.raise_after_send = raise_after_send
        self.accepted = False
        self.sent: list[dict[str, Any]] = []
        self.close_codes: list[int] = []

    async def accept(self) -> None:
        self.accepted = True
        self.application_state = WebSocketState.CONNECTED

    async def receive(self) -> dict[str, Any]:
        return self.first_frame

    async def send_json(self, event: dict[str, Any]) -> None:
        self.sent.append(event)
        if self.raise_after_send:
            raise OSError("transport accepted event before failing")

    async def close(self, code: int, reason: str = "") -> None:
        self.close_codes.append(code)
        self.application_state = WebSocketState.DISCONNECTED


def test_index_assets_health_meta_tree_file_and_reset(tmp_path: Path) -> None:
    app = create_app(settings_for(tmp_path), model=FinalOnlyModel())

    with TestClient(app) as client:
        assert client.get("/").text == "<main>app</main>"
        assert "window.loaded" in client.get("/assets/app.js").text
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/api/meta").json() == {
            "model": "gpt-4.1-mini",
            "configured": True,
            "max_run_seconds": 300.0,
        }
        assert client.get("/api/tree").json()["entries"][0]["path"] == "a.md"
        assert client.get("/api/file", params={"path": "a.md"}).json()[
            "content"
        ] == "body"
        assert post_reset(client).json() == {"status": "reset"}
        assert client.get(
            "/api/file", params={"path": "seed.md"}
        ).status_code == 200
        assert client.get("/api/file", params={"path": "a.md"}).status_code == 404


def test_html_static_and_api_responses_include_security_headers(
    tmp_path: Path,
) -> None:
    static_root = Path("static").resolve()
    app = create_app(
        settings_for(tmp_path, static_root=static_root),
        model=FinalOnlyModel(),
    )

    with TestClient(app) as client:
        responses = (
            client.get("/"),
            client.get("/assets/app.js"),
            client.get("/api/meta"),
        )

    required_csp = (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "connect-src 'self' ws: wss:; img-src 'self' data:; "
        "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; "
        "form-action 'self'"
    )
    for response in responses:
        assert response.status_code == 200
        assert response.headers["content-security-policy"] == required_csp
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["referrer-policy"] == "no-referrer"
        assert "camera=()" in response.headers["permissions-policy"]
        assert "microphone=()" in response.headers["permissions-policy"]
        assert "geolocation=()" in response.headers["permissions-policy"]


def test_tree_exposes_authenticated_pagination(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    for name in ("b.md", "c.md"):
        (settings.workspace_root / name).write_text(name, encoding="utf-8")
    app = create_app(settings, model=FinalOnlyModel())

    with TestClient(app) as client:
        first = client.get(
            "/api/tree",
            params={"path": ".", "recursive": True, "limit": 1},
        ).json()
        second = client.get(
            "/api/tree",
            params={
                "path": ".",
                "recursive": True,
                "limit": 1,
                "cursor": first["next_cursor"],
            },
        ).json()

    assert first["has_more"] is True
    assert first["next_cursor"]
    assert first["entries"][0]["path"] == "a.md"
    assert second["entries"][0]["path"] == "b.md"
    assert second["entries"] != first["entries"]


@pytest.mark.parametrize(
    ("path", "status", "error_code"),
    [
        ("missing.md", 404, "NOT_FOUND"),
        ("../outside.md", 400, "PATH_REJECTED"),
    ],
)
def test_file_errors_use_stable_http_contract(
    tmp_path: Path,
    path: str,
    status: int,
    error_code: str,
) -> None:
    app = create_app(settings_for(tmp_path), model=FinalOnlyModel())

    with TestClient(app) as client:
        response = client.get("/api/file", params={"path": path})

    assert response.status_code == status
    assert response.json() == {
        "detail": {
            "error_code": error_code,
            "message": "File request failed",
        }
    }
    assert str(tmp_path) not in response.text


def test_trace_download_validates_run_id_and_stays_in_trace_root(
    tmp_path: Path,
) -> None:
    app = create_app(settings_for(tmp_path), model=FinalOnlyModel())

    with TestClient(app) as client:
        traversal = client.get("/api/runs/..%2Fsecret/trace")
        malformed = client.get("/api/runs/not_valid/trace")
        missing = client.get("/api/runs/valid-id/trace")

    assert traversal.status_code in {400, 404}
    assert malformed.status_code == 400
    assert malformed.json()["detail"]["error_code"] == "INVALID_RUN_ID"
    assert missing.status_code == 404
    assert str(tmp_path) not in malformed.text


def test_websocket_emits_runner_events_unchanged_and_in_order(
    tmp_path: Path,
) -> None:
    app = create_app(settings_for(tmp_path), model=FinalOnlyModel())

    with TestClient(app) as client:
        with client.websocket_connect(
            "/ws/agent",
            headers={"origin": "http://testserver"},
        ) as socket:
            socket.send_json({"type": "run", "task": "Inspect files"})
            events = receive_until_terminal(socket)

        run_id = events[0]["run_id"]
        trace_response = client.get(f"/api/runs/{run_id}/trace")

    assert [event["type"] for event in events] == [
        "run_started",
        "model_call_started",
        "usage_updated",
        "assistant_message",
        "run_completed",
    ]
    assert events[2]["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 1,
        "total_tokens": 6,
    }
    assert trace_response.status_code == 200
    assert trace_response.headers["content-type"].startswith(
        "application/x-ndjson"
    )


def test_websocket_rejects_wrong_origin_before_accept(tmp_path: Path) -> None:
    app = create_app(settings_for(tmp_path), model=FinalOnlyModel())

    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as rejected:
            with client.websocket_connect(
                "/ws/agent",
                headers={"origin": "http://wrong-origin.test"},
            ):
                pass

    assert rejected.value.code == 1008


@pytest.mark.parametrize("allowed_origin", ["", " ", "\t\r\n"])
def test_create_app_rejects_blank_allowed_origin(
    tmp_path: Path,
    allowed_origin: str,
) -> None:
    settings = settings_for(tmp_path, allowed_origin=allowed_origin)

    with pytest.raises(ValueError) as rejected:
        create_app(settings, model=FinalOnlyModel())

    assert str(rejected.value) == "allowed_origin must be a valid HTTP origin"


@pytest.mark.parametrize(
    "allowed_origin",
    [
        "ftp://testserver",
        "http://user:configuration-secret@testserver",
        "http://testserver/path",
        "http://testserver?query=secret",
        "http://testserver#fragment-secret",
        "http://testserver:not-a-port",
        "http://",
        "not-an-origin",
    ],
)
def test_create_app_rejects_malformed_allowed_origin_without_leaking_it(
    tmp_path: Path,
    allowed_origin: str,
) -> None:
    settings = settings_for(tmp_path, allowed_origin=allowed_origin)

    with pytest.raises(ValueError) as rejected:
        create_app(settings, model=FinalOnlyModel())

    assert str(rejected.value) == "allowed_origin must be a valid HTTP origin"
    assert "configuration-secret" not in str(rejected.value)
    assert "fragment-secret" not in str(rejected.value)


def test_websocket_compares_normalized_origin_components(tmp_path: Path) -> None:
    app = create_app(
        settings_for(tmp_path, allowed_origin="HTTP://TESTSERVER:80"),
        model=FinalOnlyModel(),
    )

    with TestClient(app) as client:
        with client.websocket_connect(
            "/ws/agent",
            headers={"origin": "http://testserver"},
        ) as socket:
            socket.send_json({"type": "run", "task": "Inspect files"})
            events = receive_until_terminal(socket)

    assert events[-1]["type"] == "run_completed"


@pytest.mark.parametrize(
    "origin",
    [
        None,
        "null",
        "ws://testserver",
        "http://testserver.evil",
        "http://user:test@testserver",
        "http://testserver/path",
        "http://testserver?query=x",
        "http://testserver#fragment",
        "http://testserver:81",
        "not-an-origin",
    ],
)
def test_websocket_rejects_missing_malformed_and_cross_origins(
    tmp_path: Path,
    origin: str | None,
) -> None:
    app = create_app(settings_for(tmp_path), model=FinalOnlyModel())
    headers = {} if origin is None else {"origin": origin}

    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as rejected:
            with client.websocket_connect(
                "/ws/agent",
                headers=headers,
            ):
                pass

    assert rejected.value.code == 1008


def test_websocket_rate_limit_is_enforced_before_accept(tmp_path: Path) -> None:
    app = create_app(
        settings_for(tmp_path, rate_limit_per_minute=1),
        model=FinalOnlyModel(),
    )

    with TestClient(app) as client:
        with client.websocket_connect(
            "/ws/agent",
            headers={"origin": "http://testserver"},
        ) as socket:
            socket.send_json({"type": "run", "task": "First"})
            receive_until_terminal(socket)
        with pytest.raises(WebSocketDisconnect) as rejected:
            with client.websocket_connect(
                "/ws/agent",
                headers={"origin": "http://testserver"},
            ):
                pass

    assert rejected.value.code == 1013


@pytest.mark.parametrize(
    "message",
    [
        {"type": "run"},
        {"task": "valid task"},
        {"type": "run", "task": ""},
        {"type": "run", "task": " "},
        {"type": "run", "task": "x" * 4001},
        {"type": "run", "task": 123},
        {"type": "other", "task": "valid"},
    ],
)
def test_websocket_rejects_invalid_run_message(
    tmp_path: Path,
    message: dict[str, Any],
) -> None:
    app = create_app(settings_for(tmp_path), model=FinalOnlyModel())

    with TestClient(app) as client:
        with client.websocket_connect(
            "/ws/agent",
            headers={"origin": "http://testserver"},
        ) as socket:
            socket.send_json(message)
            event = socket.receive_json()
            with pytest.raises(WebSocketDisconnect) as closed:
                socket.receive_json()

    assert event == {"type": "run_failed", "message": "Invalid run request"}
    assert closed.value.code == 1008


def test_websocket_rejects_malformed_json(tmp_path: Path) -> None:
    app = create_app(settings_for(tmp_path), model=FinalOnlyModel())

    with TestClient(app) as client:
        with client.websocket_connect(
            "/ws/agent",
            headers={"origin": "http://testserver"},
        ) as socket:
            socket.send_text("{not-json")
            event = socket.receive_json()
            with pytest.raises(WebSocketDisconnect) as closed:
                socket.receive_json()

    assert event == {"type": "run_failed", "message": "Invalid run request"}
    assert closed.value.code == 1008


@pytest.mark.parametrize(
    ("frame_type", "payload"),
    [
        ("binary", b'{"type":"run","task":"binary-secret"}'),
        ("text", '"scalar-secret"'),
        ("text", '["list-secret"]'),
        ("text", '{"type":"run","task":"Inspect","extra":"extra-secret"}'),
        ("text", '{"type":"run","task":NaN}'),
    ],
)
def test_websocket_first_frame_accepts_only_strict_text_json_object(
    tmp_path: Path,
    frame_type: str,
    payload: str | bytes,
) -> None:
    app = create_app(settings_for(tmp_path), model=FinalOnlyModel())

    with TestClient(app) as client:
        with client.websocket_connect(
            "/ws/agent",
            headers={"origin": "http://testserver"},
        ) as socket:
            if frame_type == "binary":
                assert isinstance(payload, bytes)
                socket.send_bytes(payload)
            else:
                assert isinstance(payload, str)
                socket.send_text(payload)
            event = socket.receive_json()
            with pytest.raises(WebSocketDisconnect) as closed:
                socket.receive_json()

    assert event == {"type": "run_failed", "message": "Invalid run request"}
    assert closed.value.code == 1008
    serialized = json.dumps(event)
    assert "secret" not in serialized
    assert "NaN" not in serialized


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_message"),
    [
        ("invalid_frame", "Invalid run request"),
        ("model_missing", "Model is not configured"),
        ("run_slot_busy", "Server is busy"),
        ("workspace_busy", "Server is busy"),
    ],
)
async def test_route_early_failure_uses_one_terminal_after_send_oserror(
    tmp_path: Path,
    failure: str,
    expected_message: str,
) -> None:
    settings = settings_for(tmp_path)
    injected_model = None if failure == "model_missing" else FinalOnlyModel()
    app = create_app(settings, model=injected_model)
    frame = (
        {"type": "websocket.receive", "text": "{}"}
        if failure == "invalid_frame"
        else {
            "type": "websocket.receive",
            "text": json.dumps({"type": "run", "task": "Inspect"}),
        }
    )
    socket = RouteSocket(frame, raise_after_send=True)
    held_gate = None
    if failure == "run_slot_busy":
        held_gate = app.state.run_slots
    elif failure == "workspace_busy":
        held_gate = app.state.workspace_lock
    if held_gate is not None:
        assert held_gate.try_acquire()

    try:
        try:
            await websocket_route_endpoint(app)(socket)
        except OSError:
            pass
    finally:
        if held_gate is not None:
            held_gate.release()

    assert socket.accepted is True
    assert socket.sent == [
        {"type": "run_failed", "message": expected_message}
    ]


@pytest.mark.asyncio
async def test_route_outer_unexpected_error_sends_one_generic_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from workspace_agent import web

    async def fail_before_terminal(*args, **kwargs) -> None:
        raise RuntimeError("unexpected route failure")

    monkeypatch.setattr(
        web,
        "_run_with_disconnect_monitor",
        fail_before_terminal,
    )
    app = create_app(settings_for(tmp_path), model=FinalOnlyModel())
    socket = RouteSocket(
        {
            "type": "websocket.receive",
            "text": json.dumps({"type": "run", "task": "Inspect"}),
        }
    )

    await websocket_route_endpoint(app)(socket)

    assert socket.sent == [{"type": "run_failed", "message": "Run failed"}]
    assert socket.close_codes == [1011]


def test_websocket_rejects_every_frame_after_the_run_request(
    tmp_path: Path,
) -> None:
    model = BlockingModel()
    app = create_app(settings_for(tmp_path), model=model)

    with TestClient(app) as client:
        with client.websocket_connect(
            "/ws/agent",
            headers={"origin": "http://testserver"},
        ) as socket:
            socket.send_json({"type": "run", "task": "Inspect files"})
            assert socket.receive_json()["type"] == "run_started"
            assert socket.receive_json()["type"] == "model_call_started"
            assert model.started.wait(2)
            socket.send_bytes(b"second-frame-secret")
            event = socket.receive_json()
            model.release.set()
            with pytest.raises(WebSocketDisconnect) as closed:
                socket.receive_json()

    assert event == {"type": "run_failed", "message": "Invalid run request"}
    assert closed.value.code == 1008
    assert "second-frame-secret" not in json.dumps(event)


@pytest.mark.asyncio
async def test_queued_extra_frame_is_rejected_before_runner_starts() -> None:
    from workspace_agent import web

    class ExtraFrameSocket:
        def __init__(self) -> None:
            self.sent: list[dict[str, Any]] = []
            self.close_code: int | None = None

        async def receive(self) -> dict[str, Any]:
            return {
                "type": "websocket.receive",
                "text": "unexpected-second-frame-secret",
            }

        async def send_json(self, event: dict[str, Any]) -> None:
            self.sent.append(event)

        async def close(self, code: int) -> None:
            self.close_code = code

    class NeverStartedRunner:
        def __init__(self) -> None:
            self.called = False

        async def run(self, task: str, sink) -> None:
            self.called = True

    socket = ExtraFrameSocket()
    runner = NeverStartedRunner()
    channel = web._RunChannel(socket)

    await web._run_with_disconnect_monitor(
        socket,
        channel,
        runner,
        "Inspect",
    )

    assert runner.called is False
    assert socket.sent == [
        {"type": "run_failed", "message": "Invalid run request"}
    ]
    assert socket.close_code == 1008
    assert "unexpected-second-frame-secret" not in json.dumps(socket.sent)


@pytest.mark.asyncio
async def test_runner_terminal_wins_a_simultaneous_extra_frame_and_latches() -> None:
    from workspace_agent import web

    class RacingSocket:
        def __init__(self) -> None:
            self.sent: list[dict[str, Any]] = []
            self.close_code: int | None = None
            self.terminal_sent = asyncio.Event()

        async def receive(self) -> dict[str, Any]:
            await self.terminal_sent.wait()
            return {
                "type": "websocket.receive",
                "text": "simultaneous-extra-secret",
            }

        async def send_json(self, event: dict[str, Any]) -> None:
            self.sent.append(event)
            if event["type"] in {"run_completed", "run_failed"}:
                self.terminal_sent.set()
                await asyncio.sleep(0)

        async def close(self, code: int) -> None:
            self.close_code = code

    class DuplicateTerminalRunner:
        async def run(self, task: str, sink) -> None:
            await sink({"type": "run_completed", "message": "done"})
            await sink({"type": "run_failed", "message": "duplicate"})

    socket = RacingSocket()
    channel = web._RunChannel(socket)

    await web._run_with_disconnect_monitor(
        socket,
        channel,
        DuplicateTerminalRunner(),
        "Inspect",
    )

    assert socket.sent == [{"type": "run_completed", "message": "done"}]
    assert socket.close_code == 1000
    assert "simultaneous-extra-secret" not in json.dumps(socket.sent)


@pytest.mark.asyncio
async def test_terminal_slot_stays_consumed_when_transport_raises() -> None:
    from workspace_agent import web

    class RaiseAfterSendSocket:
        def __init__(self) -> None:
            self.sent: list[dict[str, Any]] = []

        async def send_json(self, event: dict[str, Any]) -> None:
            self.sent.append(event)
            if event["type"] == "run_completed":
                raise RuntimeError("transport outcome is uncertain")

        async def close(self, code: int) -> None:
            return None

    socket = RaiseAfterSendSocket()
    channel = web._RunChannel(socket)

    with pytest.raises(RuntimeError, match="transport outcome"):
        await channel.send_runner_event(
            {"type": "run_completed", "message": "done"}
        )
    sent_failure = await channel.send_failure(
        "Run failed",
        close_code=1011,
    )

    assert sent_failure is False
    assert socket.sent == [{"type": "run_completed", "message": "done"}]


@pytest.mark.asyncio
async def test_concurrent_terminal_sender_cannot_replace_cancelled_send() -> None:
    from workspace_agent import web

    class CancelAfterSendSocket:
        def __init__(self) -> None:
            self.sent: list[dict[str, Any]] = []
            self.first_sent = asyncio.Event()
            self.never = asyncio.Event()

        async def send_json(self, event: dict[str, Any]) -> None:
            self.sent.append(event)
            if len(self.sent) == 1:
                self.first_sent.set()
                await self.never.wait()

        async def close(self, code: int) -> None:
            return None

    socket = CancelAfterSendSocket()
    channel = web._RunChannel(socket)
    first = asyncio.create_task(
        channel.send_runner_event(
            {"type": "run_completed", "message": "done"}
        )
    )
    await socket.first_sent.wait()
    competing = asyncio.create_task(
        channel.send_failure("Run failed", close_code=1011)
    )
    await asyncio.sleep(0)

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    competing_result = await competing

    assert competing_result is False
    assert socket.sent == [{"type": "run_completed", "message": "done"}]


def test_websocket_run_timeout_cancels_model_and_sends_one_terminal(
    tmp_path: Path,
) -> None:
    model = SlowModel()
    app = create_app(
        settings_for(tmp_path, max_run_seconds=0.05),
        model=model,
    )

    with TestClient(app) as client:
        with client.websocket_connect(
            "/ws/agent",
            headers={"origin": "http://testserver"},
        ) as socket:
            socket.send_json({"type": "run", "task": "Wait forever"})
            events = receive_until_terminal(socket)
            with pytest.raises(WebSocketDisconnect) as closed:
                socket.receive_json()

    terminals = [
        event
        for event in events
        if event["type"] in {"run_completed", "run_failed"}
    ]
    assert terminals == [{"type": "run_failed", "message": "Run timed out"}]
    assert closed.value.code == 1011
    assert model.cancelled.wait(2)


def test_websocket_timeout_waits_for_tool_before_releasing_workspace(
    tmp_path: Path,
) -> None:
    content = "committed by timed out run"
    call = ToolCall(
        "call-timeout",
        "write_file",
        {
            "path": "timeout-write.txt",
            "content": content,
            "overwrite": False,
        },
    )
    app = create_app(
        settings_for(tmp_path, max_run_seconds=0.05),
        model=ScriptedModel(tool_reply(call)),
    )
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    original_execute = app.state.tools.execute

    def blocking_execute(name: str, arguments: dict[str, Any]) -> ToolResult:
        started.set()
        release.wait()
        try:
            return original_execute(name, arguments)
        finally:
            finished.set()

    app.state.tools.execute = blocking_execute

    with TestClient(app) as client:
        with client.websocket_connect(
            "/ws/agent",
            headers={"origin": "http://testserver"},
        ) as socket:
            socket.send_json({"type": "run", "task": "Use a slow tool"})
            while socket.receive_json()["type"] != "tool_started":
                pass
            assert started.wait(2)
            threading.Timer(0.15, release.set).start()
            terminal = socket.receive_json()
            assert finished.is_set()

        written_content = (
            app.state.tools.guard.root / "timeout-write.txt"
        ).read_text(encoding="utf-8")
        reset = post_reset(client)

    trace_files = list((tmp_path / "traces").glob("*.jsonl"))
    assert len(trace_files) == 1
    record = json.loads(trace_files[0].read_text(encoding="utf-8"))
    assert terminal == {"type": "run_failed", "message": "Run timed out"}
    assert written_content == content
    assert record["status"] == "success_after_cancel"
    assert record["result_summary"] == (
        f"Wrote {len(content.encode('utf-8'))} bytes to timeout-write.txt"
    )
    assert record["args"] == {
        "path": "timeout-write.txt",
        "overwrite": False,
        "content_bytes": len(content.encode("utf-8")),
        "content_sha256": hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest(),
    }
    assert content not in trace_files[0].read_text(encoding="utf-8")
    assert reset.status_code == 200


@pytest.mark.asyncio
async def test_run_timeout_cancels_a_stuck_event_send() -> None:
    from workspace_agent import web

    class StuckSendSocket:
        def __init__(self) -> None:
            self.sent: list[dict[str, Any]] = []
            self.close_code: int | None = None
            self.first_send_cancelled = asyncio.Event()
            self.never = asyncio.Event()
            self.calls = 0

        async def receive(self) -> dict[str, Any]:
            await self.never.wait()
            raise AssertionError("receive unexpectedly resumed")

        async def send_json(self, event: dict[str, Any]) -> None:
            self.calls += 1
            if self.calls == 1:
                try:
                    await self.never.wait()
                except asyncio.CancelledError:
                    self.first_send_cancelled.set()
                    raise
            self.sent.append(event)

        async def close(self, code: int) -> None:
            self.close_code = code

    class EventOnlyRunner:
        async def run(self, task: str, sink) -> None:
            await sink({"type": "run_started"})

    socket = StuckSendSocket()
    channel = web._RunChannel(socket)
    await asyncio.wait_for(
        web._run_with_disconnect_monitor(
            socket,
            channel,
            EventOnlyRunner(),
            "Inspect",
            max_run_seconds=0.05,
        ),
        timeout=1,
    )

    assert socket.first_send_cancelled.is_set()
    assert socket.sent == [{"type": "run_failed", "message": "Run timed out"}]
    assert socket.close_code == 1011


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_runner_before_gate_release() -> None:
    from workspace_agent import web

    class WaitingSocket:
        def __init__(self) -> None:
            self.never = asyncio.Event()

        async def receive(self) -> dict[str, Any]:
            await self.never.wait()
            raise AssertionError("receive unexpectedly resumed")

        async def send_json(self, event: dict[str, Any]) -> None:
            return None

        async def close(self, code: int) -> None:
            return None

    class SettlingRunner:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelling = asyncio.Event()
            self.release = asyncio.Event()
            self.settled = asyncio.Event()

        async def run(self, task: str, sink) -> None:
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelling.set()
                while not self.release.is_set():
                    try:
                        await asyncio.shield(self.release.wait())
                    except asyncio.CancelledError:
                        continue
                self.settled.set()
                raise

    gate = web._CapacityGate(1)
    assert gate.try_acquire()
    runner = SettlingRunner()
    socket = WaitingSocket()
    channel = web._RunChannel(socket)
    order: list[str] = []

    async def run_with_gate() -> None:
        try:
            await web._run_with_disconnect_monitor(
                socket,
                channel,
                runner,
                "Inspect",
                max_run_seconds=0.05,
            )
        finally:
            order.append("gate_release")
            gate.release()

    run_task = asyncio.create_task(run_with_gate())
    await runner.started.wait()
    await runner.cancelling.wait()
    run_task.cancel()
    await asyncio.sleep(0)
    run_task.cancel()
    await asyncio.sleep(0)

    finished_early = run_task.done()
    entered_early = gate.try_acquire()
    if entered_early:
        gate.release()

    runner.release.set()
    with pytest.raises(asyncio.CancelledError):
        await run_task

    assert finished_early is False
    assert entered_early is False
    assert runner.settled.is_set()
    assert order == ["gate_release"]
    assert gate.try_acquire()
    gate.release()


def test_websocket_reports_missing_model_without_secrets(tmp_path: Path) -> None:
    settings = settings_for(tmp_path, llm_api_key="")
    app = create_app(settings)

    with TestClient(app) as client:
        with client.websocket_connect(
            "/ws/agent",
            headers={"origin": "http://testserver"},
        ) as socket:
            socket.send_json({"type": "run", "task": "Inspect files"})
            event = socket.receive_json()

    assert event == {
        "type": "run_failed",
        "message": "Model is not configured",
    }
    assert "LLM_API_KEY" not in json.dumps(event)


def test_websocket_sanitizes_model_failures(tmp_path: Path) -> None:
    app = create_app(settings_for(tmp_path), model=FailingModel())

    with TestClient(app) as client:
        with client.websocket_connect(
            "/ws/agent",
            headers={"origin": "http://testserver"},
        ) as socket:
            socket.send_json({"type": "run", "task": "Inspect files"})
            events = receive_until_terminal(socket)

    serialized = json.dumps(events)
    assert events[-1]["type"] == "run_failed"
    assert events[-1]["message"] == "Model call failed: RuntimeError"
    assert "server-secret" not in serialized
    assert "C:/private" not in serialized


def test_run_slot_rejects_immediately_and_reset_conflicts_with_run(
    tmp_path: Path,
) -> None:
    model = BlockingModel()
    app = create_app(settings_for(tmp_path), model=model)

    with TestClient(app) as client:
        with client.websocket_connect(
            "/ws/agent",
            headers={"origin": "http://testserver"},
        ) as first:
            first.send_json({"type": "run", "task": "Hold the workspace"})
            assert first.receive_json()["type"] == "run_started"
            assert first.receive_json()["type"] == "model_call_started"
            assert model.started.wait(2)

            busy_reads = []
            for path, params in (
                ("/api/tree", {}),
                ("/api/file", {"path": "a.md"}),
            ):
                busy_reads.append(client.get(path, params=params))

            reset = post_reset(client)

            with client.websocket_connect(
                "/ws/agent",
                headers={"origin": "http://testserver"},
            ) as second:
                second.send_json({"type": "run", "task": "Second run"})
                rejected = second.receive_json()

            model.release.set()
            remaining = receive_until_terminal(first)

    for busy_read in busy_reads:
        assert busy_read.status_code == 409
        assert busy_read.json()["detail"]["error_code"] == "WORKSPACE_BUSY"
    assert reset.status_code == 409
    assert reset.json()["detail"]["error_code"] == "WORKSPACE_BUSY"
    assert rejected == {
        "type": "run_failed",
        "message": "Server is busy",
    }
    assert remaining[-1]["type"] == "run_completed"


def test_run_is_rejected_while_reset_holds_the_workspace_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from workspace_agent import web

    started = threading.Event()
    release = threading.Event()

    def blocking_reset(settings: Settings) -> None:
        started.set()
        release.wait()

    monkeypatch.setattr(web, "_reset_workspace", blocking_reset)
    app = create_app(settings_for(tmp_path), model=FinalOnlyModel())

    with TestClient(app) as client:
        with ThreadPoolExecutor(max_workers=1) as pool:
            reset_future = pool.submit(post_reset, client)
            assert started.wait(2)
            busy_reads = []
            for path, params in (
                ("/api/tree", {}),
                ("/api/file", {"path": "a.md"}),
            ):
                busy_reads.append(client.get(path, params=params))
            with client.websocket_connect(
                "/ws/agent",
                headers={"origin": "http://testserver"},
            ) as socket:
                socket.send_json({"type": "run", "task": "Inspect"})
                event = socket.receive_json()
            release.set()
            reset_response = reset_future.result(timeout=2)

    for busy_read in busy_reads:
        assert busy_read.status_code == 409
        assert busy_read.json()["detail"]["error_code"] == "WORKSPACE_BUSY"
    assert event == {"type": "run_failed", "message": "Server is busy"}
    assert reset_response.status_code == 200


@pytest.mark.parametrize(
    "origin",
    [
        None,
        "https://cross-site.example",
        "http://testserver/path",
        "http://testserver.evil.example",
    ],
)
def test_reset_rejects_non_same_origin_without_running_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    origin: str | None,
) -> None:
    from workspace_agent import web

    calls = 0

    def tracked_reset(settings: Settings) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(web, "_reset_workspace", tracked_reset)
    app = create_app(settings_for(tmp_path), model=FinalOnlyModel())

    with TestClient(app) as client:
        response = post_reset(client, origin=origin)

    assert response.status_code == 403
    assert response.json() == {
        "detail": {
            "error_code": "ORIGIN_REJECTED",
            "message": "Origin is not allowed",
        }
    }
    assert calls == 0


def test_reset_uses_an_independent_rate_limit_namespace(
    tmp_path: Path,
) -> None:
    app = create_app(
        settings_for(tmp_path, rate_limit_per_minute=1),
        model=FinalOnlyModel(),
    )

    with TestClient(app) as client:
        with client.websocket_connect(
            "/ws/agent",
            headers={"origin": "http://testserver"},
        ) as socket:
            socket.send_json({"type": "run", "task": "Inspect"})
            assert receive_until_terminal(socket)[-1]["type"] == "run_completed"

        assert post_reset(client).status_code == 200
        limited = post_reset(client)

    assert limited.status_code == 429
    assert limited.json() == {
        "detail": {
            "error_code": "RATE_LIMITED",
            "message": "Too many reset requests",
        }
    }


def test_connection_limit_rejects_idle_socket_without_waiting(
    tmp_path: Path,
) -> None:
    app = create_app(settings_for(tmp_path), model=FinalOnlyModel())

    with TestClient(app) as client:
        with client.websocket_connect(
            "/ws/agent",
            headers={"origin": "http://testserver"},
        ):
            with client.websocket_connect(
                "/ws/agent",
                headers={"origin": "http://testserver"},
            ):
                with pytest.raises(WebSocketDisconnect) as rejected:
                    with client.websocket_connect(
                        "/ws/agent",
                        headers={"origin": "http://testserver"},
                    ):
                        pass

    assert rejected.value.code == 1013


def test_reset_removes_workspace_link_without_touching_target(
    tmp_path: Path,
) -> None:
    settings = settings_for(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    protected = outside / "protected.txt"
    protected.write_text("keep", encoding="utf-8")
    link = settings.workspace_root / "outside-link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlink unavailable: {error}")
    app = create_app(settings, model=FinalOnlyModel())

    with TestClient(app) as client:
        response = post_reset(client)

    assert response.status_code == 200
    assert protected.read_text(encoding="utf-8") == "keep"
    assert not link.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction test")
def test_reset_removes_workspace_junction_without_touching_target(
    tmp_path: Path,
) -> None:
    settings = settings_for(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    protected = outside / "protected.txt"
    protected.write_text("keep", encoding="utf-8")
    junction = settings.workspace_root / "outside-junction"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        check=False,
        capture_output=True,
        text=True,
    )
    if created.returncode != 0:
        pytest.fail(created.stderr or created.stdout)
    app = create_app(settings, model=FinalOnlyModel())

    try:
        with TestClient(app) as client:
            response = post_reset(client)
        assert response.status_code == 200
        assert protected.read_text(encoding="utf-8") == "keep"
        assert not junction.exists()
    finally:
        if os.path.lexists(junction):
            os.rmdir(junction)


def test_reset_failure_is_stable_and_does_not_leak_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from workspace_agent import web

    secret = str(tmp_path / "outside-secret")

    def fail_reset(settings: Settings) -> None:
        raise RuntimeError(f"failed at {secret}")

    monkeypatch.setattr(web, "_reset_workspace", fail_reset)
    app = create_app(settings_for(tmp_path), model=FinalOnlyModel())

    with TestClient(app) as client:
        response = post_reset(client)

    assert response.status_code == 500
    assert response.json() == {
        "detail": {
            "error_code": "RESET_FAILED",
            "message": "Workspace reset failed",
        }
    }
    assert secret not in response.text


def test_create_app_rejects_missing_seed_without_creating_it(
    tmp_path: Path,
) -> None:
    settings = settings_for(tmp_path)
    (settings.seed_root / "seed.md").unlink()
    settings.seed_root.rmdir()

    with pytest.raises(ValueError, match="^seed root is invalid$") as error:
        create_app(settings, model=FinalOnlyModel())

    assert not settings.seed_root.exists()
    assert str(tmp_path) not in str(error.value)


def test_create_app_rejects_empty_seed(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    (settings.seed_root / "seed.md").unlink()

    with pytest.raises(ValueError, match="^seed root is invalid$") as error:
        create_app(settings, model=FinalOnlyModel())

    assert str(tmp_path) not in str(error.value)


def test_create_app_rejects_seed_containing_only_empty_directories(
    tmp_path: Path,
) -> None:
    settings = settings_for(tmp_path)
    (settings.seed_root / "seed.md").unlink()
    (settings.seed_root / "empty" / "nested").mkdir(parents=True)

    with pytest.raises(ValueError, match="^seed root is invalid$") as error:
        create_app(settings, model=FinalOnlyModel())

    assert str(tmp_path) not in str(error.value)


def test_create_app_rejects_seed_without_a_readable_regular_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = settings_for(tmp_path)
    seed_file = settings.seed_root / "seed.md"
    original_open = Path.open

    def deny_seed_read(path: Path, *args: Any, **kwargs: Any):
        if path == seed_file and args and "r" in str(args[0]):
            raise PermissionError("seed is unreadable")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_seed_read)

    with pytest.raises(ValueError, match="^seed root is invalid$"):
        create_app(settings, model=FinalOnlyModel())


def test_reset_rejects_seed_with_only_empty_directories_and_preserves_workspace(
    tmp_path: Path,
) -> None:
    settings = settings_for(tmp_path)
    app = create_app(settings, model=FinalOnlyModel())
    (settings.seed_root / "seed.md").unlink()
    (settings.seed_root / "empty" / "nested").mkdir(parents=True)

    with TestClient(app) as client:
        response = post_reset(client)

    assert response.status_code == 500
    assert response.json()["detail"] == {
        "error_code": "RESET_FAILED",
        "message": "Workspace reset failed",
    }
    assert (settings.workspace_root / "a.md").read_text(
        encoding="utf-8"
    ) == "body"
    assert not (settings.workspace_root / "empty").exists()


def test_create_app_rejects_seed_root_link(tmp_path: Path) -> None:
    target = tmp_path / "seed-target"
    target.mkdir()
    (target / "seed.md").write_text("seed", encoding="utf-8")
    link = tmp_path / "seed-link"
    if os.name == "nt":
        created = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=False,
            capture_output=True,
            text=True,
        )
        if created.returncode != 0:
            pytest.skip(created.stderr or created.stdout)
    else:
        link.symlink_to(target, target_is_directory=True)

    try:
        settings = settings_for(tmp_path / "app", seed_root=link)
        with pytest.raises(
            ValueError,
            match="^seed root is invalid$",
        ) as error:
            create_app(settings, model=FinalOnlyModel())
        assert str(tmp_path) not in str(error.value)
    finally:
        if os.path.lexists(link):
            os.rmdir(link)


@pytest.mark.parametrize(
    "overrides",
    [
        lambda settings: {"seed_root": settings.workspace_root},
        lambda settings: {
            "trace_root": settings.workspace_root / "nested-traces"
        },
        lambda settings: {"static_root": settings.seed_root / "assets"},
    ],
)
def test_create_app_rejects_equal_or_containing_runtime_roots(
    tmp_path: Path,
    overrides,
) -> None:
    settings = settings_for(tmp_path)
    invalid = settings.model_copy(update=overrides(settings))

    with pytest.raises(
        ValueError,
        match="^runtime roots must be separate$",
    ) as error:
        create_app(invalid, model=FinalOnlyModel())

    assert str(tmp_path) not in str(error.value)


def test_reset_rejects_seed_link_and_preserves_old_workspace(
    tmp_path: Path,
) -> None:
    settings = settings_for(tmp_path)
    app = create_app(settings, model=FinalOnlyModel())
    outside = tmp_path / "seed-outside"
    outside.mkdir()
    (outside / "payload.md").write_text("outside", encoding="utf-8")
    link = settings.seed_root / "linked"
    if os.name == "nt":
        created = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
            check=False,
            capture_output=True,
            text=True,
        )
        if created.returncode != 0:
            pytest.skip(created.stderr or created.stdout)
    else:
        link.symlink_to(outside, target_is_directory=True)

    try:
        with TestClient(app) as client:
            response = post_reset(client)

        assert response.status_code == 500
        assert response.json()["detail"] == {
            "error_code": "RESET_FAILED",
            "message": "Workspace reset failed",
        }
        assert (settings.workspace_root / "a.md").read_text(
            encoding="utf-8"
        ) == "body"
        assert not (settings.workspace_root / "seed.md").exists()
    finally:
        if os.path.lexists(link):
            os.rmdir(link)


def test_reset_accepts_a_seed_with_an_empty_directory(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    (settings.seed_root / "empty").mkdir()
    app = create_app(settings, model=FinalOnlyModel())

    with TestClient(app) as client:
        response = post_reset(client)

    assert response.status_code == 200
    assert (settings.workspace_root / "empty").is_dir()


def test_reset_rolls_back_when_workspace_exchange_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from workspace_agent import web

    settings = settings_for(tmp_path)
    original_replace = web.os.replace

    def fail_stage_exchange(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            source_path.name.startswith(".workspace-reset-stage-")
            and destination_path == settings.workspace_root
        ):
            raise OSError(f"exchange failed at {tmp_path}")
        original_replace(source, destination)

    monkeypatch.setattr(web.os, "replace", fail_stage_exchange)
    app = create_app(settings, model=FinalOnlyModel())

    with TestClient(app) as client:
        response = post_reset(client)

    assert response.status_code == 500
    assert response.json()["detail"] == {
        "error_code": "RESET_FAILED",
        "message": "Workspace reset failed",
    }
    assert str(tmp_path) not in response.text
    assert (settings.workspace_root / "a.md").read_text(
        encoding="utf-8"
    ) == "body"
    assert not (settings.workspace_root / "seed.md").exists()
    assert not any(
        entry.name.startswith(".workspace-reset-")
        for entry in tmp_path.iterdir()
    )


def test_reset_reports_backup_cleanup_failure_without_leaking_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from workspace_agent import web

    settings = settings_for(tmp_path)
    original_cleanup = web._cleanup_reset_path
    secret = str(tmp_path / "cleanup-secret")

    def fail_backup_cleanup(path: Path) -> None:
        if path.name.startswith(".workspace-reset-backup-"):
            raise OSError(f"cleanup failed at {secret}")
        original_cleanup(path)

    monkeypatch.setattr(web, "_cleanup_reset_path", fail_backup_cleanup)
    app = create_app(settings, model=FinalOnlyModel())

    with TestClient(app) as client:
        response = post_reset(client)

    backups = [
        entry
        for entry in tmp_path.iterdir()
        if entry.name.startswith(".workspace-reset-backup-")
    ]
    try:
        assert response.status_code == 500
        assert response.json()["detail"] == {
            "error_code": "RESET_CLEANUP_FAILED",
            "message": "Workspace reset cleanup failed",
        }
        assert secret not in response.text
        assert (settings.workspace_root / "seed.md").is_file()
        assert len(backups) == 1
    finally:
        for backup in backups:
            original_cleanup(backup)


def test_create_app_recovers_user_workspace_after_crash_at_first_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from workspace_agent import web

    settings = settings_for(tmp_path)
    original_replace = web.os.replace

    def crash_after_first_rename(
        source: str | Path,
        destination: str | Path,
    ) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        original_replace(source, destination)
        if (
            source_path == settings.workspace_root
            and destination_path.name.startswith(".workspace-reset-backup-")
        ):
            raise SimulatedCrash()

    with monkeypatch.context() as patcher:
        patcher.setattr(web.os, "replace", crash_after_first_rename)
        with pytest.raises(SimulatedCrash):
            web._reset_workspace(settings)

    assert not settings.workspace_root.exists()
    assert any(path.name.endswith("journal.json") for path in reset_artifacts(tmp_path))

    create_app(settings, model=FinalOnlyModel())

    assert (settings.workspace_root / "a.md").read_text(
        encoding="utf-8"
    ) == "body"
    assert not (settings.workspace_root / "seed.md").exists()
    assert reset_artifacts(tmp_path) == []


def test_create_app_conservatively_restores_backup_after_second_rename_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from workspace_agent import web

    settings = settings_for(tmp_path)
    original_replace = web.os.replace

    def crash_after_second_rename(
        source: str | Path,
        destination: str | Path,
    ) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        original_replace(source, destination)
        if (
            source_path.name.startswith(".workspace-reset-stage-")
            and destination_path == settings.workspace_root
        ):
            raise SimulatedCrash()

    with monkeypatch.context() as patcher:
        patcher.setattr(web.os, "replace", crash_after_second_rename)
        with pytest.raises(SimulatedCrash):
            web._reset_workspace(settings)

    assert (settings.workspace_root / "seed.md").is_file()
    assert any(path.name.endswith("journal.json") for path in reset_artifacts(tmp_path))

    create_app(settings, model=FinalOnlyModel())

    assert (settings.workspace_root / "a.md").read_text(
        encoding="utf-8"
    ) == "body"
    assert not (settings.workspace_root / "seed.md").exists()
    assert reset_artifacts(tmp_path) == []


def test_create_app_finishes_installed_workspace_after_cleanup_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from workspace_agent import web

    settings = settings_for(tmp_path)
    original_cleanup = web._cleanup_reset_path

    def crash_after_backup_cleanup(path: Path) -> None:
        original_cleanup(path)
        if path.name.startswith(".workspace-reset-backup-"):
            raise SimulatedCrash()

    with monkeypatch.context() as patcher:
        patcher.setattr(web, "_cleanup_reset_path", crash_after_backup_cleanup)
        with pytest.raises(SimulatedCrash):
            web._reset_workspace(settings)

    assert (settings.workspace_root / "seed.md").is_file()
    assert any(path.name.endswith("journal.json") for path in reset_artifacts(tmp_path))

    create_app(settings, model=FinalOnlyModel())

    assert (settings.workspace_root / "seed.md").read_text(
        encoding="utf-8"
    ) == "seed"
    assert not (settings.workspace_root / "a.md").exists()
    assert reset_artifacts(tmp_path) == []


def test_create_app_rejects_traversal_in_reset_journal_without_touching_outside(
    tmp_path: Path,
) -> None:
    settings = settings_for(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    marker = outside / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    journal = tmp_path / ".workspace-reset-journal.json"
    journal.write_text(
        json.dumps(
            {
                "version": 1,
                "workspace": "workspace",
                "staging": "../outside",
                "backup": ".workspace-reset-backup-" + "a" * 32,
                "phase": "backup-created",
            }
        ),
        encoding="utf-8",
    )

    try:
        with pytest.raises(
            ValueError,
            match="^workspace recovery failed$",
        ):
            create_app(settings, model=FinalOnlyModel())
        assert marker.read_text(encoding="utf-8") == "keep"
        assert (settings.workspace_root / "a.md").is_file()
    finally:
        shutil.rmtree(outside)


def test_create_app_rejects_reparse_reset_journal(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    outside = tmp_path / "journal-target"
    outside.mkdir()
    journal = tmp_path / ".workspace-reset-journal.json"
    if os.name == "nt":
        created = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(journal), str(outside)],
            check=False,
            capture_output=True,
            text=True,
        )
        if created.returncode != 0:
            pytest.skip(created.stderr or created.stdout)
    else:
        journal.symlink_to(outside, target_is_directory=True)

    try:
        with pytest.raises(
            ValueError,
            match="^workspace recovery failed$",
        ):
            create_app(settings, model=FinalOnlyModel())
        assert (settings.workspace_root / "a.md").is_file()
    finally:
        if os.path.lexists(journal):
            if os.name == "nt":
                os.rmdir(journal)
            else:
                journal.unlink()


def test_create_app_recovers_single_legacy_backup_and_cleans_staging(
    tmp_path: Path,
) -> None:
    settings = settings_for(tmp_path)
    backup = tmp_path / (".workspace-reset-backup-" + "a" * 32)
    staging = tmp_path / (".workspace-reset-stage-" + "b" * 32)
    os.replace(settings.workspace_root, backup)
    staging.mkdir()
    (staging / "partial.md").write_text("partial", encoding="utf-8")

    create_app(settings, model=FinalOnlyModel())

    assert (settings.workspace_root / "a.md").read_text(
        encoding="utf-8"
    ) == "body"
    assert reset_artifacts(tmp_path) == []


def test_create_app_rejects_multiple_legacy_backups_without_bootstrapping(
    tmp_path: Path,
) -> None:
    settings = settings_for(tmp_path)
    first = tmp_path / (".workspace-reset-backup-" + "a" * 32)
    second = tmp_path / (".workspace-reset-backup-" + "b" * 32)
    os.replace(settings.workspace_root, first)
    second.mkdir()
    (second / "other.md").write_text("other", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="^workspace recovery failed$",
    ):
        create_app(settings, model=FinalOnlyModel())

    assert not settings.workspace_root.exists()
    assert (first / "a.md").read_text(encoding="utf-8") == "body"
    assert (second / "other.md").read_text(encoding="utf-8") == "other"


def test_create_app_cleans_legacy_backup_only_when_workspace_matches_seed(
    tmp_path: Path,
) -> None:
    settings = settings_for(tmp_path)
    backup = tmp_path / (".workspace-reset-backup-" + "a" * 32)
    staging = tmp_path / (".workspace-reset-stage-" + "b" * 32)
    os.replace(settings.workspace_root, backup)
    shutil.copytree(settings.seed_root, settings.workspace_root)
    staging.mkdir()
    (staging / "partial.md").write_text("partial", encoding="utf-8")

    create_app(settings, model=FinalOnlyModel())

    assert (settings.workspace_root / "seed.md").read_text(
        encoding="utf-8"
    ) == "seed"
    assert reset_artifacts(tmp_path) == []


def test_create_app_preserves_ambiguous_legacy_backup_and_fails_stably(
    tmp_path: Path,
) -> None:
    settings = settings_for(tmp_path)
    backup = tmp_path / (".workspace-reset-backup-" + "a" * 32)
    backup.mkdir()
    (backup / "older.md").write_text("older", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="^workspace recovery failed$",
    ):
        create_app(settings, model=FinalOnlyModel())

    assert (settings.workspace_root / "a.md").read_text(
        encoding="utf-8"
    ) == "body"
    assert (backup / "older.md").read_text(encoding="utf-8") == "older"


def test_reset_journal_records_durable_phase_progression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from workspace_agent import web

    settings = settings_for(tmp_path)
    phases: list[str] = []
    fsynced: list[Path] = []
    original_write = web._write_reset_journal
    original_fsync = web._fsync_directory

    def record_phase(parent: Path, record: dict[str, Any]) -> None:
        phases.append(record["phase"])
        original_write(parent, record)

    def record_fsync(path: Path) -> None:
        fsynced.append(path)
        original_fsync(path)

    monkeypatch.setattr(web, "_write_reset_journal", record_phase)
    monkeypatch.setattr(web, "_fsync_directory", record_fsync)

    web._reset_workspace(settings)

    assert phases == ["prepared", "backup-created", "workspace-installed"]
    assert fsynced.count(tmp_path) >= 5
    assert reset_artifacts(tmp_path) == []


@pytest.mark.asyncio
async def test_cancelled_reset_settles_worker_before_releasing_workspace_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from workspace_agent import web

    worker_started = threading.Event()
    worker_release = threading.Event()
    worker_done = threading.Event()
    order: list[str] = []

    def blocking_reset(settings: Settings) -> None:
        worker_started.set()
        try:
            worker_release.wait()
        finally:
            order.append("worker_done")
            worker_done.set()

    monkeypatch.setattr(web, "_reset_workspace", blocking_reset)
    app = create_app(settings_for(tmp_path), model=FinalOnlyModel())
    original_release = app.state.workspace_lock.release

    def tracked_lock_release() -> None:
        order.append("lock_release")
        original_release()

    app.state.workspace_lock.release = tracked_lock_release
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        reset_task = asyncio.create_task(post_reset(client))
        assert await asyncio.to_thread(worker_started.wait, 2)

        reset_task.cancel()
        await asyncio.sleep(0)
        reset_task.cancel()
        await asyncio.sleep(0)

        task_finished_early = reset_task.done()
        new_run_entered = app.state.workspace_lock.try_acquire()
        if new_run_entered:
            app.state.workspace_lock.release()

        worker_release.set()
        with pytest.raises(asyncio.CancelledError):
            await reset_task

    assert task_finished_early is False
    assert new_run_entered is False
    assert worker_done.is_set()
    assert order == ["worker_done", "lock_release"]
    assert app.state.workspace_lock.try_acquire()
    app.state.workspace_lock.release()


def test_websocket_disconnect_cancels_runner_and_waits_for_tool(
    tmp_path: Path,
) -> None:
    call = ToolCall("call-1", "stat_path", {"path": "."})
    model = ScriptedModel(tool_reply(call))
    app = create_app(settings_for(tmp_path), model=model)
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocking_execute(name: str, arguments: dict[str, Any]) -> ToolResult:
        started.set()
        release.wait()
        finished.set()
        return ToolResult(ok=True, data={}, summary="settled")

    app.state.tools.execute = blocking_execute

    with TestClient(app) as client:
        with client.websocket_connect(
            "/ws/agent",
            headers={"origin": "http://testserver"},
        ) as socket:
            socket.send_json({"type": "run", "task": "Inspect"})
            while socket.receive_json()["type"] != "tool_started":
                pass
            assert started.wait(2)
            threading.Timer(0.1, release.set).start()

        assert finished.wait(2)
        assert post_reset(client).status_code == 200

    trace_files = list((tmp_path / "traces").glob("*.jsonl"))
    assert len(trace_files) == 1
    records = [
        json.loads(line)
        for line in trace_files[0].read_text(encoding="utf-8").splitlines()
    ]
    assert records[-1]["status"] == "success_after_cancel"
    assert records[-1]["result_summary"] == "settled"


def test_app_closes_only_the_model_it_constructed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from workspace_agent import web

    owned = ClosableModel()
    factory_calls: list[dict[str, Any]] = []

    def model_factory(**kwargs: Any) -> ClosableModel:
        assert kwargs["api_key"] == "server-secret"
        factory_calls.append(kwargs)
        return owned

    monkeypatch.setattr(web, "OpenAICompatibleModel", model_factory)
    configured = settings_for(tmp_path / "owned", llm_api_key="server-secret")
    owned_app = create_app(configured)
    assert factory_calls == []
    with TestClient(owned_app):
        assert len(factory_calls) == 1
        pass
    assert owned.closed is True

    injected = ClosableModel()
    with TestClient(
        create_app(settings_for(tmp_path / "injected"), model=injected)
    ):
        pass
    assert injected.closed is False


def test_untrusted_peer_cannot_spoof_client_key_with_forwarded_for() -> None:
    from workspace_agent import web

    connection = SimpleNamespace(
        client=SimpleNamespace(host="203.0.113.9"),
        headers={"x-forwarded-for": "198.51.100.7"},
    )
    trusted = web._parse_trusted_proxy_cidrs("10.0.0.0/8")

    assert web._client_key(connection, trusted) == "203.0.113.9"


def test_trusted_proxy_keys_distinct_clients_and_walks_proxy_chain() -> None:
    from workspace_agent import web

    trusted = web._parse_trusted_proxy_cidrs(
        "10.0.0.0/8, 2001:db8:1::/48"
    )
    first = SimpleNamespace(
        client=SimpleNamespace(host="10.0.0.2"),
        headers={"x-forwarded-for": "198.51.100.10"},
    )
    second = SimpleNamespace(
        client=SimpleNamespace(host="10.0.0.2"),
        headers={"x-forwarded-for": "198.51.100.11"},
    )
    chain = SimpleNamespace(
        client=SimpleNamespace(host="2001:db8:1::2"),
        headers={
            "x-forwarded-for": "2001:db8:2::7, 10.0.0.9"
        },
    )

    assert web._client_key(first, trusted) == "198.51.100.10"
    assert web._client_key(second, trusted) == "198.51.100.11"
    assert web._client_key(chain, trusted) == "2001:db8:2::7"


@pytest.mark.parametrize(
    "forwarded_for",
    [
        "198.51.100.10, malformed",
        ",".join(["198.51.100.10"] * 21),
        "198.51.100.10" + (" " * 2048),
    ],
)
def test_malformed_or_oversized_forwarded_for_falls_back_to_peer(
    forwarded_for: str,
) -> None:
    from workspace_agent import web

    connection = SimpleNamespace(
        client=SimpleNamespace(host="10.0.0.2"),
        headers={"x-forwarded-for": forwarded_for},
    )
    trusted = web._parse_trusted_proxy_cidrs("10.0.0.0/8")

    assert web._client_key(connection, trusted) == "10.0.0.2"


def test_create_app_rejects_invalid_trusted_proxy_configuration(
    tmp_path: Path,
) -> None:
    invalid = "10.0.0.0/8,secret-invalid-cidr"

    with pytest.raises(
        ValueError,
        match="^trusted proxy configuration is invalid$",
    ) as error:
        create_app(
            settings_for(tmp_path, trusted_proxy_cidrs=invalid),
            model=FinalOnlyModel(),
        )

    assert invalid not in str(error.value)


@pytest.mark.asyncio
async def test_trusted_proxy_clients_receive_independent_reset_limits(
    tmp_path: Path,
) -> None:
    app = create_app(
        settings_for(
            tmp_path,
            rate_limit_per_minute=1,
            trusted_proxy_cidrs="10.0.0.0/8",
        ),
        model=FinalOnlyModel(),
    )
    transport = ASGITransport(app=app, client=("10.0.0.2", 41000))

    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        first = await client.post(
            "/api/reset",
            headers={
                "origin": "http://testserver",
                "x-forwarded-for": "198.51.100.10",
            },
        )
        second = await client.post(
            "/api/reset",
            headers={
                "origin": "http://testserver",
                "x-forwarded-for": "198.51.100.11",
            },
        )

    assert first.status_code == 200
    assert second.status_code == 200


def test_frontend_shell_exposes_three_pane_agent_controls(
    tmp_path: Path,
) -> None:
    static_root = Path("static").resolve()
    assert static_root.joinpath("index.html").is_file()
    settings = settings_for(tmp_path, static_root=static_root)
    app = create_app(settings, model=FinalOnlyModel())

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    html = response.text
    for control_id in (
        "model-name",
        "usage-calls",
        "usage-prompt",
        "usage-completion",
        "usage-total",
        "trace-download",
        "file-tree",
        "refresh-button",
        "reset-button",
        "task-form",
        "task-input",
        "run-button",
        "run-status",
        "assistant-output",
        "file-content",
        "trace-list",
        "reset-dialog",
    ):
        assert f'id="{control_id}"' in html
    assert 'data-pane="files"' in html
    assert 'data-pane="task"' in html
    assert 'data-pane="trace"' in html
    assert 'aria-live="polite"' in html
    assert 'lang="zh-CN"' in html
    assert 'role="tree"' not in html
    assert 'role="treeitem"' not in html
    assert (
        'id="file-tree" class="file-tree" '
        'aria-label="工作区文件" tabindex="0"'
        in html
    )
    assert (
        'class="active" data-view="task" aria-current="page"'
        in html
    )


def test_frontend_assets_are_served_and_use_safe_browser_primitives(
    tmp_path: Path,
) -> None:
    static_root = Path("static").resolve()
    settings = settings_for(tmp_path, static_root=static_root)
    app = create_app(settings, model=FinalOnlyModel())

    with TestClient(app) as client:
        script_response = client.get("/assets/app.js")
        style_response = client.get("/assets/styles.css")

    assert script_response.status_code == 200
    assert style_response.status_code == 200
    script = script_response.text
    assert "textContent" in script
    assert "innerHTML" not in script
    assert "response.text()" not in script
    assert "body.getReader()" in script
    assert "response.arrayBuffer()" not in script
    assert "TextDecoder" in script
    assert "URLSearchParams" in script
    assert 'location.protocol === "https:" ? "wss:" : "ws:"' in script
    assert "new WebSocket" in script
    assert '"/api/meta"' in script
    assert '"/api/tree"' in script
    assert '"/api/file"' in script
    assert '"/api/reset"' in script
    assert "has_more" in script
    assert "next_cursor" in script
    for event_type in (
        "run_started",
        "model_call_started",
        "usage_updated",
        "tool_started",
        "tool_finished",
        "assistant_message",
        "run_completed",
        "run_failed",
    ):
        assert f'"{event_type}"' in script


def test_frontend_uses_local_fixed_lucide_asset(tmp_path: Path) -> None:
    static_root = Path("static").resolve()
    app = create_app(
        settings_for(tmp_path, static_root=static_root),
        model=FinalOnlyModel(),
    )

    with TestClient(app) as client:
        html = client.get("/").text
        vendor = client.get("/assets/vendor/lucide.min.js")

    assert 'src="/assets/vendor/lucide.min.js"' in html
    assert "unpkg.com" not in html
    assert vendor.status_code == 200
    assert len(vendor.content) > 100_000
    assert b"lucide" in vendor.content.lower()


def test_frontend_uses_native_tree_buttons_and_initializes_mobile_pane() -> (
    None
):
    script = Path("static/app.js").read_text(encoding="utf-8")

    assert 'button.setAttribute("aria-expanded"' in script
    assert 'button.setAttribute("aria-controls"' in script
    assert (
        'child.type === "directory" && child.children.size'
        not in script
    )
    assert 'item.setAttribute("aria-expanded"' not in script
    assert 'group.setAttribute("role", "group")' not in script
    assert 'activatePane("task")' in script
    assert "node.hidden = mobile && !active" in script


def test_frontend_clears_all_usage_before_opening_a_new_run() -> None:
    script = Path("static/app.js").read_text(encoding="utf-8")
    reset_start = script.index("function resetUsage()")
    reset_end = script.index("\n  }", reset_start)
    reset_body = script[reset_start:reset_end]
    start_run = script.index("function startRun(task)")
    socket_open = script.index("new WebSocket", start_run)
    reset_call = script.index("resetUsage();", start_run)

    assert "modelCalls: 0" in reset_body
    assert "promptTokens: 0" in reset_body
    assert "completionTokens: 0" in reset_body
    assert "totalTokens: 0" in reset_body
    assert 'setText("usage-calls", "0")' in reset_body
    assert 'setText("usage-prompt", "0")' in reset_body
    assert 'setText("usage-completion", "0")' in reset_body
    assert 'setText("usage-total", "0")' in reset_body
    assert start_run < reset_call < socket_open


def test_frontend_styles_define_dense_responsive_and_reduced_motion_views() -> (
    None
):
    styles = Path("static/styles.css").read_text(encoding="utf-8")

    assert "grid-template-columns:" in styles
    assert "@media (max-width: 860px)" in styles
    assert "@media (prefers-reduced-motion: reduce)" in styles
    assert ".mobile-tabs" in styles
    assert "[data-pane]" in styles
    assert "letter-spacing: 0" in styles
    assert "border-radius: 8px" in styles or "border-radius: 6px" in styles
    assert "outline: 3px solid #0b6b63" in styles
    assert "gradient" not in styles.lower()


def test_frontend_node_runtime_contracts() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for frontend runtime tests")

    completed = subprocess.run(
        [node, "--test", "tests/frontend.test.mjs"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_sliding_window_limiter_expires_isolates_keys_and_bounds_memory() -> None:
    now = [100.0]
    limiter = SlidingWindowLimiter(
        limit=1,
        window_seconds=10,
        max_keys=2,
        clock=lambda: now[0],
    )

    assert limiter.allow("a")
    assert not limiter.allow("a")
    assert limiter.allow("b")
    now[0] = 111.0
    assert limiter.allow("a")
    assert limiter.allow("c")
    assert limiter.key_count <= 2


def test_sliding_window_limiter_is_thread_safe() -> None:
    limiter = SlidingWindowLimiter(limit=5)

    with ThreadPoolExecutor(max_workers=20) as pool:
        allowed = list(pool.map(lambda _: limiter.allow("same"), range(100)))

    assert sum(allowed) == 5


def test_default_settings_initialize_an_empty_workspace_from_demo_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    for name in (
        "WORKSPACE_ROOT",
        "SEED_ROOT",
        "TRACE_ROOT",
        "STATIC_ROOT",
        "ALLOWED_ORIGIN",
    ):
        monkeypatch.delenv(name, raising=False)
    materialize_demo_seed(tmp_path / "demo_workspace_seed")
    (tmp_path / "static").mkdir()
    (tmp_path / "static" / "index.html").write_text(
        "<main>demo</main>",
        encoding="utf-8",
    )

    app = create_app(model=FinalOnlyModel())

    assert app.state.settings.workspace_root == tmp_path / "workspace"
    assert (
        app.state.settings.workspace_root / "meetings" / "falcon-kickoff.md"
    ).is_file()
    assert len(
        [
            path
            for path in app.state.settings.workspace_root.rglob("*")
            if path.is_file()
        ]
    ) >= 30


def test_custom_settings_initialize_an_empty_workspace_from_valid_seed(
    tmp_path: Path,
) -> None:
    settings = empty_settings_for(tmp_path)

    app = create_app(settings, model=FinalOnlyModel())

    assert (settings.workspace_root / "seed.md").read_text(
        encoding="utf-8"
    ) == "seed"
    assert app.state.tools.guard.root == settings.workspace_root


def test_create_app_never_overwrites_a_nonempty_workspace(
    tmp_path: Path,
) -> None:
    settings = settings_for(tmp_path)
    nested = settings.workspace_root / "empty-directory"
    nested.mkdir()

    create_app(settings, model=FinalOnlyModel())

    assert (settings.workspace_root / "a.md").read_text(
        encoding="utf-8"
    ) == "body"
    assert nested.is_dir()
    assert not (settings.workspace_root / "seed.md").exists()


def test_failed_initialization_restores_the_empty_workspace_without_path_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from workspace_agent import web

    settings = empty_settings_for(tmp_path)
    original_identity = os.stat(settings.workspace_root).st_ino
    original_replace = web.os.replace

    def fail_stage_exchange(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            source_path.name.startswith(".workspace-reset-stage-")
            and destination_path == settings.workspace_root
        ):
            raise OSError(f"initialization failed at {tmp_path}")
        original_replace(source, destination)

    monkeypatch.setattr(web.os, "replace", fail_stage_exchange)

    with pytest.raises(
        ValueError,
        match="^workspace initialization failed$",
    ) as error:
        create_app(settings, model=FinalOnlyModel())

    assert str(tmp_path) not in str(error.value)
    assert settings.workspace_root.is_dir()
    assert os.stat(settings.workspace_root).st_ino == original_identity
    assert list(settings.workspace_root.iterdir()) == []
    assert (settings.seed_root / "seed.md").read_text(
        encoding="utf-8"
    ) == "seed"
    assert not any(
        entry.name.startswith(".workspace-reset-")
        for entry in tmp_path.iterdir()
    )
