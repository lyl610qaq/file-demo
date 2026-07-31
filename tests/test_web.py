from __future__ import annotations

import asyncio
import json
import os
import subprocess
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from workspace_agent.config import Settings
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


class ClosableModel(FinalOnlyModel):
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class FailingModel:
    async def complete(self, messages, tools) -> ModelReply:
        raise RuntimeError("api-key=server-secret path=C:/private")


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


def test_index_assets_health_meta_tree_file_and_reset(tmp_path: Path) -> None:
    app = create_app(settings_for(tmp_path), model=FinalOnlyModel())

    with TestClient(app) as client:
        assert client.get("/").text == "<main>app</main>"
        assert "window.loaded" in client.get("/assets/app.js").text
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/api/meta").json() == {
            "model": "gpt-4.1-mini",
            "configured": True,
        }
        assert client.get("/api/tree").json()["entries"][0]["path"] == "a.md"
        assert client.get("/api/file", params={"path": "a.md"}).json()[
            "content"
        ] == "body"
        assert client.post("/api/reset").json() == {"status": "reset"}
        assert client.get(
            "/api/file", params={"path": "seed.md"}
        ).status_code == 200
        assert client.get("/api/file", params={"path": "a.md"}).status_code == 404


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

            reset = client.post("/api/reset")
            assert reset.status_code == 409
            assert reset.json()["detail"]["error_code"] == "WORKSPACE_BUSY"

            with client.websocket_connect(
                "/ws/agent",
                headers={"origin": "http://testserver"},
            ) as second:
                second.send_json({"type": "run", "task": "Second run"})
                rejected = second.receive_json()
            assert rejected == {
                "type": "run_failed",
                "message": "Server is busy",
            }

            model.release.set()
            remaining = receive_until_terminal(first)

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
            reset_future = pool.submit(client.post, "/api/reset")
            assert started.wait(2)
            with client.websocket_connect(
                "/ws/agent",
                headers={"origin": "http://testserver"},
            ) as socket:
                socket.send_json({"type": "run", "task": "Inspect"})
                event = socket.receive_json()
            release.set()
            reset_response = reset_future.result(timeout=2)

    assert event == {"type": "run_failed", "message": "Server is busy"}
    assert reset_response.status_code == 200


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
        response = client.post("/api/reset")

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
            response = client.post("/api/reset")
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
        response = client.post("/api/reset")

    assert response.status_code == 500
    assert response.json() == {
        "detail": {
            "error_code": "RESET_FAILED",
            "message": "Workspace reset failed",
        }
    }
    assert secret not in response.text


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
        assert client.post("/api/reset").status_code == 200

    trace_files = list((tmp_path / "traces").glob("*.jsonl"))
    assert len(trace_files) == 1
    records = [
        json.loads(line)
        for line in trace_files[0].read_text(encoding="utf-8").splitlines()
    ]
    assert records[-1]["status"] == "cancelled"


def test_app_closes_only_the_model_it_constructed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from workspace_agent import web

    owned = ClosableModel()

    def model_factory(**kwargs: Any) -> ClosableModel:
        assert kwargs["api_key"] == "server-secret"
        return owned

    monkeypatch.setattr(web, "OpenAICompatibleModel", model_factory)
    configured = settings_for(tmp_path / "owned", llm_api_key="server-secret")
    with TestClient(create_app(configured)):
        pass
    assert owned.closed is True

    injected = ClosableModel()
    with TestClient(
        create_app(settings_for(tmp_path / "injected"), model=injected)
    ):
        pass
    assert injected.closed is False


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
