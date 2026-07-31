from __future__ import annotations

import asyncio
import json
import os
import shutil
import stat
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

from workspace_agent.config import Settings
from workspace_agent.loop import AgentRunner
from workspace_agent.model import OpenAICompatibleModel
from workspace_agent.safety import WorkspaceGuard
from workspace_agent.tools import WorkspaceTools
from workspace_agent.trace import TraceStore


_TASK_LIMIT = 4000
_INITIAL_MESSAGE_TIMEOUT_SECONDS = 15.0
_DEFAULT_LIMITER_KEYS = 1024


class SlidingWindowLimiter:
    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        *,
        max_keys: int = _DEFAULT_LIMITER_KEYS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or limit < 1
        ):
            raise ValueError("limit must be a positive integer")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if (
            not isinstance(max_keys, int)
            or isinstance(max_keys, bool)
            or max_keys < 1
        ):
            raise ValueError("max_keys must be a positive integer")
        self.limit = limit
        self.window_seconds = window_seconds
        self.max_keys = max_keys
        self._clock = clock
        self._requests: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def key_count(self) -> int:
        with self._lock:
            return len(self._requests)

    def allow(self, key: str) -> bool:
        now = self._clock()
        cutoff = now - self.window_seconds
        with self._lock:
            bucket = self._requests.get(key)
            if bucket is None:
                self._purge_expired(cutoff)
                while len(self._requests) >= self.max_keys:
                    self._requests.popitem(last=False)
                bucket = deque()
                self._requests[key] = bucket
            else:
                self._requests.move_to_end(key)

            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self.limit:
                return False
            bucket.append(now)
            return True

    def _purge_expired(self, cutoff: float) -> None:
        expired: list[str] = []
        for key, bucket in self._requests.items():
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if not bucket:
                expired.append(key)
        for key in expired:
            self._requests.pop(key, None)


class _CapacityGate:
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._active = 0
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            if self._active >= self._capacity:
                return False
            self._active += 1
            return True

    def release(self) -> None:
        with self._lock:
            if self._active < 1:
                raise RuntimeError("capacity gate is not acquired")
            self._active -= 1


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(
        file_attributes & reparse_flag
    )


def _unlink_nonphysical(path: Path, metadata: os.stat_result) -> None:
    if stat.S_ISDIR(metadata.st_mode):
        os.rmdir(path)
    else:
        os.unlink(path)


def _remove_entry(path: Path) -> None:
    metadata = os.lstat(path)
    if _is_link_or_reparse(metadata):
        _unlink_nonphysical(path, metadata)
        return
    if stat.S_ISDIR(metadata.st_mode):
        with os.scandir(path) as entries:
            children = [Path(entry.path) for entry in entries]
        for child in children:
            _remove_entry(child)
        os.rmdir(path)
        return
    os.unlink(path)


def _copy_seed_directory(
    seed_guard: WorkspaceGuard,
    workspace_guard: WorkspaceGuard,
    relative: str = ".",
) -> None:
    source_directory = seed_guard.resolve(relative, must_exist=True)
    source_metadata = os.lstat(source_directory)
    if _is_link_or_reparse(source_metadata) or not stat.S_ISDIR(
        source_metadata.st_mode
    ):
        raise RuntimeError("seed contains an invalid directory")

    with os.scandir(source_directory) as entries:
        children = sorted(entries, key=lambda entry: entry.name)
    for entry in children:
        child_relative = (
            entry.name if relative == "." else f"{relative}/{entry.name}"
        )
        source = seed_guard.resolve(child_relative, must_exist=True)
        metadata = os.lstat(source)
        if _is_link_or_reparse(metadata):
            raise RuntimeError("seed contains a non-physical entry")
        destination = workspace_guard.resolve(child_relative)
        if stat.S_ISDIR(metadata.st_mode):
            destination.mkdir()
            _copy_seed_directory(
                seed_guard,
                workspace_guard,
                child_relative,
            )
        elif stat.S_ISREG(metadata.st_mode):
            with source.open("rb") as source_stream:
                with destination.open("xb") as destination_stream:
                    shutil.copyfileobj(source_stream, destination_stream)
        else:
            raise RuntimeError("seed contains an unsupported entry")


def _reset_workspace(settings: Settings) -> None:
    workspace_guard = WorkspaceGuard(settings.workspace_root)
    seed_guard = WorkspaceGuard(settings.seed_root)
    with os.scandir(workspace_guard.root) as entries:
        children = [Path(entry.path) for entry in entries]
    for child in children:
        _remove_entry(child)
    _copy_seed_directory(seed_guard, workspace_guard)


def _tool_status(error_code: str | None) -> int:
    if error_code == "NOT_FOUND":
        return 404
    if error_code == "ACCESS_DENIED":
        return 403
    if error_code in {
        "BINARY_FILE",
        "UNSUPPORTED_ENCODING",
        "CHARSET_UNDETERMINED",
    }:
        return 415
    if error_code in {"READ_ERROR", "TOOL_EXECUTION_FAILED"}:
        return 500
    return 400


def _tool_http_error(error_code: str | None, message: str) -> HTTPException:
    code = error_code or "REQUEST_FAILED"
    return HTTPException(
        status_code=_tool_status(code),
        detail={"error_code": code, "message": message},
    )


async def _send_failure(
    socket: WebSocket,
    message: str,
    *,
    close_code: int,
) -> None:
    try:
        await socket.send_json({"type": "run_failed", "message": message})
    except (RuntimeError, WebSocketDisconnect):
        return
    with suppress(RuntimeError, WebSocketDisconnect):
        await socket.close(code=close_code)


async def _watch_client(socket: WebSocket) -> str:
    try:
        message = await socket.receive()
    except (RuntimeError, WebSocketDisconnect):
        return "disconnected"
    if message.get("type") == "websocket.disconnect":
        return "disconnected"
    return "extra-message"


async def _run_with_disconnect_monitor(
    socket: WebSocket,
    runner: AgentRunner,
    task: str,
) -> None:
    async def send_event(event: dict[str, Any]) -> None:
        await socket.send_json(event)

    runner_task = asyncio.create_task(runner.run(task, send_event))
    client_task = asyncio.create_task(_watch_client(socket))
    done, _ = await asyncio.wait(
        {runner_task, client_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    if runner_task in done:
        client_task.cancel()
        with suppress(asyncio.CancelledError):
            await client_task
        try:
            await runner_task
        except (RuntimeError, WebSocketDisconnect):
            return
        return

    client_state = await client_task
    runner_task.cancel()
    with suppress(asyncio.CancelledError, RuntimeError, WebSocketDisconnect):
        await runner_task
    if client_state == "extra-message":
        await _send_failure(
            socket,
            "Only one run request is allowed",
            close_code=1008,
        )


def create_app(
    settings: Settings | None = None,
    model: Any | None = None,
) -> FastAPI:
    configured = settings or Settings()
    configured.workspace_root.mkdir(parents=True, exist_ok=True)
    configured.seed_root.mkdir(parents=True, exist_ok=True)
    configured.trace_root.mkdir(parents=True, exist_ok=True)
    configured.static_root.mkdir(parents=True, exist_ok=True)

    guard = WorkspaceGuard(configured.workspace_root)
    tools = WorkspaceTools(
        guard,
        max_read_bytes=configured.max_read_bytes,
        max_write_bytes=configured.max_write_bytes,
    )
    traces = TraceStore(configured.trace_root)
    selected_model = model
    owns_model = False
    if selected_model is None and configured.llm_api_key:
        selected_model = OpenAICompatibleModel(
            base_url=configured.llm_base_url,
            api_key=configured.llm_api_key,
            model=configured.llm_model,
            timeout_seconds=configured.request_timeout_seconds,
        )
        owns_model = True

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            yield
        finally:
            if owns_model:
                close = getattr(selected_model, "aclose", None)
                if close is not None:
                    await close()

    app = FastAPI(
        title="Workspace Agent",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.settings = configured
    app.state.tools = tools
    app.state.traces = traces
    app.state.model = selected_model
    app.state.workspace_lock = _CapacityGate(1)
    app.state.run_slots = _CapacityGate(configured.max_concurrent_runs)
    app.state.connection_slots = _CapacityGate(
        max(2, configured.max_concurrent_runs * 2)
    )
    app.state.rate_limiter = SlidingWindowLimiter(
        configured.rate_limit_per_minute
    )
    app.mount(
        "/assets",
        StaticFiles(directory=configured.static_root),
        name="assets",
    )

    @app.get("/")
    async def index() -> FileResponse:
        index_path = configured.static_root / "index.html"
        if not index_path.is_file():
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": "PAGE_NOT_FOUND",
                    "message": "Web page is not available",
                },
            )
        return FileResponse(index_path)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/meta")
    async def meta() -> dict[str, Any]:
        return {
            "model": configured.llm_model,
            "configured": app.state.model is not None,
        }

    @app.get("/api/tree")
    async def tree(
        path: Annotated[str, Query(min_length=1, max_length=1024)] = ".",
        recursive: bool = True,
        cursor: Annotated[
            str | None,
            Query(min_length=1, max_length=4096),
        ] = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 200,
    ) -> dict[str, Any]:
        result = await asyncio.to_thread(
            tools.list_dir,
            path,
            recursive,
            cursor,
            limit,
        )
        if not result.ok:
            raise _tool_http_error(result.error_code, "Tree request failed")
        return result.data

    @app.get("/api/file")
    async def file_content(
        path: Annotated[str, Query(min_length=1, max_length=1024)],
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int | None, Query(ge=1)] = None,
        cursor: Annotated[
            str | None,
            Query(min_length=1, max_length=4096),
        ] = None,
    ) -> dict[str, Any]:
        result = await asyncio.to_thread(
            tools.read_file,
            path,
            offset,
            limit,
            cursor,
        )
        if not result.ok:
            raise _tool_http_error(result.error_code, "File request failed")
        return result.data

    @app.post("/api/reset")
    async def reset() -> dict[str, str]:
        if not app.state.workspace_lock.try_acquire():
            raise HTTPException(
                status_code=409,
                detail={
                    "error_code": "WORKSPACE_BUSY",
                    "message": "Workspace is busy",
                },
            )
        try:
            try:
                await asyncio.to_thread(_reset_workspace, configured)
            except Exception:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error_code": "RESET_FAILED",
                        "message": "Workspace reset failed",
                    },
                ) from None
        finally:
            app.state.workspace_lock.release()
        return {"status": "reset"}

    @app.get("/api/runs/{run_id}/trace")
    async def trace_download(run_id: str) -> FileResponse:
        try:
            trace_path = traces.path_for(run_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "INVALID_RUN_ID",
                    "message": "Invalid run id",
                },
            ) from None
        if not trace_path.is_file():
            raise HTTPException(
                status_code=404,
                detail={
                    "error_code": "TRACE_NOT_FOUND",
                    "message": "Trace was not found",
                },
            )
        return FileResponse(
            trace_path,
            media_type="application/x-ndjson",
            filename=f"{run_id}.jsonl",
        )

    @app.websocket("/ws/agent")
    async def agent_socket(socket: WebSocket) -> None:
        origin = socket.headers.get("origin")
        if configured.allowed_origin and origin != configured.allowed_origin:
            await socket.close(code=1008, reason="origin rejected")
            return

        client_key = socket.client.host if socket.client else "unknown"
        if not app.state.rate_limiter.allow(client_key):
            await socket.close(code=1013, reason="rate limit exceeded")
            return
        if not app.state.connection_slots.try_acquire():
            await socket.close(code=1013, reason="connection limit exceeded")
            return

        accepted = False
        run_slot_acquired = False
        workspace_acquired = False
        try:
            await socket.accept()
            accepted = True
            try:
                request = await asyncio.wait_for(
                    socket.receive_json(),
                    timeout=_INITIAL_MESSAGE_TIMEOUT_SECONDS,
                )
            except WebSocketDisconnect:
                return
            except (asyncio.TimeoutError, json.JSONDecodeError, RuntimeError):
                await _send_failure(
                    socket,
                    "Invalid run request",
                    close_code=1008,
                )
                return

            task = (
                request.get("task")
                if isinstance(request, dict)
                and request.get("type") == "run"
                else None
            )
            if (
                not isinstance(task, str)
                or not task.strip()
                or len(task) > _TASK_LIMIT
            ):
                await _send_failure(
                    socket,
                    "Invalid run request",
                    close_code=1008,
                )
                return
            if app.state.model is None:
                await _send_failure(
                    socket,
                    "Model is not configured",
                    close_code=1011,
                )
                return
            if not app.state.run_slots.try_acquire():
                await _send_failure(
                    socket,
                    "Server is busy",
                    close_code=1013,
                )
                return
            run_slot_acquired = True
            if not app.state.workspace_lock.try_acquire():
                await _send_failure(
                    socket,
                    "Server is busy",
                    close_code=1013,
                )
                return
            workspace_acquired = True

            runner = AgentRunner(
                model=app.state.model,
                tools=app.state.tools,
                traces=app.state.traces,
                max_model_calls=configured.max_model_calls,
            )
            await _run_with_disconnect_monitor(socket, runner, task.strip())
        except WebSocketDisconnect:
            return
        except Exception:
            if accepted and socket.application_state == WebSocketState.CONNECTED:
                await _send_failure(
                    socket,
                    "Run failed",
                    close_code=1011,
                )
        finally:
            if workspace_acquired:
                app.state.workspace_lock.release()
            if run_slot_acquired:
                app.state.run_slots.release()
            app.state.connection_slots.release()

    return app


app = create_app()
