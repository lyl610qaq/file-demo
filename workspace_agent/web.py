from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import shutil
import stat
import tempfile
import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Callable
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlsplit

from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from workspace_agent.config import Settings
from workspace_agent.loop import AgentRunner
from workspace_agent.model import OpenAICompatibleModel
from workspace_agent.safety import WorkspaceGuard
from workspace_agent.tools import WorkspaceTools
from workspace_agent.trace import TraceStore


_TASK_LIMIT = 4000
_MAX_RUN_FRAME_CHARS = 8192
_INITIAL_MESSAGE_TIMEOUT_SECONDS = 15.0
_TERMINAL_SEND_TIMEOUT_SECONDS = 0.25
_DEFAULT_LIMITER_KEYS = 1024
_MAX_FORWARDED_FOR_CHARS = 2048
_MAX_FORWARDED_FOR_HOPS = 20
_MAX_TRUSTED_PROXY_CIDRS = 64
_ORIGIN_ERROR = "allowed_origin must be a valid HTTP origin"
_TERMINAL_EVENT_TYPES = frozenset({"run_completed", "run_failed"})
_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "connect-src 'self' ws: wss:; img-src 'self' data:; "
        "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; "
        "form-action 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": (
        "camera=(), microphone=(), geolocation=(), payment=(), "
        "usb=(), serial=(), bluetooth=()"
    ),
}


class _InvalidRunRequest(ValueError):
    pass


class _ResetError(RuntimeError):
    def __init__(self, error_code: str = "RESET_FAILED") -> None:
        super().__init__("workspace reset failed")
        self.error_code = error_code


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


def _normalize_http_origin(value: object) -> tuple[str, str, int]:
    if (
        not isinstance(value, str)
        or not value
        or any(character.isspace() for character in value)
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(_ORIGIN_ERROR)
    try:
        parsed = urlsplit(value)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        port = parsed.port
    except (UnicodeError, ValueError):
        raise ValueError(_ORIGIN_ERROR) from None

    if (
        scheme not in {"http", "https"}
        or not parsed.netloc
        or hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(_ORIGIN_ERROR)

    try:
        normalized_host = _normalize_origin_host(hostname)
    except (UnicodeError, ValueError):
        raise ValueError(_ORIGIN_ERROR) from None
    effective_port = port if port is not None else (80 if scheme == "http" else 443)
    return scheme, normalized_host, effective_port


def _normalize_origin_host(hostname: str) -> str:
    if ":" in hostname:
        address = ipaddress.ip_address(hostname)
        if address.version != 6:
            raise ValueError("invalid origin host")
        return address.compressed

    normalized = hostname.encode("idna").decode("ascii").lower()
    if len(normalized) > 253:
        raise ValueError("invalid origin host")
    labels = normalized.split(".")
    if any(
        not label
        or len(label) > 63
        or label.startswith("-")
        or label.endswith("-")
        or not all(character.isalnum() or character == "-" for character in label)
        for label in labels
    ):
        raise ValueError("invalid origin host")
    return normalized


def _parse_trusted_proxy_cidrs(
    value: str,
) -> tuple[
    ipaddress.IPv4Network | ipaddress.IPv6Network,
    ...,
]:
    if not isinstance(value, str):
        raise ValueError("trusted proxy configuration is invalid")
    stripped = value.strip()
    if not stripped:
        return ()
    if len(value) > _MAX_FORWARDED_FOR_CHARS:
        raise ValueError("trusted proxy configuration is invalid")
    parts = [part.strip() for part in value.split(",")]
    if (
        len(parts) > _MAX_TRUSTED_PROXY_CIDRS
        or any(not part for part in parts)
    ):
        raise ValueError("trusted proxy configuration is invalid")
    try:
        return tuple(
            ipaddress.ip_network(part, strict=False)
            for part in parts
        )
    except ValueError:
        raise ValueError("trusted proxy configuration is invalid") from None


def _address_is_trusted(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
    networks: tuple[
        ipaddress.IPv4Network | ipaddress.IPv6Network,
        ...,
    ],
) -> bool:
    return any(
        address.version == network.version and address in network
        for network in networks
    )


def _client_key(
    connection: Any,
    trusted_proxy_networks: tuple[
        ipaddress.IPv4Network | ipaddress.IPv6Network,
        ...,
    ],
) -> str:
    client = getattr(connection, "client", None)
    peer_value = getattr(client, "host", None)
    try:
        peer = ipaddress.ip_address(peer_value)
    except (TypeError, ValueError):
        return "unknown"
    peer_key = peer.compressed
    if not _address_is_trusted(peer, trusted_proxy_networks):
        return peer_key

    headers = getattr(connection, "headers", {})
    getlist = getattr(headers, "getlist", None)
    if callable(getlist):
        values = getlist("x-forwarded-for")
        if len(values) != 1:
            return peer_key
        forwarded_for = values[0]
    else:
        forwarded_for = headers.get("x-forwarded-for")
    if not isinstance(forwarded_for, str):
        return peer_key
    if len(forwarded_for) > _MAX_FORWARDED_FOR_CHARS:
        return peer_key
    parts = forwarded_for.split(",")
    if (
        not parts
        or len(parts) > _MAX_FORWARDED_FOR_HOPS
        or any(not part.strip() for part in parts)
    ):
        return peer_key
    try:
        addresses = [ipaddress.ip_address(part.strip()) for part in parts]
    except ValueError:
        return peer_key

    for address in reversed([*addresses, peer]):
        if not _address_is_trusted(address, trusted_proxy_networks):
            return address.compressed
    return peer_key


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


def _runtime_roots_are_separate(paths: tuple[Path, ...]) -> bool:
    canonical = [
        os.path.normcase(os.path.realpath(path))
        for path in paths
    ]
    for index, left in enumerate(canonical):
        for right in canonical[index + 1 :]:
            try:
                common = os.path.commonpath((left, right))
            except ValueError:
                continue
            if common == left or common == right:
                return False
    return True


def _validate_seed_root(seed_root: Path) -> None:
    try:
        metadata = os.lstat(seed_root)
        if (
            _is_link_or_reparse(metadata)
            or not stat.S_ISDIR(metadata.st_mode)
            or not os.access(seed_root, os.R_OK)
        ):
            raise ValueError("seed root is invalid")
        with os.scandir(seed_root) as entries:
            next(entries)
    except (OSError, StopIteration):
        raise ValueError("seed root is invalid") from None


def _prepare_runtime_roots(settings: Settings) -> tuple[
    WorkspaceGuard,
    WorkspaceGuard,
]:
    roots = (
        settings.workspace_root,
        settings.seed_root,
        settings.trace_root,
        settings.static_root,
    )
    if not _runtime_roots_are_separate(roots):
        raise ValueError("runtime roots must be separate")
    _validate_seed_root(settings.seed_root)
    try:
        workspace_guard = WorkspaceGuard(settings.workspace_root)
        seed_guard = WorkspaceGuard(settings.seed_root)
        WorkspaceGuard(settings.trace_root)
        WorkspaceGuard(settings.static_root)
    except (OSError, ValueError):
        raise ValueError("runtime root is invalid") from None
    return workspace_guard, seed_guard


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


def _validate_physical_tree(
    root: Path,
    *,
    require_nonempty: bool = False,
) -> None:
    metadata = os.lstat(root)
    if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise _ResetError()
    with os.scandir(root) as entries:
        children = [Path(entry.path) for entry in entries]
    if require_nonempty and not children:
        raise _ResetError()
    for child in children:
        child_metadata = os.lstat(child)
        if _is_link_or_reparse(child_metadata):
            raise _ResetError()
        if stat.S_ISDIR(child_metadata.st_mode):
            _validate_physical_tree(child)
        elif not stat.S_ISREG(child_metadata.st_mode):
            raise _ResetError()


def _cleanup_reset_path(path: Path) -> None:
    if os.path.lexists(path):
        _remove_entry(path)


def _reset_workspace(settings: Settings) -> None:
    workspace_guard = WorkspaceGuard(settings.workspace_root)
    seed_guard = WorkspaceGuard(settings.seed_root)
    workspace = workspace_guard.root
    parent = workspace.parent
    staging = Path(
        tempfile.mkdtemp(prefix=".workspace-reset-stage-", dir=parent)
    )
    backup = parent / f".workspace-reset-backup-{uuid.uuid4().hex}"

    try:
        staging_guard = WorkspaceGuard(staging)
        _copy_seed_directory(seed_guard, staging_guard)
        _validate_physical_tree(staging, require_nonempty=True)

        try:
            os.replace(workspace, backup)
        except OSError:
            raise _ResetError() from None

        try:
            # Single-process deployment: this checked rename sequence assumes
            # no malicious local process mutates these roots concurrently.
            os.replace(staging, workspace)
        except OSError:
            try:
                os.replace(backup, workspace)
            except OSError:
                raise _ResetError("RESET_ROLLBACK_FAILED") from None
            try:
                _cleanup_reset_path(staging)
            except OSError:
                raise _ResetError("RESET_CLEANUP_FAILED") from None
            raise _ResetError() from None

        try:
            _cleanup_reset_path(backup)
        except OSError:
            raise _ResetError("RESET_CLEANUP_FAILED") from None
    except _ResetError:
        try:
            _cleanup_reset_path(staging)
        except OSError:
            raise _ResetError("RESET_CLEANUP_FAILED") from None
        raise
    except Exception:
        try:
            _cleanup_reset_path(staging)
        except OSError:
            raise _ResetError("RESET_CLEANUP_FAILED") from None
        raise _ResetError() from None


def _directory_is_empty(root: Path) -> bool:
    try:
        metadata = os.lstat(root)
        if _is_link_or_reparse(metadata) or not stat.S_ISDIR(
            metadata.st_mode
        ):
            raise ValueError("workspace initialization failed")
        with os.scandir(root) as entries:
            return next(entries, None) is None
    except OSError:
        raise ValueError("workspace initialization failed") from None


def _initialize_workspace_if_empty(
    settings: Settings,
    guard: WorkspaceGuard,
) -> WorkspaceGuard:
    if not _directory_is_empty(guard.root):
        return guard
    try:
        _reset_workspace(settings)
        refreshed, _ = _prepare_runtime_roots(settings)
    except (OSError, ValueError, _ResetError):
        raise ValueError("workspace initialization failed") from None
    return refreshed


async def _run_thread_worker(
    function: Callable[..., Any],
    *arguments: Any,
) -> Any:
    worker = asyncio.create_task(
        asyncio.to_thread(function, *arguments)
    )
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        if worker.done() and not worker.cancelled():
            with suppress(BaseException):
                worker.result()
        raise


async def _run_reset_worker(settings: Settings) -> None:
    await _run_thread_worker(_reset_workspace, settings)


def _workspace_busy_error() -> HTTPException:
    return HTTPException(
        status_code=409,
        detail={
            "error_code": "WORKSPACE_BUSY",
            "message": "Workspace is busy",
        },
    )


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


class _RunChannel:
    def __init__(self, socket: WebSocket) -> None:
        self._socket = socket
        self._lock = asyncio.Lock()
        self._terminal_source: str | None = None

    @property
    def runner_terminal_sent(self) -> bool:
        return self._terminal_source == "runner"

    async def send_runner_event(self, event: dict[str, Any]) -> None:
        async with self._lock:
            if self._terminal_source is not None:
                return
            is_terminal = event.get("type") in _TERMINAL_EVENT_TYPES
            if is_terminal:
                self._terminal_source = "runner"
            await self._socket.send_json(event)
            if is_terminal:
                with suppress(RuntimeError, WebSocketDisconnect):
                    await self._socket.close(code=1000)

    async def send_failure(self, message: str, *, close_code: int) -> bool:
        async with self._lock:
            if self._terminal_source is not None:
                return False
            self._terminal_source = "control"
            try:
                await self._socket.send_json(
                    {"type": "run_failed", "message": message}
                )
            except (RuntimeError, WebSocketDisconnect):
                return False
            with suppress(RuntimeError, WebSocketDisconnect):
                await self._socket.close(code=close_code)
            return True


def _reject_json_constant(value: str) -> None:
    raise _InvalidRunRequest("non-finite JSON value")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _InvalidRunRequest("duplicate JSON field")
        value[key] = item
    return value


def _parse_run_request(payload: str) -> str:
    if len(payload) > _MAX_RUN_FRAME_CHARS:
        raise _InvalidRunRequest("run frame is too large")
    try:
        request = json.loads(
            payload,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (json.JSONDecodeError, RecursionError, TypeError, ValueError):
        raise _InvalidRunRequest("invalid JSON") from None
    if not isinstance(request, dict) or set(request) != {"type", "task"}:
        raise _InvalidRunRequest("invalid run object")
    task = request["task"]
    if (
        request["type"] != "run"
        or not isinstance(task, str)
        or not task.strip()
        or len(task) > _TASK_LIMIT
    ):
        raise _InvalidRunRequest("invalid run fields")
    return task.strip()


async def _receive_run_task(socket: WebSocket) -> str:
    frame = await asyncio.wait_for(
        socket.receive(),
        timeout=_INITIAL_MESSAGE_TIMEOUT_SECONDS,
    )
    if frame.get("type") == "websocket.disconnect":
        raise WebSocketDisconnect(
            code=frame.get("code", 1000),
            reason=frame.get("reason", ""),
        )
    payload = frame.get("text")
    if frame.get("type") != "websocket.receive" or not isinstance(
        payload,
        str,
    ):
        raise _InvalidRunRequest("first frame must be text")
    return _parse_run_request(payload)


async def _watch_client(socket: WebSocket) -> str:
    try:
        message = await socket.receive()
    except (RuntimeError, WebSocketDisconnect):
        return "disconnected"
    if message.get("type") == "websocket.disconnect":
        return "disconnected"
    return "extra-message"


async def _cancel_and_settle_task(
    task: asyncio.Task[Any],
    *,
    propagate_cancellation: bool = True,
) -> None:
    if not task.done():
        task.cancel()
    interrupted = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.done():
                interrupted = True
        except BaseException:
            break
    if task.done() and not task.cancelled():
        with suppress(BaseException):
            task.result()
    if interrupted and propagate_cancellation:
        raise asyncio.CancelledError


def _consume_task_result(task: asyncio.Task[Any]) -> None:
    if not task.cancelled():
        with suppress(BaseException):
            task.result()


async def _send_channel_failure_bounded(
    channel: _RunChannel,
    message: str,
    *,
    close_code: int,
) -> None:
    send_task = asyncio.create_task(
        channel.send_failure(message, close_code=close_code)
    )
    done, _ = await asyncio.wait(
        {send_task},
        timeout=_TERMINAL_SEND_TIMEOUT_SECONDS,
    )
    if send_task in done:
        _consume_task_result(send_task)
        return
    send_task.cancel()
    send_task.add_done_callback(_consume_task_result)


async def _run_with_disconnect_monitor(
    socket: WebSocket,
    channel: _RunChannel,
    runner: AgentRunner,
    task: str,
    *,
    max_run_seconds: float = 300.0,
) -> None:
    client_task = asyncio.create_task(_watch_client(socket))
    await asyncio.sleep(0)
    if client_task.done():
        client_state = await client_task
        if client_state == "extra-message":
            await _send_channel_failure_bounded(
                channel,
                "Invalid run request",
                close_code=1008,
            )
        return

    runner_task = asyncio.create_task(
        runner.run(task, channel.send_runner_event)
    )
    deadline_task = asyncio.create_task(asyncio.sleep(max_run_seconds))
    try:
        done, _ = await asyncio.wait(
            {runner_task, client_task, deadline_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if channel.runner_terminal_sent:
            await _cancel_and_settle_task(client_task)
            await _cancel_and_settle_task(deadline_task)
            if not runner_task.done():
                await _cancel_and_settle_task(runner_task)
            else:
                _consume_task_result(runner_task)
            return

        if client_task in done:
            client_state = await client_task
            await _cancel_and_settle_task(deadline_task)
            await _cancel_and_settle_task(
                runner_task,
                propagate_cancellation=client_state != "disconnected",
            )
            if client_state == "extra-message":
                await _send_channel_failure_bounded(
                    channel,
                    "Invalid run request",
                    close_code=1008,
                )
            return

        if runner_task in done:
            await _cancel_and_settle_task(client_task)
            await _cancel_and_settle_task(deadline_task)
            _consume_task_result(runner_task)
            if not channel.runner_terminal_sent:
                await _send_channel_failure_bounded(
                    channel,
                    "Run failed",
                    close_code=1011,
                )
            return

        await _cancel_and_settle_task(client_task)
        await _cancel_and_settle_task(runner_task)
        await _send_channel_failure_bounded(
            channel,
            "Run timed out",
            close_code=1011,
        )
    except asyncio.CancelledError:
        await _cancel_and_settle_task(runner_task)
        await _cancel_and_settle_task(client_task)
        await _cancel_and_settle_task(deadline_task)
        raise
    finally:
        for pending_task in (runner_task, client_task, deadline_task):
            if not pending_task.done():
                pending_task.cancel()
                pending_task.add_done_callback(_consume_task_result)


def create_app(
    settings: Settings | None = None,
    model: Any | None = None,
) -> FastAPI:
    configured = settings or Settings()
    allowed_origin = _normalize_http_origin(configured.allowed_origin)
    trusted_proxy_networks = _parse_trusted_proxy_cidrs(
        configured.trusted_proxy_cidrs
    )
    guard, _ = _prepare_runtime_roots(configured)
    guard = _initialize_workspace_if_empty(configured, guard)
    tools = WorkspaceTools(
        guard,
        max_read_bytes=configured.max_read_bytes,
        max_write_bytes=configured.max_write_bytes,
    )
    traces = TraceStore(configured.trace_root)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        owned_model: Any | None = None
        if model is None and configured.llm_api_key:
            owned_model = OpenAICompatibleModel(
                base_url=configured.llm_base_url,
                api_key=configured.llm_api_key,
                model=configured.llm_model,
                timeout_seconds=configured.request_timeout_seconds,
            )
            application.state.model = owned_model
        try:
            yield
        finally:
            if owned_model is not None:
                close = getattr(owned_model, "aclose", None)
                if close is not None:
                    await close()
            application.state.model = model

    app = FastAPI(
        title="Workspace Agent",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.settings = configured
    app.state.tools = tools
    app.state.traces = traces
    app.state.model = model
    app.state.allowed_origin = allowed_origin
    app.state.trusted_proxy_networks = trusted_proxy_networks
    app.state.workspace_lock = _CapacityGate(1)
    app.state.run_slots = _CapacityGate(configured.max_concurrent_runs)
    app.state.connection_slots = _CapacityGate(
        max(2, configured.max_concurrent_runs * 2)
    )
    app.state.rate_limiter = SlidingWindowLimiter(
        configured.rate_limit_per_minute
    )
    app.state.reset_rate_limiter = SlidingWindowLimiter(
        configured.rate_limit_per_minute
    )

    @app.middleware("http")
    async def add_security_headers(
        request: Request,
        call_next: Callable[..., Any],
    ) -> Any:
        response = await call_next(request)
        for name, value in _SECURITY_HEADERS.items():
            response.headers[name] = value
        return response

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
            "max_run_seconds": configured.max_run_seconds,
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
        if not app.state.workspace_lock.try_acquire():
            raise _workspace_busy_error()
        try:
            result = await _run_thread_worker(
                tools.list_dir,
                path,
                recursive,
                cursor,
                limit,
            )
        finally:
            app.state.workspace_lock.release()
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
        if not app.state.workspace_lock.try_acquire():
            raise _workspace_busy_error()
        try:
            result = await _run_thread_worker(
                tools.read_file,
                path,
                offset,
                limit,
                cursor,
            )
        finally:
            app.state.workspace_lock.release()
        if not result.ok:
            raise _tool_http_error(result.error_code, "File request failed")
        return result.data

    @app.post("/api/reset")
    async def reset(request: Request) -> dict[str, str]:
        try:
            request_origin = _normalize_http_origin(
                request.headers.get("origin")
            )
        except ValueError:
            request_origin = None
        if request_origin != app.state.allowed_origin:
            raise HTTPException(
                status_code=403,
                detail={
                    "error_code": "ORIGIN_REJECTED",
                    "message": "Origin is not allowed",
                },
            )
        client_key = _client_key(
            request,
            app.state.trusted_proxy_networks,
        )
        if not app.state.reset_rate_limiter.allow(client_key):
            raise HTTPException(
                status_code=429,
                detail={
                    "error_code": "RATE_LIMITED",
                    "message": "Too many reset requests",
                },
            )
        if not app.state.workspace_lock.try_acquire():
            raise _workspace_busy_error()
        try:
            try:
                await _run_reset_worker(configured)
            except _ResetError as error:
                message = (
                    "Workspace reset cleanup failed"
                    if error.error_code == "RESET_CLEANUP_FAILED"
                    else "Workspace reset rollback failed"
                    if error.error_code == "RESET_ROLLBACK_FAILED"
                    else "Workspace reset failed"
                )
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error_code": error.error_code,
                        "message": message,
                    },
                ) from None
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
        try:
            request_origin = _normalize_http_origin(socket.headers.get("origin"))
        except ValueError:
            await socket.close(code=1008, reason="origin rejected")
            return
        if request_origin != app.state.allowed_origin:
            await socket.close(code=1008, reason="origin rejected")
            return

        client_key = _client_key(
            socket,
            app.state.trusted_proxy_networks,
        )
        if not app.state.rate_limiter.allow(client_key):
            await socket.close(code=1013, reason="rate limit exceeded")
            return
        if not app.state.connection_slots.try_acquire():
            await socket.close(code=1013, reason="connection limit exceeded")
            return

        channel: _RunChannel | None = None
        run_slot_acquired = False
        workspace_acquired = False
        try:
            await socket.accept()
            channel = _RunChannel(socket)
            try:
                task = await _receive_run_task(socket)
            except WebSocketDisconnect:
                return
            except (asyncio.TimeoutError, _InvalidRunRequest, RuntimeError):
                await _send_channel_failure_bounded(
                    channel,
                    "Invalid run request",
                    close_code=1008,
                )
                return

            if app.state.model is None:
                await _send_channel_failure_bounded(
                    channel,
                    "Model is not configured",
                    close_code=1011,
                )
                return
            if not app.state.run_slots.try_acquire():
                await _send_channel_failure_bounded(
                    channel,
                    "Server is busy",
                    close_code=1013,
                )
                return
            run_slot_acquired = True
            if not app.state.workspace_lock.try_acquire():
                await _send_channel_failure_bounded(
                    channel,
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
            await _run_with_disconnect_monitor(
                socket,
                channel,
                runner,
                task,
                max_run_seconds=configured.max_run_seconds,
            )
        except WebSocketDisconnect:
            return
        except Exception:
            if channel is not None:
                await _send_channel_failure_bounded(
                    channel,
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


class _LazyApplication:
    def __init__(self) -> None:
        self._application: FastAPI | None = None
        self._lock = threading.Lock()

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[..., Any],
        send: Callable[..., Any],
    ) -> None:
        application = self._application
        if application is None:
            with self._lock:
                if self._application is None:
                    self._application = create_app()
                application = self._application
        await application(scope, receive, send)


app = _LazyApplication()
