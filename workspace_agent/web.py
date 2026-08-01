from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import os
import re
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
_RESET_JOURNAL_NAME = ".workspace-reset-journal.json"
_RESET_JOURNAL_TEMP_PATTERN = re.compile(
    r"\.workspace-reset-journal-tmp-[0-9a-f]{32}\Z",
    flags=re.ASCII,
)
_RESET_STAGING_PATTERN = re.compile(
    r"\.workspace-reset-stage-[0-9a-f]{32}\Z",
    flags=re.ASCII,
)
_RESET_BACKUP_PATTERN = re.compile(
    r"\.workspace-reset-backup-[0-9a-f]{32}\Z",
    flags=re.ASCII,
)
_RESET_SEED_FINGERPRINT_PATTERN = re.compile(
    r"[0-9a-f]{64}\Z",
    flags=re.ASCII,
)
_RESET_INSTALL_PHASES = frozenset(
    {"prepared", "backup-created", "workspace-installed"}
)
_RESET_ROLLBACK_PHASES = frozenset(
    {"rollback-restoring", "rollback-restored"}
)
_RESET_PHASES = _RESET_INSTALL_PHASES | _RESET_ROLLBACK_PHASES
_MAX_RESET_JOURNAL_BYTES = 4096
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
        if _count_readable_regular_files(seed_root) < 1:
            raise ValueError("seed root is invalid")
    except (OSError, ValueError):
        raise ValueError("seed root is invalid") from None


def _count_readable_regular_files(root: Path) -> int:
    metadata = os.lstat(root)
    if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("tree root is invalid")
    count = 0
    with os.scandir(root) as entries:
        children = [Path(entry.path) for entry in entries]
    for child in children:
        child_metadata = os.lstat(child)
        if _is_link_or_reparse(child_metadata):
            raise ValueError("tree contains a non-physical entry")
        if stat.S_ISDIR(child_metadata.st_mode):
            count += _count_readable_regular_files(child)
        elif stat.S_ISREG(child_metadata.st_mode):
            with child.open("rb") as stream:
                stream.read(1)
            count += 1
        else:
            raise ValueError("tree contains an unsupported entry")
    return count


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
                    _fsync_file(destination_stream)
        else:
            raise RuntimeError("seed contains an unsupported entry")
    _fsync_directory(workspace_guard.resolve(relative, must_exist=True))


def _validate_physical_tree(
    root: Path,
    *,
    require_nonempty: bool = False,
) -> None:
    try:
        file_count = _count_readable_regular_files(root)
        if require_nonempty and file_count < 1:
            raise _ResetError()
    except _ResetError:
        raise
    except (OSError, ValueError):
        raise _ResetError() from None


def _cleanup_reset_path(path: Path) -> None:
    if os.path.lexists(path):
        _remove_entry(path)


def _fsync_file(stream: Any) -> None:
    stream.flush()
    os.fsync(stream.fileno())


def _fsync_windows_directory(path: Path) -> None:
    import ctypes
    from ctypes import wintypes

    generic_read = 0x80000000
    generic_write = 0x40000000
    file_share_read = 0x00000001
    file_share_write = 0x00000002
    file_share_delete = 0x00000004
    open_existing = 3
    file_flag_backup_semantics = 0x02000000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    flush = kernel32.FlushFileBuffers
    flush.argtypes = (wintypes.HANDLE,)
    flush.restype = wintypes.BOOL
    close = kernel32.CloseHandle
    close.argtypes = (wintypes.HANDLE,)
    close.restype = wintypes.BOOL
    handle = create_file(
        str(path),
        generic_read | generic_write,
        file_share_read | file_share_write | file_share_delete,
        None,
        open_existing,
        file_flag_backup_semantics,
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise OSError(
            ctypes.get_last_error(),
            "CreateFileW failed",
            str(path),
        )
    flush_error: OSError | None = None
    try:
        if not flush(handle):
            flush_error = OSError(
                ctypes.get_last_error(),
                "FlushFileBuffers failed",
                str(path),
            )
            raise flush_error
    finally:
        if not close(handle) and flush_error is None:
            raise OSError(
                ctypes.get_last_error(),
                "CloseHandle failed",
                str(path),
            )


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        _fsync_windows_directory(path)
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_reset_journal(parent: Path, record: dict[str, Any]) -> None:
    _validate_reset_journal_record(record, record.get("workspace"))
    journal = parent / _RESET_JOURNAL_NAME
    temporary = parent / (
        f".workspace-reset-journal-tmp-{uuid.uuid4().hex}"
    )
    payload = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(payload) > _MAX_RESET_JOURNAL_BYTES:
        raise _ResetError()
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, journal)
        _fsync_directory(parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if os.path.lexists(temporary):
            metadata = os.lstat(temporary)
            if _is_link_or_reparse(metadata) or not stat.S_ISREG(
                metadata.st_mode
            ):
                raise _ResetError()
            os.unlink(temporary)


def _validate_reset_journal_record(
    record: object,
    workspace_name: object,
) -> dict[str, Any]:
    required = {
        "version",
        "workspace",
        "staging",
        "backup",
        "phase",
    }
    if not isinstance(record, dict):
        raise ValueError("invalid reset journal")
    version = record.get("version")
    if type(version) is not int:
        raise ValueError("invalid reset journal")
    if version == 1 and set(record) == required:
        normalized = dict(record)
        normalized["workspace_was_empty"] = None
        normalized["seed_fingerprint"] = None
    elif (
        version == 2
        and set(record) == required | {"workspace_was_empty"}
        and isinstance(record.get("workspace_was_empty"), bool)
    ):
        normalized = dict(record)
        normalized["seed_fingerprint"] = None
    elif (
        version == 3
        and set(record)
        == required | {"workspace_was_empty", "seed_fingerprint"}
        and isinstance(record.get("workspace_was_empty"), bool)
        and isinstance(record.get("seed_fingerprint"), str)
        and _RESET_SEED_FINGERPRINT_PATTERN.fullmatch(
            record["seed_fingerprint"]
        )
        is not None
    ):
        normalized = dict(record)
    else:
        raise ValueError("invalid reset journal")
    if (
        not isinstance(workspace_name, str)
        or normalized.get("workspace") != workspace_name
        or not isinstance(normalized.get("staging"), str)
        or _RESET_STAGING_PATTERN.fullmatch(normalized["staging"])
        is None
        or not isinstance(normalized.get("backup"), str)
        or _RESET_BACKUP_PATTERN.fullmatch(normalized["backup"])
        is None
        or normalized["staging"] == normalized["backup"]
        or not isinstance(normalized.get("phase"), str)
        or normalized["phase"]
        not in (
            _RESET_PHASES
            if normalized["version"] == 3
            else _RESET_INSTALL_PHASES
        )
    ):
        raise ValueError("invalid reset journal")
    return normalized


def _load_reset_journal(parent: Path, workspace_name: str) -> dict[str, Any]:
    journal = parent / _RESET_JOURNAL_NAME
    metadata = os.lstat(journal)
    if (
        _is_link_or_reparse(metadata)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size > _MAX_RESET_JOURNAL_BYTES
    ):
        raise ValueError("invalid reset journal")
    with journal.open("rb") as stream:
        payload = stream.read(_MAX_RESET_JOURNAL_BYTES + 1)
    if len(payload) > _MAX_RESET_JOURNAL_BYTES:
        raise ValueError("invalid reset journal")
    try:
        record = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise ValueError("invalid reset journal") from None
    return _validate_reset_journal_record(record, workspace_name)


def _remove_reset_journal(parent: Path) -> None:
    journal = parent / _RESET_JOURNAL_NAME
    if not os.path.lexists(journal):
        return
    metadata = os.lstat(journal)
    if _is_link_or_reparse(metadata) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("invalid reset journal")
    os.unlink(journal)
    _fsync_directory(parent)


def _reset_candidate(
    parent: Path,
    basename: str,
    pattern: re.Pattern[str],
) -> Path:
    if pattern.fullmatch(basename) is None:
        raise ValueError("invalid reset candidate")
    return parent / basename


def _validate_recovery_directory(path: Path) -> None:
    _count_readable_regular_files(path)


def _cleanup_recovery_directory(
    path: Path,
    pattern: re.Pattern[str],
) -> None:
    if pattern.fullmatch(path.name) is None:
        raise ValueError("invalid reset candidate")
    if not os.path.lexists(path):
        return
    _validate_recovery_directory(path)
    _cleanup_reset_path(path)


def _cleanup_journal_temps(parent: Path) -> None:
    with os.scandir(parent) as entries:
        paths = [Path(entry.path) for entry in entries]
    for path in paths:
        if not path.name.startswith(".workspace-reset-journal-tmp-"):
            continue
        if _RESET_JOURNAL_TEMP_PATTERN.fullmatch(path.name) is None:
            raise ValueError("invalid reset journal temporary")
        metadata = os.lstat(path)
        if _is_link_or_reparse(metadata) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError("invalid reset journal temporary")
        os.unlink(path)
        _fsync_directory(parent)


def _tree_fingerprint(root: Path) -> dict[str, tuple[str, int, str]]:
    result: dict[str, tuple[str, int, str]] = {}
    metadata = os.lstat(root)
    if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("invalid physical tree")
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            children = [Path(entry.path) for entry in entries]
        for child in children:
            child_metadata = os.lstat(child)
            if _is_link_or_reparse(child_metadata):
                raise ValueError("invalid physical tree")
            relative = child.relative_to(root).as_posix()
            if stat.S_ISDIR(child_metadata.st_mode):
                result[relative] = ("directory", 0, "")
                pending.append(child)
            elif stat.S_ISREG(child_metadata.st_mode):
                digest = hashlib.sha256()
                with child.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(65536), b""):
                        digest.update(chunk)
                result[relative] = (
                    "file",
                    child_metadata.st_size,
                    digest.hexdigest(),
                )
            else:
                raise ValueError("invalid physical tree")
    return result


def _physical_trees_equal(left: Path, right: Path) -> bool:
    return _tree_fingerprint(left) == _tree_fingerprint(right)


def _physical_tree_fingerprint(root: Path) -> str:
    records = [
        {
            "path": path,
            "type": entry_type,
            "bytes": byte_count,
            "sha256": digest,
        }
        for path, (entry_type, byte_count, digest) in sorted(
            _tree_fingerprint(root).items()
        )
    ]
    payload = json.dumps(
        records,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _allow_empty_workspace_for_recovery(
    record: dict[str, Any],
    backup: Path,
    *,
    backup_exists: bool,
) -> bool:
    workspace_was_empty = record["workspace_was_empty"]
    if workspace_was_empty is not None:
        return workspace_was_empty
    if not backup_exists:
        return False
    return _count_readable_regular_files(backup) == 0


def _workspace_matches_seed_fingerprint(
    workspace: Path,
    seed_fingerprint: str,
) -> bool:
    return (
        _count_readable_regular_files(workspace) > 0
        and _physical_tree_fingerprint(workspace) == seed_fingerprint
    )


def _can_finish_v1_empty_backup_restore(
    record: dict[str, Any],
    settings: Settings,
    workspace: Path,
    staging: Path,
    *,
    staging_exists: bool,
) -> bool:
    if record["version"] != 1 or record["phase"] != "backup-created":
        return False
    if _count_readable_regular_files(workspace) != 0:
        return False
    return not staging_exists or _physical_trees_equal(
        staging,
        settings.seed_root,
    )


def _restore_reset_backup(
    parent: Path,
    workspace: Path,
    staging: Path,
    backup: Path,
    *,
    allow_empty_workspace: bool,
) -> None:
    _validate_physical_tree(
        backup,
        require_nonempty=not allow_empty_workspace,
    )
    if os.path.lexists(workspace):
        if os.path.lexists(staging):
            _cleanup_recovery_directory(staging, _RESET_STAGING_PATTERN)
        _validate_recovery_directory(workspace)
        os.replace(workspace, staging)
        _fsync_directory(parent)
    os.replace(backup, workspace)
    _fsync_directory(parent)
    _finish_restored_backup(
        parent,
        workspace,
        staging,
        allow_empty_workspace=allow_empty_workspace,
    )


def _finish_restored_backup(
    parent: Path,
    workspace: Path,
    staging: Path,
    *,
    allow_empty_workspace: bool,
) -> None:
    _validate_physical_tree(
        workspace,
        require_nonempty=not allow_empty_workspace,
    )
    if os.path.lexists(staging):
        _cleanup_recovery_directory(staging, _RESET_STAGING_PATTERN)
        _fsync_directory(parent)
    _remove_reset_journal(parent)
    _cleanup_journal_temps(parent)


def _complete_v3_rollback_restore(
    parent: Path,
    workspace: Path,
    staging: Path,
    backup: Path,
    record: dict[str, Any],
) -> None:
    allow_empty_workspace = record["workspace_was_empty"]
    if os.path.lexists(backup):
        _validate_physical_tree(
            backup,
            require_nonempty=not allow_empty_workspace,
        )
        if os.path.lexists(workspace):
            if os.path.lexists(staging):
                _cleanup_recovery_directory(staging, _RESET_STAGING_PATTERN)
            _validate_recovery_directory(workspace)
            os.replace(workspace, staging)
            _fsync_directory(parent)
        os.replace(backup, workspace)
        _fsync_directory(parent)
    elif not os.path.lexists(workspace):
        raise ValueError("inconsistent reset journal")

    _validate_physical_tree(
        workspace,
        require_nonempty=not allow_empty_workspace,
    )
    record["phase"] = "rollback-restored"
    _write_reset_journal(parent, record)
    _finish_restored_backup(
        parent,
        workspace,
        staging,
        allow_empty_workspace=allow_empty_workspace,
    )


def _start_v3_rollback_restore(
    parent: Path,
    workspace: Path,
    staging: Path,
    backup: Path,
    record: dict[str, Any],
) -> None:
    record["phase"] = "rollback-restoring"
    _write_reset_journal(parent, record)
    _complete_v3_rollback_restore(
        parent,
        workspace,
        staging,
        backup,
        record,
    )


def _recover_journaled_reset(
    settings: Settings,
    parent: Path,
    record: dict[str, Any],
) -> None:
    workspace = settings.workspace_root
    staging = _reset_candidate(
        parent,
        record["staging"],
        _RESET_STAGING_PATTERN,
    )
    backup = _reset_candidate(
        parent,
        record["backup"],
        _RESET_BACKUP_PATTERN,
    )
    phase = record["phase"]
    workspace_exists = os.path.lexists(workspace)
    staging_exists = os.path.lexists(staging)
    backup_exists = os.path.lexists(backup)

    for candidate, exists in (
        (workspace, workspace_exists),
        (staging, staging_exists),
        (backup, backup_exists),
    ):
        if exists:
            _validate_recovery_directory(candidate)

    allow_empty_workspace = _allow_empty_workspace_for_recovery(
        record,
        backup,
        backup_exists=backup_exists,
    )

    if phase == "prepared":
        if backup_exists:
            _restore_reset_backup(
                parent,
                workspace,
                staging,
                backup,
                allow_empty_workspace=allow_empty_workspace,
            )
            return
        if phase == "prepared" and workspace_exists:
            if staging_exists:
                _cleanup_recovery_directory(staging, _RESET_STAGING_PATTERN)
                _fsync_directory(parent)
            _remove_reset_journal(parent)
            _cleanup_journal_temps(parent)
            return
        raise ValueError("inconsistent reset journal")

    if phase == "backup-created":
        if backup_exists:
            _restore_reset_backup(
                parent,
                workspace,
                staging,
                backup,
                allow_empty_workspace=allow_empty_workspace,
            )
            return
        if workspace_exists:
            if _can_finish_v1_empty_backup_restore(
                record,
                settings,
                workspace,
                staging,
                staging_exists=staging_exists,
            ):
                _finish_restored_backup(
                    parent,
                    workspace,
                    staging,
                    allow_empty_workspace=True,
                )
                return
            _finish_restored_backup(
                parent,
                workspace,
                staging,
                allow_empty_workspace=allow_empty_workspace,
            )
            return
        raise ValueError("inconsistent reset journal")

    if phase == "rollback-restoring":
        _complete_v3_rollback_restore(
            parent,
            workspace,
            staging,
            backup,
            record,
        )
        return

    if phase == "rollback-restored":
        if backup_exists or not workspace_exists:
            raise ValueError("inconsistent reset journal")
        _finish_restored_backup(
            parent,
            workspace,
            staging,
            allow_empty_workspace=allow_empty_workspace,
        )
        return

    if record["seed_fingerprint"] is None:
        if backup_exists:
            _restore_reset_backup(
                parent,
                workspace,
                staging,
                backup,
                allow_empty_workspace=allow_empty_workspace,
            )
            return
        if workspace_exists:
            workspace_file_count = _count_readable_regular_files(workspace)
            staging_file_count = (
                _count_readable_regular_files(staging)
                if staging_exists
                else 0
            )
            if workspace_file_count > 0 or staging_file_count == 0:
                # Legacy journals have neither a seed fingerprint nor a
                # remaining backup here. An empty physical workspace with no
                # nonempty staging source is the only usable terminal state.
                if staging_exists:
                    _cleanup_recovery_directory(staging, _RESET_STAGING_PATTERN)
                _fsync_directory(parent)
                _remove_reset_journal(parent)
                _cleanup_journal_temps(parent)
                return
        raise ValueError("installed workspace cannot be verified")

    if (
        workspace_exists
        and _workspace_matches_seed_fingerprint(
            workspace,
            record["seed_fingerprint"],
        )
    ):
        if staging_exists:
            _cleanup_recovery_directory(staging, _RESET_STAGING_PATTERN)
        if backup_exists:
            _cleanup_recovery_directory(backup, _RESET_BACKUP_PATTERN)
        _fsync_directory(parent)
        _remove_reset_journal(parent)
        _cleanup_journal_temps(parent)
        return
    if backup_exists:
        _start_v3_rollback_restore(
            parent,
            workspace,
            staging,
            backup,
            record,
        )
        return
    raise ValueError("inconsistent reset journal")


def _legacy_reset_candidates(
    parent: Path,
) -> tuple[list[Path], list[Path], list[Path]]:
    staging: list[Path] = []
    backups: list[Path] = []
    journal_temps: list[Path] = []
    with os.scandir(parent) as entries:
        paths = [Path(entry.path) for entry in entries]
    for path in paths:
        name = path.name
        if name.startswith(".workspace-reset-stage-"):
            if _RESET_STAGING_PATTERN.fullmatch(name) is None:
                raise ValueError("invalid legacy reset candidate")
            _validate_recovery_directory(path)
            staging.append(path)
        elif name.startswith(".workspace-reset-backup-"):
            if _RESET_BACKUP_PATTERN.fullmatch(name) is None:
                raise ValueError("invalid legacy reset candidate")
            _validate_recovery_directory(path)
            backups.append(path)
        elif name.startswith(".workspace-reset-journal-tmp-"):
            if _RESET_JOURNAL_TEMP_PATTERN.fullmatch(name) is None:
                raise ValueError("invalid legacy journal temporary")
            metadata = os.lstat(path)
            if _is_link_or_reparse(metadata) or not stat.S_ISREG(
                metadata.st_mode
            ):
                raise ValueError("invalid legacy journal temporary")
            journal_temps.append(path)
    return staging, backups, journal_temps


def _recover_legacy_reset(settings: Settings, parent: Path) -> None:
    staging, backups, journal_temps = _legacy_reset_candidates(parent)
    if not staging and not backups and not journal_temps:
        return
    workspace = settings.workspace_root
    workspace_exists = os.path.lexists(workspace)
    if workspace_exists:
        _validate_recovery_directory(workspace)

    if not workspace_exists:
        if len(backups) == 1:
            for candidate in staging:
                _cleanup_recovery_directory(candidate, _RESET_STAGING_PATTERN)
            os.replace(backups[0], workspace)
            _fsync_directory(parent)
            for temporary in journal_temps:
                os.unlink(temporary)
            if journal_temps:
                _fsync_directory(parent)
            return
        if backups or staging or journal_temps:
            raise ValueError("ambiguous legacy reset state")
        return

    if len(backups) > 1:
        raise ValueError("ambiguous legacy reset state")
    if backups:
        if not _physical_trees_equal(workspace, settings.seed_root):
            raise ValueError("ambiguous legacy reset state")
        _cleanup_recovery_directory(backups[0], _RESET_BACKUP_PATTERN)
        _fsync_directory(parent)
    for candidate in staging:
        _cleanup_recovery_directory(candidate, _RESET_STAGING_PATTERN)
        _fsync_directory(parent)
    for temporary in journal_temps:
        os.unlink(temporary)
        _fsync_directory(parent)


def _recover_workspace_state(settings: Settings) -> None:
    parent = settings.workspace_root.parent
    if not os.path.lexists(parent):
        return
    try:
        parent_guard = WorkspaceGuard(parent)
        parent = parent_guard.root
        journal = parent / _RESET_JOURNAL_NAME
        if os.path.lexists(journal):
            record = _load_reset_journal(
                parent,
                settings.workspace_root.name,
            )
            _recover_journaled_reset(settings, parent, record)
        else:
            _recover_legacy_reset(settings, parent)
    except (OSError, ValueError, _ResetError):
        raise ValueError("workspace recovery failed") from None


def _reset_workspace(settings: Settings) -> None:
    try:
        _recover_workspace_state(settings)
        workspace_guard = WorkspaceGuard(settings.workspace_root)
        seed_guard = WorkspaceGuard(settings.seed_root)
        workspace = workspace_guard.root
        parent = workspace.parent
        workspace_was_empty = _directory_is_empty(workspace)
        staging = parent / f".workspace-reset-stage-{uuid.uuid4().hex}"
        backup = parent / f".workspace-reset-backup-{uuid.uuid4().hex}"
        os.mkdir(staging)
        staging_guard = WorkspaceGuard(staging)
        _copy_seed_directory(seed_guard, staging_guard)
        _validate_physical_tree(staging, require_nonempty=True)
        seed_fingerprint = _physical_tree_fingerprint(seed_guard.root)
        if _physical_tree_fingerprint(staging_guard.root) != seed_fingerprint:
            raise _ResetError()
    except (OSError, ValueError, _ResetError):
        try:
            if "staging" in locals() and os.path.lexists(staging):
                _cleanup_recovery_directory(staging, _RESET_STAGING_PATTERN)
        except OSError:
            raise _ResetError("RESET_CLEANUP_FAILED") from None
        raise _ResetError() from None

    journal = {
        "version": 3,
        "workspace": workspace.name,
        "staging": staging.name,
        "backup": backup.name,
        "phase": "prepared",
        "workspace_was_empty": workspace_was_empty,
        "seed_fingerprint": seed_fingerprint,
    }
    try:
        _write_reset_journal(parent, journal)
        os.replace(workspace, backup)
        _fsync_directory(parent)
        journal["phase"] = "backup-created"
        _write_reset_journal(parent, journal)
        os.replace(staging, workspace)
        _fsync_directory(parent)
        journal["phase"] = "workspace-installed"
        _write_reset_journal(parent, journal)
    except Exception:
        try:
            _recover_workspace_state(settings)
        except ValueError:
            raise _ResetError("RESET_ROLLBACK_FAILED") from None
        raise _ResetError() from None

    try:
        if not _workspace_matches_seed_fingerprint(
            workspace,
            journal["seed_fingerprint"],
        ):
            _start_v3_rollback_restore(
                parent,
                workspace,
                staging,
                backup,
                journal,
            )
            raise _ResetError()
        _cleanup_reset_path(backup)
        _fsync_directory(parent)
        _remove_reset_journal(parent)
        _cleanup_journal_temps(parent)
    except Exception as error:
        if isinstance(error, _ResetError):
            raise
        raise _ResetError("RESET_CLEANUP_FAILED") from None


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
    _recover_workspace_state(configured)
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
            "max_read_bytes": configured.max_read_bytes,
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
