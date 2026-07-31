import base64
import binascii
import codecs
import hashlib
import hmac
import json
import os
import secrets
import stat
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from workspace_agent.safety import PathRejected, WorkspaceGuard
from workspace_agent.schemas import ToolResult


_DETECTION_BYTES = 4096
_HASH_CHUNK_BYTES = 65536
_MAX_WARNINGS = 100
_MAX_WARNING_CHARS = 500
_MAX_SNIPPET_CHARS = 500
_SEARCH_SEGMENT_CHARS = 4096
_MAX_CHARACTER_PROBE_BYTES = 8
_MAX_AMBIGUOUS_TEXT_CHARS = 8
_CURSOR_VERSION = 1


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List workspace directory entries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative directory path.",
                        "default": ".",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Include entries in nested directories.",
                        "default": False,
                    },
                    "cursor": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "description": "Authenticated continuation cursor.",
                        "default": None,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum entries to return.",
                        "minimum": 1,
                        "maximum": 200,
                        "default": 100,
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search workspace files for text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Non-empty text to find.",
                        "minLength": 1,
                    },
                    "path": {
                        "type": "string",
                        "description": "Relative file or directory path.",
                        "default": ".",
                    },
                    "cursor": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "description": "Authenticated continuation cursor.",
                        "default": None,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum matches to return.",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 50,
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Match letter case exactly.",
                        "default": True,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a bounded text segment from a workspace file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file path.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Byte offset at a character boundary.",
                        "minimum": 0,
                        "maximum": 2**63 - 1,
                        "default": 0,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum bytes to read.",
                        "minimum": 1,
                        "maximum": 2**31 - 1,
                    },
                    "cursor": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "description": "Authenticated continuation cursor.",
                        "default": None,
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stat_path",
            "description": "Inspect a workspace file or directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file or directory path.",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Atomically write UTF-8 text to a workspace file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative destination file path.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text to encode as UTF-8.",
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Atomically replace a safe existing file.",
                        "default": False,
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_file",
            "description": "Move a workspace file without replacing a target.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Relative source file path.",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Relative destination file path.",
                    },
                },
                "required": ["source", "destination"],
                "additionalProperties": False,
            },
        },
    },
]


class ToolInputError(ValueError):
    def __init__(
        self,
        message: str,
        code: str = "INVALID_INPUT",
        data: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.data = data or {}


def _detect_encoding(
    path: Path,
    *,
    revalidate: Callable[[], Path] | None = None,
) -> str:
    if revalidate is None:
        absolute = Path(os.path.abspath(path))
        local_guard = WorkspaceGuard(absolute.parent)
        revalidate = lambda: local_guard.resolve(
            absolute.name,
            must_exist=True,
        )

    sample_path = revalidate()
    with sample_path.open("rb") as handle:
        sample = handle.read(_DETECTION_BYTES)

    if sample.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        raise ToolInputError(
            "UTF-32 is not supported",
            "UNSUPPORTED_ENCODING",
        )

    has_utf16_bom = sample.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE))
    if b"\x00" in sample and not has_utf16_bom:
        raise ToolInputError("file appears to be binary", "BINARY_FILE")

    size_path = revalidate()
    size = os.lstat(size_path).st_size
    final = size <= len(sample)
    if has_utf16_bom:
        candidates = ("utf-16",)
    else:
        candidates = ("utf-8-sig", "gb18030")

    for encoding in candidates:
        decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
        try:
            decoder.decode(sample, final=final)
        except UnicodeDecodeError:
            continue
        if (
            encoding == "utf-8-sig"
            and size > len(sample)
            and sample.isascii()
        ):
            return "utf-8-or-gb18030"
        return encoding

    raise ToolInputError(
        "file charset could not be determined",
        "CHARSET_UNDETERMINED",
    )


def _relative(root: Path, path: Path) -> str:
    relative = path.relative_to(root).as_posix()
    return relative or "."


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(
        file_attributes & reparse_flag
    )


def _modified_at(metadata: os.stat_result) -> str:
    return datetime.fromtimestamp(
        metadata.st_mtime,
        tz=timezone.utc,
    ).isoformat()


def _path_type(metadata: os.stat_result) -> str:
    if stat.S_ISREG(metadata.st_mode):
        return "file"
    if stat.S_ISDIR(metadata.st_mode):
        return "directory"
    return "other"


def _base64_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _base64_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    try:
        decoded = base64.b64decode(
            value + padding,
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, ValueError) as error:
        raise ToolInputError("cursor is malformed", "INVALID_CURSOR") from error
    if _base64_encode(decoded) != value:
        raise ToolInputError("cursor is malformed", "INVALID_CURSOR")
    return decoded


def _encode_cursor(payload: dict[str, Any], key: bytes) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    signature = hmac.new(key, raw, hashlib.sha256).digest()
    return f"{_base64_encode(raw)}.{_base64_encode(signature)}"


def _decode_cursor(cursor: str, key: bytes) -> dict[str, Any]:
    if not isinstance(cursor, str):
        raise ToolInputError("cursor must be a string", "INVALID_CURSOR")
    try:
        payload_part, signature_part = cursor.split(".")
    except ValueError as error:
        raise ToolInputError("cursor is malformed", "INVALID_CURSOR") from error
    raw = _base64_decode(payload_part)
    signature = _base64_decode(signature_part)
    expected = hmac.new(key, raw, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected):
        raise ToolInputError("cursor authentication failed", "INVALID_CURSOR")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ToolInputError("cursor is malformed", "INVALID_CURSOR") from error
    if not isinstance(payload, dict) or payload.get("v") != _CURSOR_VERSION:
        raise ToolInputError("cursor version is invalid", "INVALID_CURSOR")
    return payload


def _bounded_int(
    value: Any,
    *,
    name: str,
    minimum: int,
    maximum: int,
    code: str = "INVALID_LIMIT",
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not minimum <= value <= maximum
    ):
        raise ToolInputError(
            f"{name} must be between {minimum} and {maximum}",
            code,
        )
    return value


def _warning(warnings: list[str], message: str) -> None:
    if len(warnings) < _MAX_WARNINGS:
        warnings.append(message[:_MAX_WARNING_CHARS])


def _common_prefix(values: list[str]) -> str:
    if not values:
        return ""
    prefix = values[0]
    for value in values[1:]:
        while not value.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


def _match_span(
    source: str,
    needle: str,
    case_sensitive: bool,
) -> tuple[int, int] | None:
    if case_sensitive:
        start = source.find(needle)
        return None if start < 0 else (start, start + len(needle))

    folded_parts: list[str] = []
    source_indexes: list[int] = []
    for source_index, character in enumerate(source):
        folded = character.casefold()
        folded_parts.append(folded)
        source_indexes.extend([source_index] * len(folded))
    start = "".join(folded_parts).find(needle)
    if start < 0:
        return None
    end = start + len(needle)
    return source_indexes[start], source_indexes[end - 1] + 1


def _success(data: dict[str, Any], summary: str) -> ToolResult:
    return ToolResult(ok=True, data=data, summary=summary[:500])


def _failure(
    code: str,
    summary: str,
    data: dict[str, Any] | None = None,
) -> ToolResult:
    return ToolResult(
        ok=False,
        data=data or {},
        summary=summary[:500],
        error_code=code,
    )


def _error_result(operation: str, error: Exception) -> ToolResult:
    if isinstance(error, ToolInputError):
        return _failure(
            error.code,
            f"{operation} failed: {error}",
            error.data,
        )
    if isinstance(error, PathRejected):
        return _failure("PATH_REJECTED", f"{operation} failed: path rejected")
    if isinstance(error, FileNotFoundError):
        return _failure("NOT_FOUND", f"{operation} failed: path not found")
    if isinstance(error, PermissionError):
        return _failure(
            "ACCESS_DENIED",
            f"{operation} failed: access denied",
        )
    if isinstance(error, UnicodeError):
        return _failure(
            "DECODE_ERROR",
            f"{operation} failed: text could not be decoded",
        )
    if isinstance(error, (TypeError, ValueError)):
        return _failure(
            "INVALID_INPUT",
            f"{operation} failed: invalid input",
        )
    return _failure("READ_ERROR", f"{operation} failed: filesystem error")


def _mutation_error_result(operation: str, error: Exception) -> ToolResult:
    if isinstance(error, ToolInputError):
        return _failure(
            error.code,
            f"{operation} failed: {error}",
            error.data,
        )
    if isinstance(error, PathRejected):
        return _failure("PATH_REJECTED", f"{operation} failed: path rejected")
    if isinstance(error, FileNotFoundError):
        return _failure("NOT_FOUND", f"{operation} failed: path not found")
    if isinstance(error, PermissionError):
        return _failure(
            "ACCESS_DENIED",
            f"{operation} failed: access denied",
        )
    if isinstance(error, (TypeError, ValueError)):
        return _failure(
            "INVALID_INPUT",
            f"{operation} failed: invalid input",
        )
    code = "WRITE_FAILED" if operation == "write_file" else "MOVE_FAILED"
    return _failure(code, f"{operation} failed: filesystem error")


def _dispatch_error_result(name: str, error: Exception) -> ToolResult:
    if isinstance(error, ToolInputError):
        return _failure(
            error.code,
            f"{name} failed: invalid input",
        )
    if isinstance(error, PathRejected):
        return _failure("PATH_REJECTED", f"{name} failed: path rejected")
    if isinstance(error, FileNotFoundError):
        return _failure("NOT_FOUND", f"{name} failed: path not found")
    if isinstance(error, PermissionError):
        return _failure("ACCESS_DENIED", f"{name} failed: access denied")
    if isinstance(error, (TypeError, ValueError)):
        return _failure(
            "INVALID_ARGUMENTS",
            f"{name} failed: invalid arguments",
        )
    return _failure(
        "TOOL_EXECUTION_FAILED",
        f"{name} failed: filesystem error",
    )


class WorkspaceTools:
    def __init__(
        self,
        guard: WorkspaceGuard,
        *,
        max_read_bytes: int,
        max_write_bytes: int,
    ) -> None:
        if not isinstance(guard, WorkspaceGuard):
            raise ToolInputError("guard must be a WorkspaceGuard")
        self.max_read_bytes = _bounded_int(
            max_read_bytes,
            name="max_read_bytes",
            minimum=1,
            maximum=2**31 - 1,
        )
        self.max_write_bytes = _bounded_int(
            max_write_bytes,
            name="max_write_bytes",
            minimum=1,
            maximum=2**31 - 1,
        )
        self.guard = guard
        self._cursor_key = secrets.token_bytes(32)

    def list_dir(
        self,
        path: str = ".",
        recursive: bool = False,
        cursor: str | None = None,
        limit: int = 100,
    ) -> ToolResult:
        try:
            page_size = _bounded_int(
                limit,
                name="limit",
                minimum=1,
                maximum=200,
            )
            if not isinstance(recursive, bool):
                raise ToolInputError("recursive must be a boolean")

            resolved, metadata = self._lstat(path)
            if _is_link_or_reparse(metadata):
                raise PathRejected("path contains a non-physical component")
            if not stat.S_ISDIR(metadata.st_mode):
                raise ToolInputError(
                    "path must be a directory",
                    "NOT_A_DIRECTORY",
                )

            relative_scope = _relative(self.guard.root, resolved)
            after: str | None = None
            if cursor is not None:
                payload = _decode_cursor(cursor, self._cursor_key)
                if (
                    payload.get("op") != "list"
                    or payload.get("path") != relative_scope
                    or payload.get("recursive") is not recursive
                    or not isinstance(payload.get("last"), str)
                ):
                    raise ToolInputError(
                        "cursor does not match list parameters",
                        "INVALID_CURSOR",
                    )
                after = payload["last"]

            warnings: list[str] = []
            entries: list[dict[str, Any]] = []
            iterator = self._iter_entries(
                relative_scope,
                recursive=recursive,
                warnings=warnings,
                after=after,
            )
            try:
                for entry in iterator:
                    entries.append(entry)
                    if len(entries) == page_size + 1:
                        break
            finally:
                iterator.close()

            has_more = len(entries) > page_size
            page = entries[:page_size]
            next_cursor = None
            if has_more and page:
                next_cursor = _encode_cursor(
                    {
                        "v": _CURSOR_VERSION,
                        "op": "list",
                        "path": relative_scope,
                        "recursive": recursive,
                        "last": page[-1]["path"],
                    },
                    self._cursor_key,
                )
            return _success(
                {
                    "entries": page,
                    "warnings": warnings,
                    "has_more": has_more,
                    "next_cursor": next_cursor,
                },
                f"Listed {len(page)} workspace entries",
            )
        except Exception as error:
            return _error_result("list_dir", error)

    def search_files(
        self,
        query: str,
        path: str = ".",
        cursor: str | None = None,
        limit: int = 50,
        case_sensitive: bool = True,
    ) -> ToolResult:
        try:
            if not isinstance(query, str) or not query:
                raise ToolInputError("query must be a non-empty string")
            if not isinstance(case_sensitive, bool):
                raise ToolInputError("case_sensitive must be a boolean")
            page_size = _bounded_int(
                limit,
                name="limit",
                minimum=1,
                maximum=100,
            )

            resolved, metadata = self._lstat(path)
            if _is_link_or_reparse(metadata):
                raise PathRejected("path contains a non-physical component")

            warnings: list[str] = []
            relative_scope = _relative(self.guard.root, resolved)
            after_path: str | None = None
            after_line = 0
            if cursor is not None:
                payload = _decode_cursor(cursor, self._cursor_key)
                query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
                if (
                    payload.get("op") != "search"
                    or payload.get("path") != relative_scope
                    or payload.get("query_sha256") != query_hash
                    or payload.get("case_sensitive") is not case_sensitive
                    or not isinstance(payload.get("last_path"), str)
                    or not isinstance(payload.get("last_line"), int)
                    or not isinstance(payload.get("size"), int)
                    or not isinstance(payload.get("mtime_ns"), int)
                ):
                    raise ToolInputError(
                        "cursor does not match search parameters",
                        "INVALID_CURSOR",
                    )
                after_path = payload["last_path"]
                after_line = payload["last_line"]
                try:
                    _, anchor_metadata = self._resolve_regular_file(after_path)
                except (FileNotFoundError, OSError, PathRejected, ToolInputError):
                    raise ToolInputError(
                        "search cursor anchor changed",
                        "STALE_CURSOR",
                    )
                if (
                    anchor_metadata.st_size != payload["size"]
                    or anchor_metadata.st_mtime_ns != payload["mtime_ns"]
                ):
                    raise ToolInputError(
                        "search cursor anchor changed",
                        "STALE_CURSOR",
                    )

            if stat.S_ISREG(metadata.st_mode):
                files: Iterator[str] = iter((relative_scope,))
            elif stat.S_ISDIR(metadata.st_mode):
                files = self._iter_files(
                    relative_scope,
                    warnings,
                    after_path=after_path,
                )
            else:
                raise ToolInputError(
                    "path must be a regular file or directory",
                    "INVALID_PATH_TYPE",
                )

            needle = query if case_sensitive else query.casefold()
            matches: list[dict[str, Any]] = []
            iterator = self._iter_matches(
                files,
                needle,
                case_sensitive,
                warnings,
                after_path=after_path,
                after_line=after_line,
            )
            try:
                for match in iterator:
                    matches.append(match)
                    if len(matches) == page_size + 1:
                        break
            finally:
                iterator.close()

            has_more = len(matches) > page_size
            page = matches[:page_size]
            next_cursor = None
            if has_more and page:
                last = page[-1]
                next_cursor = _encode_cursor(
                    {
                        "v": _CURSOR_VERSION,
                        "op": "search",
                        "path": relative_scope,
                        "query_sha256": hashlib.sha256(
                            query.encode("utf-8")
                        ).hexdigest(),
                        "case_sensitive": case_sensitive,
                        "last_path": last["path"],
                        "last_line": last["line"],
                        "size": last["_size"],
                        "mtime_ns": last["_mtime_ns"],
                    },
                    self._cursor_key,
                )
            public_matches = [
                {
                    "path": match["path"],
                    "line": match["line"],
                    "snippet": match["snippet"],
                }
                for match in page
            ]
            return _success(
                {
                    "matches": public_matches,
                    "warnings": warnings,
                    "has_more": has_more,
                    "next_cursor": next_cursor,
                },
                f"Found {len(public_matches)} workspace matches",
            )
        except Exception as error:
            return _error_result("search_files", error)

    def read_file(
        self,
        path: str,
        offset: int = 0,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> ToolResult:
        try:
            byte_offset = _bounded_int(
                offset,
                name="offset",
                minimum=0,
                maximum=2**63 - 1,
                code="INVALID_OFFSET",
            )
            requested_limit = (
                min(16384, self.max_read_bytes)
                if limit is None
                else limit
            )
            byte_limit = _bounded_int(
                requested_limit,
                name="limit",
                minimum=1,
                maximum=self.max_read_bytes,
            )

            resolved, metadata = self._resolve_regular_file(path)
            relative_path = _relative(self.guard.root, resolved)
            cursor_payload: dict[str, Any] | None = None
            if cursor is not None:
                cursor_payload = _decode_cursor(cursor, self._cursor_key)
                if (
                    cursor_payload.get("op") != "read"
                    or cursor_payload.get("path") != relative_path
                    or cursor_payload.get("offset") != byte_offset
                    or cursor_payload.get("encoding")
                    not in {
                        "utf-8-sig",
                        "utf-16",
                        "gb18030",
                        "utf-8-or-gb18030",
                    }
                    or not isinstance(cursor_payload.get("size"), int)
                    or not isinstance(cursor_payload.get("mtime_ns"), int)
                ):
                    raise ToolInputError(
                        "cursor does not match read parameters",
                        "INVALID_CURSOR",
                    )
                if (
                    metadata.st_size != cursor_payload["size"]
                    or metadata.st_mtime_ns != cursor_payload["mtime_ns"]
                ):
                    raise ToolInputError(
                        "read cursor file changed",
                        "STALE_CURSOR",
                    )
                encoding = cursor_payload["encoding"]
                if encoding == "utf-8-or-gb18030":
                    raw_candidates = cursor_payload.get("candidates")
                    if raw_candidates != ["utf-8-sig", "gb18030"]:
                        raise ToolInputError(
                            "cursor decoder state is invalid",
                            "INVALID_CURSOR",
                        )
                    candidates = tuple(raw_candidates)
                    raw_pending = cursor_payload.get("pending")
                    if not isinstance(raw_pending, str):
                        raise ToolInputError(
                            "cursor decoder state is invalid",
                            "INVALID_CURSOR",
                        )
                    pending_input = _base64_decode(raw_pending)
                    if len(pending_input) > _MAX_CHARACTER_PROBE_BYTES:
                        raise ToolInputError(
                            "cursor decoder state is invalid",
                            "INVALID_CURSOR",
                        )
                else:
                    candidates = (encoding,)
                    pending_input = b""
            else:
                encoding = _detect_encoding(
                    resolved,
                    revalidate=lambda: self.guard.resolve(
                        path,
                        must_exist=True,
                    ),
                )
                candidates = self._encoding_candidates(encoding)
                pending_input = b""

            size = metadata.st_size
            if byte_offset > size:
                raise ToolInputError(
                    "offset is not a valid character boundary",
                    "INVALID_OFFSET",
                )
            if cursor_payload is None and byte_offset:
                if encoding == "gb18030":
                    raise ToolInputError(
                        "GB18030 offsets require a continuation cursor",
                        "OFFSET_REQUIRES_CURSOR",
                    )
                if not self._is_character_boundary(
                    path,
                    byte_offset,
                    encoding,
                ):
                    code = (
                        "OFFSET_REQUIRES_CURSOR"
                        if encoding == "utf-8-or-gb18030"
                        else "INVALID_OFFSET"
                    )
                    raise ToolInputError(
                        "offset is not a valid character boundary",
                        code,
                    )
                if (
                    encoding == "utf-8-or-gb18030"
                    and not self._next_character_is_utf8(
                        path,
                        byte_offset,
                        size,
                    )
                ):
                    raise ToolInputError(
                        "GB18030 offsets require a continuation cursor",
                        "OFFSET_REQUIRES_CURSOR",
                    )

            if byte_offset + len(pending_input) > size:
                raise ToolInputError(
                    "cursor decoder state is invalid",
                    "INVALID_CURSOR",
                )
            included_pending = pending_input[:byte_limit]
            _, handle = self._open_file(path, "rb")
            with handle:
                handle.seek(byte_offset + len(included_pending))
                remaining_limit = byte_limit - len(included_pending)
                chunk = included_pending + handle.read(
                    min(
                        remaining_limit,
                        metadata.st_size
                        - byte_offset
                        - len(included_pending),
                    )
                )
            self._reject_binary_chunk(chunk, encoding)

            (
                content,
                consumed,
                selected_encoding,
                surviving_candidates,
            ) = self._decode_read_chunk(
                path,
                candidates,
                byte_offset,
                chunk,
                byte_offset + len(chunk) == metadata.st_size,
            )
            if (
                cursor_payload is None
                and byte_offset
                and encoding == "utf-8-or-gb18030"
                and selected_encoding == "gb18030"
            ):
                raise ToolInputError(
                    "GB18030 offsets require a continuation cursor",
                    "OFFSET_REQUIRES_CURSOR",
                )

            next_offset = byte_offset + consumed
            has_more = next_offset < metadata.st_size
            next_pending = (
                chunk[consumed:]
                if selected_encoding == "utf-8-or-gb18030"
                else b""
            )
            if len(next_pending) > _MAX_CHARACTER_PROBE_BYTES:
                raise ToolInputError(
                    "file charset remains ambiguous",
                    "CHARSET_UNDETERMINED",
                )
            if consumed == 0 and has_more:
                minimum_bytes = self._minimum_character_bytes(
                    path,
                    byte_offset,
                    surviving_candidates,
                )
                data = (
                    {"minimum_bytes": minimum_bytes}
                    if minimum_bytes is not None
                    else {}
                )
                raise ToolInputError(
                    "limit does not contain the next complete character",
                    "LIMIT_SPLITS_CHARACTER",
                    data,
                )
            next_cursor = None
            if has_more:
                next_cursor = _encode_cursor(
                    {
                        "v": _CURSOR_VERSION,
                        "op": "read",
                        "path": relative_path,
                        "offset": next_offset,
                        "encoding": selected_encoding,
                        **(
                            {
                                "candidates": list(surviving_candidates),
                                "pending": _base64_encode(next_pending),
                            }
                            if selected_encoding == "utf-8-or-gb18030"
                            else {}
                        ),
                        "size": metadata.st_size,
                        "mtime_ns": metadata.st_mtime_ns,
                    },
                    self._cursor_key,
                )
            return _success(
                {
                    "path": relative_path,
                    "content": content,
                    "offset": byte_offset,
                    "next_offset": next_offset,
                    "has_more": has_more,
                    "encoding": selected_encoding,
                    "next_cursor": next_cursor,
                },
                f"Read {consumed} bytes from {relative_path}",
            )
        except Exception as error:
            return _error_result("read_file", error)

    def stat_path(self, path: str) -> ToolResult:
        try:
            resolved, metadata = self._lstat(path)
            if _is_link_or_reparse(metadata):
                raise PathRejected("path contains a non-physical component")

            kind = _path_type(metadata)
            digest: str | None = None
            if kind == "file":
                resolved, metadata = self._resolve_regular_file(path)
                hasher = hashlib.sha256()
                _, handle = self._open_file(path, "rb")
                with handle:
                    for chunk in iter(
                        lambda: handle.read(_HASH_CHUNK_BYTES),
                        b"",
                    ):
                        hasher.update(chunk)
                digest = hasher.hexdigest()

            relative_path = _relative(self.guard.root, resolved)
            return _success(
                {
                    "path": relative_path,
                    "type": kind,
                    "size": metadata.st_size,
                    "modified_at": _modified_at(metadata),
                    "sha256": digest,
                },
                f"Inspected {relative_path}",
            )
        except Exception as error:
            return _error_result("stat_path", error)

    def write_file(
        self,
        *,
        path: str,
        content: str,
        overwrite: bool = False,
    ) -> ToolResult:
        temp_relative: str | None = None
        temp_identity: tuple[int, int] | None = None
        descriptor: int | None = None
        try:
            if not isinstance(content, str):
                raise ToolInputError("content must be a string")
            if not isinstance(overwrite, bool):
                raise ToolInputError("overwrite must be a boolean")
            payload = content.encode("utf-8")
            if len(payload) > self.max_write_bytes:
                raise ToolInputError(
                    "content exceeds the write byte limit",
                    "WRITE_TOO_LARGE",
                )

            relative_path, parent_relative = self._prepare_target(path)
            target, target_metadata = self._target_state(relative_path)
            if target_metadata is not None:
                if not overwrite:
                    raise ToolInputError(
                        "target already exists",
                        "TARGET_EXISTS",
                    )
                if not stat.S_ISREG(target_metadata.st_mode):
                    raise ToolInputError(
                        "existing target must be a regular file",
                        "INVALID_TARGET",
                    )

            (
                temp_relative,
                descriptor,
                temp_identity,
            ) = self._create_temp_file(parent_relative, target.name)
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                descriptor = None
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())

            temp = self._resolve_owned_file(
                temp_relative,
                temp_identity,
            )
            self._validate_directory(parent_relative)
            target, target_metadata = self._target_state(relative_path)
            if overwrite:
                if (
                    target_metadata is not None
                    and not stat.S_ISREG(target_metadata.st_mode)
                ):
                    raise ToolInputError(
                        "existing target must be a regular file",
                        "INVALID_TARGET",
                    )
                os.replace(temp, target)
                temp_relative = None
            else:
                if target_metadata is not None:
                    raise ToolInputError(
                        "target already exists",
                        "TARGET_EXISTS",
                    )
                try:
                    os.link(temp, target)
                except FileExistsError as error:
                    raise ToolInputError(
                        "target already exists",
                        "TARGET_EXISTS",
                    ) from error
                if not self._cleanup_owned_file(
                    temp_relative,
                    temp_identity,
                ):
                    rollback_succeeded = self._remove_owned_file(
                        relative_path,
                        temp_identity,
                    )
                    self._cleanup_owned_file(
                        temp_relative,
                        temp_identity,
                    )
                    target_exists = self._safe_path_exists(relative_path)
                    temp_exists = self._safe_path_exists(temp_relative)
                    temp_relative = None
                    rollback_complete = (
                        rollback_succeeded and not target_exists
                    )
                    code = (
                        "WRITE_CLEANUP_FAILED"
                        if rollback_complete
                        else "WRITE_ROLLBACK_FAILED"
                    )
                    return _failure(
                        code,
                        "write_file failed: publication cleanup failed",
                        {
                            "path": relative_path,
                            "target_exists": target_exists,
                            "temp_exists": temp_exists,
                        },
                    )
                temp_relative = None

            return _success(
                {
                    "path": relative_path,
                    "bytes": len(payload),
                },
                f"Wrote {len(payload)} bytes to {relative_path}",
            )
        except (
            FileNotFoundError,
            PathRejected,
            ToolInputError,
            TypeError,
            ValueError,
            OSError,
        ) as error:
            if temp_relative is not None and temp_identity is not None:
                if not self._cleanup_owned_file(
                    temp_relative,
                    temp_identity,
                ):
                    target_exists = self._safe_path_exists(relative_path)
                    temp_exists = self._safe_path_exists(temp_relative)
                    temp_relative = None
                    return _failure(
                        "WRITE_CLEANUP_FAILED",
                        "write_file failed: temporary cleanup failed",
                        {
                            "path": relative_path,
                            "target_exists": target_exists,
                            "temp_exists": temp_exists,
                        },
                    )
                temp_relative = None
            return _mutation_error_result("write_file", error)
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if temp_relative is not None and temp_identity is not None:
                self._cleanup_owned_file(temp_relative, temp_identity)

    def move_file(
        self,
        *,
        source: str,
        destination: str,
    ) -> ToolResult:
        destination_created = False
        source_identity: tuple[int, int] | None = None
        source_relative: str | None = None
        destination_relative: str | None = None
        try:
            source_path, source_metadata = self._resolve_regular_file(source)
            source_relative = _relative(self.guard.root, source_path)
            source_identity = (
                source_metadata.st_dev,
                source_metadata.st_ino,
            )
            destination_relative, parent_relative = self._prepare_target(
                destination
            )

            self._validate_directory(parent_relative)
            source_path, source_metadata = self._resolve_regular_file(
                source_relative
            )
            if (
                source_metadata.st_dev,
                source_metadata.st_ino,
            ) != source_identity:
                raise ToolInputError(
                    "source changed before move",
                    "MOVE_FAILED",
                )
            destination_path, destination_metadata = self._target_state(
                destination_relative
            )
            if destination_metadata is not None:
                raise ToolInputError(
                    "target already exists",
                    "TARGET_EXISTS",
                )

            try:
                os.link(source_path, destination_path)
            except FileExistsError as error:
                raise ToolInputError(
                    "target already exists",
                    "TARGET_EXISTS",
                ) from error
            destination_created = True

            try:
                source_path, source_metadata = self._resolve_regular_file(
                    source_relative
                )
                if (
                    source_metadata.st_dev,
                    source_metadata.st_ino,
                ) != source_identity:
                    raise OSError("source changed before removal")
                os.unlink(source_path)
            except (
                FileNotFoundError,
                PathRejected,
                ToolInputError,
                TypeError,
                ValueError,
                OSError,
            ):
                rollback_succeeded = self._rollback_move(
                    destination_relative,
                    source_identity,
                )
                destination_created = not rollback_succeeded
                if not rollback_succeeded:
                    return _failure(
                        "MOVE_ROLLBACK_FAILED",
                        "move_file failed: destination rollback failed",
                        {
                            "source": source_relative,
                            "destination": destination_relative,
                            "source_exists": self._safe_path_exists(
                                source_relative
                            ),
                            "destination_exists": self._safe_path_exists(
                                destination_relative
                            ),
                        },
                    )
                return _failure(
                    "MOVE_FAILED",
                    "move_file failed: source could not be removed",
                )

            destination_created = False
            return _success(
                {
                    "source": source_relative,
                    "destination": destination_relative,
                },
                f"Moved {source_relative} to {destination_relative}",
            )
        except (
            FileNotFoundError,
            PathRejected,
            ToolInputError,
            TypeError,
            ValueError,
            OSError,
        ) as error:
            if (
                destination_created
                and destination_relative is not None
                and source_identity is not None
            ):
                rollback_succeeded = self._rollback_move(
                    destination_relative,
                    source_identity,
                )
                destination_created = not rollback_succeeded
                if not rollback_succeeded:
                    return _failure(
                        "MOVE_ROLLBACK_FAILED",
                        "move_file failed: destination rollback failed",
                        {
                            "source": source_relative,
                            "destination": destination_relative,
                            "source_exists": self._safe_path_exists(
                                source_relative
                            ),
                            "destination_exists": self._safe_path_exists(
                                destination_relative
                            ),
                        },
                    )
            return _mutation_error_result("move_file", error)

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        if not isinstance(name, str):
            return _failure(
                "INVALID_TOOL_NAME",
                "tool execution failed: name must be a string",
            )
        handlers: dict[str, Callable[..., ToolResult]] = {
            "list_dir": self.list_dir,
            "search_files": self.search_files,
            "read_file": self.read_file,
            "stat_path": self.stat_path,
            "write_file": self.write_file,
            "move_file": self.move_file,
        }
        if not isinstance(arguments, dict):
            return _failure(
                "INVALID_ARGUMENTS",
                "tool execution failed: arguments must be an object",
            )
        handler = handlers.get(name)
        if handler is None:
            return _failure(
                "UNKNOWN_TOOL",
                "tool execution failed: unknown tool",
            )
        try:
            return handler(**arguments)
        except (
            FileNotFoundError,
            PathRejected,
            ToolInputError,
            TypeError,
            ValueError,
            OSError,
        ) as error:
            return _dispatch_error_result(name, error)
        except MemoryError:
            raise
        except Exception:
            return _failure(
                "TOOL_EXECUTION_FAILED",
                f"{name} failed: unexpected tool error",
            )

    def _prepare_target(self, relative_path: str) -> tuple[str, str]:
        resolved = self.guard.resolve(relative_path, must_exist=False)
        normalized = _relative(self.guard.root, resolved)
        if normalized == ".":
            raise ToolInputError(
                "target must name a file",
                "INVALID_TARGET",
            )

        components = normalized.split("/")
        parent_relative = "."
        for component in components[:-1]:
            self._validate_directory(parent_relative)
            child_relative = (
                component
                if parent_relative == "."
                else f"{parent_relative}/{component}"
            )
            child, child_metadata = self._optional_lstat(child_relative)
            if child_metadata is None:
                child = self.guard.resolve(
                    child_relative,
                    must_exist=False,
                )
                try:
                    os.mkdir(child)
                except FileExistsError:
                    pass
                child, child_metadata = self._lstat(child_relative)
            if (
                _is_link_or_reparse(child_metadata)
                or not stat.S_ISDIR(child_metadata.st_mode)
            ):
                raise PathRejected(
                    "target parent is not a physical directory"
                )
            parent_relative = child_relative

        self._validate_directory(parent_relative)
        self._target_state(normalized)
        return normalized, parent_relative

    def _optional_lstat(
        self,
        relative_path: str,
    ) -> tuple[Path, os.stat_result | None]:
        resolved = self.guard.resolve(relative_path, must_exist=False)
        try:
            return resolved, os.lstat(resolved)
        except FileNotFoundError:
            return resolved, None

    def _target_state(
        self,
        relative_path: str,
    ) -> tuple[Path, os.stat_result | None]:
        resolved, metadata = self._optional_lstat(relative_path)
        if metadata is not None and _is_link_or_reparse(metadata):
            raise PathRejected("path contains a non-physical component")
        return resolved, metadata

    def _validate_directory(self, relative_path: str) -> Path:
        resolved, metadata = self._lstat(relative_path)
        if (
            _is_link_or_reparse(metadata)
            or not stat.S_ISDIR(metadata.st_mode)
        ):
            raise PathRejected("path is not a physical directory")
        return resolved

    def _create_temp_file(
        self,
        parent_relative: str,
        target_name: str,
    ) -> tuple[str, int, tuple[int, int]]:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_BINARY", 0)
        for _ in range(128):
            temp_name = f".{target_name}.{secrets.token_hex(16)}.tmp"
            temp_relative = (
                temp_name
                if parent_relative == "."
                else f"{parent_relative}/{temp_name}"
            )
            temp = self.guard.resolve(temp_relative, must_exist=False)
            try:
                descriptor = os.open(temp, flags, 0o600)
            except FileExistsError:
                continue
            try:
                metadata = os.fstat(descriptor)
                ownership = (
                    temp_relative,
                    descriptor,
                    (metadata.st_dev, metadata.st_ino),
                )
            except BaseException as error:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                cleanup_succeeded = False
                try:
                    temp = self.guard.resolve(
                        temp_relative,
                        must_exist=True,
                    )
                    os.unlink(temp)
                    _, remaining = self._optional_lstat(temp_relative)
                    cleanup_succeeded = remaining is None
                except (
                    FileNotFoundError,
                    PathRejected,
                    ToolInputError,
                    TypeError,
                    ValueError,
                    OSError,
                ):
                    pass
                if not cleanup_succeeded:
                    raise ToolInputError(
                        "temporary file cleanup failed",
                        "WRITE_CLEANUP_FAILED",
                        {"temp_exists": True},
                    ) from error
                raise
            return ownership
        raise OSError("could not allocate temporary file")

    def _resolve_owned_file(
        self,
        relative_path: str,
        identity: tuple[int, int],
    ) -> Path:
        resolved, metadata = self._resolve_regular_file(relative_path)
        if (metadata.st_dev, metadata.st_ino) != identity:
            raise PathRejected("temporary file changed")
        return resolved

    def _cleanup_owned_file(
        self,
        relative_path: str,
        identity: tuple[int, int],
    ) -> bool:
        return self._remove_owned_file(relative_path, identity)

    def _remove_owned_file(
        self,
        relative_path: str,
        identity: tuple[int, int],
    ) -> bool:
        try:
            resolved, metadata = self._resolve_regular_file(relative_path)
            if (
                _is_link_or_reparse(metadata)
                or (metadata.st_dev, metadata.st_ino) != identity
            ):
                return False
            os.unlink(resolved)
            _, remaining = self._optional_lstat(relative_path)
            return remaining is None
        except FileNotFoundError:
            return True
        except (
            PathRejected,
            ToolInputError,
            TypeError,
            ValueError,
            OSError,
        ):
            return False

    def _rollback_move(
        self,
        destination: str,
        source_identity: tuple[int, int],
    ) -> bool:
        return self._remove_owned_file(destination, source_identity)

    def _safe_path_exists(self, relative_path: str) -> bool:
        try:
            _, metadata = self._optional_lstat(relative_path)
            return metadata is not None
        except (
            FileNotFoundError,
            PathRejected,
            ToolInputError,
            TypeError,
            ValueError,
            OSError,
        ):
            return True

    def _resolve_regular_file(
        self,
        relative_path: str,
    ) -> tuple[Path, os.stat_result]:
        resolved, metadata = self._lstat(relative_path)
        if _is_link_or_reparse(metadata):
            raise PathRejected("path contains a non-physical component")
        if not stat.S_ISREG(metadata.st_mode):
            raise ToolInputError(
                "path must be a regular file",
                "NOT_A_FILE",
            )
        return resolved, metadata

    def _lstat(
        self,
        relative_path: str,
    ) -> tuple[Path, os.stat_result]:
        resolved = self.guard.resolve(relative_path, must_exist=True)
        return resolved, os.lstat(resolved)

    def _open_file(self, relative_path: str, mode: str, **kwargs):
        resolved = self.guard.resolve(relative_path, must_exist=True)
        return resolved, resolved.open(mode, **kwargs)

    def _scan_names(self, relative_path: str) -> list[str]:
        resolved = self.guard.resolve(relative_path, must_exist=True)
        with os.scandir(resolved) as scan:
            return sorted(entry.name for entry in scan)

    def _iter_entries(
        self,
        directory: str,
        *,
        recursive: bool,
        warnings: list[str],
        _nested: bool = False,
        after: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        try:
            _, metadata = self._lstat(directory)
            if _is_link_or_reparse(metadata) or not stat.S_ISDIR(
                metadata.st_mode
            ):
                return
            names = self._scan_names(directory)
        except (FileNotFoundError, OSError, PathRejected):
            if not _nested:
                raise
            _warning(
                warnings,
                f"Skipped unreadable directory: {directory}",
            )
            return

        candidates: list[tuple[str, str, dict[str, Any] | str]] = []
        for name in names:
            relative_path = (
                name if directory == "." else f"{directory.rstrip('/')}/{name}"
            )
            try:
                child, child_metadata = self._lstat(relative_path)
                if _is_link_or_reparse(child_metadata):
                    _warning(
                        warnings,
                        f"Skipped unsafe entry: {relative_path}",
                    )
                    continue
            except (FileNotFoundError, OSError, PathRejected):
                _warning(
                    warnings,
                    f"Skipped unsafe or unreadable entry: {relative_path}",
                )
                continue

            kind = _path_type(child_metadata)
            entry = {
                "path": _relative(self.guard.root, child),
                "type": kind,
                "size": child_metadata.st_size,
                "modified_at": _modified_at(child_metadata),
            }
            candidates.append((relative_path, "emit", entry))
            if recursive and kind == "directory":
                candidates.append((f"{relative_path}/", "recurse", relative_path))

        for _, action, value in sorted(candidates, key=lambda item: item[0]):
            if action == "emit":
                entry = value  # type: ignore[assignment]
                if after is None or entry["path"] > after:
                    yield entry
            else:
                prefix = f"{value}/"
                if (
                    after is not None
                    and prefix <= after
                    and not after.startswith(prefix)
                ):
                    continue
                yield from self._iter_entries(
                    value,  # type: ignore[arg-type]
                    recursive=True,
                    warnings=warnings,
                    _nested=True,
                    after=after,
                )

    def _iter_files(
        self,
        directory: str,
        warnings: list[str],
        _nested: bool = False,
        after_path: str | None = None,
    ) -> Iterator[str]:
        try:
            _, metadata = self._lstat(directory)
            if _is_link_or_reparse(metadata) or not stat.S_ISDIR(
                metadata.st_mode
            ):
                return
            names = self._scan_names(directory)
        except (FileNotFoundError, OSError, PathRejected):
            if not _nested:
                raise
            _warning(
                warnings,
                f"Skipped unreadable directory: {directory}",
            )
            return

        candidates: list[tuple[str, str, str]] = []
        for name in names:
            relative_path = (
                name if directory == "." else f"{directory.rstrip('/')}/{name}"
            )
            try:
                _, child_metadata = self._lstat(relative_path)
                if _is_link_or_reparse(child_metadata):
                    _warning(
                        warnings,
                        f"Skipped unsafe entry: {relative_path}",
                    )
                    continue
            except (FileNotFoundError, OSError, PathRejected):
                _warning(
                    warnings,
                    f"Skipped unsafe or unreadable entry: {relative_path}",
                )
                continue

            kind = _path_type(child_metadata)
            if kind == "file":
                candidates.append((relative_path, relative_path, kind))
            elif kind == "directory":
                candidates.append((f"{relative_path}/", relative_path, kind))

        for _, relative_path, kind in sorted(candidates):
            if kind == "file":
                if after_path is None or relative_path >= after_path:
                    yield relative_path
            else:
                prefix = f"{relative_path}/"
                if (
                    after_path is not None
                    and prefix < after_path
                    and not after_path.startswith(prefix)
                ):
                    continue
                yield from self._iter_files(
                    relative_path,
                    warnings,
                    _nested=True,
                    after_path=after_path,
                )

    def _iter_matches(
        self,
        files: Iterator[str],
        needle: str,
        case_sensitive: bool,
        warnings: list[str],
        *,
        after_path: str | None = None,
        after_line: int = 0,
    ) -> Iterator[dict[str, Any]]:
        for relative_path in files:
            if after_path is not None and relative_path < after_path:
                continue
            try:
                resolved, metadata = self._resolve_regular_file(relative_path)
                encoding = _detect_encoding(
                    resolved,
                    revalidate=lambda: self.guard.resolve(
                        relative_path,
                        must_exist=True,
                    ),
                )
                decoded_chunks = self._iter_decoded_chunks(
                    relative_path,
                    encoding,
                )
                for line_number, snippet in self._bounded_line_matches(
                    self._normalize_newlines(decoded_chunks),
                    needle,
                    case_sensitive,
                ):
                    if (
                        relative_path == after_path
                        and line_number <= after_line
                    ):
                        continue
                    yield {
                        "path": relative_path,
                        "line": line_number,
                        "snippet": snippet,
                        "_size": metadata.st_size,
                        "_mtime_ns": metadata.st_mtime_ns,
                    }
            except ToolInputError as error:
                _warning(
                    warnings,
                    f"Skipped {relative_path}: {error}",
                )
            except UnicodeError:
                _warning(
                    warnings,
                    f"Skipped {relative_path}: text could not be decoded",
                )
            except (OSError, PathRejected):
                _warning(
                    warnings,
                    f"Skipped {relative_path}: file could not be read safely",
                )

    def _iter_decoded_chunks(
        self,
        relative_path: str,
        encoding: str,
    ) -> Iterator[str]:
        candidates = self._encoding_candidates(encoding)
        decoders = {
            candidate: codecs.getincrementaldecoder(candidate)(
                errors="strict"
            )
            for candidate in candidates
        }
        pending = {candidate: "" for candidate in candidates}
        _, handle = self._open_file(relative_path, "rb")
        with handle:
            while True:
                chunk = handle.read(_SEARCH_SEGMENT_CHARS)
                final = not chunk
                if chunk:
                    self._reject_binary_chunk(chunk, encoding)

                decoded: dict[str, str] = {}
                last_error: UnicodeDecodeError | None = None
                for candidate in tuple(decoders):
                    try:
                        decoded[candidate] = decoders[candidate].decode(
                            chunk,
                            final=final,
                        )
                    except UnicodeDecodeError as error:
                        last_error = error
                        del decoders[candidate]
                        del pending[candidate]

                if not decoders:
                    assert last_error is not None
                    raise last_error
                if len(decoders) == 1:
                    selected = next(iter(decoders))
                    text = pending[selected] + decoded.get(selected, "")
                    pending[selected] = ""
                    if text:
                        yield text
                else:
                    for candidate, text in decoded.items():
                        pending[candidate] += text
                    common = _common_prefix(list(pending.values()))
                    if common:
                        yield common
                        for candidate in pending:
                            pending[candidate] = pending[candidate][
                                len(common):
                            ]
                    if final:
                        preferred = next(iter(decoders))
                        if pending[preferred]:
                            yield pending[preferred]
                        return
                    if any(
                        len(text) > _MAX_AMBIGUOUS_TEXT_CHARS
                        for text in pending.values()
                    ):
                        raise ToolInputError(
                            "file charset remains ambiguous",
                            "CHARSET_UNDETERMINED",
                        )
                if final:
                    return

    @staticmethod
    def _normalize_newlines(chunks: Iterator[str]) -> Iterator[str]:
        pending_carriage_return = False
        for chunk in chunks:
            if pending_carriage_return:
                if chunk.startswith("\n"):
                    chunk = chunk[1:]
                chunk = "\n" + chunk
                pending_carriage_return = False
            if chunk.endswith("\r"):
                chunk = chunk[:-1]
                pending_carriage_return = True
            normalized = chunk.replace("\r\n", "\n").replace("\r", "\n")
            if normalized:
                yield normalized
        if pending_carriage_return:
            yield "\n"

    @staticmethod
    def _bounded_line_matches(
        chunks: Iterator[str],
        needle: str,
        case_sensitive: bool,
    ) -> Iterator[tuple[int, str]]:
        line_number = 1
        line_tail = ""
        matched = False
        tail_size = max(_MAX_SNIPPET_CHARS, len(needle))

        for chunk in chunks:
            for segment in chunk.splitlines(keepends=True):
                snippet_source = line_tail + segment
                if not matched:
                    span = _match_span(
                        snippet_source,
                        needle,
                        case_sensitive,
                    )
                    if span is not None:
                        match_start, match_end = span
                        start = max(0, match_start - 100)
                        if match_end - start > _MAX_SNIPPET_CHARS:
                            start = max(
                                0,
                                match_end - _MAX_SNIPPET_CHARS,
                            )
                        snippet = snippet_source[
                            start : start + _MAX_SNIPPET_CHARS
                        ].rstrip("\n")
                        yield line_number, snippet
                        matched = True

                if segment.endswith("\n"):
                    line_number += 1
                    line_tail = ""
                    matched = False
                else:
                    line_tail = snippet_source[-tail_size:]

    def _is_character_boundary(
        self,
        relative_path: str,
        offset: int,
        encoding: str,
    ) -> bool:
        if offset == 0:
            return True

        _, metadata = self._resolve_regular_file(relative_path)
        if offset > metadata.st_size:
            return False
        if encoding in {"utf-8-sig", "utf-8-or-gb18030"}:
            start = max(0, offset - 4)
            probe = self._read_bytes(
                relative_path,
                start,
                min(8, metadata.st_size - start),
            )
            index = offset - start
            if index < len(probe) and probe[index] & 0xC0 == 0x80:
                return False
            prefix = probe[:index]
            if not prefix:
                return True
            continuation_count = 0
            position = len(prefix) - 1
            while position >= 0 and prefix[position] & 0xC0 == 0x80:
                continuation_count += 1
                position -= 1
            if position < 0:
                return False
            lead = prefix[position]
            if lead < 0x80:
                expected = 1
            elif 0xC2 <= lead <= 0xDF:
                expected = 2
            elif 0xE0 <= lead <= 0xEF:
                expected = 3
            elif 0xF0 <= lead <= 0xF4:
                expected = 4
            else:
                return False
            return continuation_count + 1 == expected
        if encoding == "utf-16":
            if offset < 2 or (offset - 2) % 2:
                return False
            bom = self._read_bytes(relative_path, 0, 2)
            byteorder = (
                "little" if bom == codecs.BOM_UTF16_LE else "big"
            )
            start = max(2, offset - 2)
            probe = self._read_bytes(
                relative_path,
                start,
                min(4, metadata.st_size - start),
            )
            if offset == 2:
                return True
            previous = int.from_bytes(probe[:2], byteorder)
            current = (
                int.from_bytes(probe[2:4], byteorder)
                if len(probe) >= 4
                else None
            )
            return not (
                0xD800 <= previous <= 0xDBFF
                and current is not None
                and 0xDC00 <= current <= 0xDFFF
            )
        return False

    def _read_bytes(
        self,
        relative_path: str,
        offset: int,
        size: int,
    ) -> bytes:
        _, handle = self._open_file(relative_path, "rb")
        with handle:
            handle.seek(offset)
            return handle.read(size)

    def _next_character_is_utf8(
        self,
        relative_path: str,
        offset: int,
        file_size: int,
    ) -> bool:
        probe = self._read_bytes(
            relative_path,
            offset,
            min(4, file_size - offset),
        )
        if not probe:
            return True

        first = probe[0]
        if first < 0x80:
            needed = 1
        elif 0xC2 <= first <= 0xDF:
            needed = 2
        elif 0xE0 <= first <= 0xEF:
            needed = 3
        elif 0xF0 <= first <= 0xF4:
            needed = 4
        else:
            return False

        if len(probe) < needed:
            return False
        try:
            probe[:needed].decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return False
        return True

    @staticmethod
    def _encoding_candidates(encoding: str) -> tuple[str, ...]:
        if encoding == "utf-8-or-gb18030":
            return ("utf-8-sig", "gb18030")
        return (encoding,)

    @staticmethod
    def _reject_binary_chunk(chunk: bytes, encoding: str) -> None:
        if encoding != "utf-16" and b"\x00" in chunk:
            raise ToolInputError(
                "file appears to be binary",
                "BINARY_FILE",
            )

    def _decode_read_chunk(
        self,
        relative_path: str,
        candidates: tuple[str, ...],
        offset: int,
        chunk: bytes,
        reaches_eof: bool,
    ) -> tuple[str, int, str, tuple[str, ...]]:
        results: dict[str, tuple[str, int]] = {}
        last_error: UnicodeDecodeError | None = None
        for candidate in candidates:
            decoder_encoding = self._decoder_encoding(
                relative_path,
                candidate,
                offset,
            )
            decoder = codecs.getincrementaldecoder(decoder_encoding)(
                errors="strict"
            )
            try:
                content = decoder.decode(chunk, final=False)
                buffered, _ = decoder.getstate()
                consumed = len(chunk) - len(buffered)
                if reaches_eof:
                    content += decoder.decode(b"", final=True)
                    consumed = len(chunk)
            except UnicodeDecodeError as error:
                last_error = error
                continue
            results[candidate] = (content, consumed)

        if not results:
            assert last_error is not None
            raise last_error
        if len(results) == 1:
            selected = next(iter(results))
            content, consumed = results[selected]
            return content, consumed, selected, (selected,)

        contents = [result[0] for result in results.values()]
        consumed_values = {result[1] for result in results.values()}
        if reaches_eof and len(set(contents)) > 1:
            selected = next(
                candidate
                for candidate in candidates
                if candidate in results
            )
            content, consumed = results[selected]
            return content, consumed, selected, (selected,)
        if len(set(contents)) == 1 and len(consumed_values) == 1:
            consumed = consumed_values.pop()
            return (
                contents[0],
                consumed,
                "utf-8-or-gb18030",
                tuple(results),
            )

        shared_content, shared_bytes = self._shared_decode_boundary(
            relative_path,
            tuple(results),
            offset,
            chunk,
        )
        return (
            shared_content,
            shared_bytes,
            "utf-8-or-gb18030",
            tuple(results),
        )

    def _shared_decode_boundary(
        self,
        relative_path: str,
        candidates: tuple[str, ...],
        offset: int,
        chunk: bytes,
    ) -> tuple[str, int]:
        decoders = {
            candidate: codecs.getincrementaldecoder(
                self._decoder_encoding(
                    relative_path,
                    candidate,
                    offset,
                )
            )(errors="strict")
            for candidate in candidates
        }
        outputs = {candidate: "" for candidate in candidates}
        shared_content = ""
        shared_bytes = 0

        for byte_count, byte in enumerate(chunk, start=1):
            for candidate in tuple(decoders):
                try:
                    outputs[candidate] += decoders[candidate].decode(
                        bytes((byte,)),
                        final=False,
                    )
                except UnicodeDecodeError:
                    del decoders[candidate]
                    del outputs[candidate]
            if len(decoders) < 2:
                break
            if len(set(outputs.values())) == 1 and all(
                not decoder.getstate()[0]
                for decoder in decoders.values()
            ):
                shared_content = next(iter(outputs.values()))
                shared_bytes = byte_count
        return shared_content, shared_bytes

    def _decoder_encoding(
        self,
        relative_path: str,
        encoding: str,
        offset: int,
    ) -> str:
        if offset == 0:
            return encoding
        if encoding == "utf-8-sig":
            return "utf-8"
        if encoding == "utf-16":
            _, handle = self._open_file(relative_path, "rb")
            with handle:
                bom = handle.read(2)
            return "utf-16-le" if bom == codecs.BOM_UTF16_LE else "utf-16-be"
        return encoding

    def _minimum_character_bytes(
        self,
        relative_path: str,
        offset: int,
        candidates: tuple[str, ...],
    ) -> int | None:
        _, handle = self._open_file(relative_path, "rb")
        with handle:
            handle.seek(offset)
            probe = handle.read(_MAX_CHARACTER_PROBE_BYTES)

        minimums: list[int] = []
        for candidate in candidates:
            decoder_encoding = self._decoder_encoding(
                relative_path,
                candidate,
                offset,
            )
            decoder = codecs.getincrementaldecoder(decoder_encoding)(
                errors="strict"
            )
            try:
                for byte_count, byte in enumerate(probe, start=1):
                    if decoder.decode(bytes((byte,)), final=False):
                        minimums.append(byte_count)
                        break
            except UnicodeDecodeError:
                continue
        return min(minimums) if minimums else None
