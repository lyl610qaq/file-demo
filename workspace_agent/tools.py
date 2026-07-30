import codecs
import hashlib
import os
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

    has_utf16_bom = sample.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE))
    if b"\x00" in sample and not has_utf16_bom:
        raise ToolInputError("file appears to be binary", "BINARY_FILE")

    if has_utf16_bom:
        candidates = ("utf-16",)
    else:
        candidates = ("utf-8-sig", "gb18030")

    size_path = revalidate()
    size = os.lstat(size_path).st_size
    final = size <= len(sample)
    for encoding in candidates:
        decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
        try:
            decoder.decode(sample, final=final)
        except UnicodeDecodeError:
            continue
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


def _parse_cursor(cursor: str | None) -> int:
    if cursor is None:
        return 0
    if not isinstance(cursor, str) or not cursor.isdecimal():
        raise ToolInputError("cursor must be a decimal string", "INVALID_CURSOR")
    return int(cursor)


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

    def list_dir(
        self,
        path: str = ".",
        recursive: bool = False,
        cursor: str | None = None,
        limit: int = 100,
    ) -> ToolResult:
        try:
            start = _parse_cursor(cursor)
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

            warnings: list[str] = []
            entries = list(
                self._iter_entries(path, recursive=recursive, warnings=warnings)
            )
            entries.sort(key=lambda entry: entry["path"])
            page = entries[start : start + page_size]
            has_more = start + page_size < len(entries)
            next_cursor = str(start + page_size) if has_more else None
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
            start = _parse_cursor(cursor)
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
            if stat.S_ISREG(metadata.st_mode):
                files: Iterator[str] = iter((relative_scope,))
            elif stat.S_ISDIR(metadata.st_mode):
                files = self._iter_files(relative_scope, warnings)
            else:
                raise ToolInputError(
                    "path must be a regular file or directory",
                    "INVALID_PATH_TYPE",
                )

            needle = query if case_sensitive else query.casefold()
            matches: list[dict[str, Any]] = []
            seen = 0
            iterator = self._iter_matches(
                files,
                needle,
                case_sensitive,
                warnings,
            )
            try:
                for match in iterator:
                    if seen >= start:
                        matches.append(match)
                        if len(matches) == page_size + 1:
                            break
                    seen += 1
            finally:
                iterator.close()

            has_more = len(matches) > page_size
            if has_more:
                matches.pop()
            next_cursor = str(start + page_size) if has_more else None
            return _success(
                {
                    "matches": matches,
                    "warnings": warnings,
                    "has_more": has_more,
                    "next_cursor": next_cursor,
                },
                f"Found {len(matches)} workspace matches",
            )
        except Exception as error:
            return _error_result("search_files", error)

    def read_file(
        self,
        path: str,
        offset: int = 0,
        limit: int = 16384,
    ) -> ToolResult:
        try:
            byte_offset = _bounded_int(
                offset,
                name="offset",
                minimum=0,
                maximum=2**63 - 1,
                code="INVALID_OFFSET",
            )
            byte_limit = _bounded_int(
                limit,
                name="limit",
                minimum=1,
                maximum=self.max_read_bytes,
            )

            resolved, _ = self._resolve_regular_file(path)
            encoding = _detect_encoding(
                resolved,
                revalidate=lambda: self.guard.resolve(
                    path,
                    must_exist=True,
                ),
            )
            resolved, metadata = self._resolve_regular_file(path)
            size = metadata.st_size
            if byte_offset > size or not self._is_character_boundary(
                path,
                byte_offset,
                encoding,
            ):
                raise ToolInputError(
                    "offset is not a valid character boundary",
                    "INVALID_OFFSET",
                )

            _, handle = self._open_file(path, "rb")
            with handle:
                handle.seek(byte_offset)
                chunk = handle.read(min(byte_limit, metadata.st_size - byte_offset))

            decoder_encoding = self._decoder_encoding(
                path,
                encoding,
                byte_offset,
            )
            decoder = codecs.getincrementaldecoder(decoder_encoding)(
                errors="strict"
            )
            content = decoder.decode(chunk, final=False)
            buffered, _ = decoder.getstate()
            consumed = len(chunk) - len(buffered)
            reaches_eof = byte_offset + len(chunk) == metadata.st_size
            if reaches_eof:
                content += decoder.decode(b"", final=True)
                consumed = len(chunk)

            next_offset = byte_offset + consumed
            has_more = next_offset < metadata.st_size
            if consumed == 0 and has_more:
                minimum_bytes = self._minimum_character_bytes(
                    path,
                    byte_offset,
                    encoding,
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
            relative_path = _relative(self.guard.root, resolved)
            return _success(
                {
                    "path": relative_path,
                    "content": content,
                    "offset": byte_offset,
                    "next_offset": next_offset,
                    "has_more": has_more,
                    "encoding": encoding,
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
    ) -> Iterator[dict[str, Any]]:
        _, metadata = self._lstat(directory)
        if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
            return

        for name in self._scan_names(directory):
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
            yield {
                "path": _relative(self.guard.root, child),
                "type": kind,
                "size": child_metadata.st_size,
                "modified_at": _modified_at(child_metadata),
            }
            if recursive and kind == "directory":
                yield from self._iter_entries(
                    relative_path,
                    recursive=True,
                    warnings=warnings,
                )

    def _iter_files(
        self,
        directory: str,
        warnings: list[str],
    ) -> Iterator[str]:
        _, metadata = self._lstat(directory)
        if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
            return

        candidates: list[tuple[str, str, str]] = []
        for name in self._scan_names(directory):
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
                yield relative_path
            else:
                yield from self._iter_files(relative_path, warnings)

    def _iter_matches(
        self,
        files: Iterator[str],
        needle: str,
        case_sensitive: bool,
        warnings: list[str],
    ) -> Iterator[dict[str, Any]]:
        for relative_path in files:
            try:
                resolved, _ = self._resolve_regular_file(relative_path)
                encoding = _detect_encoding(
                    resolved,
                    revalidate=lambda: self.guard.resolve(
                        relative_path,
                        must_exist=True,
                    ),
                )
                _, handle = self._open_file(
                    relative_path,
                    "r",
                    encoding=encoding,
                    errors="strict",
                    newline=None,
                )
                with handle:
                    for line_number, snippet in self._bounded_line_matches(
                        handle,
                        needle,
                        case_sensitive,
                    ):
                        yield {
                            "path": relative_path,
                            "line": line_number,
                            "snippet": snippet,
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

    @staticmethod
    def _bounded_line_matches(
        handle,
        needle: str,
        case_sensitive: bool,
    ) -> Iterator[tuple[int, str]]:
        line_number = 1
        carry = ""
        snippet = ""
        matched = False
        carry_size = max(len(needle) - 1, 0)

        while True:
            segment = handle.readline(_SEARCH_SEGMENT_CHARS)
            if segment == "":
                return

            if len(snippet) < _MAX_SNIPPET_CHARS:
                remaining = _MAX_SNIPPET_CHARS - len(snippet)
                snippet += segment[:remaining]

            searchable = segment if case_sensitive else segment.casefold()
            combined = carry + searchable
            if not matched and needle in combined:
                yield line_number, snippet.rstrip("\r\n")
                matched = True

            if segment.endswith("\n"):
                line_number += 1
                carry = ""
                snippet = ""
                matched = False
            elif carry_size:
                carry = combined[-carry_size:]

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

        decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
        remaining = offset
        _, handle = self._open_file(relative_path, "rb")
        with handle:
            while remaining:
                chunk = handle.read(min(_HASH_CHUNK_BYTES, remaining))
                if not chunk:
                    return False
                decoder.decode(chunk, final=False)
                remaining -= len(chunk)
        buffered, _ = decoder.getstate()
        return not buffered

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
        encoding: str,
    ) -> int | None:
        decoder_encoding = self._decoder_encoding(
            relative_path,
            encoding,
            offset,
        )
        _, handle = self._open_file(relative_path, "rb")
        with handle:
            handle.seek(offset)
            probe = handle.read(_MAX_CHARACTER_PROBE_BYTES)

        decoder = codecs.getincrementaldecoder(decoder_encoding)(
            errors="strict"
        )
        for byte_count, byte in enumerate(probe, start=1):
            if decoder.decode(bytes((byte,)), final=False):
                return byte_count
        return None
