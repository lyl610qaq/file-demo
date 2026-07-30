from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, IO


MAX_RUN_ID_LENGTH = 64
_RUN_ID_PATTERN = re.compile(
    rf"[A-Za-z0-9-]{{1,{MAX_RUN_ID_LENGTH}}}",
    flags=re.ASCII,
)


def sanitize_args(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(args)
    if tool == "write_file":
        content = sanitized.pop("content", "")
        encoded = content.encode("utf-8")
        sanitized["content_bytes"] = len(encoded)
        sanitized["content_sha256"] = hashlib.sha256(encoded).hexdigest()
    return sanitized


class TraceWriter:
    def __init__(
        self,
        path: Path,
        lock: threading.Lock,
        handle: IO[str] | None = None,
    ) -> None:
        self.path = path
        self._lock = lock
        self._handle = handle or path.open(
            "a",
            encoding="utf-8",
            newline="\n",
        )

    def append(
        self,
        *,
        step: int,
        tool: str,
        args: dict[str, Any],
        result_summary: str,
        status: str,
    ) -> None:
        record = {
            "step": step,
            "tool": tool,
            "args": sanitize_args(tool, args),
            "result_summary": result_summary,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        line = json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._lock:
            if self._handle.closed:
                raise ValueError("trace writer is closed")
            self._handle.write(line)
            self._handle.write("\n")
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.close()

    def __enter__(self) -> TraceWriter:
        with self._lock:
            if self._handle.closed:
                raise ValueError("trace writer is closed")
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()


class TraceStore:
    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, run_id: str) -> Path:
        """Return the JSONL path for a 1-64 character ASCII run ID."""
        if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(
            run_id
        ):
            raise ValueError("invalid run ID")
        return self.root / f"{run_id}.jsonl"

    def create(self, run_id: str) -> TraceWriter:
        path = self.path_for(run_id)
        handle = path.open("x", encoding="utf-8", newline="\n")
        return TraceWriter(path, threading.Lock(), handle)
