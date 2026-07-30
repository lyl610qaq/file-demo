import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9-]+", flags=re.ASCII)


def sanitize_args(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(args)
    if tool == "write_file":
        content = sanitized.pop("content", "")
        encoded = content.encode("utf-8")
        sanitized["content_bytes"] = len(encoded)
        sanitized["content_sha256"] = hashlib.sha256(encoded).hexdigest()
    return sanitized


class TraceWriter:
    def __init__(self, path: Path, lock: threading.Lock) -> None:
        self.path = path
        self._lock = lock

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
        )
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as trace:
                trace.write(line)
                trace.write("\n")


class TraceStore:
    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser().resolve(strict=False)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(
            run_id
        ):
            raise ValueError("invalid run ID")
        return self.root / f"{run_id}.jsonl"

    def create(self, run_id: str) -> TraceWriter:
        path = self.path_for(run_id)
        path.touch(exist_ok=False)
        return TraceWriter(path, threading.Lock())
