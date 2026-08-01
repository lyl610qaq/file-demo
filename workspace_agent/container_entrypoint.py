"""Minimal container process bootstrap; this is not an Agent CLI."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path


DATA_ROOT = Path("/data")
RUNTIME_UID = 10001
RUNTIME_GID = 10001
_PORT_PATTERN = re.compile(r"[0-9]+\Z", flags=re.ASCII)


def resolve_port(value: str | None) -> int:
    """Return a validated TCP port, defaulting to the public container port."""
    if value is None:
        return 8000
    if _PORT_PATTERN.fullmatch(value) is None:
        raise ValueError("PORT must be an ASCII decimal integer from 1 to 65535")
    port = int(value)
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be an ASCII decimal integer from 1 to 65535")
    return port


def _current_uid() -> int:
    geteuid = getattr(os, "geteuid", None)
    if geteuid is None:
        return RUNTIME_UID
    return int(geteuid())


def _is_reparse_point(metadata: os.stat_result) -> bool:
    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(getattr(metadata, "st_file_attributes", 0) & reparse_point)


def _require_physical_directory(path: Path, *, create: bool) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        if not create:
            raise RuntimeError(f"required runtime directory is missing: {path}") from None
        path.mkdir(mode=0o700)
        metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or _is_reparse_point(metadata)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise RuntimeError(f"runtime path must be a physical directory: {path}")


def initialize_runtime_directories() -> None:
    """Prepare only the two fixed, mutable container directories."""
    _require_physical_directory(DATA_ROOT, create=False)
    is_root = _current_uid() == 0
    for path in (DATA_ROOT / "workspace", DATA_ROOT / "traces"):
        _require_physical_directory(path, create=True)
        if is_root:
            chown = getattr(os, "chown", None)
            if chown is None:
                raise RuntimeError("root privilege drop is unavailable on this platform")
            chown(path, RUNTIME_UID, RUNTIME_GID)


def _drop_root_privileges() -> None:
    if _current_uid() != 0:
        return
    setgroups = getattr(os, "setgroups", None)
    setgid = getattr(os, "setgid", None)
    setuid = getattr(os, "setuid", None)
    if setgroups is None or setgid is None or setuid is None:
        raise RuntimeError("root privilege drop is unavailable on this platform")
    setgroups([])
    setgid(RUNTIME_GID)
    setuid(RUNTIME_UID)


def main() -> None:
    port = resolve_port(os.environ.get("PORT"))
    initialize_runtime_directories()
    _drop_root_privileges()
    os.execvp(
        "uvicorn",
        [
            "uvicorn",
            "workspace_agent.web:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
        ],
    )


if __name__ == "__main__":
    main()
