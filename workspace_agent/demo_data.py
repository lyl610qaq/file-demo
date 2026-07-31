from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path, PurePosixPath


CORE_FILES: dict[str, str] = {
    "meetings/falcon-kickoff.md": (
        "---\n"
        "date: 2026-01-12\n"
        "---\n"
        "# Project Falcon kickoff\n\n"
        "The team started under the working name Project Falcon and agreed "
        "on the initial scope.\n"
    ),
    "meetings/falcon-rename.md": (
        "---\n"
        "date: 2026-03-20\n"
        "---\n"
        "# Project Falcon naming decision\n\n"
        "The current formal project name is Aquila, replacing the earlier "
        "Project Falcon working name.\n"
    ),
    "notes/falcon-risk.md": (
        "---\n"
        "date: 2026-02-04\n"
        "---\n"
        "# Project Falcon risk note\n\n"
        "Project Falcon depends on completion of the vendor security review.\n"
    ),
    "drafts/old-outline.md": (
        "---\n"
        "status: obsolete\n"
        "---\n"
        "# Old outline\n\n"
        "Replace this draft with the approved structure.\n"
    ),
    "drafts/current-name.md": (
        "---\n"
        "status: obsolete\n"
        "---\n"
        "# Naming draft\n\n"
        "This proposal was superseded by the Aquila decision.\n"
    ),
    "drafts/obsolete-by-name.md": (
        "---\n"
        "status: active\n"
        "---\n"
        "# Current migration draft\n\n"
        "Keep this file. Its filename is a classification trap.\n"
    ),
    "drafts/active-plan.md": (
        "---\n"
        "status: active\n"
        "---\n"
        "# Active delivery plan\n\n"
        "Keep this current plan in the drafts folder.\n"
    ),
    "data/owners.csv": (
        "team,owner\n"
        "platform,Mei\n"
        "security,Arun\n"
        "product,Lina\n"
    ),
    "logs/agent.log": (
        "2026-03-20 INFO demo workspace initialized\n"
        "2026-03-21 INFO validation completed\n"
    ),
}


def demo_files() -> dict[str, str]:
    files = dict(CORE_FILES)
    for number in range(1, 25):
        files[f"notes/general-{number:02d}.md"] = (
            "---\n"
            f"date: 2026-04-{number:02d}\n"
            "---\n"
            f"# General note {number:02d}\n\n"
            "This note is unrelated to the tracked project.\n"
        )
    return files


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0) & reparse_flag
    )


def materialize_demo_seed(root: Path) -> None:
    target_root = Path(os.path.abspath(root))
    _ensure_physical_directory_path(target_root.parent)
    target_existed = _validate_empty_target(target_root)
    staging: Path | None = Path(
        tempfile.mkdtemp(
            prefix=".demo-seed-stage-",
            dir=target_root.parent,
        )
    )
    removed_empty_target = False
    try:
        files = demo_files()
        _write_demo_files(staging, files)
        _validate_demo_tree(staging, files)
        if target_existed:
            os.rmdir(target_root)
            removed_empty_target = True
        try:
            os.replace(staging, target_root)
        except OSError:
            if removed_empty_target and not os.path.lexists(target_root):
                os.mkdir(target_root)
            raise
        staging = None
    finally:
        if staging is not None and os.path.lexists(staging):
            _remove_physical_tree(staging)


def _ensure_physical_directory_path(directory: Path) -> None:
    missing: list[Path] = []
    current = directory
    while not os.path.lexists(current):
        missing.append(current)
        parent = current.parent
        if parent == current:
            raise ValueError("demo seed ancestor must be a physical directory")
        current = parent

    ancestor = current
    while True:
        metadata = os.lstat(ancestor)
        if _is_link_or_reparse(metadata) or not stat.S_ISDIR(
            metadata.st_mode
        ):
            raise ValueError("demo seed ancestor must be a physical directory")
        parent = ancestor.parent
        if parent == ancestor:
            break
        ancestor = parent

    for path in reversed(missing):
        os.mkdir(path)
        metadata = os.lstat(path)
        if _is_link_or_reparse(metadata) or not stat.S_ISDIR(
            metadata.st_mode
        ):
            raise ValueError("demo seed ancestor must be a physical directory")


def _validate_empty_target(target: Path) -> bool:
    if not os.path.lexists(target):
        return False
    metadata = os.lstat(target)
    if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("demo seed target must be a physical directory")
    with os.scandir(target) as entries:
        if next(entries, None) is not None:
            raise FileExistsError("demo seed target is nonempty")
    return True


def _write_demo_files(root: Path, files: dict[str, str]) -> None:
    for relative, content in files.items():
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts:
            raise ValueError("demo seed contains an invalid path")
        destination = root.joinpath(*pure.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content.encode("utf-8"))


def _validate_demo_tree(root: Path, files: dict[str, str]) -> None:
    expected = {
        relative: content.encode("utf-8")
        for relative, content in files.items()
    }
    actual: dict[str, bytes] = {}
    pending = [root]
    while pending:
        directory = pending.pop()
        metadata = os.lstat(directory)
        if _is_link_or_reparse(metadata) or not stat.S_ISDIR(
            metadata.st_mode
        ):
            raise ValueError("demo seed staging is invalid")
        with os.scandir(directory) as entries:
            children = [Path(entry.path) for entry in entries]
        for child in children:
            child_metadata = os.lstat(child)
            if _is_link_or_reparse(child_metadata):
                raise ValueError("demo seed staging is invalid")
            if stat.S_ISDIR(child_metadata.st_mode):
                pending.append(child)
            elif stat.S_ISREG(child_metadata.st_mode):
                relative = child.relative_to(root).as_posix()
                actual[relative] = child.read_bytes()
            else:
                raise ValueError("demo seed staging is invalid")
    if actual != expected:
        raise ValueError("demo seed staging content is invalid")


def _remove_physical_tree(path: Path) -> None:
    metadata = os.lstat(path)
    if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
        if stat.S_ISDIR(metadata.st_mode):
            os.rmdir(path)
        else:
            os.unlink(path)
        return
    with os.scandir(path) as entries:
        children = [Path(entry.path) for entry in entries]
    for child in children:
        child_metadata = os.lstat(child)
        if stat.S_ISDIR(child_metadata.st_mode) and not _is_link_or_reparse(
            child_metadata
        ):
            _remove_physical_tree(child)
        elif stat.S_ISDIR(child_metadata.st_mode):
            os.rmdir(child)
        else:
            os.unlink(child)
    os.rmdir(path)
