import importlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from workspace_agent.safety import WorkspaceGuard


def _tools_module():
    return importlib.import_module("workspace_agent.tools")


def _tools(root: Path, *, max_write_bytes: int = 1024):
    return _tools_module().WorkspaceTools(
        WorkspaceGuard(root),
        max_read_bytes=16384,
        max_write_bytes=max_write_bytes,
    )


def _create_directory_symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")


def _create_windows_junction(junction: Path, target: Path) -> None:
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(target)],
        check=False,
        capture_output=True,
        text=True,
    )
    if created.returncode != 0:
        failure = (created.stderr or created.stdout).strip() or "no output"
        pytest.fail(f"junction creation failed: {failure}")


def _replace_with_directory_link(link: Path, target: Path) -> None:
    os.rmdir(link)
    if os.name == "nt":
        _create_windows_junction(link, target)
    else:
        _create_directory_symlink(link, target)


def _remove_directory_link(link: Path) -> None:
    if os.name == "nt":
        os.rmdir(link)
    else:
        link.unlink()


def test_write_file_creates_parents_and_reports_utf8_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()

    result = _tools(root).write_file(
        path="nested/deep/note.txt",
        content="A\u00e9\u4e2d",
    )

    assert result.ok
    assert result.data == {
        "path": "nested/deep/note.txt",
        "bytes": len("A\u00e9\u4e2d".encode("utf-8")),
    }
    assert (root / "nested" / "deep" / "note.txt").read_text(
        encoding="utf-8"
    ) == "A\u00e9\u4e2d"
    assert list((root / "nested" / "deep").iterdir()) == [
        root / "nested" / "deep" / "note.txt"
    ]
    assert str(root) not in repr(result)


def test_write_file_without_overwrite_preserves_existing_target(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "note.txt"
    target.write_text("original", encoding="utf-8")

    result = _tools(root).write_file(path="note.txt", content="replacement")

    assert result.ok is False
    assert result.error_code == "TARGET_EXISTS"
    assert target.read_text(encoding="utf-8") == "original"
    assert list(root.iterdir()) == [target]


def test_write_file_overwrite_uses_atomic_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "note.txt"
    target.write_text("original", encoding="utf-8")
    tools_module = _tools_module()
    original_replace = tools_module.os.replace
    observations: list[tuple[bytes, bytes]] = []

    def observing_replace(source, destination) -> None:
        observations.append(
            (Path(source).read_bytes(), Path(destination).read_bytes())
        )
        original_replace(source, destination)

    monkeypatch.setattr(tools_module.os, "replace", observing_replace)

    result = _tools(root).write_file(
        path="note.txt",
        content="replacement",
        overwrite=True,
    )

    assert result.ok
    assert observations == [(b"replacement", b"original")]
    assert target.read_bytes() == b"replacement"
    assert list(root.iterdir()) == [target]


def test_write_file_cleans_temp_after_publish_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "note.txt"
    target.write_text("original", encoding="utf-8")
    tools_module = _tools_module()

    def failing_replace(source, destination) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(tools_module.os, "replace", failing_replace)

    result = _tools(root).write_file(
        path="note.txt",
        content="replacement",
        overwrite=True,
    )

    assert result.ok is False
    assert result.error_code == "WRITE_FAILED"
    assert target.read_text(encoding="utf-8") == "original"
    assert list(root.iterdir()) == [target]


def test_write_file_rejects_non_string_and_oversized_utf8_content(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    tools = _tools(root, max_write_bytes=3)

    non_string = tools.write_file(path="wrong.txt", content=b"bytes")
    oversized = tools.write_file(path="large.txt", content="\u00e9\u00e9")

    assert non_string.ok is False
    assert non_string.error_code == "INVALID_INPUT"
    assert oversized.ok is False
    assert oversized.error_code == "WRITE_TOO_LARGE"
    assert list(root.iterdir()) == []


def test_write_file_rejects_path_escape(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()

    result = _tools(root).write_file(
        path="../escape.txt",
        content="blocked",
    )

    assert result.ok is False
    assert result.error_code == "PATH_REJECTED"
    assert not (tmp_path / "escape.txt").exists()


def test_write_file_rejects_directory_symlink(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    _create_directory_symlink(root / "linked", outside)

    result = _tools(root).write_file(
        path="linked/escape.txt",
        content="blocked",
    )

    assert result.ok is False
    assert result.error_code == "PATH_REJECTED"
    assert not (outside / "escape.txt").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction test")
def test_write_file_rejects_directory_junction(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    junction = root / "junction"
    _create_windows_junction(junction, outside)
    try:
        result = _tools(root).write_file(
            path="junction/escape.txt",
            content="blocked",
        )

        assert result.ok is False
        assert result.error_code == "PATH_REJECTED"
        assert not (outside / "escape.txt").exists()
    finally:
        os.rmdir(junction)


def test_move_file_creates_parents_removes_source_and_preserves_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    source = root / "source.bin"
    payload = b"\x00\xffpreserved"
    source.write_bytes(payload)

    result = _tools(root).move_file(
        source="source.bin",
        destination="nested/deep/destination.bin",
    )

    assert result.ok
    assert result.data == {
        "source": "source.bin",
        "destination": "nested/deep/destination.bin",
    }
    assert not source.exists()
    assert (root / "nested" / "deep" / "destination.bin").read_bytes() == payload
    assert str(root) not in repr(result)


def test_move_file_refuses_existing_destination(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    source = root / "source.txt"
    destination = root / "destination.txt"
    source.write_text("source", encoding="utf-8")
    destination.write_text("destination", encoding="utf-8")

    result = _tools(root).move_file(
        source="source.txt",
        destination="destination.txt",
    )

    assert result.ok is False
    assert result.error_code == "TARGET_EXISTS"
    assert source.read_text(encoding="utf-8") == "source"
    assert destination.read_text(encoding="utf-8") == "destination"


def test_move_file_rolls_back_destination_when_source_unlink_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    source = root / "source.txt"
    destination = root / "destination.txt"
    source.write_text("source", encoding="utf-8")
    tools_module = _tools_module()
    original_unlink = tools_module.os.unlink

    def failing_source_unlink(path, *args, **kwargs) -> None:
        if Path(path) == source:
            raise OSError("simulated source unlink failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(tools_module.os, "unlink", failing_source_unlink)

    result = _tools(root).move_file(
        source="source.txt",
        destination="destination.txt",
    )

    assert result.ok is False
    assert result.error_code == "MOVE_FAILED"
    assert source.read_text(encoding="utf-8") == "source"
    assert not destination.exists()


def test_mutations_revalidate_guard_before_publication_operations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "source.txt").write_text("source", encoding="utf-8")
    events: list[tuple[str, str]] = []

    class RecordingGuard(WorkspaceGuard):
        def resolve(self, relative: str, must_exist: bool = False) -> Path:
            resolved = super().resolve(relative, must_exist=must_exist)
            events.append(("resolve", relative.replace("\\", "/")))
            return resolved

    tools_module = _tools_module()
    original_link = tools_module.os.link
    original_replace = tools_module.os.replace

    def recording_link(source, destination, *args, **kwargs) -> None:
        events.append(
            (
                "link",
                f"{Path(source).name}->{Path(destination).name}",
            )
        )
        original_link(source, destination, *args, **kwargs)

    def recording_replace(source, destination, *args, **kwargs) -> None:
        events.append(
            (
                "replace",
                f"{Path(source).name}->{Path(destination).name}",
            )
        )
        original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(tools_module.os, "link", recording_link)
    monkeypatch.setattr(tools_module.os, "replace", recording_replace)
    tools = tools_module.WorkspaceTools(
        RecordingGuard(root),
        max_read_bytes=1024,
        max_write_bytes=1024,
    )

    written = tools.write_file(
        path="written.txt",
        content="written",
        overwrite=True,
    )
    moved = tools.move_file(
        source="source.txt",
        destination="nested/moved.txt",
    )

    assert written.ok
    assert moved.ok
    replace_index = next(
        index for index, event in enumerate(events) if event[0] == "replace"
    )
    move_link_index = max(
        index for index, event in enumerate(events) if event[0] == "link"
    )
    assert ("resolve", "written.txt") in events[max(0, replace_index - 4) : replace_index]
    assert ("resolve", "source.txt") in events[max(0, move_link_index - 8) : move_link_index]
    assert ("resolve", "nested/moved.txt") in events[
        max(0, move_link_index - 8) : move_link_index
    ]
    assert sum(event == ("resolve", "written.txt") for event in events) >= 2
    assert sum(event == ("resolve", "source.txt") for event in events) >= 2
    assert sum(
        event == ("resolve", "nested/moved.txt") for event in events
    ) >= 2


@pytest.mark.parametrize("operation", ["write", "move"])
def test_mutations_reject_parent_replaced_with_nonphysical_link(
    tmp_path: Path,
    operation: str,
) -> None:
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    if operation == "move":
        (root / "source.txt").write_text("source", encoding="utf-8")

    class SwappingGuard(WorkspaceGuard):
        swapped = False

        def resolve(self, relative: str, must_exist: bool = False) -> Path:
            normalized = relative.replace("\\", "/")
            parent = root / "nested"
            if (
                normalized == "nested/destination.txt"
                and parent.is_dir()
                and not self.swapped
            ):
                _replace_with_directory_link(parent, outside)
                self.swapped = True
            return super().resolve(relative, must_exist=must_exist)

    guard = SwappingGuard(root)
    tools = _tools_module().WorkspaceTools(
        guard,
        max_read_bytes=1024,
        max_write_bytes=1024,
    )
    try:
        if operation == "write":
            result = tools.write_file(
                path="nested/destination.txt",
                content="blocked",
            )
        else:
            result = tools.move_file(
                source="source.txt",
                destination="nested/destination.txt",
            )

        assert result.ok is False
        assert result.error_code == "PATH_REJECTED"
        assert not (outside / "destination.txt").exists()
        if operation == "move":
            assert (root / "source.txt").read_text(encoding="utf-8") == "source"
    finally:
        if guard.swapped:
            _remove_directory_link(root / "nested")


def test_tool_schemas_define_exact_public_surface() -> None:
    schemas = _tools_module().TOOL_SCHEMAS
    names = [schema["function"]["name"] for schema in schemas]

    assert names == [
        "list_dir",
        "search_files",
        "read_file",
        "stat_path",
        "write_file",
        "move_file",
    ]
    assert len(schemas) == 6
    for schema in schemas:
        parameters = schema["function"]["parameters"]
        assert schema["type"] == "function"
        assert parameters["type"] == "object"
        assert parameters["additionalProperties"] is False

    read_properties = schemas[2]["function"]["parameters"]["properties"]
    assert "cursor" in read_properties
    assert {"type": "null"} in read_properties["cursor"]["anyOf"]

    serialized = json.dumps(schemas).lower()
    for forbidden in ("root", "workspaceid", "delete", "shell"):
        assert forbidden not in serialized


def test_execute_dispatches_all_six_tools(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "source.txt").write_text("needle", encoding="utf-8")
    tools = _tools(root)
    calls: list[tuple[str, dict[str, Any]]] = [
        ("list_dir", {}),
        ("search_files", {"query": "needle"}),
        ("read_file", {"path": "source.txt"}),
        ("stat_path", {"path": "source.txt"}),
        ("write_file", {"path": "written.txt", "content": "written"}),
        (
            "move_file",
            {"source": "written.txt", "destination": "moved.txt"},
        ),
    ]

    results = [tools.execute(name, arguments) for name, arguments in calls]

    assert all(result.ok for result in results)
    assert (root / "moved.txt").read_text(encoding="utf-8") == "written"


def test_execute_rejects_unknown_tool_non_dict_and_missing_arguments(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    tools = _tools(root)

    unknown = tools.execute("delete_file", {})
    non_dict = tools.execute("read_file", ["note.txt"])
    missing = tools.execute("write_file", {"path": "note.txt"})

    assert unknown.ok is False
    assert unknown.error_code == "UNKNOWN_TOOL"
    assert non_dict.ok is False
    assert non_dict.error_code == "INVALID_ARGUMENTS"
    assert missing.ok is False
    assert missing.error_code == "INVALID_ARGUMENTS"


def test_execute_converts_expected_errors_without_leaking_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    tools = _tools(root)

    def failing_write_file(**arguments):
        raise OSError(f"cannot access {root}")

    monkeypatch.setattr(tools, "write_file", failing_write_file)

    result = tools.execute(
        "write_file",
        {"path": "note.txt", "content": "content"},
    )

    assert result.ok is False
    assert result.error_code == "TOOL_EXECUTION_FAILED"
    assert str(root) not in result.summary
    assert str(root) not in repr(result.model_payload())
