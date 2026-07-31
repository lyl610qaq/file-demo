import importlib
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


def test_write_file_cleans_descriptor_and_temp_when_fstat_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    tools_module = _tools_module()
    original_open = tools_module.os.open
    original_fstat = tools_module.os.fstat
    opened_descriptors: list[int] = []

    def recording_open(path, flags, mode=0o777, *args, **kwargs) -> int:
        descriptor = original_open(path, flags, mode, *args, **kwargs)
        opened_descriptors.append(descriptor)
        return descriptor

    def failing_fstat(descriptor: int):
        raise OSError("simulated fstat failure")

    monkeypatch.setattr(tools_module.os, "open", recording_open)
    monkeypatch.setattr(tools_module.os, "fstat", failing_fstat)

    result = _tools(root).write_file(path="note.txt", content="content")

    assert result.ok is False
    assert result.error_code == "WRITE_FAILED"
    assert len(opened_descriptors) == 1
    with pytest.raises(OSError):
        original_fstat(opened_descriptors[0])
    assert list(root.iterdir()) == []


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


def test_mutations_resolve_operands_immediately_before_each_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "source.txt").write_text("source", encoding="utf-8")
    (root / "rollback-source.txt").write_text("rollback", encoding="utf-8")
    (root / "replace.txt").write_text("old", encoding="utf-8")
    events: list[tuple[str, ...]] = []

    class RecordingGuard(WorkspaceGuard):
        def resolve(self, relative: str, must_exist: bool = False) -> Path:
            resolved = super().resolve(relative, must_exist=must_exist)
            events.append(("resolve", relative.replace("\\", "/")))
            return resolved

    tools_module = _tools_module()
    original_mkdir = tools_module.os.mkdir
    original_open = tools_module.os.open
    original_link = tools_module.os.link
    original_replace = tools_module.os.replace
    original_unlink = tools_module.os.unlink
    failing_unlink: str | None = None

    def relative(path) -> str:
        return Path(path).relative_to(root).as_posix()

    def recording_mkdir(path, *args, **kwargs) -> None:
        events.append(("mkdir", relative(path)))
        original_mkdir(path, *args, **kwargs)

    def recording_open(path, flags, mode=0o777, *args, **kwargs) -> int:
        events.append(("open", relative(path)))
        return original_open(path, flags, mode, *args, **kwargs)

    def recording_link(source, destination, *args, **kwargs) -> None:
        events.append(("link", relative(source), relative(destination)))
        original_link(source, destination, *args, **kwargs)

    def recording_replace(source, destination, *args, **kwargs) -> None:
        events.append(("replace", relative(source), relative(destination)))
        original_replace(source, destination, *args, **kwargs)

    def recording_unlink(path, *args, **kwargs) -> None:
        path_relative = relative(path)
        events.append(("unlink", path_relative))
        if path_relative == failing_unlink:
            raise OSError("simulated source unlink failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(tools_module.os, "mkdir", recording_mkdir)
    monkeypatch.setattr(tools_module.os, "open", recording_open)
    monkeypatch.setattr(tools_module.os, "link", recording_link)
    monkeypatch.setattr(tools_module.os, "replace", recording_replace)
    monkeypatch.setattr(tools_module.os, "unlink", recording_unlink)
    tools = tools_module.WorkspaceTools(
        RecordingGuard(root),
        max_read_bytes=1024,
        max_write_bytes=1024,
    )
    events.clear()

    published = tools.write_file(
        path="nested/deep/published.txt",
        content="published",
    )
    replaced = tools.write_file(
        path="replace.txt",
        content="new",
        overwrite=True,
    )
    moved = tools.move_file(
        source="source.txt",
        destination="nested/moved.txt",
    )
    failing_unlink = "rollback-source.txt"
    rolled_back = tools.move_file(
        source="rollback-source.txt",
        destination="nested/rollback.txt",
    )

    assert published.ok
    assert replaced.ok
    assert moved.ok
    assert rolled_back.ok is False
    assert rolled_back.error_code == "MOVE_FAILED"

    mutation_indexes = [
        index
        for index, event in enumerate(events)
        if event[0] in {"mkdir", "open", "link", "replace", "unlink"}
    ]
    operation_counts = {
        operation: sum(events[index][0] == operation for index in mutation_indexes)
        for operation in {"mkdir", "open", "link", "replace", "unlink"}
    }
    assert operation_counts == {
        "mkdir": 2,
        "open": 2,
        "link": 3,
        "replace": 1,
        "unlink": 4,
    }, f"mutation operation coverage was {operation_counts!r}"
    for index in mutation_indexes:
        event = events[index]
        operation = event[0]
        if operation in {"mkdir", "open", "unlink"}:
            assert events[index - 1] == ("resolve", event[1]), (
                f"{operation}({event[1]}) was preceded by "
                f"{events[index - 1]!r}"
            )
        elif operation == "link":
            source_relative, destination_relative = event[1:]
            if Path(source_relative).name.startswith(".published.txt."):
                expected = [
                    ("resolve", source_relative),
                    ("resolve", "nested/deep"),
                    ("resolve", destination_relative),
                ]
            else:
                expected = [
                    ("resolve", "nested"),
                    ("resolve", source_relative),
                    ("resolve", destination_relative),
                ]
            assert events[index - 3 : index] == expected, (
                f"link({source_relative}, {destination_relative}) had "
                f"safety events {events[index - 3 : index]!r}"
            )
        else:
            source_relative, destination_relative = event[1:]
            assert events[index - 3 : index] == [
                ("resolve", source_relative),
                ("resolve", "."),
                ("resolve", destination_relative),
            ], (
                f"replace({source_relative}, {destination_relative}) had "
                f"safety events {events[index - 3 : index]!r}"
            )


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


@pytest.mark.skipif(os.name != "nt", reason="Windows junction test")
@pytest.mark.parametrize("operation", ["write", "move"])
@pytest.mark.parametrize("checkpoint", ["outer_parent", "inner_parent"])
def test_mutations_reject_multi_parent_junction_substitution(
    tmp_path: Path,
    operation: str,
    checkpoint: str,
) -> None:
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    if operation == "move":
        (root / "source.txt").write_text("source", encoding="utf-8")
    destination = "level1/level2/destination.txt"

    class SwappingGuard(WorkspaceGuard):
        swapped_path: Path | None = None

        def resolve(self, relative: str, must_exist: bool = False) -> Path:
            normalized = relative.replace("\\", "/")
            outer = root / "level1"
            inner = outer / "level2"
            if (
                checkpoint == "outer_parent"
                and normalized == "level1/level2"
                and outer.is_dir()
                and not inner.exists()
                and self.swapped_path is None
            ):
                _replace_with_directory_link(outer, outside)
                self.swapped_path = outer
            elif (
                checkpoint == "inner_parent"
                and normalized == destination
                and inner.is_dir()
                and self.swapped_path is None
            ):
                _replace_with_directory_link(inner, outside)
                self.swapped_path = inner
            return super().resolve(relative, must_exist=must_exist)

    guard = SwappingGuard(root)
    tools = _tools_module().WorkspaceTools(
        guard,
        max_read_bytes=1024,
        max_write_bytes=1024,
    )
    try:
        if operation == "write":
            result = tools.write_file(path=destination, content="blocked")
        else:
            result = tools.move_file(
                source="source.txt",
                destination=destination,
            )

        assert result.ok is False, f"{operation}/{checkpoint} unexpectedly succeeded"
        assert result.error_code == "PATH_REJECTED", (
            f"{operation}/{checkpoint} returned {result.error_code}"
        )
        assert list(outside.iterdir()) == [], (
            f"{operation}/{checkpoint} created an outside artifact"
        )
        if operation == "move":
            assert (root / "source.txt").read_text(
                encoding="utf-8"
            ) == "source"
    finally:
        if guard.swapped_path is not None:
            _remove_directory_link(guard.swapped_path)

    assert not (root / destination).exists()
    assert not list(root.rglob("*.tmp"))


def test_tool_schemas_match_exact_contract() -> None:
    schemas = _tools_module().TOOL_SCHEMAS
    expected_names = {
        "list_dir",
        "search_files",
        "read_file",
        "stat_path",
        "write_file",
        "move_file",
    }
    by_name = {
        schema["function"]["name"]: schema
        for schema in schemas
    }
    assert len(schemas) == 6, f"schema count was {len(schemas)}"
    assert set(by_name) == expected_names, (
        f"schema names were {sorted(by_name)}"
    )

    required = {
        "list_dir": [],
        "search_files": ["query"],
        "read_file": ["path"],
        "stat_path": ["path"],
        "write_file": ["path", "content"],
        "move_file": ["source", "destination"],
    }
    defaults = {
        "list_dir": {
            "path": ".",
            "recursive": False,
            "cursor": None,
            "limit": 100,
        },
        "search_files": {
            "path": ".",
            "cursor": None,
            "limit": 50,
            "case_sensitive": True,
        },
        "read_file": {"offset": 0, "cursor": None},
        "stat_path": {},
        "write_file": {"overwrite": False},
        "move_file": {},
    }
    numeric_bounds = {
        "list_dir": {"limit": (1, 200)},
        "search_files": {"limit": (1, 100)},
        "read_file": {
            "offset": (0, 2**63 - 1),
            "limit": (1, 2**31 - 1),
        },
    }

    for name in sorted(expected_names):
        schema = by_name[name]
        parameters = schema["function"]["parameters"]
        properties = parameters["properties"]
        assert schema["type"] == "function", f"{name}.type"
        assert parameters["type"] == "object", f"{name}.parameters.type"
        assert parameters["additionalProperties"] is False, (
            f"{name}.parameters.additionalProperties"
        )
        assert parameters["required"] == required[name], f"{name}.required"
        actual_defaults = {
            field: definition["default"]
            for field, definition in properties.items()
            if "default" in definition
        }
        assert actual_defaults == defaults[name], (
            f"{name}.defaults were {actual_defaults!r}"
        )
        for field, (minimum, maximum) in numeric_bounds.get(
            name,
            {},
        ).items():
            assert properties[field]["minimum"] == minimum, (
                f"{name}.{field}.minimum"
            )
            assert properties[field]["maximum"] == maximum, (
                f"{name}.{field}.maximum"
            )

    for name in ("list_dir", "search_files", "read_file"):
        cursor = by_name[name]["function"]["parameters"]["properties"][
            "cursor"
        ]
        cursor_types = {
            choice["type"] for choice in cursor["anyOf"]
        }
        assert cursor_types == {"string", "null"}, f"{name}.cursor.anyOf"

    forbidden = {"root", "workspaceid", "shell", "delete"}

    def assert_allowed_keys(value, location: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                assert key.lower() not in forbidden, f"{location}.{key}"
                assert_allowed_keys(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                assert_allowed_keys(child, f"{location}[{index}]")

    assert_allowed_keys(schemas, "TOOL_SCHEMAS")


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
