import hashlib
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from datetime import datetime
from pathlib import Path

import pytest

from workspace_agent.safety import PathRejected, WorkspaceGuard
from workspace_agent.schemas import (
    ModelReply,
    RunResult,
    ToolCall,
    ToolResult,
    Usage,
)
from workspace_agent.trace import TraceStore, TraceWriter, sanitize_args


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
        pytest.fail(
            "directory junction creation failed "
            f"with exit code {created.returncode}: {failure}"
        )


def test_workspace_guard_accepts_existing_file_below_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    child = root / "docs" / "note.txt"
    child.parent.mkdir(parents=True)
    child.write_text("hello", encoding="utf-8")

    guard = WorkspaceGuard(root)

    assert guard.resolve("docs/note.txt", must_exist=True) == child.resolve()


@pytest.mark.parametrize(
    "invalid_path",
    [
        "../outside.txt",
        "..\\outside.txt",
        "",
        "nul\x00byte.txt",
    ],
)
def test_workspace_guard_rejects_unsafe_relative_paths(
    tmp_path: Path,
    invalid_path: str,
) -> None:
    guard = WorkspaceGuard(tmp_path / "workspace")

    with pytest.raises(PathRejected):
        guard.resolve(invalid_path)


def test_workspace_guard_rejects_absolute_paths(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path / "workspace")

    with pytest.raises(PathRejected):
        guard.resolve(str(tmp_path / "outside.txt"))


def test_workspace_guard_allows_dot(tmp_path: Path) -> None:
    root = tmp_path / "workspace"

    assert WorkspaceGuard(root).resolve(".") == root.resolve()


def test_workspace_guard_rejects_symlink_component(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    target = tmp_path / "target"
    root.mkdir()
    target.mkdir()
    (target / "secret.txt").write_text("secret", encoding="utf-8")
    _create_directory_symlink(root / "linked", target)

    guard = WorkspaceGuard(root)

    with pytest.raises(PathRejected):
        guard.resolve("linked/secret.txt", must_exist=True)


@pytest.mark.skipif(os.name != "nt", reason="Windows reparse-point test")
def test_workspace_guard_rejects_junction_component(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    junction = root / "junction"
    _create_windows_junction(junction, outside)

    try:
        with pytest.raises(PathRejected):
            WorkspaceGuard(root).resolve("junction/secret.txt")
    finally:
        os.rmdir(junction)


def test_workspace_guard_rejects_symlink_root(tmp_path: Path) -> None:
    target = tmp_path / "real-workspace"
    target.mkdir()
    linked_root = tmp_path / "linked-workspace"
    _create_directory_symlink(linked_root, target)

    with pytest.raises(PathRejected):
        WorkspaceGuard(linked_root)


def test_workspace_guard_rejects_symlink_ancestor(tmp_path: Path) -> None:
    target = tmp_path / "real-parent"
    target.mkdir()
    linked_parent = tmp_path / "linked-parent"
    _create_directory_symlink(linked_parent, target)

    try:
        with pytest.raises(PathRejected):
            WorkspaceGuard(linked_parent / "workspace")
    finally:
        linked_parent.unlink()


@pytest.mark.skipif(os.name != "nt", reason="Windows reparse-point test")
def test_workspace_guard_rejects_reparse_point_root(tmp_path: Path) -> None:
    target = tmp_path / "real-workspace"
    target.mkdir()
    junction = tmp_path / "junction-workspace"
    _create_windows_junction(junction, target)

    try:
        with pytest.raises(PathRejected):
            WorkspaceGuard(junction)
    finally:
        os.rmdir(junction)


@pytest.mark.skipif(os.name != "nt", reason="Windows reparse-point test")
def test_workspace_guard_rejects_junction_ancestor(tmp_path: Path) -> None:
    target = tmp_path / "real-parent"
    target.mkdir()
    junction = tmp_path / "junction-parent"
    _create_windows_junction(junction, target)

    try:
        with pytest.raises(PathRejected):
            WorkspaceGuard(junction / "workspace")
    finally:
        os.rmdir(junction)


def test_workspace_guard_requires_existing_path(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path / "workspace")

    with pytest.raises(FileNotFoundError):
        guard.resolve("missing/file.txt", must_exist=True)


def test_usage_addition_propagates_unknown_components() -> None:
    left = Usage(prompt_tokens=2, completion_tokens=None, total_tokens=5)
    right = Usage(prompt_tokens=3, completion_tokens=7, total_tokens=None)

    result = left.plus(right)

    assert result == Usage(
        prompt_tokens=5,
        completion_tokens=None,
        total_tokens=None,
    )
    assert result.as_dict() == {
        "prompt_tokens": 5,
        "completion_tokens": None,
        "total_tokens": None,
    }


def test_schema_dataclasses_are_frozen_and_have_expected_defaults() -> None:
    call = ToolCall(id="call-1", name="read_file", arguments={"path": "a.txt"})
    reply = ModelReply(message={"role": "assistant"}, tool_calls=(call,))
    run = RunResult(
        run_id="run-1",
        status="completed",
        message="done",
        model_calls=1,
        usage=Usage(total_tokens=4),
    )

    assert reply.usage == Usage()
    assert run.usage.total_tokens == 4
    with pytest.raises(FrozenInstanceError):
        call.name = "write_file"  # type: ignore[misc]


def test_tool_result_model_payload_marks_workspace_data_untrusted() -> None:
    result = ToolResult(
        ok=False,
        data={"content": "workspace text"},
        summary="read failed",
        error_code="read_error",
    )

    assert result.model_payload() == {
        "trust_boundary": "UNTRUSTED_WORKSPACE_DATA",
        "ok": False,
        "data": {"content": "workspace text"},
        "error_code": "read_error",
    }


def test_trace_store_create_immediately_creates_empty_file(
    tmp_path: Path,
) -> None:
    store = TraceStore(tmp_path / "traces")

    writer = store.create("run-123")

    try:
        assert writer.path == tmp_path / "traces" / "run-123.jsonl"
        assert writer.path.is_file()
        assert writer.path.read_bytes() == b""
    finally:
        writer.close()


def test_trace_writer_appends_one_compact_utf8_json_line(
    tmp_path: Path,
) -> None:
    writer = TraceStore(tmp_path / "traces").create("run-1")

    try:
        writer.append(
            step=2,
            tool="read_file",
            args={"path": "资料.txt"},
            result_summary="读取完成",
            status="ok",
        )
    finally:
        writer.close()

    raw = writer.path.read_bytes()
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 1
    assert b": " not in raw
    assert b", " not in raw
    record = json.loads(raw.decode("utf-8"))
    assert record["step"] == 2
    assert record["tool"] == "read_file"
    assert record["args"] == {"path": "资料.txt"}
    assert record["result_summary"] == "读取完成"
    assert record["status"] == "ok"
    assert datetime.fromisoformat(record["timestamp"]).tzinfo is not None


def test_trace_writer_replaces_write_content_with_size_and_sha256(
    tmp_path: Path,
) -> None:
    content = "sensitive 内容"
    writer = TraceStore(tmp_path / "traces").create("run-1")

    try:
        writer.append(
            step=1,
            tool="write_file",
            args={"path": "note.txt", "content": content, "overwrite": True},
            result_summary="written",
            status="ok",
        )
    finally:
        writer.close()

    record = json.loads(writer.path.read_text(encoding="utf-8"))
    assert "content" not in record["args"]
    assert record["args"] == {
        "path": "note.txt",
        "overwrite": True,
        "content_bytes": len(content.encode("utf-8")),
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


def test_sanitize_args_does_not_mutate_original_arguments() -> None:
    original = {
        "path": "note.txt",
        "content": "secret",
        "overwrite": True,
    }

    sanitized = sanitize_args("write_file", original)

    assert original == {
        "path": "note.txt",
        "content": "secret",
        "overwrite": True,
    }
    assert sanitized is not original


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_trace_writer_rejects_nonfinite_numbers_without_appending(
    tmp_path: Path,
    nonfinite: float,
) -> None:
    writer = TraceStore(tmp_path / "traces").create("run-1")

    try:
        with pytest.raises(ValueError, match="Out of range float values"):
            writer.append(
                step=1,
                tool="read_file",
                args={"value": nonfinite},
                result_summary="invalid",
                status="error",
            )
        assert writer.path.read_bytes() == b""
    finally:
        writer.close()


def test_trace_writer_keeps_writing_to_exclusively_created_file(
    tmp_path: Path,
) -> None:
    writer = TraceStore(tmp_path / "traces").create("run-1")
    original_file = writer.path.with_name("original.jsonl")
    replacement = writer.path.with_name("replacement.jsonl")
    replacement.write_text("", encoding="utf-8")

    try:
        try:
            os.replace(writer.path, original_file)
        except PermissionError:
            replacement_blocked = True
        else:
            replacement_blocked = False
            os.replace(replacement, writer.path)
        writer.append(
            step=1,
            tool="read_file",
            args={"path": "note.txt"},
            result_summary="read",
            status="ok",
        )
    finally:
        writer.close()

    trace_file = writer.path if replacement_blocked else original_file
    if not replacement_blocked:
        assert writer.path.read_bytes() == b""
    records = [
        json.loads(line)
        for line in trace_file.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["step"] for record in records] == [1]


def test_trace_writer_context_manager_closes_and_rejects_append(
    tmp_path: Path,
) -> None:
    with TraceStore(tmp_path / "traces").create("run-1") as writer:
        writer.append(
            step=1,
            tool="read_file",
            args={},
            result_summary="read",
            status="ok",
        )

    with pytest.raises(ValueError, match="closed"):
        writer.append(
            step=2,
            tool="read_file",
            args={},
            result_summary="read",
            status="ok",
        )


def test_trace_writer_retains_handle_when_constructed_directly(
    tmp_path: Path,
) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text("", encoding="utf-8")

    with TraceWriter(path, threading.Lock()) as writer:
        writer.append(
            step=1,
            tool="read_file",
            args={},
            result_summary="read",
            status="ok",
        )

    assert json.loads(path.read_text(encoding="utf-8"))["step"] == 1


def test_trace_writer_serializes_concurrent_appends(
    tmp_path: Path,
) -> None:
    writer = TraceStore(tmp_path / "traces").create("run-1")

    def append(step: int) -> None:
        writer.append(
            step=step,
            tool="read_file",
            args={"path": f"{step}.txt"},
            result_summary="read",
            status="ok",
        )

    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(append, range(100)))
    finally:
        writer.close()

    records = [
        json.loads(line)
        for line in writer.path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 100
    assert {record["step"] for record in records} == set(range(100))


def test_trace_store_accepts_run_id_at_maximum_length(
    tmp_path: Path,
) -> None:
    store = TraceStore(tmp_path / "traces")
    run_id = "r" * 64

    with store.create(run_id) as writer:
        assert writer.path == store.root / f"{run_id}.jsonl"


def test_trace_store_rejects_run_id_above_maximum_length(
    tmp_path: Path,
) -> None:
    store = TraceStore(tmp_path / "traces")
    run_id = "r" * 65

    with pytest.raises(ValueError, match="invalid run ID"):
        store.path_for(run_id)
    with pytest.raises(ValueError, match="invalid run ID"):
        store.create(run_id)


@pytest.mark.parametrize(
    "run_id",
    [
        "",
        "../run",
        "..\\run",
        "run/name",
        "run_name",
        "run.jsonl",
        "rún",
        " run ",
    ],
)
def test_trace_store_rejects_invalid_run_ids(
    tmp_path: Path,
    run_id: str,
) -> None:
    store = TraceStore(tmp_path / "traces")

    with pytest.raises(ValueError):
        store.path_for(run_id)
    with pytest.raises(ValueError):
        store.create(run_id)


def test_trace_store_refuses_to_overwrite_existing_run(
    tmp_path: Path,
) -> None:
    store = TraceStore(tmp_path / "traces")
    writer = store.create("same-run")

    try:
        with pytest.raises(FileExistsError):
            store.create("same-run")
    finally:
        writer.close()
