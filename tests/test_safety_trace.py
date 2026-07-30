import hashlib
import json
import os
import subprocess
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
from workspace_agent.trace import TraceStore


def _create_directory_symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")


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
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        check=False,
        capture_output=True,
        text=True,
    )
    if created.returncode != 0:
        failure = (created.stderr or created.stdout).strip()
        pytest.skip(
            "directory junction creation failed "
            f"with exit code {created.returncode}: {failure}"
        )

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


@pytest.mark.skipif(os.name != "nt", reason="Windows reparse-point test")
def test_workspace_guard_rejects_reparse_point_root(tmp_path: Path) -> None:
    target = tmp_path / "real-workspace"
    target.mkdir()
    junction = tmp_path / "junction-workspace"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(target)],
        check=False,
        capture_output=True,
        text=True,
    )
    if created.returncode != 0:
        pytest.skip(f"directory junctions are unavailable: {created.stderr}")

    try:
        with pytest.raises(PathRejected):
            WorkspaceGuard(junction)
    finally:
        junction.rmdir()


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

    assert writer.path == tmp_path / "traces" / "run-123.jsonl"
    assert writer.path.is_file()
    assert writer.path.read_bytes() == b""


def test_trace_writer_appends_one_compact_utf8_json_line(
    tmp_path: Path,
) -> None:
    writer = TraceStore(tmp_path / "traces").create("run-1")

    writer.append(
        step=2,
        tool="read_file",
        args={"path": "资料.txt"},
        result_summary="读取完成",
        status="ok",
    )

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

    writer.append(
        step=1,
        tool="write_file",
        args={"path": "note.txt", "content": content, "overwrite": True},
        result_summary="written",
        status="ok",
    )

    record = json.loads(writer.path.read_text(encoding="utf-8"))
    assert "content" not in record["args"]
    assert record["args"] == {
        "path": "note.txt",
        "overwrite": True,
        "content_bytes": len(content.encode("utf-8")),
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


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
    store.create("same-run")

    with pytest.raises(FileExistsError):
        store.create("same-run")
