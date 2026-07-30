import codecs
import hashlib
import importlib
import os
import subprocess
from pathlib import Path

import pytest

from workspace_agent.safety import WorkspaceGuard


def _tools_module():
    return importlib.import_module("workspace_agent.tools")


def _tools(root: Path, *, max_read_bytes: int = 16384):
    return _tools_module().WorkspaceTools(
        WorkspaceGuard(root),
        max_read_bytes=max_read_bytes,
        max_write_bytes=1024,
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


def test_list_dir_sorts_nonrecursive_entries_and_paginates(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "z.txt").write_text("z", encoding="utf-8")
    (root / "a").mkdir()
    (root / "m.txt").write_text("middle", encoding="utf-8")
    tools = _tools(root)

    first = tools.list_dir(limit=2)
    second = tools.list_dir(cursor=first.data["next_cursor"], limit=2)

    assert first.ok
    assert [entry["path"] for entry in first.data["entries"]] == [
        "a",
        "m.txt",
    ]
    assert first.data["has_more"] is True
    assert isinstance(first.data["next_cursor"], str)
    assert not first.data["next_cursor"].isdecimal()
    assert [entry["path"] for entry in second.data["entries"]] == ["z.txt"]
    assert second.data["has_more"] is False
    assert second.data["next_cursor"] is None


def test_list_dir_recursively_sorts_workspace_relative_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    (root / "b").mkdir(parents=True)
    (root / "a").mkdir()
    (root / "b" / "two.txt").write_text("2", encoding="utf-8")
    (root / "a" / "one.txt").write_text("1", encoding="utf-8")

    result = _tools(root).list_dir(recursive=True)

    assert result.ok
    assert [entry["path"] for entry in result.data["entries"]] == [
        "a",
        "a/one.txt",
        "b",
        "b/two.txt",
    ]
    assert {entry["type"] for entry in result.data["entries"]} == {
        "directory",
        "file",
    }


def test_list_dir_skips_and_does_not_traverse_directory_symlink(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    _create_directory_symlink(root / "linked", outside)

    result = _tools(root).list_dir(recursive=True)

    assert result.ok
    assert result.data["entries"] == []
    assert any("linked" in warning for warning in result.data["warnings"])
    assert "secret" not in repr(result.data)


@pytest.mark.skipif(os.name != "nt", reason="Windows junction test")
def test_list_dir_skips_and_does_not_traverse_windows_junction(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    junction = root / "junction"
    _create_windows_junction(junction, outside)

    try:
        result = _tools(root).list_dir(recursive=True)
        assert result.ok
        assert result.data["entries"] == []
        assert any(
            "junction" in warning for warning in result.data["warnings"]
        )
    finally:
        os.rmdir(junction)


def test_search_files_supports_exact_and_casefold_literal_matching(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "notes.txt").write_text(
        "Needle\nneedle\nneed.le\n",
        encoding="utf-8",
    )
    tools = _tools(root)

    exact = tools.search_files("Needle")
    folded = tools.search_files("NEEDLE", case_sensitive=False)
    literal = tools.search_files("need.le")

    assert [(match["line"], match["snippet"]) for match in exact.data["matches"]] == [
        (1, "Needle")
    ]
    assert [match["line"] for match in folded.data["matches"]] == [1, 2]
    assert [match["line"] for match in literal.data["matches"]] == [3]


def test_search_files_paginates_without_duplicate_or_missing_matches(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "a.txt").write_text("hit 1\nhit 2\n", encoding="utf-8")
    (root / "b.txt").write_text("hit 3\nhit 4\n", encoding="utf-8")
    tools = _tools(root)

    first = tools.search_files("hit", limit=3)
    second = tools.search_files(
        "hit",
        cursor=first.data["next_cursor"],
        limit=3,
    )
    combined = first.data["matches"] + second.data["matches"]

    assert first.data["has_more"] is True
    assert isinstance(first.data["next_cursor"], str)
    assert not first.data["next_cursor"].isdecimal()
    assert second.data["has_more"] is False
    assert [(item["path"], item["line"]) for item in combined] == [
        ("a.txt", 1),
        ("a.txt", 2),
        ("b.txt", 1),
        ("b.txt", 2),
    ]


def test_recursive_list_stops_after_limit_plus_one_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "workspace"
    deep = root / "a" / "first"
    deep.mkdir(parents=True)
    for index in range(2_000):
        (deep / f"{index:04}.txt").write_text("x", encoding="utf-8")
    tools = _tools(root)
    original_scan_names = tools._scan_names
    scanned: list[str] = []

    def recording_scan_names(relative_path: str) -> list[str]:
        scanned.append(relative_path)
        if relative_path == "a/first":
            raise AssertionError("early page consumed the deep subtree")
        return original_scan_names(relative_path)

    monkeypatch.setattr(tools, "_scan_names", recording_scan_names)

    result = tools.list_dir(recursive=True, limit=1)

    assert result.ok
    assert [entry["path"] for entry in result.data["entries"]] == ["a"]
    assert result.data["has_more"] is True
    assert "a/first" not in scanned


def test_list_cursor_is_bound_and_ignores_insertions_before_page(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "b.txt").write_text("b", encoding="utf-8")
    (root / "c.txt").write_text("c", encoding="utf-8")
    tools = _tools(root)

    first = tools.list_dir(limit=1)
    (root / "a.txt").write_text("a", encoding="utf-8")
    second = tools.list_dir(cursor=first.data["next_cursor"], limit=2)
    mismatch = tools.list_dir(
        recursive=True,
        cursor=first.data["next_cursor"],
        limit=1,
    )

    assert [entry["path"] for entry in first.data["entries"]] == ["b.txt"]
    assert [entry["path"] for entry in second.data["entries"]] == ["c.txt"]
    assert mismatch.ok is False
    assert mismatch.error_code == "INVALID_CURSOR"


def test_search_cursor_is_bound_and_ignores_insertions_before_page(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "b.txt").write_text("hit b", encoding="utf-8")
    (root / "c.txt").write_text("hit c", encoding="utf-8")
    tools = _tools(root)

    first = tools.search_files("hit", limit=1)
    (root / "a.txt").write_text("hit a", encoding="utf-8")
    second = tools.search_files(
        "hit",
        cursor=first.data["next_cursor"],
        limit=2,
    )
    mismatch = tools.search_files(
        "different",
        cursor=first.data["next_cursor"],
        limit=1,
    )

    assert [(item["path"], item["line"]) for item in first.data["matches"]] == [
        ("b.txt", 1)
    ]
    assert [(item["path"], item["line"]) for item in second.data["matches"]] == [
        ("c.txt", 1)
    ]
    assert mismatch.ok is False
    assert mismatch.error_code == "INVALID_CURSOR"


def test_search_cursor_detects_changed_last_file(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "a.txt"
    target.write_text("hit one\nhit two\n", encoding="utf-8")
    tools = _tools(root)

    first = tools.search_files("hit", limit=1)
    target.write_text("hit changed\nhit two\n", encoding="utf-8")
    metadata = target.stat()
    os.utime(
        target,
        ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1_000_000),
    )
    resumed = tools.search_files(
        "hit",
        cursor=first.data["next_cursor"],
        limit=1,
    )

    assert resumed.ok is False
    assert resumed.error_code == "STALE_CURSOR"


def test_search_files_bounds_snippets_and_warns_for_binary_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "long.txt").write_text("match " + ("x" * 800), encoding="utf-8")
    (root / "binary.dat").write_bytes(b"match\x00hidden")

    result = _tools(root).search_files("match")

    assert result.ok
    assert len(result.data["matches"][0]["snippet"]) == 500
    assert any(
        "binary.dat" in warning and "binary" in warning.lower()
        for warning in result.data["warnings"]
    )


def test_search_files_streams_to_a_large_file_tail(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    large = root / "large.txt"
    with large.open("w", encoding="utf-8") as handle:
        for _ in range(100_000):
            handle.write("ordinary bounded line\n")
        handle.write("the target is at the tail\n")

    result = _tools(root).search_files("target")

    assert result.ok
    assert result.data["matches"] == [
        {
            "path": "large.txt",
            "line": 100_001,
            "snippet": "the target is at the tail",
        }
    ]
    assert len(repr(result.data)) < 1_000


def test_search_files_uses_global_lexical_file_order(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    (root / "a").mkdir(parents=True)
    (root / "a" / "nested.txt").write_text("hit", encoding="utf-8")
    (root / "a.txt").write_text("hit", encoding="utf-8")
    (root / "aa.txt").write_text("hit", encoding="utf-8")
    (root / "b.txt").write_text("hit", encoding="utf-8")

    result = _tools(root).search_files("hit")

    assert result.ok
    assert [match["path"] for match in result.data["matches"]] == [
        "a.txt",
        "a/nested.txt",
        "aa.txt",
        "b.txt",
    ]


def test_search_files_bounds_multi_megabyte_logical_lines_and_carry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    segment_size = 4096
    query = "boundary"
    prefix = "x" * (segment_size - 3)
    (root / "huge.txt").write_text(
        prefix + query + ("y" * (3 * 1024 * 1024)),
        encoding="utf-8",
    )
    tools_module = _tools_module()
    original_open_file = tools_module.WorkspaceTools._open_file
    readline_sizes: list[int] = []

    class BoundedTextReader:
        def __init__(self, handle) -> None:
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def __iter__(self):
            raise AssertionError("logical lines must not use unbounded iteration")

        def readline(self, size: int = -1) -> str:
            assert 0 < size <= segment_size
            readline_sizes.append(size)
            return self.handle.readline(size)

    def bounded_open_file(self, relative_path: str, mode: str, **kwargs):
        resolved, handle = original_open_file(
            self,
            relative_path,
            mode,
            **kwargs,
        )
        if relative_path == "huge.txt" and mode == "r":
            handle = BoundedTextReader(handle)
        return resolved, handle

    monkeypatch.setattr(
        tools_module.WorkspaceTools,
        "_open_file",
        bounded_open_file,
    )

    result = _tools(root).search_files(query)

    assert result.ok
    assert result.data["matches"][0]["path"] == "huge.txt"
    assert result.data["matches"][0]["line"] == 1
    assert len(result.data["matches"][0]["snippet"]) <= 500
    assert readline_sizes


def test_late_huge_line_match_is_in_bounded_snippet(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    query = "late-boundary-query"
    (root / "late.txt").write_text(
        ("x" * (2 * 1024 * 1024)) + query + ("y" * 2_000),
        encoding="utf-8",
    )

    result = _tools(root).search_files(query)

    snippet = result.data["matches"][0]["snippet"]
    assert len(snippet) <= 500
    assert query in snippet


def test_nested_scan_errors_warn_and_preserve_readable_siblings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "workspace"
    (root / "blocked").mkdir(parents=True)
    (root / "blocked" / "secret.txt").write_text("hit", encoding="utf-8")
    (root / "readable").mkdir()
    (root / "readable" / "good.txt").write_text("hit", encoding="utf-8")
    tools = _tools(root)
    original_scan_names = tools._scan_names

    def denying_scan_names(relative_path: str) -> list[str]:
        if relative_path == "blocked":
            raise PermissionError("denied")
        return original_scan_names(relative_path)

    monkeypatch.setattr(tools, "_scan_names", denying_scan_names)

    listed = tools.list_dir(recursive=True)
    searched = tools.search_files("hit")

    assert listed.ok
    assert "readable/good.txt" in {
        entry["path"] for entry in listed.data["entries"]
    }
    assert any("blocked" in warning for warning in listed.data["warnings"])
    assert searched.ok
    assert [item["path"] for item in searched.data["matches"]] == [
        "readable/good.txt"
    ]
    assert any("blocked" in warning for warning in searched.data["warnings"])


def test_read_file_default_limit_respects_configured_maximum(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "large.txt").write_text("x" * 2_000, encoding="utf-8")

    result = _tools(root, max_read_bytes=1024).read_file("large.txt")

    assert result.ok
    assert len(result.data["content"].encode("utf-8")) == 1024
    assert result.data["has_more"] is True


def test_read_file_continues_on_utf8_character_boundaries(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "utf8.txt").write_text("A你B好C", encoding="utf-8")
    tools = _tools(root, max_read_bytes=4)

    first = tools.read_file("utf8.txt", limit=4)
    second = tools.read_file("utf8.txt", offset=first.data["next_offset"], limit=4)

    assert first.ok
    assert first.data["content"] == "A你"
    assert first.data["next_offset"] == 4
    assert first.data["has_more"] is True
    assert second.data["content"] == "B好"
    assert second.data["offset"] == 4
    assert second.data["next_offset"] == 8
    assert second.data["encoding"] == "utf-8-sig"


def test_read_file_rejects_mid_character_offset_and_excessive_limit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "utf8.txt").write_text("A你B", encoding="utf-8")
    tools = _tools(root, max_read_bytes=4)

    invalid_offset = tools.read_file("utf8.txt", offset=2, limit=2)
    excessive_limit = tools.read_file("utf8.txt", limit=5)

    assert invalid_offset.ok is False
    assert invalid_offset.error_code == "INVALID_OFFSET"
    assert excessive_limit.ok is False
    assert excessive_limit.error_code == "INVALID_LIMIT"


@pytest.mark.parametrize(
    ("name", "encoding", "content", "minimum_bytes"),
    [
        ("utf8.txt", "utf-8", "你", 3),
        ("gb.txt", "gb18030", "中", 2),
    ],
)
def test_read_file_fails_when_limit_cannot_make_character_progress(
    tmp_path: Path,
    name: str,
    encoding: str,
    content: str,
    minimum_bytes: int,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / name).write_bytes(content.encode(encoding))

    result = _tools(root, max_read_bytes=8).read_file(name, limit=1)

    assert result.ok is False
    assert result.error_code == "LIMIT_SPLITS_CHARACTER"
    assert result.data == {"minimum_bytes": minimum_bytes}
    assert "next_offset" not in result.data


def test_gb18030_reads_require_and_accept_authenticated_continuation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "gb.txt"
    target.write_bytes("A😀B".encode("gb18030"))
    tools = _tools(root, max_read_bytes=8)

    first = tools.read_file("gb.txt", limit=1)
    without_cursor = tools.read_file("gb.txt", offset=1, limit=4)
    interior_offsets = [
        tools.read_file("gb.txt", offset=offset, limit=1)
        for offset in (2, 3, 4)
    ]
    second = tools.read_file(
        "gb.txt",
        offset=1,
        limit=4,
        cursor=first.data["next_cursor"],
    )
    third = tools.read_file(
        "gb.txt",
        offset=second.data["next_offset"],
        limit=1,
        cursor=second.data["next_cursor"],
    )

    assert first.data["content"] == "A"
    assert first.data["next_offset"] == 1
    assert without_cursor.error_code == "OFFSET_REQUIRES_CURSOR"
    assert {
        result.error_code for result in interior_offsets
    } == {"OFFSET_REQUIRES_CURSOR"}
    assert second.ok
    assert second.data["content"] == "😀"
    assert second.data["next_offset"] == 5
    assert third.data["content"] == "B"


def test_read_cursor_rejects_tampering_wrong_path_and_changed_file(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "gb.txt"
    target.write_bytes("A中B".encode("gb18030"))
    (root / "other.txt").write_text("other", encoding="utf-8")
    tools = _tools(root, max_read_bytes=8)
    first = tools.read_file("gb.txt", limit=1)
    cursor = first.data["next_cursor"]
    replacement = "A文B".encode("gb18030")

    tampered = tools.read_file(
        "gb.txt",
        offset=1,
        limit=2,
        cursor=cursor[:-1] + ("A" if cursor[-1] != "A" else "B"),
    )
    wrong_path = tools.read_file(
        "other.txt",
        offset=1,
        limit=2,
        cursor=cursor,
    )
    target.write_bytes(replacement)
    metadata = target.stat()
    os.utime(
        target,
        ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1_000_000),
    )
    stale = tools.read_file(
        "gb.txt",
        offset=1,
        limit=2,
        cursor=cursor,
    )

    assert tampered.error_code == "INVALID_CURSOR"
    assert wrong_path.error_code == "INVALID_CURSOR"
    assert stale.error_code == "STALE_CURSOR"


def test_deep_utf8_boundary_probe_does_not_read_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "large.txt"
    size = 8 * 1024 * 1024
    target.write_bytes(b"x" * (size - 1) + b"z")
    tools = _tools(root)
    original_open_file = tools._open_file
    bytes_read = 0

    class CountingReader:
        def __init__(self, handle) -> None:
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def read(self, size: int = -1) -> bytes:
            nonlocal bytes_read
            data = self.handle.read(size)
            bytes_read += len(data)
            return data

    def counting_open_file(relative_path: str, mode: str, **kwargs):
        resolved, handle = original_open_file(relative_path, mode, **kwargs)
        if relative_path == "large.txt" and mode == "rb":
            handle = CountingReader(handle)
        return resolved, handle

    monkeypatch.setattr(tools, "_open_file", counting_open_file)

    result = tools.read_file("large.txt", offset=size - 1, limit=1)

    assert result.ok
    assert result.data["content"] == "z"
    assert bytes_read <= 32


def test_utf16_surrogate_pair_boundaries_are_local_and_strict(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "utf16.txt").write_bytes("A😀B".encode("utf-16"))
    tools = _tools(root)

    middle = tools.read_file("utf16.txt", offset=6, limit=2)
    after_pair = tools.read_file("utf16.txt", offset=8, limit=2)

    assert middle.error_code == "INVALID_OFFSET"
    assert after_pair.ok
    assert after_pair.data["content"] == "B"


def test_ascii_probe_then_gb18030_content_reads_and_searches(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    content = ("a" * 5_000) + "中文"
    (root / "late-gb.txt").write_bytes(content.encode("gb18030"))
    tools = _tools(root)

    read = tools.read_file("late-gb.txt")
    searched = tools.search_files("中文")
    direct_offset = tools.read_file(
        "late-gb.txt",
        offset=5_000,
        limit=1,
    )

    assert read.ok
    assert read.data["content"] == content
    assert read.data["encoding"] == "gb18030"
    assert searched.ok
    assert [(item["path"], item["line"]) for item in searched.data["matches"]] == [
        ("late-gb.txt", 1)
    ]
    assert direct_offset.error_code == "OFFSET_REQUIRES_CURSOR"


@pytest.mark.parametrize(
    "payload",
    [
        codecs.BOM_UTF32_LE + "hello".encode("utf-32-le"),
        codecs.BOM_UTF32_BE + "hello".encode("utf-32-be"),
    ],
)
def test_utf32_boms_are_rejected_explicitly(
    tmp_path: Path,
    payload: bytes,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "utf32.txt").write_bytes(payload)

    result = _tools(root).read_file("utf32.txt")

    assert result.ok is False
    assert result.error_code == "UNSUPPORTED_ENCODING"


@pytest.mark.parametrize(
    ("name", "encoding", "content", "expected_encoding"),
    [
        ("utf16.txt", "utf-16", "hello 世界", "utf-16"),
        ("gb.txt", "gb18030", "中文内容", "gb18030"),
    ],
)
def test_read_file_detects_supported_non_utf8_encodings(
    tmp_path: Path,
    name: str,
    encoding: str,
    content: str,
    expected_encoding: str,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / name).write_bytes(content.encode(encoding))

    result = _tools(root).read_file(name)

    assert result.ok
    assert result.data["content"] == content
    assert result.data["encoding"] == expected_encoding


@pytest.mark.parametrize(
    ("name", "payload", "error_code"),
    [
        ("binary.dat", b"text\x00more", "BINARY_FILE"),
        ("unknown.dat", b"\xff\xff\xff", "CHARSET_UNDETERMINED"),
    ],
)
def test_read_file_returns_encoding_errors(
    tmp_path: Path,
    name: str,
    payload: bytes,
    error_code: str,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / name).write_bytes(payload)

    result = _tools(root).read_file(name)

    assert result.ok is False
    assert result.error_code == error_code


def test_encoding_detection_never_uses_path_read_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "note.txt").write_text("hello", encoding="utf-8")

    def forbidden_read_bytes(self: Path) -> bytes:
        raise AssertionError("whole-file reads are forbidden")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read_bytes)

    assert _tools(root).read_file("note.txt").data["content"] == "hello"


def test_tools_revalidate_same_relative_path_before_every_filesystem_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "workspace"
    (root / "nested").mkdir(parents=True)
    target = root / "nested" / "note.txt"
    target.write_text("hello target", encoding="utf-8")
    events: list[tuple[str, str]] = []

    class RecordingGuard(WorkspaceGuard):
        resolving = False
        pending: str | None = None

        def resolve(self, relative: str, must_exist: bool = False) -> Path:
            self.resolving = True
            try:
                resolved = super().resolve(relative, must_exist=must_exist)
            finally:
                self.resolving = False
            self.pending = resolved.relative_to(self.root).as_posix() or "."
            return resolved

    guard = RecordingGuard(root)
    tools_module = _tools_module()
    original_lstat = os.lstat
    original_scandir = os.scandir
    original_open = Path.open
    original_stat = Path.stat

    def relative_path(path) -> str | None:
        try:
            relative = Path(os.path.abspath(path)).relative_to(root)
        except ValueError:
            return None
        return relative.as_posix() or "."

    def record_access(kind: str, path) -> None:
        relative = relative_path(path)
        if relative is None or guard.resolving:
            return
        assert guard.pending == relative, (
            f"{kind} for {relative} was not immediately preceded by "
            "a same-path guard resolution"
        )
        events.append((kind, relative))
        guard.pending = None

    def recording_lstat(path, *args, **kwargs):
        record_access("lstat", path)
        return original_lstat(path, *args, **kwargs)

    def recording_scandir(path):
        record_access("scandir", path)
        return original_scandir(path)

    def recording_open(self: Path, *args, **kwargs):
        record_access("open", self)
        return original_open(self, *args, **kwargs)

    def recording_stat(self: Path, *args, **kwargs):
        record_access("stat", self)
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(tools_module.os, "lstat", recording_lstat)
    monkeypatch.setattr(tools_module.os, "scandir", recording_scandir)
    monkeypatch.setattr(Path, "open", recording_open)
    monkeypatch.setattr(Path, "stat", recording_stat)
    tools = tools_module.WorkspaceTools(
        guard,
        max_read_bytes=16,
        max_write_bytes=16,
    )

    listed = tools.list_dir(recursive=True)
    searched = tools.search_files("target")
    read = tools.read_file("nested/note.txt", limit=8)
    inspected = tools.stat_path("nested/note.txt")

    assert listed.ok
    assert searched.ok
    assert read.ok
    assert inspected.ok
    assert {"lstat", "scandir", "open"} <= {kind for kind, _ in events}


def test_stat_path_hashes_files_and_never_leaks_absolute_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    payload = b"bounded hash content"
    (root / "note.bin").write_bytes(payload)
    (root / "folder").mkdir()
    tools = _tools(root)

    file_result = tools.stat_path("note.bin")
    dir_result = tools.stat_path("folder")

    assert file_result.ok
    assert file_result.data["path"] == "note.bin"
    assert file_result.data["sha256"] == hashlib.sha256(payload).hexdigest()
    assert dir_result.data["sha256"] is None
    assert str(root) not in repr(file_result)
    assert str(root) not in repr(dir_result)


@pytest.mark.parametrize(
    "call",
    [
        lambda tools: tools.list_dir(cursor="not-decimal"),
        lambda tools: tools.list_dir(limit=0),
        lambda tools: tools.list_dir(limit=201),
        lambda tools: tools.search_files("", limit=1),
        lambda tools: tools.search_files("x", cursor="-1"),
        lambda tools: tools.search_files("x", limit=101),
        lambda tools: tools.read_file("note.txt", offset=-1),
        lambda tools: tools.read_file(123),
        lambda tools: tools.stat_path(None),
    ],
)
def test_invalid_inputs_return_tool_result_errors(
    tmp_path: Path,
    call,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "note.txt").write_text("hello", encoding="utf-8")

    result = call(_tools(root))

    assert result.ok is False
    assert result.error_code is not None
    assert result.data == {}
