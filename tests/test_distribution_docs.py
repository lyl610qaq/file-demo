from __future__ import annotations

import importlib
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import pytest
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def _entrypoint_module():
    return importlib.import_module("workspace_agent.container_entrypoint")


def _frontmatter(text: str) -> dict[str, str]:
    lines = text.splitlines()
    assert lines[0] == "---"
    closing_index = lines.index("---", 1)
    metadata: dict[str, str] = {}
    for line in lines[1:closing_index]:
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"')
    return metadata


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _docker_daemon_is_available() -> bool:
    if shutil.which("docker") is None:
        return False
    completed = subprocess.run(
        ["docker", "version", "--format", "{{.Server.Version}}"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return completed.returncode == 0 and bool(completed.stdout.strip())


def _container_pid_one_uid(container: str) -> str:
    completed = subprocess.run(
        ["docker", "exec", container, "cat", "/proc/1/status"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    uid_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("Uid:")
    )
    return uid_line.split()[1]


@pytest.mark.parametrize("raw, expected", [(None, 8000), ("1", 1), ("8000", 8000), ("65535", 65535)])
def test_container_entrypoint_accepts_ascii_decimal_ports(
    raw: str | None,
    expected: int,
) -> None:
    entrypoint = _entrypoint_module()

    assert entrypoint.resolve_port(raw) == expected


@pytest.mark.parametrize("raw", ["", "0", "65536", "-1", "+8000", "8.0", " 8000", "\u0668\u0660\u0660\u0660"])
def test_container_entrypoint_rejects_invalid_ports(raw: str) -> None:
    entrypoint = _entrypoint_module()

    with pytest.raises(ValueError, match="PORT"):
        entrypoint.resolve_port(raw)


def test_root_entrypoint_initializes_only_fixed_runtime_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entrypoint = _entrypoint_module()
    data_root = tmp_path / "data"
    data_root.mkdir()
    app_root = tmp_path / "app"
    app_root.mkdir()
    sentinel = app_root / "keep.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    ownership_changes: list[tuple[Path, int, int]] = []

    monkeypatch.setattr(entrypoint, "DATA_ROOT", data_root)
    monkeypatch.setattr(entrypoint.os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(
        entrypoint.os,
        "chown",
        lambda path, uid, gid: ownership_changes.append((Path(path), uid, gid)),
        raising=False,
    )

    entrypoint.initialize_runtime_directories()

    assert (data_root / "workspace").is_dir()
    assert (data_root / "traces").is_dir()
    assert ownership_changes == [
        (data_root / "workspace", entrypoint.RUNTIME_UID, entrypoint.RUNTIME_GID),
        (data_root / "traces", entrypoint.RUNTIME_UID, entrypoint.RUNTIME_GID),
    ]
    assert sentinel.read_text(encoding="utf-8") == "unchanged"


def test_non_root_entrypoint_uses_writable_data_without_chowning_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entrypoint = _entrypoint_module()
    data_root = tmp_path / "data"
    data_root.mkdir()
    app_root = tmp_path / "app"
    app_root.mkdir()
    sentinel = app_root / "keep.txt"
    sentinel.write_text("unchanged", encoding="utf-8")

    monkeypatch.setattr(entrypoint, "DATA_ROOT", data_root)
    monkeypatch.setattr(entrypoint.os, "geteuid", lambda: 1000, raising=False)
    monkeypatch.setattr(
        entrypoint.os,
        "chown",
        lambda *_: pytest.fail("non-root entrypoint must not chown"),
        raising=False,
    )

    entrypoint.initialize_runtime_directories()

    assert (data_root / "workspace").is_dir()
    assert (data_root / "traces").is_dir()
    assert sentinel.read_text(encoding="utf-8") == "unchanged"


def test_entrypoint_rejects_linked_runtime_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entrypoint = _entrypoint_module()
    data_root = tmp_path / "data"
    data_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    workspace = data_root / "workspace"
    try:
        workspace.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlinks are unavailable: {error}")

    monkeypatch.setattr(entrypoint, "DATA_ROOT", data_root)
    monkeypatch.setattr(entrypoint.os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(entrypoint.os, "chown", lambda *_: None, raising=False)

    with pytest.raises(RuntimeError, match="physical directory"):
        entrypoint.initialize_runtime_directories()


def test_root_entrypoint_drops_privileges_before_exec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entrypoint = _entrypoint_module()
    data_root = tmp_path / "data"
    data_root.mkdir()
    calls: list[tuple[object, ...]] = []

    monkeypatch.setattr(entrypoint, "DATA_ROOT", data_root)
    monkeypatch.setattr(entrypoint.os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(entrypoint.os, "chown", lambda *_: None, raising=False)
    monkeypatch.setattr(
        entrypoint.os,
        "setgroups",
        lambda groups: calls.append(("setgroups", groups)),
        raising=False,
    )
    monkeypatch.setattr(
        entrypoint.os,
        "setgid",
        lambda gid: calls.append(("setgid", gid)),
        raising=False,
    )
    monkeypatch.setattr(
        entrypoint.os,
        "setuid",
        lambda uid: calls.append(("setuid", uid)),
        raising=False,
    )

    def record_exec(program: str, arguments: list[str]) -> None:
        calls.append(("exec", program, *arguments))
        raise RuntimeError("exec intercepted")

    monkeypatch.setattr(entrypoint.os, "execvp", record_exec)
    monkeypatch.setenv("PORT", "8765")

    with pytest.raises(RuntimeError, match="exec intercepted"):
        entrypoint.main()

    assert calls == [
        ("setgroups", []),
        ("setgid", entrypoint.RUNTIME_GID),
        ("setuid", entrypoint.RUNTIME_UID),
        (
            "exec",
            "uvicorn",
            "uvicorn",
            "workspace_agent.web:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8765",
        ),
    ]


def test_dockerfile_uses_the_validated_entrypoint_and_preserves_app_assets() -> None:
    dockerfile = ROOT / "Dockerfile"

    assert dockerfile.is_file()
    text = dockerfile.read_text(encoding="utf-8")

    assert "FROM python:3.12-slim" in text
    assert "pip install --no-cache-dir ." in text
    assert "COPY workspace_agent ./workspace_agent" in text
    assert "COPY static ./static" in text
    assert "COPY demo_workspace_seed ./demo_workspace_seed" in text
    assert "ENV WORKSPACE_ROOT=/data/workspace" in text
    assert "TRACE_ROOT=/data/traces" in text
    assert "mkdir -p /data" in text
    assert "chmod 1777 /data" in text
    assert "USER root" in text
    assert "USER 10001" not in text
    assert "chown -R /app" not in text
    assert "ENTRYPOINT [\"python\", \"-m\", \"workspace_agent.container_entrypoint\"]" in text
    assert "sh\", \"-c" not in text
    assert "${PORT" not in text
    assert "HEALTHCHECK" in text
    assert "/health" in text


def test_dockerignore_is_a_deny_all_allowlist() -> None:
    lines = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    meaningful = [line for line in lines if line and not line.startswith("#")]

    assert meaningful[0] == "**"
    assert {
        "!Dockerfile",
        "!.dockerignore",
        "!pyproject.toml",
        "!README.md",
        "!.env.example",
        "!workspace_agent/**",
        "!static/**",
        "!demo_workspace_seed/**",
    }.issubset(meaningful)
    assert not any(line.startswith("!.env") and line != "!.env.example" for line in meaningful)
    assert "!tests/**" not in meaningful
    assert "!.git/**" not in meaningful
    assert "!workspace/**" not in meaningful
    assert "!traces/**" not in meaningful


def test_dockerignore_reexcludes_nested_secrets_after_source_allowlists() -> None:
    lines = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    meaningful = [line for line in lines if line and not line.startswith("#")]
    source_allowlists = (
        "!workspace_agent/**",
        "!static/**",
        "!demo_workspace_seed/**",
    )
    nested_secret_exclusions = (
        "**/.env",
        "**/.env.*",
        "**/*.pem",
        "**/*.key",
        "**/id_rsa",
    )

    assert set(source_allowlists).issubset(meaningful)
    last_source_allowlist = max(meaningful.index(pattern) for pattern in source_allowlists)
    for pattern in nested_secret_exclusions:
        assert pattern in meaningful
        assert meaningful.index(pattern) > last_source_allowlist
    last_example_allowlist = max(
        index
        for index, pattern in enumerate(meaningful)
        if pattern == "!.env.example"
    )
    assert last_example_allowlist > max(
        meaningful.index(pattern) for pattern in nested_secret_exclusions
    )


def test_railway_configuration_declares_docker_healthcheck_and_restart_policy() -> None:
    railway = ROOT / "railway.toml"

    assert railway.is_file()
    config = tomllib.loads(railway.read_text(encoding="utf-8"))

    assert config["build"]["builder"] == "DOCKERFILE"
    assert config["build"]["dockerfilePath"] == "Dockerfile"
    assert config["deploy"]["healthcheckPath"] == "/health"
    assert config["deploy"]["restartPolicyType"] == "ON_FAILURE"


def test_readme_has_hugging_face_frontmatter_and_portable_platform_facts() -> None:
    readme = ROOT / "README.md"

    text = readme.read_text(encoding="utf-8")
    frontmatter = _frontmatter(text)

    assert frontmatter["title"] == "Workspace Agent"
    assert frontmatter["emoji"]
    allowed_colors = {
        "red",
        "orange",
        "yellow",
        "green",
        "blue",
        "indigo",
        "purple",
        "pink",
        "gray",
    }
    assert frontmatter["colorFrom"] in allowed_colors
    assert frontmatter["colorTo"] in allowed_colors
    assert frontmatter["sdk"] == "docker"
    assert frontmatter["app_port"] == "8000"
    for phrase in (
        "ALLOWED_ORIGIN",
        "TRUSTED_PROXY_CIDRS",
        "https://docs.railway.com/volumes",
        "默认以 root 启动入口",
        "UID 10001",
        "UID 1000",
        "主机目录或数据卷",
        "app_port: 8000",
        "https://huggingface.co/docs/hub/main/spaces-sdks-docker",
        "WebSocket",
        "https://vercel.com/kb/guide/do-vercel-serverless-functions-support-websocket-connections",
        "external storage",
        "distributed coordination",
        "runtime rewrite",
        "单进程",
        "非多租户",
        "不受信任",
        "reset journal v3",
        "Python 3.12",
    ):
        assert phrase in text
    assert "RAILWAY_RUN_UID=0" not in text


def test_notes_are_chinese_and_do_not_contain_absolute_local_paths_or_api_key_values() -> None:
    notes = ROOT / "NOTES.md"
    text = notes.read_text(encoding="utf-8")

    assert "部署" in text
    assert "API Key" in text
    assert "C:\\Users\\" not in text
    assert "LLM_API_KEY=" not in text


def test_gitignore_keeps_versioned_assets_but_ignores_runtime_data_and_reset_artifacts() -> None:
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "workspace/" in ignored
    assert "traces/" in ignored
    assert ".workspace-reset-" in ignored
    assert "demo_workspace_seed/" not in ignored


def test_container_image_initializes_a_root_owned_named_volume_then_drops_privileges() -> None:
    if not _docker_daemon_is_available():
        pytest.skip("Docker daemon unavailable; volume initialization integration test skipped")

    image = f"workspace-agent-test:{uuid.uuid4().hex}"
    container = f"workspace-agent-test-{uuid.uuid4().hex}"
    volume = f"workspace-agent-data-{uuid.uuid4().hex}"
    host_port = _free_local_port()
    try:
        subprocess.run(
            ["docker", "build", "--tag", image, str(ROOT)],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        subprocess.run(
            ["docker", "volume", "create", volume],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        root_owned = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--user",
                "0",
                "--mount",
                f"type=volume,src={volume},dst=/data",
                "--entrypoint",
                "python",
                image,
                "-c",
                "import os; print(os.stat('/data').st_uid)",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert root_owned.stdout.strip() == "0"
        subprocess.run(
            [
                "docker",
                "run",
                "--detach",
                "--name",
                container,
                "--publish",
                f"127.0.0.1:{host_port}:8000",
                "--mount",
                f"type=volume,src={volume},dst=/data",
                image,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        deadline = time.monotonic() + 30
        while True:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{host_port}/health",
                    timeout=2,
                ) as response:
                    assert response.status == 200
                    break
            except (urllib.error.URLError, TimeoutError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.2)
        assert _container_pid_one_uid(container) == "10001"
    finally:
        subprocess.run(
            ["docker", "rm", "--force", container],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        subprocess.run(
            ["docker", "volume", "rm", volume],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        subprocess.run(
            ["docker", "image", "rm", "--force", image],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
