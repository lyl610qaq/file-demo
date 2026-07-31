from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_builds_a_non_root_python_312_web_image() -> None:
    dockerfile = ROOT / "Dockerfile"

    assert dockerfile.is_file()
    text = dockerfile.read_text(encoding="utf-8")

    assert "FROM python:3.12-slim" in text
    assert "pip install --no-cache-dir ." in text
    assert "workspace_agent" in text
    assert "static" in text
    assert "demo_workspace_seed" in text
    assert "USER 10001" in text
    assert "/app/workspace" in text
    assert "/app/traces" in text
    assert "uvicorn" in text
    assert "0.0.0.0" in text
    assert "${PORT:-8000}" in text


def test_railway_configuration_declares_docker_healthcheck_and_restart_policy() -> None:
    railway = ROOT / "railway.toml"

    assert railway.is_file()
    text = railway.read_text(encoding="utf-8")
    config = tomllib.loads(text)

    assert config["build"]["builder"] == "DOCKERFILE"
    assert config["build"]["dockerfilePath"] == "Dockerfile"
    assert config["deploy"]["healthcheckPath"] == "/health"
    assert config["deploy"]["restartPolicyType"] == "ON_FAILURE"


def test_readme_documents_the_deployment_and_security_contract() -> None:
    readme = ROOT / "README.md"

    assert readme.is_file()
    text = readme.read_text(encoding="utf-8")

    for phrase in (
        "ALLOWED_ORIGIN",
        "TRUSTED_PROXY_CIDRS",
        "Railway",
        "Fly.io",
        "HF Spaces",
        "Vercel",
        "WebSocket",
        "单进程",
        "非多租户",
        "不受信任",
        "reset journal v3",
        "Python 3.12",
    ):
        assert phrase in text


def test_notes_are_chinese_and_do_not_contain_absolute_local_paths_or_api_key_values() -> None:
    notes = ROOT / "NOTES.md"

    assert notes.is_file()
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
