from pathlib import Path

import pytest
from pydantic import ValidationError

from workspace_agent.config import Settings


def test_settings_normalize_paths_and_apply_run_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    settings = Settings(
        workspace_root=Path("workspace"),
        seed_root=Path("seed"),
        trace_root=Path("traces"),
        static_root=Path("static"),
        llm_base_url="https://api.siliconflow.cn/v1",
        llm_model="Qwen/Qwen3-8B",
    )

    assert settings.workspace_root == tmp_path / "workspace"
    assert settings.seed_root == tmp_path / "seed"
    assert settings.trace_root == tmp_path / "traces"
    assert settings.static_root == tmp_path / "static"
    assert settings.max_model_calls == 30
    assert settings.max_concurrent_runs == 1
    assert settings.max_read_bytes == 16384
    assert settings.max_write_bytes == 262144
    assert settings.request_timeout_seconds == 60.0
    assert settings.rate_limit_per_minute == 10


def test_settings_allow_an_empty_api_key(tmp_path: Path) -> None:
    settings = Settings(
        workspace_root=tmp_path / "workspace",
        seed_root=tmp_path / "seed",
        trace_root=tmp_path / "traces",
        static_root=tmp_path / "static",
    )

    assert settings.llm_api_key == ""


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("max_model_calls", 0),
        ("max_model_calls", 101),
        ("max_concurrent_runs", 0),
        ("max_concurrent_runs", 9),
        ("max_read_bytes", 1023),
        ("max_read_bytes", 65537),
        ("max_write_bytes", 1023),
        ("max_write_bytes", 1048577),
        ("request_timeout_seconds", 4.9),
        ("request_timeout_seconds", 180.1),
        ("rate_limit_per_minute", 0),
        ("rate_limit_per_minute", 121),
    ],
)
def test_settings_reject_values_outside_limits(
    field_name: str,
    invalid_value: int | float,
) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field_name: invalid_value})
