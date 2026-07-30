from pathlib import Path

from workspace_agent.config import Settings


def test_settings_normalize_paths_and_apply_run_defaults(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    seed_root = tmp_path / "seed"
    trace_root = tmp_path / "traces"
    static_root = tmp_path / "static"

    settings = Settings(
        workspace_root=workspace_root,
        seed_root=seed_root,
        trace_root=trace_root,
        static_root=static_root,
        llm_base_url="https://api.siliconflow.cn/v1",
        llm_model="Qwen/Qwen3-8B",
    )

    assert settings.workspace_root == workspace_root.absolute()
    assert settings.seed_root == seed_root.absolute()
    assert settings.trace_root == trace_root.absolute()
    assert settings.static_root == static_root.absolute()
    assert all(
        path.is_absolute()
        for path in (
            settings.workspace_root,
            settings.seed_root,
            settings.trace_root,
            settings.static_root,
        )
    )
    assert settings.max_model_calls == 30
    assert settings.max_concurrent_runs == 1


def test_settings_allow_an_empty_api_key(tmp_path: Path) -> None:
    settings = Settings(
        workspace_root=tmp_path / "workspace",
        seed_root=tmp_path / "seed",
        trace_root=tmp_path / "traces",
        static_root=tmp_path / "static",
    )

    assert settings.llm_api_key == ""
