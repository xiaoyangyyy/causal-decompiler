"""CrisisGrid and ReleaseOps scenario packs + scripted MRI smoke."""

from __future__ import annotations

from src.engine.causal import CausalDecompiler
from src.engine.simulation import SimConfig
from src.world.loader import load_events, load_world, scenario_config_dir
from src.world.scenario_packs import write_all


def setup_module() -> None:
    write_all()


def test_scenario_dirs_exist():
    write_all()
    assert (scenario_config_dir("crisisgrid") / "world.yaml").exists()
    assert (scenario_config_dir("releaseops") / "world.yaml").exists()
    assert scenario_config_dir("labwars").name in {"config", "labwars"}


def test_crisisgrid_has_independent_cast():
    world = load_world("crisisgrid")
    assert "dispatcher" in world.agents
    assert "phd_a" not in world.agents
    types = {e.type for e in load_events("crisisgrid")}
    assert "bridge_closed" in types
    assert "sensor_report" in types


def test_releaseops_has_workflow_cast():
    world = load_world("releaseops")
    assert "deployer" in world.agents
    assert "phd_a" not in world.agents
    types = {e.type for e in load_events("releaseops")}
    assert {"stale_config", "test_skipped", "alert", "rollback"} <= types


def test_crisisgrid_scripted_mri_smoke():
    report = CausalDecompiler().decompile(
        SimConfig(
            max_rounds=6,
            seed=11,
            mvp=False,
            llm_provider="scripted",
            scenario="crisisgrid",
            disable_state_events=True,
        ),
        outcome="stranded",
        auto_battery=False,
        include_toy_shapley=False,
        include_story_shapley=False,
        memory_rounds=[3],
        blame_limit=1,
    )
    assert report.identity_twin_ok
    assert "y_private" in report.split_y
    assert report.social_ir.get("by_layer", {}).get("E", 0) >= 1
    text = " ".join(report.findings)
    assert "public protest" not in text
    assert "Planted AND" not in text
    assert "hidden transcript" not in text
    assert not report.shapley_toy
    assert report.channels.get("stranded", 0) > 0


def test_releaseops_scripted_mri_smoke():
    report = CausalDecompiler().decompile(
        SimConfig(
            max_rounds=6,
            seed=11,
            mvp=False,
            llm_provider="scripted",
            scenario="releaseops",
            disable_state_events=True,
        ),
        outcome="task_y",
        auto_battery=False,
        include_toy_shapley=False,
        include_story_shapley=False,
        memory_rounds=[3],
        blame_limit=1,
    )
    assert report.identity_twin_ok
    assert report.channels.get("ppg") == 0.0
    assert "task_y" in report.split_y or report.channels.get("task_y") is not None
