import sys
import shutil
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "python" / "pedestrian" / "pedestrian"))

from datasources.sdd import load_sdd_scenario


def test_scene_014_normalized_scenario_config_loads_in_sim_coordinates():
    scenario = load_sdd_scenario("outputs/sdd_processed", 14, "test")

    np.testing.assert_allclose(
        scenario.metadata["robot_start_zone"],
        np.asarray(
            [
                [0.768, 4.995],
                [1.282, 4.995],
                [1.282, 5.623],
                [0.768, 5.623],
            ]
        ),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        scenario.metadata["robot_goal"],
        np.asarray([6.175, 3.624]),
        atol=1e-6,
    )


def test_scene_014_pixel_scenario_config_converts_to_sim_coordinates():
    scenario = load_sdd_scenario("outputs/sdd_processed", 14, "test2")
    scale = 10.0 / 381.0

    np.testing.assert_allclose(
        scenario.metadata["robot_start_zone"],
        np.asarray(
            [
                [105.821 * scale, 295.602 * scale],
                [125.423 * scale, 295.602 * scale],
                [125.423 * scale, 316.713 * scale],
                [105.821 * scale, 316.713 * scale],
            ]
        ),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        scenario.metadata["robot_goal"],
        np.asarray([225.7 * scale, 135.009 * scale]),
        atol=1e-6,
    )


def test_sdd_scenario_grid_prefers_transition_state_space_metadata():
    scenario = load_sdd_scenario(
        "outputs/sdd_processed",
        14,
        "test",
        state_space_metadata={
            "cell_size": 0.123,
            "grid_size_meters": 0.456,
            "rows": 10,
            "cols": 20,
            "state_count": 99,
        },
    )

    grid = scenario.metadata["grid"]
    assert grid["resolution"] == 0.123
    assert grid["resolution_meters"] == 0.456
    assert grid["transition_rows"] == 10
    assert grid["transition_cols"] == 20
    assert grid["transition_state_count"] == 99
    assert grid["source"] == "sdd_transition_model"


def test_sdd_scenario_loads_targets_of_interest(tmp_path):
    scenario = load_sdd_scenario("outputs/sdd_processed", 14, "test")

    assert scenario.metadata["targets_of_interest_track_ids"] is None

    processed_root = tmp_path / "sdd_processed"
    shutil.copytree(
        ROOT / "outputs" / "sdd_processed" / "scene_014",
        processed_root / "scene_014",
    )
    scenario_path = processed_root / "scene_014" / "scenarios" / "test.json"
    text = scenario_path.read_text()
    data = json.loads(text)
    data["targetsOfInterestTrackIds"] = [2, 3]
    scenario_path.write_text(json.dumps(data))

    scenario = load_sdd_scenario(processed_root, 14, "test")

    assert scenario.metadata["targets_of_interest_track_ids"] == [2, 3]
