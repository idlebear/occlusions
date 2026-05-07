import sys
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
