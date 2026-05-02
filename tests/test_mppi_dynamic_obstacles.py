import sys
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "python" / "pedestrian" / "pedestrian"))
sys.path.insert(0, str(ROOT / "src" / "thirdParty" / "warp_mppi"))

sys.modules.setdefault("pygame", types.ModuleType("pygame"))
sys.modules.setdefault("visilibity", types.ModuleType("visilibity"))
shapely_stub = types.ModuleType("shapely")
shapely_geometry_stub = types.ModuleType("shapely.geometry")
shapely_geometry_stub.LineString = object
shapely_geometry_stub.MultiPolygon = object
shapely_geometry_stub.Point = object
shapely_geometry_stub.Polygon = object
shapely_ops_stub = types.ModuleType("shapely.ops")
shapely_ops_stub.unary_union = lambda polygons: None
sys.modules.setdefault("shapely", shapely_stub)
sys.modules.setdefault("shapely.geometry", shapely_geometry_stub)
sys.modules.setdefault("shapely.ops", shapely_ops_stub)
skimage_stub = types.ModuleType("skimage")
skimage_morphology_stub = types.ModuleType("skimage.morphology")
skimage_morphology_stub.binary_dilation = lambda *args, **kwargs: None
skimage_morphology_stub.binary_opening = lambda *args, **kwargs: None
sys.modules.setdefault("skimage", skimage_stub)
sys.modules.setdefault("skimage.morphology", skimage_morphology_stub)
velocity_prediction_stub = types.ModuleType("controller.velocity_prediction")
velocity_prediction_stub.ControlVariations = object
sys.modules.setdefault("controller.velocity_prediction", velocity_prediction_stub)
entropy_stub = types.ModuleType("entropy")
entropy_stub.evaluate_method = lambda *args, **kwargs: None
sys.modules.setdefault("entropy", entropy_stub)
hmm_stub = types.ModuleType("hmm")
hmm_stub.HMM = object
sys.modules.setdefault("hmm", hmm_stub)
specialk_stub = types.ModuleType("specialk")
specialk_stub.generate_specialk_trajectories = lambda *args, **kwargs: []
sys.modules.setdefault("specialk", specialk_stub)
sys.modules.setdefault("pandas", types.ModuleType("pandas"))
polycheck_stub = types.ModuleType("polycheck")
polycheck_stub.load_polycheck = lambda *args, **kwargs: None
polycheck_stub.faux_scan = None
polycheck_stub.visibility_from_region = lambda *args, **kwargs: None
sys.modules.setdefault("polycheck", polycheck_stub)
warp_mppi_stub = types.ModuleType("warp_mppi")
legacy_stub = types.ModuleType("warp_mppi.legacy")
legacy_stub.PyCudaMPPI = object
legacy_stub.evaluate_trajectories_by_entropy_gpu = lambda *args, **kwargs: None
legacy_stub.evaluate_discrete_oce_gpu = lambda *args, **kwargs: None
warp_mppi_stub.legacy = legacy_stub
sys.modules.setdefault("warp_mppi", warp_mppi_stub)
sys.modules.setdefault("warp_mppi.legacy", legacy_stub)

try:
    import main as pedestrian_main
except Exception as exc:  # pragma: no cover - depends on optional CUDA stack
    pytest.skip(
        f"pedestrian main could not be imported: {exc}", allow_module_level=True
    )


class StraightLineVehicle:
    L = 1.0
    W = 0.5

    def ode(self, state, control):
        return np.asarray([1.0, float(control[0]), 0.0, 0.0], dtype=float)


def test_agent_prediction_sequence_uses_future_prediction_after_current_state():
    agent = {
        "id": 7,
        "pos": np.asarray([1.0, 2.0, 0.5, 0.25], dtype=float),
        "extent": 0.3,
    }
    prediction = np.asarray(
        [
            [
                [1.0, 2.0, 0.5, 0.25],
                [1.1, 2.0, 0.5, 0.25],
                [1.2, 2.0, 0.5, 0.25],
                [1.3, 2.0, 0.5, 0.25],
            ],
            [
                [1.0, 2.0, 0.5, 0.25],
                [1.3, 2.0, 0.5, 0.25],
                [1.4, 2.0, 0.5, 0.25],
                [1.5, 2.0, 0.5, 0.25],
            ],
        ],
        dtype=np.float32,
    )

    states, source = pedestrian_main._agent_prediction_sequence(
        agent,
        {7: prediction},
        horizon=3,
    )

    assert source == "prediction"
    assert states.shape == (3, 4)
    assert np.allclose(states[:, 0], [1.2, 1.3, 1.4])


def test_dynamic_collision_filter_recomputes_safe_sample_weights():
    agent = {
        "id": 1,
        "pos": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
        "extent": 0.1,
    }
    u_nom = np.zeros((2, 2), dtype=np.float32)
    u_variations = np.zeros((2, 2, 2), dtype=np.float32)
    u_variations[1, :, 0] = 2.0
    weights = np.ones(2, dtype=np.float32)
    agent_predictions = {
        1: np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    }

    filtered, debug = pedestrian_main.filter_dynamic_collision_samples(
        robot_model=StraightLineVehicle(),
        initial_state=np.asarray([0.0, 0.0, 1.0, 0.0], dtype=float),
        u_nom=u_nom,
        u_variations=u_variations,
        weights=weights,
        agents=[agent],
        agent_predictions=agent_predictions,
        horizon=2,
        dt=1.0,
    )

    assert debug["collision_free"].tolist() == [False, True]
    assert np.allclose(filtered, [0.0, 1.0])


def test_dynamic_collision_filter_preserves_weights_when_all_samples_violate_clearance():
    agent = {
        "id": 1,
        "pos": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
        "extent": 0.1,
    }
    u_nom = np.zeros((2, 2), dtype=np.float32)
    u_variations = np.zeros((2, 2, 2), dtype=np.float32)
    weights = np.asarray([0.25, 0.75], dtype=np.float32)
    agent_predictions = {
        1: np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    }

    filtered, debug = pedestrian_main.filter_dynamic_collision_samples(
        robot_model=StraightLineVehicle(),
        initial_state=np.asarray([0.0, 0.0, 1.0, 0.0], dtype=float),
        u_nom=u_nom,
        u_variations=u_variations,
        weights=weights,
        agents=[agent],
        agent_predictions=agent_predictions,
        horizon=2,
        dt=1.0,
    )

    assert debug["collision_free"].tolist() == [False, False]
    assert debug["all_samples_violate_clearance"] is True
    assert np.allclose(filtered, weights)


def test_dynamic_obstacle_buffer_uses_scaled_hard_clearance_without_robot_double_count():
    agent = {
        "id": 4,
        "pos": np.asarray([0.0, 0.0, 0.0, 0.0], dtype=float),
        "extent": 0.2,
    }
    robot = StraightLineVehicle()
    robot.size_scale = 2.0

    obstacles, debug = pedestrian_main.build_dynamic_obstacles_for_mppi(
        [agent],
        agent_predictions=None,
        horizon=2,
        robot_model=robot,
        hard_clearance_margin=pedestrian_main.MIN_SEPARATION,
    )

    expected_buffer = pedestrian_main.MIN_SEPARATION * robot.size_scale
    assert np.isclose(obstacles[0]["collision_buffer"], expected_buffer)
    assert np.isclose(debug[0]["collision_buffer"], expected_buffer)

    physical_obstacles, physical_debug = (
        pedestrian_main.build_dynamic_obstacles_for_mppi(
            [agent],
            agent_predictions=None,
            horizon=2,
            robot_model=robot,
        )
    )
    assert np.isclose(physical_obstacles[0]["collision_buffer"], 0.0)
    assert np.isclose(physical_debug[0]["collision_buffer"], 0.0)


def test_dynamic_clearance_margin_uses_actor_dimension_scale():
    robot = StraightLineVehicle()
    robot.size_scale = 0.25

    assert np.isclose(
        pedestrian_main.mppi_dynamic_clearance_margin(robot, 0.5),
        0.125,
    )
    assert np.isclose(
        pedestrian_main.mppi_static_clearance_margin(robot, 0.5),
        0.125,
    )


def test_prepare_obstacle_batches_keeps_multistep_dynamic_actor():
    sys.modules.pop("warp_mppi", None)
    sys.modules.pop("warp_mppi.legacy", None)
    sys.modules.pop("warp_mppi.legacy.mppi_pycuda", None)
    try:
        from warp_mppi.legacy import mppi_pycuda
    except Exception as exc:  # pragma: no cover - depends on optional CUDA stack
        pytest.skip(f"PyCUDA MPPI could not be imported: {exc}")

    obstacles = [
        {
            "states": np.asarray(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            "extent": [0.2, 0.2],
            "collision_buffer": 0.5,
        }
    ]

    static_extents, static_states, dynamic_actors, vertices, offsets = (
        mppi_pycuda._prepare_obstacle_batches(obstacles, horizon=4)
    )

    assert static_extents.shape == (0, 2)
    assert static_states.shape == (0, 4)
    assert vertices.shape == (0,)
    assert offsets.tolist() == [0]
    assert len(dynamic_actors) == 1
    assert dynamic_actors[0]["states"].shape == (4, 4)
    assert np.allclose(dynamic_actors[0]["states"][-1, :2], [3.0, 0.0])
    assert dynamic_actors[0]["collision_geometry"][0] > 0.2
