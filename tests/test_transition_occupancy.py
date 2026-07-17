import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PEDESTRIAN_ROOT = ROOT / "src" / "python" / "pedestrian" / "pedestrian"
sys.path.insert(0, str(PEDESTRIAN_ROOT))


class _Point:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)


class _Polygon:
    def __init__(self, points):
        self.points = np.asarray(points, dtype=float)
        self.is_valid = True
        self.is_empty = self.points.size == 0
        self.bounds = (
            float(np.min(self.points[:, 0])),
            float(np.min(self.points[:, 1])),
            float(np.max(self.points[:, 0])),
            float(np.max(self.points[:, 1])),
        )

    def buffer(self, _distance):
        return self

    def covers(self, point):
        minx, miny, maxx, maxy = self.bounds
        return minx <= point.x <= maxx and miny <= point.y <= maxy


shapely_stub = types.ModuleType("shapely")
shapely_geometry_stub = types.ModuleType("shapely.geometry")
shapely_geometry_stub.Point = _Point
shapely_geometry_stub.Polygon = _Polygon
shapely_geometry_stub.MultiPolygon = object
shapely_geometry_stub.LineString = object
shapely_ops_stub = types.ModuleType("shapely.ops")
shapely_ops_stub.unary_union = lambda polygons: None
sys.modules.setdefault("shapely", shapely_stub)
sys.modules.setdefault("shapely.geometry", shapely_geometry_stub)
sys.modules.setdefault("shapely.ops", shapely_ops_stub)

import Grid.OccupancyGrid as occupancy_grid_module
occupancy_grid_module.Point = _Point
occupancy_grid_module.Polygon = _Polygon
OccupancyGrid = occupancy_grid_module.OccupancyGrid
from Grid.predicted_occupancy import (
    build_transition_occupancy_horizon,
    line_is_occluded_by_occupancy,
    save_oce_debug_review_png,
)


def test_occupancy_grid_rasterizes_static_polygons_lower_left():
    polygon = {
        "blocking": True,
        "points": np.asarray(
            [
                [1.0, 1.0],
                [3.0, 1.0],
                [3.0, 3.0],
                [1.0, 3.0],
            ],
            dtype=float,
        ),
    }
    grid = OccupancyGrid(
        dim=4.0,
        resolution=1.0,
        origin=(0.0, 0.0),
        static_polygons=[polygon],
        origin_mode="lower_left",
    )

    probabilities = grid.probabilityMap()

    assert probabilities[1, 1] > 0.99
    assert probabilities[2, 2] > 0.99
    assert np.isclose(probabilities[0, 0], 0.5)


def test_transition_occupancy_treats_unknown_static_base_as_free():
    polygon = {
        "blocking": True,
        "points": np.asarray(
            [
                [1.0, 1.0],
                [2.0, 1.0],
                [2.0, 2.0],
                [1.0, 2.0],
            ],
            dtype=float,
        ),
    }
    base_grid = OccupancyGrid(
        dim=3.0,
        resolution=1.0,
        origin=(0.0, 0.0),
        static_polygons=[polygon],
        origin_mode="lower_left",
    )

    horizon = build_transition_occupancy_horizon(
        tracker=None,
        horizon=0,
        origin=(0.0, 0.0),
        resolution=1.0,
        rows=3,
        cols=3,
        base_grid=base_grid,
    )

    assert np.isclose(horizon.probability_grids[0, 0, 0], 0.0)
    assert horizon.probability_grids[0, 1, 1] > 0.99


class _FakeHMM:
    def __init__(self, state_distribution):
        self.state_distribution = np.asarray(state_distribution, dtype=np.float32)
        self.mode_distribution = np.asarray([1.0], dtype=np.float32)


class _FakeTracker:
    state_centers_sim = np.asarray([[0.5, 0.5]], dtype=np.float32)

    def __init__(self):
        self.agent_hmms = {
            10: _FakeHMM([0.4]),
            20: _FakeHMM([0.5]),
        }

    def agent_mixed_transition_csr(self, agent_ids, horizon):
        prefix = np.zeros((len(agent_ids), int(horizon) + 1, 1), dtype=np.float32)
        for idx, agent_id in enumerate(agent_ids):
            prefix[idx, :, 0] = self.agent_hmms[agent_id].state_distribution[0]
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0, 2), dtype=np.int32),
            prefix,
        )


def test_transition_occupancy_unions_probabilities_and_tracks_overlap():
    horizon = build_transition_occupancy_horizon(
        tracker=_FakeTracker(),
        horizon=1,
        origin=(0.0, 0.0),
        resolution=1.0,
        rows=2,
        cols=2,
        base_grid=np.zeros((2, 2), dtype=np.float32),
    )

    assert np.isclose(horizon.probability_grids[0, 0, 0], 0.7)
    assert horizon.overlap_count_grids[0, 0, 0] == 2
    assert horizon.collision_grids[0, 0, 0]
    assert horizon.owner_mask_grids[0, 0, 0] == np.uint64(0b11)


class _FakeCoarseTracker(_FakeTracker):
    state_centers_sim = np.asarray([[1.0, 1.0]], dtype=np.float32)

    def __init__(self):
        self.agent_hmms = {10: _FakeHMM([1.0])}
        self.state_grid_indices = np.asarray([[0, 0]], dtype=float)
        self.cell_size = 2.0
        self.bounds = {"min_x": 0.0, "min_y": 0.0, "max_y": 2.0}


def test_transition_occupancy_rasterizes_full_discrete_state_cell():
    horizon = build_transition_occupancy_horizon(
        tracker=_FakeCoarseTracker(),
        horizon=0,
        origin=(0.0, 0.0),
        resolution=1.0,
        rows=2,
        cols=2,
        base_grid=np.zeros((2, 2), dtype=np.float32),
    )

    assert np.allclose(horizon.probability_grids[0], 1.0)
    assert np.all(horizon.owner_mask_grids[0] == np.uint64(0b1))
    assert np.all(horizon.collision_grids[0])


def test_visibility_ignores_self_occlusion_but_blocks_other_owners():
    probability = np.asarray([[0.8, 0.0, 0.0]], dtype=np.float32)
    self_owner = np.asarray([[0b01, 0, 0]], dtype=np.uint64)
    overlap_owner = np.asarray([[0b11, 0, 0]], dtype=np.uint64)

    assert not line_is_occluded_by_occupancy(
        start=(0.1, 0.5),
        end=(2.5, 0.5),
        probability_grid=probability,
        owner_mask_grid=self_owner,
        target_owner_bit=np.uint64(0b01),
        origin=(0.0, 0.0),
        resolution=1.0,
        threshold=0.25,
    )
    assert line_is_occluded_by_occupancy(
        start=(0.1, 0.5),
        end=(2.5, 0.5),
        probability_grid=probability,
        owner_mask_grid=overlap_owner,
        target_owner_bit=np.uint64(0b01),
        origin=(0.0, 0.0),
        resolution=1.0,
        threshold=0.25,
    )


def test_oce_debug_review_png_writes_multi_panel_overlay(tmp_path):
    horizon = build_transition_occupancy_horizon(
        tracker=_FakeTracker(),
        horizon=5,
        origin=(0.0, 0.0),
        resolution=1.0,
        rows=4,
        cols=4,
        base_grid=np.zeros((4, 4), dtype=np.float32),
    )
    output = tmp_path / "oce_debug.png"

    save_oce_debug_review_png(
        horizon,
        output,
        robot_state=np.asarray([0.5, 0.5, 0.0, 0.0], dtype=float),
        targets=[
            {
                "id": 1,
                "pos": np.asarray([1.5, 1.5, 0.0, 0.0], dtype=float),
                "extent": 0.2,
                "visible": True,
            }
        ],
        static_polygons=[
            {
                "blocking": True,
                "points": np.asarray(
                    [[2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 3.0]],
                    dtype=float,
                ),
            }
        ],
        step_stride=5,
    )

    assert output.exists()
    assert output.stat().st_size > 0


def _install_pycuda_stubs():
    pycuda = types.ModuleType("pycuda")
    driver = types.ModuleType("pycuda.driver")
    compiler = types.ModuleType("pycuda.compiler")
    characterize = types.ModuleType("pycuda.characterize")
    gpuarray = types.ModuleType("pycuda.gpuarray")

    class _Context:
        @staticmethod
        def get_current():
            return None

        @staticmethod
        def synchronize():
            return None

    class _PrimaryContext:
        def push(self):
            return None

        def pop(self):
            return None

        def detach(self):
            return None

    class _Device:
        def __init__(self, _index):
            pass

        def retain_primary_context(self):
            return _PrimaryContext()

    class _SourceModule:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_function(self, _name):
            return lambda *args, **kwargs: None

    driver.init = lambda: None
    driver.Context = _Context
    driver.Device = _Device
    driver.LogicError = RuntimeError
    driver.Error = RuntimeError
    driver.CompileError = RuntimeError
    compiler.SourceModule = _SourceModule
    characterize.sizeof = lambda *_args, **_kwargs: 4

    sys.modules["pycuda"] = pycuda
    sys.modules["pycuda.driver"] = driver
    sys.modules["pycuda.compiler"] = compiler
    sys.modules["pycuda.characterize"] = characterize
    sys.modules["pycuda.gpuarray"] = gpuarray


def test_mppi_occupancy_cost_helper_soft_and_hard_costs():
    _install_pycuda_stubs()
    module_path = (
        ROOT
        / "src"
        / "thirdParty"
        / "warp_mppi"
        / "warp_mppi"
        / "legacy"
        / "mppi_pycuda.py"
    )
    spec = importlib.util.spec_from_file_location("mppi_pycuda_test_stub", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    grids = np.asarray([[[0.2, 0.8]]], dtype=np.float32)
    soft = module.occupancy_grid_cost_for_state(
        [0.2, 0.5, 0.0, 0.0],
        grids,
        origin=(0.0, 0.0),
        resolution=1.0,
        ego_collision_geometry=(0.0, 0.0, 0.0),
        soft_weight=10.0,
        collision_cost=1000.0,
        hard_threshold=0.65,
    )
    hard = module.occupancy_grid_cost_for_state(
        [1.2, 0.5, 0.0, 0.0],
        grids,
        origin=(0.0, 0.0),
        resolution=1.0,
        ego_collision_geometry=(0.0, 0.0, 0.0),
        soft_weight=10.0,
        collision_cost=1000.0,
        hard_threshold=0.65,
    )

    assert np.isclose(soft, 2.0)
    assert np.isclose(hard, 1008.0)


def test_mppi_rollout_obstacle_batches_disable_legacy_geometry_with_occupancy():
    _install_pycuda_stubs()
    module_path = (
        ROOT
        / "src"
        / "thirdParty"
        / "warp_mppi"
        / "warp_mppi"
        / "legacy"
        / "mppi_pycuda.py"
    )
    spec = importlib.util.spec_from_file_location("mppi_pycuda_test_stub", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    occupancy_payload = module._prepare_occupancy_grid_stack(
        {
            "probability_grids": np.zeros((2, 4, 4), dtype=np.float32),
            "origin": (0.0, 0.0),
            "resolution": 1.0,
        }
    )
    obstacles = [
        {"polygon": np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])},
        {
            "states": np.asarray(
                [[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            "extent": [0.2, 0.2],
        },
    ]

    static_extents, static_states, dynamic_actors, vertices, offsets = (
        module._prepare_rollout_obstacle_batches(
            obstacles, horizon=3, occupancy_payload=occupancy_payload
        )
    )

    assert static_extents.shape == (0, 2)
    assert static_states.shape == (0, 4)
    assert dynamic_actors == []
    assert vertices.shape == (0,)
    assert offsets.tolist() == [0]
