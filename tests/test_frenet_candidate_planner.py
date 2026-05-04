import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "python" / "pedestrian" / "pedestrian"))
sys.path.insert(0, str(ROOT / "src" / "thirdParty" / "warp_mppi"))

from trajectory_planner import cubic_spline_planner
from trajectory_planner.frenet_candidate import (
    frenet_lateral_offsets,
    frenet_prediction_schedule,
    project_pose_to_spline_frenet,
)
from trajectory_planner.frenet_optimal_trajectory import (
    PlannerArgs,
    frenet_optimal_planning,
)


def test_frenet_lateral_offsets_cover_nominal_and_bounds():
    assert frenet_lateral_offsets(1, 2.0) == [0.0]

    odd = frenet_lateral_offsets(5, 2.0)
    assert odd[0] == 0.0
    assert np.allclose(sorted(odd), [-2.0, -1.0, 0.0, 1.0, 2.0])

    even = frenet_lateral_offsets(4, 2.0)
    assert even[0] == 0.0
    assert -2.0 in even
    assert 2.0 in even
    assert any(offset < 0.0 for offset in even)
    assert any(offset > 0.0 for offset in even)


def test_project_pose_to_spline_frenet_straight_line():
    csp = cubic_spline_planner.Spline2D([0.0, 10.0], [0.0, 0.0])
    state = project_pose_to_spline_frenet(
        csp,
        np.asarray([4.2, 1.5, 0.0, 0.0]),
        search_step=0.01,
    )

    assert state["s"] == pytest.approx(4.2, abs=0.02)
    assert state["d"] == pytest.approx(1.5, abs=0.02)
    assert state["yaw"] == pytest.approx(0.0, abs=1.0e-6)


def test_project_pose_to_spline_frenet_curved_path():
    csp = cubic_spline_planner.Spline2D(
        [0.0, 2.0, 4.0],
        [0.0, 2.0, 0.0],
    )
    state = project_pose_to_spline_frenet(
        csp,
        np.asarray([2.0, 3.0, 0.0, 0.0]),
        search_step=0.01,
    )

    assert state["s"] > 0.0
    assert np.isfinite(state["yaw"])
    assert state["d"] > 0.5


def test_frenet_planner_explicit_offsets_start_from_current_lateral_state():
    csp = cubic_spline_planner.Spline2D([0.0, 10.0], [0.0, 0.0])
    offsets = [0.0, -1.0, 1.0]
    args = PlannerArgs(
        min_predict_time=1.0,
        max_predict_time=1.0,
        time_tick=0.1,
        target_speed=1.0,
        stopping_time=None,
        generate_planning_path=False,
        trajectory_offsets=offsets,
    )

    paths = frenet_optimal_planning(
        csp,
        2.0,
        1.0,
        0.5,
        0.0,
        0.0,
        0.0,
        args,
    )[1]

    assert len(paths) == len(offsets)
    for path, target_offset in zip(paths, offsets):
        assert path.y[0] == pytest.approx(0.5, abs=1.0e-6)
        assert path.d[-1] == pytest.approx(target_offset, abs=1.0e-6)
        assert path.y[-1] == pytest.approx(target_offset, abs=1.0e-6)


def test_frenet_prediction_schedule_uses_route_length_not_control_horizon():
    schedule = frenet_prediction_schedule(
        remaining_s=6.0,
        target_speed=0.5,
        current_s_speed=0.0,
        control_dt=0.01,
        resolution=0.1,
        control_horizon=10,
    )

    assert schedule["predict_time"] > 10.0
    assert schedule["time_tick"] > 0.01
    assert schedule["sample_spacing"] == pytest.approx(0.05)
    assert schedule["planning_speed"] == pytest.approx(0.5)


def test_frenet_offset_path_on_curve_has_dense_continuous_samples():
    csp = cubic_spline_planner.Spline2D(
        [0.0, 1.5, 3.0, 4.5],
        [0.0, 1.0, 1.0, 0.0],
    )
    schedule = frenet_prediction_schedule(
        remaining_s=float(csp.s[-1]),
        target_speed=0.5,
        current_s_speed=0.5,
        control_dt=0.01,
        resolution=0.1,
        control_horizon=10,
    )
    args = PlannerArgs(
        min_predict_time=schedule["predict_time"],
        max_predict_time=schedule["predict_time"],
        time_tick=schedule["time_tick"],
        target_speed=schedule["planning_speed"],
        stopping_time=None,
        generate_planning_path=False,
        trajectory_offsets=[0.0, 0.35],
    )

    paths = frenet_optimal_planning(
        csp,
        0.0,
        0.5,
        0.0,
        0.0,
        0.0,
        0.0,
        args,
    )[1]

    assert len(paths[1].x) > 20
    gaps = np.linalg.norm(
        np.diff(np.column_stack([paths[1].x, paths[1].y]), axis=0),
        axis=1,
    )
    assert float(np.max(gaps)) < 0.2


def test_generate_trajectories_frenet_smoke_when_sim_deps_available():
    pytest.importorskip("pygame")
    pytest.importorskip("visilibity")
    pytest.importorskip("warp_mppi.legacy")
    import main

    args = Namespace(
        trajectory_generator="frenet",
        trajectory_count=3,
        frenet_max_d=1.0,
        robot_speed=1.0,
        tick_time=0.1,
        horizon=5,
        max_initial_route_heading_error_deg=35.0,
        recovery_route_lookahead=1.0,
        debug_paths=False,
        debug_steering=False,
    )
    paths = main.generate_trajectories(
        np.asarray([0.0, 0.0, 1.0, 0.0]),
        np.asarray([5.0, 0.0]),
        args,
        static_polygons=[],
        display_offset=np.asarray([-1.0, -1.0]),
        display_diff=8.0,
        vehicle_length=0.7,
        vehicle_width=0.7,
        vehicle_scale=1.0,
        resolution=0.1,
    )

    assert len(paths) == 3
    assert paths[0]["generator"] == "frenet"
    assert paths[0]["nominal"]
    assert all("path" in path and "route" in path for path in paths)
