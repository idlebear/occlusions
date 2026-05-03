import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "python" / "pedestrian" / "pedestrian"))
sys.path.insert(0, str(ROOT / "src" / "thirdParty" / "warp_mppi"))

from datasources import load_scenario
from entropy_validation import (
    DEFAULT_DESTINATION_CLASS,
    DEFAULT_HORIZON,
    DEFAULT_ROBOT_GOAL,
    DEFAULT_ROBOT_HEADING,
    DEFAULT_ROBOT_START,
    DEFAULT_ROLLOUT_STEPS,
    DEFAULT_SCAN_RANGE,
    DEFAULT_SCENE_ID,
    DEFAULT_TRACK_ID,
    compare_cpu_gpu,
    load_sdd_model_bundle,
    load_track_state_sequence,
    linear_robot_path,
    point_is_free,
    pycuda_available,
    realized_class_belief,
    run_gpu_entropy,
    run_validation,
    shapely_occlusion_schedule,
    sparse_exact_entropy,
    target_visibility_over_rollout,
    validate_track_class,
    blocking_union,
)


@pytest.fixture(scope="module")
def validation_context():
    bundle = load_sdd_model_bundle("outputs/sdd_models", DEFAULT_SCENE_ID)
    scenario = load_scenario(
        "sdd",
        sdd_processed_root="outputs/sdd_processed",
        sdd_scene_id=DEFAULT_SCENE_ID,
    )
    target_states = load_track_state_sequence(
        "outputs/sdd_models",
        DEFAULT_SCENE_ID,
        DEFAULT_TRACK_ID,
    )
    initial_belief = np.zeros(bundle.state_centers_sim.shape[0], dtype=np.float32)
    initial_belief[int(target_states[0])] = 1.0
    static_union = blocking_union(scenario.static_polygons)
    return bundle, scenario, target_states, initial_belief, static_union


def test_default_track_class_and_robot_points_are_valid(validation_context):
    bundle, scenario, _target_states, _initial_belief, _static_union = validation_context

    validate_track_class(bundle, DEFAULT_TRACK_ID, DEFAULT_DESTINATION_CLASS)
    assert point_is_free(DEFAULT_ROBOT_START, scenario.static_polygons)
    assert point_is_free(DEFAULT_ROBOT_GOAL, scenario.static_polygons)


def test_sparse_cpu_entropy_horizon_30_is_finite_and_grows(validation_context):
    bundle, _scenario, _target_states, initial_belief, static_union = validation_context
    robot_path = np.repeat(
        np.asarray(DEFAULT_ROBOT_START, dtype=float).reshape(1, 2),
        DEFAULT_HORIZON + 1,
        axis=0,
    )
    schedule = shapely_occlusion_schedule(
        robot_path,
        bundle.state_centers_sim,
        static_union,
        DEFAULT_SCAN_RANGE,
    )

    results = sparse_exact_entropy(
        initial_belief,
        bundle.mixed_transition,
        schedule,
        DEFAULT_HORIZON,
    )
    entropies = np.asarray([row["entropy"] for row in results], dtype=float)

    assert len(results) == DEFAULT_HORIZON
    assert np.all(np.isfinite(entropies))
    assert np.all(entropies >= 0.0)
    assert entropies[-1] > entropies[4]


def test_realized_class_belief_converges_after_visibility(validation_context):
    bundle, _scenario, target_states, _initial_belief, static_union = validation_context
    rollout_path = linear_robot_path(
        DEFAULT_ROBOT_START,
        DEFAULT_ROBOT_GOAL,
        DEFAULT_ROLLOUT_STEPS,
    )
    visibility = target_visibility_over_rollout(
        rollout_path,
        target_states,
        bundle.state_centers_sim,
        static_union,
        DEFAULT_SCAN_RANGE,
    )
    rows = realized_class_belief(bundle, target_states, visibility)
    true_class_idx = bundle.class_ids.index(DEFAULT_DESTINATION_CLASS)

    assert any(visibility)
    assert not all(visibility)
    assert rows[-1]["mode_distribution"][true_class_idx] >= 0.95


def test_gpu_entropy_matches_sparse_cpu_when_available(validation_context):
    if not pycuda_available():
        pytest.skip("PyCUDA is not available")

    bundle, _scenario, _target_states, initial_belief, _static_union = validation_context
    robot_path = np.repeat(
        np.asarray(DEFAULT_ROBOT_START, dtype=float).reshape(1, 2),
        DEFAULT_HORIZON + 1,
        axis=0,
    )
    gpu_result = run_gpu_entropy(
        bundle,
        robot_path,
        initial_belief,
        bundle.mixed_transition,
        DEFAULT_HORIZON,
        DEFAULT_SCAN_RANGE,
    )
    gpu_occlusion = [
        (1.0 - gpu_result.visibility_tensor[0, step]).astype(float)
        for step in range(DEFAULT_HORIZON + 1)
    ]
    cpu_results = sparse_exact_entropy(
        initial_belief,
        bundle.mixed_transition,
        gpu_occlusion,
        DEFAULT_HORIZON,
    )

    compare_cpu_gpu(cpu_results, gpu_result, rtol=1.0e-4, atol=1.0e-5)


def test_cli_smoke_generates_all_outputs_when_gpu_available(tmp_path):
    if not pycuda_available():
        pytest.skip("PyCUDA is not available")

    out_dir = tmp_path / "entropy_validation"
    summary = run_validation(
        Namespace(
            sdd_processed_root="outputs/sdd_processed",
            sdd_model_root="outputs/sdd_models",
            scene_id=DEFAULT_SCENE_ID,
            destination_class=DEFAULT_DESTINATION_CLASS,
            track_id=DEFAULT_TRACK_ID,
            horizon=DEFAULT_HORIZON,
            interval=5,
            rollout_steps=DEFAULT_ROLLOUT_STEPS,
            robot_start=list(DEFAULT_ROBOT_START),
            robot_goal=list(DEFAULT_ROBOT_GOAL),
            robot_heading=DEFAULT_ROBOT_HEADING,
            scan_range=DEFAULT_SCAN_RANGE,
            out_dir=str(out_dir),
            rtol=1.0e-4,
            atol=1.0e-5,
            cpu_compare_stride=1,
        )
    )

    assert len(summary["outputs"]["current_belief_pngs"]) == DEFAULT_ROLLOUT_STEPS
    assert len(summary["outputs"]["horizon_contact_sheet_pngs"]) == DEFAULT_ROLLOUT_STEPS
    for output in summary["outputs"]["current_belief_pngs"]:
        assert Path(output).exists()
    for output in summary["outputs"]["horizon_contact_sheet_pngs"]:
        assert Path(output).exists()
    assert Path(summary["outputs"]["current_belief_dir"]).exists()
    assert Path(summary["outputs"]["horizon_contact_sheet_dir"]).exists()
    assert Path(summary["outputs"]["class_belief"]).exists()
    assert Path(summary["outputs"]["entropy_cpu_gpu_csv"]).exists()
    assert Path(summary["outputs"]["class_belief_csv"]).exists()
    assert Path(summary["outputs"]["summary"]).exists()
