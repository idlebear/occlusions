import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "python" / "pedestrian" / "pedestrian"))
sys.path.insert(0, str(ROOT / "src" / "thirdParty" / "warp_mppi"))

from entropy import evaluate_method
from hmm import HMM
from warp_mppi.legacy.discrete_oce_pycuda import (
    PYCUDA_AVAILABLE,
    evaluate_discrete_oce_gpu,
)


def _cpu_score(P, belief, occlusion):
    hmm = HMM(
        num_states=P.shape[0],
        num_observations=P.shape[0],
        num_modes=1,
        transitions=P.reshape(1, P.shape[0], P.shape[1]),
        emission_probabilities=np.eye(P.shape[0]),
        distributions={
            "state": belief,
            "mode": np.ones(1),
        },
        precompute_diagnostics=False,
    )
    result = evaluate_method(
        "discrete_exact_entropy",
        trial=0,
        k=len(occlusion) - 1,
        hmm=hmm,
        I_s=occlusion,
        state_coords=np.column_stack([np.arange(P.shape[0]), np.zeros(P.shape[0])]),
    )
    return float(result[-1]["cumulative_entropy"])


def test_cpu_discrete_exact_entropy_smoke():
    P = np.asarray(
        [
            [0.7, 0.3, 0.0],
            [0.0, 0.6, 0.4],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    belief = np.asarray([1.0, 0.0, 0.0], dtype=float)
    occlusion = [
        np.ones(3, dtype=float),
        np.asarray([1.0, 0.0, 1.0]),
        np.asarray([1.0, 1.0, 0.0]),
    ]

    score = _cpu_score(P, belief, occlusion)
    assert np.isfinite(score)
    assert score >= 0.0


@pytest.mark.skipif(not PYCUDA_AVAILABLE, reason="PyCUDA is not available")
def test_gpu_discrete_exact_matches_cpu_for_open_grid():
    P = np.asarray(
        [
            [0.7, 0.3, 0.0],
            [0.0, 0.6, 0.4],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    belief = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    paths = np.asarray([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]], dtype=np.float32)
    centers = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    static_grid = np.zeros((1, 3), dtype=np.uint8)

    gpu = evaluate_discrete_oce_gpu(
        paths=paths,
        state_centers=centers,
        static_grid=static_grid,
        grid_origin=(0.0, 0.0),
        grid_resolution=1.0,
        transition_matrices=P.reshape(1, 3, 3),
        beliefs=belief,
        horizon=2,
        scan_range=10.0,
        return_visibility=True,
    )

    occlusion = [
        1.0 - gpu.visibility_tensor[0, 0],
        1.0 - gpu.visibility_tensor[0, 1],
        1.0 - gpu.visibility_tensor[0, 2],
    ]
    cpu_score = _cpu_score(P.astype(float), belief[0].astype(float), occlusion)
    assert np.allclose(gpu.scores[0], cpu_score, rtol=1.0e-4, atol=1.0e-5)

    hmm = HMM(
        num_states=P.shape[0],
        num_observations=P.shape[0],
        num_modes=1,
        transitions=P.reshape(1, P.shape[0], P.shape[1]),
        emission_probabilities=np.eye(P.shape[0]),
        distributions={
            "state": belief[0].astype(float),
            "mode": np.ones(1),
        },
        precompute_diagnostics=False,
    )
    cpu = evaluate_method(
        "discrete_exact_entropy",
        trial=0,
        k=2,
        hmm=hmm,
        I_s=occlusion,
        state_coords=np.column_stack([np.arange(P.shape[0]), np.zeros(P.shape[0])]),
    )
    expected_entropy = np.asarray([row["entropy"] for row in cpu], dtype=np.float32)
    expected_probability = np.asarray([row["prob"] for row in cpu], dtype=np.float32)
    expected_e_state = np.asarray([row["E_state"] for row in cpu], dtype=np.float32)
    expected_a_state = np.asarray([row["A_state"] for row in cpu], dtype=np.float32)

    assert gpu.score_components.shape == (1, 1, 2, 4)
    assert np.allclose(gpu.step_entropy[0, 0], expected_entropy, rtol=1.0e-4, atol=1.0e-5)
    assert np.allclose(
        gpu.step_probability[0, 0],
        expected_probability,
        rtol=1.0e-4,
        atol=1.0e-5,
    )
    assert np.allclose(gpu.step_e_state[0, 0], expected_e_state, rtol=1.0e-4, atol=1.0e-5)
    assert np.allclose(gpu.step_a_state[0, 0], expected_a_state, rtol=1.0e-4, atol=1.0e-5)
