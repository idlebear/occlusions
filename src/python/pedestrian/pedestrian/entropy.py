"""
Given a set of observations, calculate the expected entropy of the system
at each step k.

"""

from dataclasses import dataclass
from typing import List

from time import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

try:  # Optional dependency for sparse suffix computation
    from scipy import sparse
except ImportError:  # pragma: no cover - SciPy not available in some environments
    sparse = None

try:
    from util.stats import MonteCarloIntegration
except ImportError:
    class MonteCarloIntegration:
        def __init__(self, *args, **kwargs):
            raise ImportError("util.stats.MonteCarloIntegration is unavailable")

from util.gaussian import create_gaussian
try:
    from noisy_observations import (
        construct_noisy_probability,
        construct_noisy_probability_with_caching,
    )
except ImportError:
    def construct_noisy_probability(*args, **kwargs):
        raise ImportError("noisy_observations.construct_noisy_probability is unavailable")

    def construct_noisy_probability_with_caching(*args, **kwargs):
        raise ImportError(
            "noisy_observations.construct_noisy_probability_with_caching is unavailable"
        )
from hmm import HMM


TOLERANCE = 1e-10  # TOLERANCE for numerical stability
BELIEF_THRESHOLD = 1e-8  # Threshold for belief probability to consider them non-zero
# 1e-6 prunes too much, execution time becomes obviously truncated (should look exponential)
# 1e-10 is too small, still allows the exhaustive entropy to fail due to memory overflow


@dataclass
class SensorConfig:
    mode: str = "noisy"
    noise_sigma: float = 0.3
    noise_kernel_size: int = 3
    noise_scale: float = 1.0
    noise_sigma_min: float = 0.1
    noise_sigma_rate: float = 0.025
    noise_sigma_max: float = 0.3


@dataclass
class GridConfig:
    height: int
    width: int


def _precompute_observation_probabilities(
    k: int, I_s: List[np.ndarray], sensor_config: SensorConfig, grid_config: GridConfig
) -> List[np.ndarray]:
    O_vis = []
    for i in range(k):
        O_vis.append(
            construct_noisy_probability(
                height=grid_config.height,
                width=grid_config.width,
                I_occ=I_s[i],
                sigma=sensor_config.noise_sigma,
                kernel_size=sensor_config.noise_kernel_size,
                scale=sensor_config.noise_scale,
            )
        )
    return O_vis


def _precompute_observation_probabilities_with_variable_noise(
    I_s: List[np.ndarray],
    sensor_config: SensorConfig,
    grid_config: GridConfig,
    sensor_pos: List[float],
) -> List[np.ndarray]:
    O_vis = []
    for pos, occ in zip(sensor_pos, I_s):
        O_vis.append(
            construct_noisy_probability_with_caching(
                height=grid_config.height,
                width=grid_config.width,
                I_occ=occ,
                sensor_pos=pos,
                noise_sigma_min=sensor_config.noise_sigma_min,
                noise_sigma_rate=sensor_config.noise_sigma_rate,
                noise_sigma_max=sensor_config.noise_sigma_max,
                kernel_size=sensor_config.noise_kernel_size,
            )
        )
    return O_vis


def calc_entropy(x):
    """
    Calculate the entropy of a distribution.

    Args:
        x (np.array): The distribution to calculate the entropy of.

    Returns:
        float: The entropy of the distribution.
    """
    log_x = np.zeros_like(x)
    valid_x = x > 0
    log_x[valid_x] = np.log(x[valid_x])
    entropy = -np.sum(x * log_x)
    return entropy


def evaluate_method(
    method,
    **kwargs,
):
    if method == "naive":
        entropy_fn = _naive
    elif method == "noisy_naive":
        entropy_fn = _noisy_naive
    elif method == "noisy":
        entropy_fn = _noisy
    elif method == "noisy_3":
        entropy_fn = _noisy
        kwargs["k"] = min(3, kwargs.get("k", 1))
    elif method == "noisy_5":
        entropy_fn = _noisy
        kwargs["k"] = min(5, kwargs.get("k", 1))
    elif method == "noisy_mc":
        entropy_fn = _mc_entropy
    elif method == "exact_entropy":
        entropy_fn = _exact_entropy
    elif method == "discrete_exact_entropy":
        entropy_fn = _discrete_exact_entropy
    elif method == "distance_aware_exact_entropy":
        entropy_fn = _distance_aware_exact_entropy
    elif method == "noisy_exact_entropy":
        entropy_fn = _noisy_exact_entropy
    elif method == "approximate_entropy":
        entropy_fn = _approximate_entropy
    elif method == "distance_aware_approximate_entropy":
        entropy_fn = _distance_aware_approximate_entropy
    elif method == "noisy_approximate_entropy":
        entropy_fn = _noisy_approximate_entropy
    elif method == "random":
        entropy_fn = _random_entropy
    elif method == "noisy_mode_mc":
        entropy_fn = _noisy_mode_mc
    elif method == "noisy_steps_mc":
        entropy_fn = _noisy_steps_mc
    elif method == "noisy_visibility":
        entropy_fn = _noisy_occlusion_probability
    elif method == "visibility":
        entropy_fn = _occlusion_probability
    elif method == "greedy":
        entropy_fn = _greedy_entropy
    elif method == "noisy_greedy":
        entropy_fn = _noisy_greedy_entropy
    elif method == "combined_entropy":
        entropy_fn = _combined_entropy
        kwargs["mode"] = "exact"
    elif method == "combined_approximate_entropy":
        entropy_fn = _combined_entropy
        kwargs["mode"] = "approximate"
    elif method == "combined_jsd_entropy":
        entropy_fn = _combined_entropy
        kwargs["mode"] = "exact"
        kwargs["separation_metric"] = "jsd"
    elif method == "combined_approximate_jsd_entropy":
        entropy_fn = _combined_entropy
        kwargs["mode"] = "approximate"
        kwargs["separation_metric"] = "jsd"
    else:
        raise ValueError(f"Unknown method: {method}")

    result = entropy_fn(
        **kwargs,
    )
    return result


def _random_entropy(trial, k, P, b, I_s, **kwargs):
    """
    Return a random value for the entropy.
    """

    # random is well, random and must be different between paths - repeating the same
    # seed for each path gives identical results and P0 is always chosen

    results = []
    aggregate_entropy = 0
    for step in range(1, k + 1):  # skip current step/position
        # construct a random entropy based on the number of occluded states
        occluded_states = np.sum(I_s[step])
        max_entropy = np.log(occluded_states) if occluded_states > 0 else 0.0
        entropy = np.random.uniform() * max_entropy
        prob = np.random.uniform()
        aggregate_entropy += entropy
        results.append(
            {
                "trial": trial,
                "step": step,
                "name": "Random",
                "entropy": entropy,
                "mean_entropy": aggregate_entropy / step,
                "cumulative_entropy": aggregate_entropy,
                "prob": prob,
                "time": 0,
            },
        )

    return results


def _noisy(trial, k, P, b, I_s, **kwargs):
    """
    Evaluate the entropy of the transition matrix P, starting with belief b and
    occluded states defined by I_s.
    Args:
        trial (int): The trial number.
        k (int): The number of steps to evaluate. If k < len(I_s), then after the first
                 k steps, the beliefs are propagated forward without branching.
        P (Matrix): The symbolic transition matrix to evaluate.
        b (Matrix): The symbolic initial belief vector.
        I_s (np.array): A list of lists, each defining the occluded states
                        at each step.
    Returns:
        Dictionary containing the evaluated entropy and calculation time.
    """
    total_steps = len(I_s) - 1
    assert k <= total_steps, "entropy step out of range!"
    M = len(I_s[0])

    beliefs = [b]
    probs = [1]

    tic = time()

    # construct the observation probabilities if a noisy observer is used.
    grid_height = kwargs.get("grid_height", None)
    grid_width = kwargs.get("grid_width", None)
    if grid_height is None or grid_width is None:
        raise ValueError(
            "Grid height and width must be specified for noisy naive entropy."
        )
    grid_config = GridConfig(height=grid_height, width=grid_width)

    sensor_config = SensorConfig(  # Populate from kwargs or defaults
        noise_kernel_size=kwargs.get("noise_kernel_size", 3),
        noise_scale=kwargs.get("noise_scale", 1),
        noise_sigma_min=kwargs.get("noise_sigma_min", 0.1),
        noise_sigma_rate=kwargs.get("noise_sigma_rate", 0.025),
        noise_sigma_max=kwargs.get("noise_sigma_max", 0.3),
    )
    sensor_pos = kwargs.get("sensor_pos", None)
    assert sensor_pos is not None, "Sensor position must be provided for noisy entropy."

    O_vis = _precompute_observation_probabilities_with_variable_noise(
        I_s, sensor_config, grid_config, sensor_pos
    )

    entropies = {}
    for i in range(total_steps):
        entropies[i] = []
    step_occlusion_observation_probs = []

    for step in range(1, k + 1):  # skip current step/position
        next_beliefs = []
        next_probs = []
        step_occlusion_observation_probs.append([])

        for belief, prob in zip(beliefs, probs):
            # Calculate P(X_s | Obs_hist_{s-1}) for this belief path
            pred_belief = belief @ P

            # Calculate the next belief vector based on the previous
            # belief and the observations
            # Accumulate P(Y_s = occluded_obs | Obs_hist_{s-1}) for this belief path
            # M is num_states, so O_vis[step][:, M] is P(Y_s=occ_obs | X_s)
            for i in range(M + 1):
                post_obs_belief = pred_belief * O_vis[step][:, i]
                eta = np.sum(post_obs_belief)
                if eta < TOLERANCE:
                    continue
                post_obs_belief /= eta
                post_obs_prob = eta * prob

                # If the post observation belief is below the threshold, skip it
                if post_obs_prob < BELIEF_THRESHOLD:
                    continue

                # If this is the occluded observation type, add its probability to the step total
                if i == M:
                    # P(Y_s=occ_obs | specific Obs_hist_{s-1})
                    step_occlusion_observation_probs[-1].append(post_obs_prob)

                next_beliefs.append(post_obs_belief)
                next_probs.append(post_obs_prob)

                # Use the full post-observation belief to capture sensor noise effects.
                entropy = calc_entropy(post_obs_belief)
                entropies[step].append(
                    {
                        "belief": post_obs_belief,
                        "prob": post_obs_prob,  # This is P(Y_s=i | Obs_hist_{s-1}) for this path
                        "entropy": entropy,
                    }
                )

        beliefs = next_beliefs
        probs = next_probs

    # propagate the beliefs forward for the remaining steps - no more branches
    for step in range(k + 1, total_steps + 1):
        next_beliefs = []
        next_probs = []
        step_occlusion_observation_probs.append([])

        for belief, prob in zip(beliefs, probs):
            # Calculate P(X_s | Obs_hist_{s-1}) for this belief path
            pred_belief = belief @ P
            occ_belief = belief * I_s[step]
            occ_prob = np.sum(occ_belief)
            if occ_prob < TOLERANCE:
                continue

            entropy = calc_entropy(occ_belief / occ_prob)
            step_occlusion_observation_probs[-1].append(occ_prob)

            entropies[step].append(
                {
                    "prob": prob,
                    "entropy": entropy,
                }
            )
        beliefs = next_beliefs
        probs = next_probs

    true_entropy_time = time() - tic

    # aggregate the probability of occlusion for each step
    aggregated_occlusion_probs = []
    for occlusion_probs in step_occlusion_observation_probs:
        aggregated_occlusion_probs.append(np.sum(occlusion_probs))

    # calculate the final entropy
    results = []
    aggregate_entropy = 0
    for step, values in entropies.items():
        entropy = 0
        # The 'prob' for the result dict should be P(Y_s=occluded_obs | Obs_hist_{s-1})
        occlusion_prob = aggregated_occlusion_probs[step]
        for entry in values:
            entropy += (
                entry["entropy"] * entry["prob"]
            )  # entry["prob"] is P(Y_s=i | Obs_hist_{s-1})
        aggregate_entropy += entropy
        results.append(
            {
                "trial": trial,
                "step": step + 1,
                "name": "True Entropy (Noisy)",
                "entropy": entropy,
                "mean_entropy": aggregate_entropy / step,
                "cumulative_entropy": aggregate_entropy,
                "prob": occlusion_prob,
                "time": true_entropy_time,
            },
        )

    return results


def _naive(trial, k, P, b, I_s, **kwargs):
    result = []

    tic = time()
    b_naive = b.copy()
    aggregate_entropy = 0
    for step in range(1, k + 1):
        b_naive = b_naive @ P
        b_occ = b_naive @ np.diag(I_s[step])
        b_prob = b_occ.sum()
        if b_prob == 0:
            naive_entropy = 0
        else:
            naive_entropy = calc_entropy(b_occ / b_prob) * b_prob
        naive_entropy_time = time() - tic
        aggregate_entropy += naive_entropy
        result.append(
            {
                "trial": trial,
                "step": step + 1,
                "name": "Naive Entropy",
                "entropy": naive_entropy,
                "mean_entropy": aggregate_entropy / step,
                "cumulative_entropy": aggregate_entropy,
                "prob": b_prob,
                "time": naive_entropy_time,
            }
        )

    return result


def _greedy_entropy(trial, k, hmm: HMM, I_s: List[np.ndarray], **kwargs):
    """
    Estimate the entropy of the next step using a greedy approach.
    """

    result = []

    tic = time()

    greedy_entropy = 0.0

    # for each mode in the HMM, calculate the visibility
    for index, P in enumerate(hmm.transition_matrices):
        b_greedy = hmm.state_distribution @ P
        b_occ = b_greedy @ np.diag(I_s[1])  # one step ahead
        b_prob = b_occ.sum()
        greedy_entropy = 0
        if b_prob > TOLERANCE:
            greedy_entropy = (
                calc_entropy(b_occ / b_prob) * b_prob * hmm.mode_distribution[index]
            )

    greedy_entropy_time = time() - tic
    result.append(
        {
            "trial": trial,
            "step": 1,
            "name": "Greedy Entropy",
            "entropy": greedy_entropy,
            "prob": b_prob,
            "time": greedy_entropy_time,
        }
    )

    return result


def _noisy_greedy_entropy(trial, k, hmm: HMM, I_s: List[np.ndarray], **kwargs):
    """
    Estimate the entropy of the next step using a greedy approach.
    """

    tic = time()

    grid_height = kwargs.get("grid_height", None)
    grid_width = kwargs.get("grid_width", None)
    if grid_height is None or grid_width is None:
        raise ValueError(
            "Grid height and width must be specified for noisy naive entropy."
        )
    grid_config = GridConfig(height=grid_height, width=grid_width)

    sensor_config = SensorConfig(  # Populate from kwargs or defaults
        noise_kernel_size=kwargs.get("noise_kernel_size", 3),
        noise_scale=kwargs.get("noise_scale", 1),
        noise_sigma_min=kwargs.get("noise_sigma_min", 0.1),
        noise_sigma_rate=kwargs.get("noise_sigma_rate", 0.025),
        noise_sigma_max=kwargs.get("noise_sigma_max", 0.3),
    )
    sensor_pos = kwargs.get("sensor_pos", None)
    assert sensor_pos is not None, "Sensor position must be provided for noisy entropy."

    O_vis = _precompute_observation_probabilities_with_variable_noise(
        I_s, sensor_config, grid_config, sensor_pos
    )

    greedy_entropy = 0.0

    # for each mode in the HMM, calculate the visibility
    for index, P in enumerate(hmm.transition_matrices):
        b_greedy = hmm.state_distribution @ P

        b_occ = b_greedy * O_vis[1][:, -1]  # one step ahead
        b_prob = b_occ.sum()
        greedy_entropy = 0
        if b_prob > TOLERANCE:
            greedy_entropy += (
                calc_entropy(b_occ / b_prob) * b_prob * hmm.mode_distribution[index]
            )

    greedy_entropy_time = time() - tic

    result = []
    result.append(
        {
            "trial": trial,
            "step": 1,
            "name": "Greedy Entropy",
            "entropy": greedy_entropy,
            "prob": b_prob,
            "time": greedy_entropy_time,
        }
    )

    return result


def _build_distance_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError("Kernel size must be a positive odd integer.")
    origin_offset = -(kernel_size // 2)
    return create_gaussian(
        size=kernel_size,
        origin=(origin_offset, origin_offset),
        sigma=sigma,
        scale=1,
    )


def _apply_distance_aware_blur(
    belief: np.ndarray, grid_height: int, grid_width: int, kernel: np.ndarray
) -> np.ndarray:
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2

    belief_grid = belief.reshape(grid_height, grid_width)
    padded_belief = np.pad(belief_grid, pad_width=pad, mode="constant")
    belief_windows = sliding_window_view(padded_belief, (kernel_size, kernel_size))
    blurred = np.einsum("ij,xyij->xy", kernel, belief_windows, optimize=True)

    ones_grid = np.ones_like(belief_grid)
    padded_ones = np.pad(ones_grid, pad_width=pad, mode="constant")
    ones_windows = sliding_window_view(padded_ones, (kernel_size, kernel_size))
    normalizers = np.einsum("ij,xyij->xy", kernel, ones_windows, optimize=True)
    normalizers = np.clip(normalizers, TOLERANCE, None)

    blurred /= normalizers
    blurred = np.clip(blurred, 0.0, None)
    total = blurred.sum()
    if total <= TOLERANCE:
        return np.zeros_like(belief)
    blurred /= total
    return blurred.ravel()


def _exact_entropy_calc(
    step,
    b_0,
    occ_vectors,
    P_cache,
    occ_suffixes,
    entropy_fn,
    spatial_fn=None,
    next_sensor_step=0,
    sensing_interval=1,
    planning_speed=1,
):
    """
    Calculate the entropy and probability by walking backwards from the
    final step, subtracting off the occluded paths.  This method calculates the exact
    entropy if all steps are used (N = len(I_s)).

    Args:
        N (int): The number of steps.
        P (np.array): The transition matrix.
        b (np.array): The belief vector.
        I_s (np.array): The occluded states.
        depth (int): The number of steps to use, with more steps leading to a more accurate result.

    Returns:
        dict: A unified entropy result containing:
            - "entropy": legacy weighted sum of partition entropies
            - "state_entropy": entropy of the mixed conditional belief
            - "mode_entropy": mode uncertainty term (0.0 for single-mode methods)
            - "spatial_separation": weighted spatial separation
            - "belief": mixed conditional belief
            - "prob": total occluded probability
            - "A_state": average within-partition entropy
            - "E_state": state_entropy - A_state
    """

    M = len(b_0)

    beliefs = []
    probs = []

    def _get_visibility_beliefs_for_partition(partition_step):
        """Get all visibility beliefs for a given partition."""
        partition_beliefs = []
        partition_probs = []

        initial_belief = np.zeros_like(b_0)

        b_partition = b_0 @ P_cache[partition_step]

        for state in range(M):
            # Check if state is visible (occlusion vector value is 0)
            if occ_vectors[partition_step][state] >= TOLERANCE:
                continue
            if b_partition[state] < TOLERANCE:
                continue

            initial_belief.fill(0)  # reset initial belief
            initial_belief[state] = b_partition[
                state
            ]  # set the current state as visible

            final_belief = initial_belief @ occ_suffixes[partition_step + 1]
            final_prob = np.sum(final_belief)
            if final_prob > TOLERANCE:
                partition_beliefs.append(final_belief)
                partition_probs.append(final_prob)

        return partition_beliefs, partition_probs

    if _has_sensor_opportunity(
        step,
        next_sensor_step=next_sensor_step,
        sensing_interval=sensing_interval,
        planning_speed=planning_speed,
    ):
        final_belief = b_0 @ occ_suffixes[1]
        final_prob = np.sum(final_belief)
        if final_prob > TOLERANCE:
            beliefs.append([final_belief])
            probs.append([final_prob])

    for partition_step in _iter_sensor_partition_steps(
        step,
        next_sensor_step=next_sensor_step,
        sensing_interval=sensing_interval,
        planning_speed=planning_speed,
    ):
        partition_beliefs, partition_probs = _get_visibility_beliefs_for_partition(
            partition_step
        )
        if partition_beliefs:  # Only add non-empty results
            beliefs.append(partition_beliefs)
            probs.append(partition_probs)

            # _draw_belief_for_partition(
            #     beliefs=partition_beliefs,
            #     partition=partition,
            #     occlusion=occ_vectors[partition],
            # )

    if len(beliefs) == 0:
        return {
            "entropy": 0,
            "state_entropy": 0,
            "mode_entropy": 0.0,
            "spatial_separation": 0,
            "belief": np.zeros_like(b_0),
            "prob": 0,
            "E_state": 0,
            "A_state": 0,
        }

    spatial_separation = 0.0
    legacy_entropy = 0
    A_state = 0.0

    sum_state = np.zeros_like(b_0)
    total_prob = sum([sum(p) for p in probs])

    for b, p in zip(beliefs, probs):
        for sb, sp in zip(b, p):
            if sp:
                entropy = entropy_fn(sb / sp) * sp
                legacy_entropy += entropy
                if spatial_fn is not None:
                    spatial_separation += spatial_fn(sb / sp) * sp
                sum_state += sb / total_prob
                A_state += entropy / total_prob

    state_entropy = entropy_fn(sum_state)
    E_state = max(0.0, state_entropy - A_state)

    assert total_prob <= 1.001, "Expected probability exceeds 1.0"

    # # normalize the entropy and spatial separation by the total expected probability
    # exact_entropy /= prob_expected
    # spatial_separation /= prob_expected

    return {
        # Legacy entropy is the weighted sum of the partition entropies.
        "entropy": float(legacy_entropy),
        "state_entropy": float(state_entropy),
        # Single-mode exact entropy has no mode uncertainty term.
        "mode_entropy": 0.0,
        "spatial_separation": float(spatial_separation),
        "belief": sum_state,
        "prob": float(total_prob),
        "E_state": float(E_state),
        "A_state": float(A_state),
    }


def _draw_belief_for_partition(beliefs, partition, occlusion, prefix=""):
    import matplotlib.pyplot as plt

    if type(beliefs) is list:
        # sum all beliefs
        belief = np.sum(beliefs, axis=0)
    else:
        belief = beliefs

    grid_size = int(np.sqrt(len(belief)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    belief_grid = belief.reshape(grid_size, grid_size)
    occlusion_grid = occlusion.reshape(grid_size, grid_size)

    im = axes[0].imshow(
        occlusion_grid,
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    axes[0].set_title(f"Partition {partition} Occlusion")

    im = axes[1].imshow(
        belief_grid,
        cmap="viridis",
        vmin=0,
        vmax=np.max(belief_grid),
    )
    axes[1].set_title(f"Partition {partition} Belief")
    fig.colorbar(im, ax=axes[1])

    plt.savefig(f"{prefix}belief_partition_{partition}.png")
    plt.close()


def _get_distance_aware_entropy_fn(grid_height, grid_width, kernel):

    def distance_aware_entropy_fn(belief):
        blurred = _apply_distance_aware_blur(belief, grid_height, grid_width, kernel)
        return calc_entropy(blurred)

    return distance_aware_entropy_fn


def _regularize_covariance(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mat = 0.5 * (mat + mat.T)
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.clip(eigvals, eps, None)
    return (eigvecs * eigvals) @ eigvecs.T


def _get_spatial_separation_fn(grid_height=None, grid_width=None, state_coords=None):
    if state_coords is not None:
        state_coords = np.asarray(state_coords, dtype=float)

        def spatial_separation_fn(belief):
            active = belief > TOLERANCE
            if not np.any(active):
                return 0.0

            coords = state_coords[active].T
            weights = belief[active]
            weights_total = weights.sum()
            if weights_total <= TOLERANCE:
                return 0.0
            mean = coords @ weights / weights_total
            centered = coords - mean[:, None]
            cov = centered * weights / weights_total @ centered.T
            cov = _regularize_covariance(cov)
            return np.trace(cov)

        return spatial_separation_fn

    def spatial_separation_fn(belief):

        belief_grid = belief.reshape(int(grid_height), int(grid_width))

        occupied = np.where(belief_grid > TOLERANCE)

        coords = np.vstack(occupied).astype(float)
        weights = belief_grid[occupied]
        weights_total = weights.sum()
        if weights_total <= TOLERANCE:
            return 0.0
        mean = coords @ weights / weights_total
        centered = coords - mean[:, None]
        cov = centered * weights / weights_total @ centered.T

        cov = _regularize_covariance(cov)

        # the spatial separation is the trace of the covariance matrix
        return np.trace(cov)

    return spatial_separation_fn


def _validate_sensor_schedule(next_sensor_step=0, sensing_interval=1, planning_speed=1):
    sensing_interval = int(sensing_interval)
    planning_speed = int(planning_speed)
    next_sensor_step = int(next_sensor_step)

    if planning_speed <= 0:
        raise ValueError("planning_speed must be a positive integer.")
    if sensing_interval <= 0:
        raise ValueError("sensing_interval must be a positive integer.")
    if not 0 <= next_sensor_step < sensing_interval:
        raise ValueError(
            "next_sensor_step must satisfy 0 <= next_sensor_step < sensing_interval."
        )

    return next_sensor_step, sensing_interval, planning_speed


def _sensor_active_at_step(
    step, next_sensor_step=0, sensing_interval=1, planning_speed=1
):
    if step <= 0:
        return False

    next_sensor_step, sensing_interval, planning_speed = _validate_sensor_schedule(
        next_sensor_step=next_sensor_step,
        sensing_interval=sensing_interval,
        planning_speed=planning_speed,
    )
    phase = (int(step) - 1) - next_sensor_step
    return phase >= 0 and (phase % sensing_interval) < planning_speed


def _has_sensor_opportunity(
    step, next_sensor_step=0, sensing_interval=1, planning_speed=1
):
    next_sensor_step, _, _ = _validate_sensor_schedule(
        next_sensor_step=next_sensor_step,
        sensing_interval=sensing_interval,
        planning_speed=planning_speed,
    )
    return int(step) > int(next_sensor_step)


def _iter_sensor_partition_steps(
    step, next_sensor_step=0, sensing_interval=1, planning_speed=1
):
    for partition_step in range(1, int(step)):
        if _sensor_active_at_step(
            partition_step,
            next_sensor_step=next_sensor_step,
            sensing_interval=sensing_interval,
            planning_speed=planning_speed,
        ):
            yield partition_step


def _compute_transition_prefixes(P, k, planning_speed: int = 1):
    total_steps = int(k)
    if total_steps <= 0:
        return {}

    planning_speed = int(planning_speed)
    if planning_speed <= 0:
        raise ValueError("planning_speed must be a positive integer.")

    identity = np.eye(P.shape[0], dtype=P.dtype)
    running = identity.copy()
    prefixes = {}
    for step in range(1, total_steps + 1):
        if (step - 1) % planning_speed == 0:
            running = running @ P
        prefixes[step] = running.copy()

    return prefixes


def _compute_occ_suffixes(
    occ_vectors,
    P,
    k,
    sensing_interval: int = 1,
    next_sensor_step: int = 0,
    planning_speed: int = 1,
):
    """
    Precompute suffix products S_step[idx] = P@diag(occ[idx])@...@P@diag(occ[step])
    for every step up to min(k, len(occ_vectors) - 1) where observations are available.
    """
    total_steps = min(k, len(occ_vectors) - 1)
    if total_steps <= 0:
        return {}

    next_sensor_step, sensing_interval, planning_speed = _validate_sensor_schedule(
        next_sensor_step=next_sensor_step,
        sensing_interval=sensing_interval,
        planning_speed=planning_speed,
    )
    identity = np.eye(P.shape[0], dtype=P.dtype)
    P_diag = {}
    for i in range(1, total_steps + 1):
        step_transition = P if (i - 1) % planning_speed == 0 else identity
        if _sensor_active_at_step(
            i,
            next_sensor_step=next_sensor_step,
            sensing_interval=sensing_interval,
            planning_speed=planning_speed,
        ):
            P_diag[i] = step_transition * occ_vectors[i].reshape(
                1, -1
            )  # same as step_transition @ np.diag(occ_vectors[i])
        else:
            P_diag[i] = step_transition

    occ_suffixes = {}
    for step in range(1, total_steps + 1):
        suffix = {}
        running = None
        for idx in range(step, 0, -1):
            if running is None:
                running = P_diag[idx]
            else:
                running = P_diag[idx] @ running
            suffix[idx] = running
        occ_suffixes[step] = suffix

    return occ_suffixes


def _compute_occ_suffixes_sparse(
    occ_vectors,
    P,
    k,
    matrix_format: str = "csr",
    sensing_interval: int = 1,
    next_sensor_step: int = 0,
    planning_speed: int = 1,
):
    """
    Sparse variant of `_compute_occ_suffixes` that keeps every product as a SciPy
    sparse matrix so we can compare timings against the dense implementation.
    """
    if sparse is None:
        raise ImportError(
            "_compute_occ_suffixes_sparse requires SciPy; install scipy to use it."
        )

    total_steps = min(k, len(occ_vectors) - 1)
    if total_steps <= 0:
        return {}

    if sparse.issparse(P):
        P_sparse = P.asformat(matrix_format)
    else:
        P_sparse = sparse.csr_matrix(P).asformat(matrix_format)

    next_sensor_step, sensing_interval, planning_speed = _validate_sensor_schedule(
        next_sensor_step=next_sensor_step,
        sensing_interval=sensing_interval,
        planning_speed=planning_speed,
    )
    identity_sparse = sparse.identity(P_sparse.shape[0], format=matrix_format)
    diag_products = {}
    for i in range(1, total_steps + 1):
        step_transition = P_sparse if (i - 1) % planning_speed == 0 else identity_sparse
        if _sensor_active_at_step(
            i,
            next_sensor_step=next_sensor_step,
            sensing_interval=sensing_interval,
            planning_speed=planning_speed,
        ):
            occ_diag = sparse.diags(occ_vectors[i], format=matrix_format)
            diag_products[i] = step_transition @ occ_diag
        else:
            diag_products[i] = step_transition

    occ_suffixes = {}
    for step in range(1, total_steps + 1):
        suffix = {}
        running = None
        for idx in range(step, 0, -1):
            if running is None:
                running = diag_products[idx]
            else:
                running = diag_products[idx] @ running
            suffix[idx] = running
        occ_suffixes[step] = suffix

    return occ_suffixes


def _compute_occ_suffixes_by_mode(
    occ_vectors,
    P_list,
    k,
    sensing_interval: int = 1,
    next_sensor_step: int = 0,
    planning_speed: int = 1,
):
    # Precompute promotion under occlusion keeping each mode separate
    occ_suffixes_by_mode = {}
    for c, P_c in enumerate(P_list):
        occ_suffixes = _compute_occ_suffixes(
            occ_vectors,
            P_c,
            k,
            sensing_interval=sensing_interval,
            next_sensor_step=next_sensor_step,
            planning_speed=planning_speed,
        )
        occ_suffixes_by_mode[c] = occ_suffixes

    return occ_suffixes_by_mode


def _exact_entropy(trial, k, P, b, I_s, depth=None, seed=None, **kwargs):
    """
    Estimate the entropy of the transition matrix P, starting with belief b and
    occluded states defined by I_s.

    This method finds the entropy at N for various depths, from 1 to N.

    Args:
        trial (int): The trial number.
        k (int): The number of steps to evaluate.
        P (Matrix): The symbolic transition matrix to evaluate.
        b (Matrix): The symbolic initial belief vector.
        I_s (np.array): A list of lists, each defining the occluded states
                        at each step.
    Returns:
        A list of dictionaries containing the estimated entropy and calculation time.

    """

    result = []

    tic = time()

    # Pull state entropy weight from kwargs or from an argparse-style args object, default 0.5
    alpha = kwargs.get("alpha", None)
    if alpha is None:
        alpha = 1.0
    alpha = float(alpha)
    zeta = 1.0 - alpha
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")

    state_coords = kwargs.get("state_coords")
    grid_height = kwargs.get("grid_height")
    grid_width = kwargs.get("grid_width")
    spatial_separation_fn = None
    if state_coords is not None:
        spatial_separation_fn = _get_spatial_separation_fn(state_coords=state_coords)
    elif grid_height is not None and grid_width is not None:
        spatial_separation_fn = _get_spatial_separation_fn(
            grid_height=grid_height,
            grid_width=grid_width,
        )

    # Store occlusion vectors directly
    occ_vectors = [np.asarray(occ, dtype=float) for occ in I_s]

    next_sensor_step = kwargs.get("next_sensor_step", 0)
    sensing_interval = kwargs.get("sensing_interval", 1)
    planning_speed = kwargs.get("planning_speed", 1)
    next_sensor_step, sensing_interval, planning_speed = _validate_sensor_schedule(
        next_sensor_step=next_sensor_step,
        sensing_interval=sensing_interval,
        planning_speed=planning_speed,
    )

    P_cache = _compute_transition_prefixes(P, k, planning_speed=planning_speed)

    occ_suffixes = _compute_occ_suffixes(
        occ_vectors,
        P,
        k,
        sensing_interval=sensing_interval,
        next_sensor_step=next_sensor_step,
        planning_speed=planning_speed,
    )

    aggregate_entropy = 0
    aggregate_combined_entropy = 0
    aggregate_occ_prob = 0
    for step in range(1, k + 1):
        entropy_result = _exact_entropy_calc(
            step=step,
            b_0=b,
            occ_vectors=occ_vectors,
            P_cache=P_cache,
            occ_suffixes=occ_suffixes[step],
            entropy_fn=calc_entropy,
            spatial_fn=spatial_separation_fn,
            sensing_interval=sensing_interval,
            next_sensor_step=next_sensor_step,
            planning_speed=planning_speed,
        )
        combined_entropy = (
            alpha * entropy_result["entropy"]
            + zeta * entropy_result["spatial_separation"]
        )
        aggregate_entropy += entropy_result["entropy"]
        aggregate_combined_entropy += combined_entropy
        aggregate_occ_prob += entropy_result["prob"]
        entropy_time = time() - tic
        result.append(
            {
                "trial": trial,
                "step": step,
                "name": "Expected Entropy",
                "entropy": entropy_result["entropy"],
                "state_entropy": entropy_result["state_entropy"],
                "mode_entropy": entropy_result["mode_entropy"],
                "combined_entropy": combined_entropy,
                "mean_entropy": aggregate_entropy / step,
                "cumulative_entropy": aggregate_entropy,
                "mean_combined_entropy": aggregate_combined_entropy / step,
                "cumulative_combined_entropy": aggregate_combined_entropy,
                "E_state": entropy_result["E_state"],
                "A_state": entropy_result["A_state"],
                "spatial_separation": entropy_result["spatial_separation"],
                "belief": entropy_result["belief"],
                "prob": entropy_result["prob"],
                "mean_occ_probability": aggregate_occ_prob / step,
                "time": entropy_time,
            }
        )

    return result


def _discrete_exact_entropy(trial, k, hmm: HMM, I_s, depth=None, seed=None, **kwargs):
    """
    Evaluate discrete OCE for an HMM belief by delegating to the legacy exact
    entropy implementation with the HMM's current mixed transition model.
    """

    P = hmm.get_mixed_transition_matrix()
    b = np.asarray(hmm.state_distribution, dtype=float).copy()
    return _exact_entropy(
        trial=trial,
        k=k,
        P=P,
        b=b,
        I_s=I_s,
        depth=depth,
        seed=seed,
        **kwargs,
    )


def _distance_aware_exact_entropy(trial, k, P, b, I_s, depth=None, seed=None, **kwargs):
    result = []

    tic = time()

    # Pull state entropy weight from kwargs or from an argparse-style args object, default 0.5
    alpha = kwargs.get("alpha", None)
    if alpha is None:
        alpha = 1.0
    alpha = float(alpha)
    zeta = 1.0 - alpha
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")

    occ_vectors = [np.asarray(occ, dtype=float) for occ in I_s]

    next_sensor_step = kwargs.get("next_sensor_step", 0)
    sensing_interval = kwargs.get("sensing_interval", 1)
    planning_speed = kwargs.get("planning_speed", 1)
    next_sensor_step, sensing_interval, planning_speed = _validate_sensor_schedule(
        next_sensor_step=next_sensor_step,
        sensing_interval=sensing_interval,
        planning_speed=planning_speed,
    )

    P_cache = _compute_transition_prefixes(P, k, planning_speed=planning_speed)

    occ_suffixes = _compute_occ_suffixes(
        occ_vectors,
        P,
        k,
        sensing_interval=sensing_interval,
        next_sensor_step=next_sensor_step,
        planning_speed=planning_speed,
    )

    noise_kernel_size = kwargs.get("noise_kernel_size")
    if noise_kernel_size is None:
        raise ValueError("Kernel size must be provided for distance-aware entropy.")

    noise_sigma_max = kwargs.get("noise_sigma_max")
    if noise_sigma_max is None:
        raise ValueError("noise_sigma_max must be provided for distance-aware entropy.")

    kernel = _build_distance_kernel(noise_kernel_size, noise_sigma_max)

    grid_height = kwargs.get("grid_height")
    grid_width = kwargs.get("grid_width")
    if grid_height is None or grid_width is None:
        raise ValueError(
            "Grid height and width must be specified for distance-aware entropy."
        )
    entropy_fn = _get_distance_aware_entropy_fn(
        grid_height=grid_height,
        grid_width=grid_width,
        kernel=kernel,
    )
    spatial_separation_fn = _get_spatial_separation_fn(
        grid_height=grid_height,
        grid_width=grid_width,
    )

    aggregate_entropy = 0
    aggregate_combined_entropy = 0
    aggregate_occ_prob = 0
    for step in range(1, k + 1):
        entropy_result = _exact_entropy_calc(
            step=step,
            b_0=b,
            occ_vectors=occ_vectors,
            P_cache=P_cache,
            occ_suffixes=occ_suffixes[step],
            entropy_fn=entropy_fn,
            spatial_fn=spatial_separation_fn,
            sensing_interval=sensing_interval,
            next_sensor_step=next_sensor_step,
            planning_speed=planning_speed,
        )
        combined_entropy = (
            alpha * entropy_result["entropy"]
            + zeta * entropy_result["spatial_separation"]
        )
        aggregate_entropy += entropy_result["entropy"]
        aggregate_combined_entropy += combined_entropy
        aggregate_occ_prob += entropy_result["prob"]
        entropy_time = time() - tic
        result.append(
            {
                "trial": trial,
                "step": step,
                "name": "Distance-Aware Entropy",
                "entropy": entropy_result["entropy"],
                "state_entropy": entropy_result["state_entropy"],
                "mode_entropy": entropy_result["mode_entropy"],
                "combined_entropy": combined_entropy,
                "mean_entropy": aggregate_entropy / step,
                "cumulative_entropy": aggregate_entropy,
                "mean_combined_entropy": aggregate_combined_entropy / step,
                "cumulative_combined_entropy": aggregate_combined_entropy,
                "spatial_separation": entropy_result["spatial_separation"],
                "belief": entropy_result["belief"],
                "E_state": entropy_result["E_state"],
                "A_state": entropy_result["A_state"],
                "prob": entropy_result["prob"],
                "mean_occ_probability": aggregate_occ_prob / step,
                "time": entropy_time,
            }
        )

    return result


def _noisy_exact_entropy_calc(
    step,
    b_0,
    occ_vectors,
    P_cache,
    O_vis,
    occ_suffixes,
    entropy_fn=calc_entropy,
    spatial_fn=None,
    next_sensor_step=0,
    sensing_interval=1,
    planning_speed=1,
):
    """
    Calculate the entropy and probability by walking backwards from the
    final step, subtracting off the occluded paths.  This method calculates the exact
    entropy if all steps are used (N = len(I_s)).

    This version of the function is designed to handle noisy observations.

    Args:
        N (int): The number of steps.
        P (np.array): The transition matrix.
        b (np.array): The belief vector.
        I_s (np.array): The occluded states.
        depth (int): The number of steps to use, with more steps leading to a more accurate result.

    Returns:
        dict: Same unified entropy result schema as _exact_entropy_calc().
    """

    M = len(b_0)

    beliefs = []
    probs = []

    def _get_visibility_beliefs_for_partition(partition_step):
        """Get all visibility beliefs for a given partition."""
        partition_beliefs = []
        partition_probs = []

        b_partition = b_0 @ P_cache[partition_step]

        for obs in range(M):
            b_state = b_partition * O_vis[partition_step][:, obs]
            prob_state = b_state.sum()
            if prob_state < TOLERANCE:
                continue  # Skip if belief is zero

            final_belief = b_state @ occ_suffixes[partition_step + 1]
            final_prob = np.sum(final_belief)
            if final_prob > TOLERANCE:
                partition_beliefs.append(final_belief)
                partition_probs.append(final_prob)

        return partition_beliefs, partition_probs

    if _has_sensor_opportunity(
        step,
        next_sensor_step=next_sensor_step,
        sensing_interval=sensing_interval,
        planning_speed=planning_speed,
    ):
        final_belief = b_0 @ occ_suffixes[1]
        final_prob = np.sum(final_belief)
        if final_prob > TOLERANCE:
            beliefs.append([final_belief])
            probs.append([final_prob])

    for partition_step in _iter_sensor_partition_steps(
        step,
        next_sensor_step=next_sensor_step,
        sensing_interval=sensing_interval,
        planning_speed=planning_speed,
    ):
        partition_beliefs, partition_probs = _get_visibility_beliefs_for_partition(
            partition_step
        )

        if partition_beliefs:  # Only add non-empty results
            beliefs.append(partition_beliefs)
            probs.append(partition_probs)

    if len(beliefs) == 0:
        return {
            "entropy": 0.0,
            "state_entropy": 0.0,
            "mode_entropy": 0.0,
            "spatial_separation": 0.0,
            "belief": np.zeros_like(b_0),
            "prob": 0.0,
            "E_state": 0.0,
            "A_state": 0.0,
        }

    legacy_entropy = 0.0
    spatial_separation = 0.0
    A_state = 0.0

    sum_state = np.zeros_like(b_0)
    total_prob = sum(sum(p) for p in probs)

    for b, p in zip(beliefs, probs):
        for sb, sp in zip(b, p):
            if sp <= TOLERANCE:
                continue
            entropy = entropy_fn(sb / sp) * sp
            legacy_entropy += entropy
            if spatial_fn is not None:
                spatial_separation += spatial_fn(sb / sp) * sp
            sum_state += sb / total_prob
            A_state += entropy / total_prob

    state_entropy = entropy_fn(sum_state)
    E_state = max(0.0, state_entropy - A_state)

    assert total_prob <= 1.001, "Expected probability exceeds 1.0"

    return {
        # Legacy entropy is the weighted sum of the partition entropies.
        "entropy": float(legacy_entropy),
        "state_entropy": float(state_entropy),
        # Single-mode noisy exact entropy has no mode uncertainty term.
        "mode_entropy": 0.0,
        "spatial_separation": float(spatial_separation),
        "belief": sum_state,
        "prob": float(total_prob),
        "E_state": float(E_state),
        "A_state": float(A_state),
    }


def _noisy_exact_entropy(trial, k, P, b, I_s, depth=None, seed=None, **kwargs):
    """
    Estimate the entropy of the transition matrix P, starting with belief b and
    occluded states defined by I_s.

    This method finds the entropy at N for various depths, from 1 to N.

    Args:
        trial (int): The trial number.
        k (int): The number of steps to evaluate.
        P (Matrix): The symbolic transition matrix to evaluate.
        b (Matrix): The symbolic initial belief vector.
        I_s (np.array): A list of lists, each defining the occluded states
                        at each step.
    Returns:
        A list of dictionaries containing the estimated entropy and calculation time.

    """

    result = []

    tic = time()

    alpha = kwargs.get("alpha", None)
    if alpha is None:
        alpha = 1.0
    alpha = float(alpha)
    zeta = 1.0 - alpha
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")

    grid_height = kwargs.get("grid_height")
    grid_width = kwargs.get("grid_width")
    if grid_height is None or grid_width is None:
        raise ValueError(
            "Grid height and width must be specified for noisy exact entropy."
        )
    grid_config = GridConfig(height=grid_height, width=grid_width)
    spatial_separation_fn = _get_spatial_separation_fn(
        grid_height=grid_height,
        grid_width=grid_width,
    )

    # Store occlusion vectors directly
    occ_vectors = [np.asarray(occ, dtype=float) for occ in I_s]

    next_sensor_step = kwargs.get("next_sensor_step", 0)
    sensing_interval = kwargs.get("sensing_interval", 1)
    planning_speed = kwargs.get("planning_speed", 1)
    next_sensor_step, sensing_interval, planning_speed = _validate_sensor_schedule(
        next_sensor_step=next_sensor_step,
        sensing_interval=sensing_interval,
        planning_speed=planning_speed,
    )

    P_cache = _compute_transition_prefixes(P, k, planning_speed=planning_speed)

    sensor_config = SensorConfig(
        mode=kwargs.get("sensor_mode", "noisy"),
        noise_kernel_size=kwargs.get("noise_kernel_size", 3),
        noise_scale=kwargs.get("noise_scale", 1),
        noise_sigma_min=kwargs.get("noise_sigma_min", 0.1),
        noise_sigma_rate=kwargs.get("noise_sigma_rate", 0.025),
        noise_sigma_max=kwargs.get("noise_sigma_max", 0.3),
    )
    sensor_pos = kwargs.get("sensor_pos", None)
    assert sensor_pos is not None, "Sensor position must be provided for noisy entropy."

    O_vis = _precompute_observation_probabilities_with_variable_noise(
        I_s, sensor_config, grid_config, sensor_pos
    )

    # Precompute promotion under occlusion
    occ_suffixes = _compute_occ_suffixes(
        occ_vectors,
        P,
        k,
        sensing_interval=sensing_interval,
        next_sensor_step=next_sensor_step,
        planning_speed=planning_speed,
    )

    aggregate_entropy = 0.0
    aggregate_combined_entropy = 0.0
    aggregate_occ_prob = 0.0
    for step in range(1, k + 1):
        entropy_result = _noisy_exact_entropy_calc(
            step=step,
            b_0=b,
            occ_vectors=occ_vectors,
            P_cache=P_cache,
            O_vis=O_vis,
            occ_suffixes=occ_suffixes[step],
            entropy_fn=calc_entropy,
            spatial_fn=spatial_separation_fn,
            next_sensor_step=next_sensor_step,
            sensing_interval=sensing_interval,
            planning_speed=planning_speed,
        )
        combined_entropy = (
            alpha * entropy_result["entropy"]
            + zeta * entropy_result["spatial_separation"]
        )
        aggregate_entropy += entropy_result["entropy"]
        aggregate_combined_entropy += combined_entropy
        aggregate_occ_prob += entropy_result["prob"]
        entropy_time = time() - tic
        result.append(
            {
                "trial": trial,
                "step": step,
                "name": "Expected Entropy",
                "entropy": entropy_result["entropy"],
                "state_entropy": entropy_result["state_entropy"],
                "mode_entropy": entropy_result["mode_entropy"],
                "combined_entropy": combined_entropy,
                "mean_entropy": aggregate_entropy / step,
                "cumulative_entropy": aggregate_entropy,
                "mean_combined_entropy": aggregate_combined_entropy / step,
                "cumulative_combined_entropy": aggregate_combined_entropy,
                "E_state": entropy_result["E_state"],
                "A_state": entropy_result["A_state"],
                "spatial_separation": entropy_result["spatial_separation"],
                "belief": entropy_result["belief"],
                "prob": entropy_result["prob"],
                "mean_occ_probability": aggregate_occ_prob / step,
                "time": entropy_time,
            }
        )

    return result


# def _draw_belief_for_partition(partition, belief, V_occ_partition, grid):
#     """Draw a belief for a given partition."""
#     import matplotlib.pyplot as plt

#     fig, ax = plt.subplots()
#     M = len(belief)

#     data = np.ones((int(np.sqrt(M)), int(np.sqrt(M)), 3), dtype=int) * 255
#     if grid is not None:
#         obstacles = np.where(
#             grid == 1
#         )  # Assuming grid is a 2D array with obstacles marked as 1
#         for i in range(len(obstacles[0])):
#             x, y = obstacles[0][i], obstacles[1][i]
#             data[x, y, :] = (5, 5, 5)  # Mark obstacles in dark gray

#     for m in range(M):
#         if V_occ_partition[m, m]:
#             data[m // int(np.sqrt(M)), m % int(np.sqrt(M)), :] = (60, 60, 60)

#     b_active = np.where(belief > TOLERANCE)[0]
#     if len(b_active):
#         for i, state in enumerate(b_active):
#             row = state // int(np.sqrt(M))
#             col = state % int(np.sqrt(M))
#             data[row, col, :] = (
#                 int(255 * belief[i]) // 2 + 128,
#                 128,
#                 128,
#             )

#     ax.imshow(
#         data,
#         aspect="auto",
#     )
#     ax.set_title(f"Belief for Partition {partition}")
#     ax.set_xlabel("State")
#     ax.set_ylabel("Belief Probability")
#     plt.savefig(f"belief_partition_{partition}.png")
#     plt.close(fig)


def _approximate_entropy_calc(
    step,
    b_0,
    occ_vectors,
    P_cache,
    occ_suffixes,
    spatial_fn,
    grid,
    next_sensor_step=0,
    sensing_interval=1,
    planning_speed=1,
):
    """
    Calculate the entropy and probability by walking backwards from the
    final step, subtracting off the occluded paths. Any sequences or paths that
    are visible in the same step are aggregated, resulting in an approximation
    of the exact partition walk.

    Args:
        N (int): The number of steps.
        P (np.array): The transition matrix.
        b (np.array): The belief vector.
        I_s (np.array): The occluded states.
    Returns:
        dict: A unified entropy result containing:
            - "entropy": legacy weighted sum of partition entropies
            - "state_entropy": entropy of the mixed conditional belief
            - "mode_entropy": mode uncertainty term (0.0 for single-mode methods)
            - "spatial_separation": weighted spatial separation
            - "belief": mixed conditional belief
            - "prob": total occluded probability
            - "A_state": average within-partition entropy
            - "E_state": state_entropy - A_state
    """

    beliefs = []
    probs = []

    if _has_sensor_opportunity(
        step,
        next_sensor_step=next_sensor_step,
        sensing_interval=sensing_interval,
        planning_speed=planning_speed,
    ):
        zero_belief = b_0 @ occ_suffixes[1]
        zero_prob = np.sum(zero_belief)
        if zero_prob > TOLERANCE:
            beliefs.append(zero_belief)
            probs.append(zero_prob)

    # for each path where the target is visible when a sensor is available,
    # remove that path from the initial estimate and add a new entry
    for partition_step in _iter_sensor_partition_steps(
        step,
        next_sensor_step=next_sensor_step,
        sensing_interval=sensing_interval,
        planning_speed=planning_speed,
    ):
        # Apply visibility: (1 - occ_vector) gives visible states
        vis_mask = 1.0 - occ_vectors[partition_step]
        b_partition = (b_0 @ P_cache[partition_step]) * vis_mask
        if np.sum(b_partition) < TOLERANCE:
            continue

        final_belief = b_partition @ occ_suffixes[partition_step + 1]
        final_prob = np.sum(final_belief)
        if final_prob > TOLERANCE:
            beliefs.append(final_belief)
            probs.append(final_prob)

    if len(beliefs) == 0:
        return {
            "entropy": 0,
            "state_entropy": 0,
            "mode_entropy": 0.0,
            "spatial_separation": 0,
            "belief": np.zeros_like(b_0),
            "prob": 0.0,
            "E_state": 0.0,
            "A_state": 0.0,
        }

    legacy_entropy = 0
    spatial_separation = 0

    total_prob = sum(probs)
    sum_state = np.zeros_like(b_0)
    A_state = 0.0

    for b, p in zip(beliefs, probs):
        if p <= TOLERANCE:
            continue
        entropy = calc_entropy(b / p) * p
        legacy_entropy += entropy
        if spatial_fn is not None:
            spatial_separation += spatial_fn(b / p) * p

        sum_state += b / total_prob
        A_state += entropy / total_prob

    state_entropy = calc_entropy(sum_state)
    E_state = max(0.0, state_entropy - A_state)

    assert total_prob <= 1.001, "Expected probability exceeds 1.0"

    # # normalize the entropy and spatial separation by the total expected probability
    # approximate_entropy /= total_prob
    # spatial_separation /= total_prob

    return {
        # Legacy entropy is the weighted sum of the partition entropies.
        "entropy": float(legacy_entropy),
        "state_entropy": float(state_entropy),
        # Single-mode approximate entropy has no mode uncertainty term.
        "mode_entropy": 0.0,
        "spatial_separation": float(spatial_separation),
        "belief": sum_state,
        "prob": float(total_prob),
        "E_state": float(E_state),
        "A_state": float(A_state),
    }


def _distance_aware_approximate_entropy_calc(
    step,
    b_0,
    occ_vectors,
    P_cache,
    occ_suffixes,
    grid_height,
    grid_width,
    kernel,
    next_sensor_step=0,
    sensing_interval=1,
    planning_speed=1,
):
    beliefs = []
    probs = []

    if _has_sensor_opportunity(
        step,
        next_sensor_step=next_sensor_step,
        sensing_interval=sensing_interval,
        planning_speed=planning_speed,
    ):
        zero_belief = b_0 @ occ_suffixes[1]
        zero_prob = np.sum(zero_belief)
        if zero_prob > TOLERANCE:
            beliefs.append(zero_belief)
            probs.append(zero_prob)

    for partition_step in _iter_sensor_partition_steps(
        step,
        next_sensor_step=next_sensor_step,
        sensing_interval=sensing_interval,
        planning_speed=planning_speed,
    ):
        vis_mask = 1.0 - occ_vectors[partition_step]
        b_partition = (b_0 @ P_cache[partition_step]) * vis_mask
        if np.sum(b_partition) < TOLERANCE:
            continue

        final_belief = b_partition @ occ_suffixes[partition_step + 1]
        final_prob = np.sum(final_belief)
        if final_prob > TOLERANCE:
            beliefs.append(final_belief)
            probs.append(final_prob)

    if len(beliefs) == 0:
        return 0, 0

    approximate_entropy = 0
    prob_expected = 0
    for belief, prob in zip(beliefs, probs):
        if prob <= TOLERANCE:
            continue
        normalized = belief / prob
        blurred = _apply_distance_aware_blur(
            normalized, grid_height, grid_width, kernel
        )
        approximate_entropy += calc_entropy(blurred) * prob
        prob_expected += prob

    assert prob_expected <= 1.001, "Expected probability exceeds 1.0"

    # normalize the entropy and spatial separation by the total expected probability
    approximate_entropy /= prob_expected

    return approximate_entropy, prob_expected


def _approximate_entropy(trial, k, P, b, I_s, depth=None, seed=None, **kwargs):
    """
    Estimate the entropy of the transition matrix P, starting with belief b and
    occluded states defined by I_s.

    This method finds the entropy at N for various depths, from 1 to N.

    Args:
        trial (int): The trial number.
        k (int): The number of steps to evaluate.
        P (Matrix): The symbolic transition matrix to evaluate.
        b (Matrix): The symbolic initial belief vector.
        I_s (np.array): A list of lists, each defining the occluded states
                        at each step.
    Returns:
        Dictionary containing the estimated entropy and calculation time.

    """

    result = []

    tic = time()

    # Pull state entropy weight from kwargs or from an argparse-style args object, default 0.5
    alpha = kwargs.get("alpha", None)
    if alpha is None:
        alpha = 1.0
    alpha = float(alpha)
    zeta = 1.0 - alpha
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")

    grid_height = kwargs.get("grid_height")
    grid_width = kwargs.get("grid_width")
    if grid_height is None or grid_width is None:
        raise ValueError(
            "Grid height and width must be specified for spatial separation."
        )
    spatial_separation_fn = _get_spatial_separation_fn(
        grid_height=grid_height,
        grid_width=grid_width,
    )

    # Store occlusion vectors directly
    occ_vectors = [np.asarray(occ, dtype=float) for occ in I_s]

    next_sensor_step = kwargs.get("next_sensor_step", 0)
    sensing_interval = kwargs.get("sensing_interval", 1)
    planning_speed = kwargs.get("planning_speed", 1)
    next_sensor_step, sensing_interval, planning_speed = _validate_sensor_schedule(
        next_sensor_step=next_sensor_step,
        sensing_interval=sensing_interval,
        planning_speed=planning_speed,
    )

    P_cache = _compute_transition_prefixes(P, k, planning_speed=planning_speed)

    # Precompute promotion under occlusion
    occ_suffixes = _compute_occ_suffixes(
        occ_vectors,
        P,
        k,
        sensing_interval=sensing_interval,
        next_sensor_step=next_sensor_step,
        planning_speed=planning_speed,
    )

    aggregate_entropy = 0
    aggregate_combined_entropy = 0
    aggregate_occ_prob = 0
    for step in range(1, k + 1):
        step_result = _approximate_entropy_calc(
            step=step,
            b_0=b,
            occ_vectors=occ_vectors,
            P_cache=P_cache,
            occ_suffixes=occ_suffixes[step],
            spatial_fn=spatial_separation_fn,
            grid=kwargs.get("grid_data", None),  # Optional grid for visualization
            next_sensor_step=next_sensor_step,
            sensing_interval=sensing_interval,
            planning_speed=planning_speed,
        )
        combined_entropy = (
            alpha * step_result["entropy"] + zeta * step_result["spatial_separation"]
        )
        spatial_separation = step_result["spatial_separation"]
        entropy_prob = step_result["prob"]
        aggregate_entropy += step_result["entropy"]
        aggregate_combined_entropy += combined_entropy
        aggregate_occ_prob += entropy_prob
        entropy_time = time() - tic
        result.append(
            {
                "trial": trial,
                "step": step,
                "name": "Entropy Expected",
                "entropy": step_result["entropy"],
                "state_entropy": step_result["state_entropy"],
                "mode_entropy": step_result["mode_entropy"],
                "combined_entropy": combined_entropy,
                "mean_entropy": aggregate_entropy / step,
                "cumulative_entropy": aggregate_entropy,
                "mean_combined_entropy": aggregate_combined_entropy / step,
                "cumulative_combined_entropy": aggregate_combined_entropy,
                "spatial_separation": spatial_separation,
                "belief": step_result["belief"],
                "prob": entropy_prob,
                "mean_occ_probability": aggregate_occ_prob / step,
                "time": entropy_time,
                "E_state": step_result["E_state"],
                "A_state": step_result["A_state"],
            }
        )

    return result


def _distance_aware_approximate_entropy(
    trial, k, P, b, I_s, depth=None, seed=None, **kwargs
):
    result = []

    tic = time()

    occ_vectors = [np.asarray(occ, dtype=float) for occ in I_s]

    next_sensor_step = kwargs.get("next_sensor_step", 0)
    sensing_interval = kwargs.get("sensing_interval", 1)
    planning_speed = kwargs.get("planning_speed", 1)
    next_sensor_step, sensing_interval, planning_speed = _validate_sensor_schedule(
        next_sensor_step=next_sensor_step,
        sensing_interval=sensing_interval,
        planning_speed=planning_speed,
    )

    P_cache = _compute_transition_prefixes(P, k, planning_speed=planning_speed)
    occ_suffixes = _compute_occ_suffixes(
        occ_vectors,
        P,
        k,
        sensing_interval=sensing_interval,
        next_sensor_step=next_sensor_step,
        planning_speed=planning_speed,
    )

    grid_height = kwargs.get("grid_height")
    grid_width = kwargs.get("grid_width")
    if grid_height is None or grid_width is None:
        raise ValueError(
            "Grid height and width must be specified for distance-aware entropy."
        )

    noise_kernel_size = kwargs.get("noise_kernel_size")
    if noise_kernel_size is None:
        raise ValueError("Kernel size must be provided for distance-aware entropy.")

    noise_sigma_max = kwargs.get("noise_sigma_max")
    if noise_sigma_max is None:
        raise ValueError("noise_sigma_max must be provided for distance-aware entropy.")

    kernel = _build_distance_kernel(noise_kernel_size, noise_sigma_max)

    aggregate_entropy = 0
    aggregate_occ_prob = 0
    for step in range(1, k + 1):
        entropy, entropy_prob = _distance_aware_approximate_entropy_calc(
            step=step,
            b_0=b,
            occ_vectors=occ_vectors,
            P_cache=P_cache,
            occ_suffixes=occ_suffixes[step],
            grid_height=grid_height,
            grid_width=grid_width,
            kernel=kernel,
            next_sensor_step=next_sensor_step,
            sensing_interval=sensing_interval,
            planning_speed=planning_speed,
        )
        aggregate_entropy += entropy
        aggregate_occ_prob += entropy_prob
        entropy_time = time() - tic
        result.append(
            {
                "trial": trial,
                "step": step,
                "name": "Distance-Aware Approximate Entropy",
                "entropy": entropy,
                "mean_entropy": aggregate_entropy / step,
                "cumulative_entropy": aggregate_entropy,
                "prob": entropy_prob,
                "mean_occ_probability": aggregate_occ_prob / step,
                "time": entropy_time,
            }
        )

    return result


def _noisy_approximate_entropy_calc(
    step,
    b_0,
    occ_vectors,
    P_cache,
    S_vis,
    occ_suffixes,
    next_sensor_step=0,
    sensing_interval=1,
    planning_speed=1,
):
    """
    Calculate the entropy and probability by walking backwards from the
    final step, subtracting off the occluded paths.  Any sequences/paths that
    are visible in the same step are aggregated resulting in an
    approximtion of the true answer.

    Args:
        N (int): The number of steps.
        P (np.array): The transition matrix.
        b (np.array): The belief vector.
        I_s (np.array): The occluded states.
    Returns:
        tuple: The entropy and probability.
    """

    beliefs = []
    probs = []

    if _has_sensor_opportunity(
        step,
        next_sensor_step=next_sensor_step,
        sensing_interval=sensing_interval,
        planning_speed=planning_speed,
    ):
        zero_belief = b_0 @ occ_suffixes[1]
        zero_prob = np.sum(zero_belief)
        if zero_prob > TOLERANCE:
            beliefs.append(zero_belief)
            probs.append(zero_prob)

    for partition_step in _iter_sensor_partition_steps(
        step,
        next_sensor_step=next_sensor_step,
        sensing_interval=sensing_interval,
        planning_speed=planning_speed,
    ):
        b_partition = (b_0 @ P_cache[partition_step]) * S_vis[partition_step]
        if np.sum(b_partition) < TOLERANCE:
            continue

        final_belief = b_partition @ occ_suffixes[partition_step + 1]
        final_prob = np.sum(final_belief)
        if final_prob > TOLERANCE:
            beliefs.append(final_belief)
            probs.append(final_prob)

    if len(beliefs) == 0:
        return 0, 0

    approximate_entropy = 0
    prob_expected = 0
    # and the remaining occluded paths
    for b, p in zip(beliefs, probs):
        entropy = calc_entropy(b / p) if p else 0
        approximate_entropy += entropy * p
        prob_expected += p

    # normalize the entropy and spatial separation by the total expected probability
    approximate_entropy /= prob_expected

    return approximate_entropy, prob_expected


def _noisy_approximate_entropy(trial, k, P, b, I_s, depth=None, seed=None, **kwargs):
    """
    Estimate the entropy of the transition matrix P, starting with belief b and
    occluded states defined by I_s.

    This method finds the entropy at N for various depths, from 1 to N.

    This method accounts for noisy observations.

    Args:
        trial (int): The trial number.
        k (int): The number of steps to evaluate.
        P (Matrix): The symbolic transition matrix to evaluate.
        b (Matrix): The symbolic initial belief vector.
        I_s (np.array): A list of lists, each defining the occluded states
                        at each step.
    Returns:
        Dictionary containing the estimated entropy and calculation time.

    """

    result = []

    tic = time()

    # Store occlusion vectors directly
    occ_vectors = [np.asarray(occ, dtype=float) for occ in I_s]

    next_sensor_step = kwargs.get("next_sensor_step", 0)
    sensing_interval = kwargs.get("sensing_interval", 1)
    planning_speed = kwargs.get("planning_speed", 1)
    next_sensor_step, sensing_interval, planning_speed = _validate_sensor_schedule(
        next_sensor_step=next_sensor_step,
        sensing_interval=sensing_interval,
        planning_speed=planning_speed,
    )

    P_cache = _compute_transition_prefixes(P, k, planning_speed=planning_speed)

    grid_height = kwargs.get("grid_height", None)
    grid_width = kwargs.get("grid_width", None)
    if grid_height is None or grid_width is None:
        raise ValueError(
            "Grid height and width must be specified for noisy naive entropy."
        )
    grid_config = GridConfig(height=grid_height, width=grid_width)

    sensor_config = SensorConfig(  # Populate from kwargs or defaults
        mode=kwargs.get("sensor_mode", "noisy"),
        noise_kernel_size=kwargs.get("noise_kernel_size", 3),
        noise_scale=kwargs.get("noise_scale", 1),
        noise_sigma_min=kwargs.get("noise_sigma_min", 0.1),
        noise_sigma_rate=kwargs.get("noise_sigma_rate", 0.025),
        noise_sigma_max=kwargs.get("noise_sigma_max", 0.3),
    )
    sensor_pos = kwargs.get("sensor_pos", None)
    assert sensor_pos is not None, "Sensor position must be provided for noisy entropy."

    O_vis = _precompute_observation_probabilities_with_variable_noise(
        I_s, sensor_config, grid_config, sensor_pos
    )
    assert np.all(np.isclose(np.array(O_vis).sum(axis=2), 1.0))

    # preprocess O_vis into probability for each state skipping the last state (occluded)
    S_vis = [np.sum(o[:, :-1], axis=1) for o in O_vis]

    # Precompute promotion under occlusion
    occ_suffixes = _compute_occ_suffixes(
        occ_vectors,
        P,
        k,
        sensing_interval=sensing_interval,
        next_sensor_step=next_sensor_step,
        planning_speed=planning_speed,
    )

    aggregate_entropy = 0
    for step in range(1, k + 1):
        entropy, entropy_prob = _noisy_approximate_entropy_calc(
            step=step,
            b_0=b,
            occ_vectors=occ_vectors,
            P_cache=P_cache,
            S_vis=S_vis,
            occ_suffixes=occ_suffixes[step],
            next_sensor_step=next_sensor_step,
            sensing_interval=sensing_interval,
            planning_speed=planning_speed,
        )
        aggregate_entropy += entropy
        entropy_time = time() - tic
        result.append(
            {
                "trial": trial,
                "step": step,
                "name": "Entropy Expected",
                "entropy": entropy,
                "mean_entropy": aggregate_entropy / step,
                "cumulative_entropy": aggregate_entropy,
                "prob": entropy_prob,
                "time": entropy_time,
            }
        )

    return result


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Numerically stable KL divergence KL(p || q) for discrete distributions."""
    valid = p > TOLERANCE
    if not np.any(valid):
        return 0.0
    q_safe = np.clip(q[valid], TOLERANCE, None)
    return float(np.sum(p[valid] * (np.log(p[valid]) - np.log(q_safe))))


def _jensen_shannon_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon distance between discrete distributions."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum <= TOLERANCE or q_sum <= TOLERANCE:
        return 0.0

    p = p / p_sum
    q = q / q_sum
    m = 0.5 * (p + q)
    js_div = 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)
    return float(np.sqrt(max(js_div, 0.0)))


def _s_js_disc_partition(mode_dim, mode_partition_results):
    """
    Partition-local discrete JS separation:

        S_JS^disc(p, k) = (1/2) * sum_{m,n} w_m(p) w_n(p) D_JS(pi_k^{(m,p)}, pi_k^{(n,p)})

    where w_m(p) are posterior mode weights for the partition p and
    pi_k^{(m,p)} are per-mode state distributions at step k within that partition.
    """
    p_part = 0.0
    for w, _, p in mode_partition_results:
        if p > TOLERANCE:
            p_part += w * p
    if p_part <= TOLERANCE:
        return 0.0, 0.0

    mode_post = np.zeros(mode_dim, dtype=float)
    pi_by_mode = {}
    for idx, (w, b, p) in enumerate(mode_partition_results):
        if b is None or p <= TOLERANCE or w <= TOLERANCE:
            continue
        mode_post[idx] = w * p
        pi_by_mode[idx] = b / p

    post_mass = mode_post.sum()
    if post_mass <= TOLERANCE:
        return 0.0, p_part
    mode_post /= post_mass

    active_modes = [idx for idx in pi_by_mode if mode_post[idx] > TOLERANCE]
    if len(active_modes) < 2:
        return 0.0, p_part

    # 0.5 * sum_{m,n} (...) over a symmetric distance equals sum_{m<n} (...)
    separation = 0.0
    for i_pos, i_mode in enumerate(active_modes[:-1]):
        for j_mode in active_modes[i_pos + 1 :]:
            separation += (
                mode_post[i_mode]
                * mode_post[j_mode]
                * _jensen_shannon_distance(pi_by_mode[i_mode], pi_by_mode[j_mode])
            )

    return float(separation), p_part


# mixture probability and mixture final belief for this partition
def _mixture_probability(
    state_dim,
    mode_dim,
    mode_partition_results,
    spatial_separation_fn=None,
    separation_metric="spatial",
):
    mode_entropy_contribution = 0.0
    state_mix_entropy_contribution = 0.0
    separation_contribution = 0.0
    state_post_entropy = 0.0

    p_part = 0.0
    state_mix = np.zeros(state_dim)
    for w, b, p in mode_partition_results:
        if p > TOLERANCE:
            p_part += w * p
            state_mix += w * b

    state_entropy_contributions = []

    if p_part >= TOLERANCE:
        # State-entropy contribution (for each mode given this partition)

        for w, belief, p in mode_partition_results:
            if belief is None or p <= TOLERANCE or w <= TOLERANCE:
                continue

            part_prob = w * p
            entropy = calc_entropy(belief / p)
            state_entropy_contributions.append((part_prob, belief / p, entropy))

        state_mix_post = state_mix / p_part
        state_post_entropy = calc_entropy(state_mix_post)
        state_mix_entropy_contribution = state_post_entropy * p_part

        if separation_metric == "spatial":
            if spatial_separation_fn is not None:
                separation_contribution = spatial_separation_fn(state_mix_post) * p_part
        elif separation_metric == "jsd":
            s_js_disc, _ = _s_js_disc_partition(mode_dim, mode_partition_results)
            separation_contribution = s_js_disc * p_part
        else:
            raise ValueError("separation_metric must be 'spatial' or 'jsd'")

        # Mode posterior for this partition and mode-entropy
        mode_post = np.zeros(mode_dim)
        for index, (w, _, p) in enumerate(mode_partition_results):
            if p > TOLERANCE:
                mode_post[index] = w * p
        s = mode_post.sum()
        if s > TOLERANCE:
            mode_post /= s
            mode_entropy_contribution = calc_entropy(mode_post) * p_part

    return {
        "state_mix_entropy_contribution": state_mix_entropy_contribution,
        "state_entropy_contributions": state_entropy_contributions,
        "mode_entropy_contribution": mode_entropy_contribution,
        "separation_contribution": separation_contribution,
        "p_part": p_part,
        "state_post_entropy": state_post_entropy if p_part >= TOLERANCE else 0.0,
        "state_post": state_mix_post if p_part >= TOLERANCE else None,
        "mode_post": mode_post if p_part >= TOLERANCE else None,
    }


def _combined_partition_exact_calc(
    step,
    b_0,
    occ_vectors,
    P_caches_by_mode,
    occ_suffixes_by_mode,
    mode_weights,
    spatial_separation_fn=None,
    separation_metric="spatial",
    next_sensor_step=0,
    sensing_interval=1,
    planning_speed=1,
):
    """
    Multi-mode exact partition walk.

    This returns both the legacy partition-based entropy and the state-space
    decomposition terms:
      (a) "entropy" = sum_p P(p) H(b_t | p), the legacy weighted sum of
          partition entropies,
      (b) "state_entropy" = H(sum_p P(p | occ) b_t^p), the entropy of the
          mixed conditional belief,
      (c) "A_state" = sum_p P(p | occ) H(b_t | p),
      (d) "E_state" = state_entropy - A_state,
      (e) "mode_entropy" = expected mode entropy under partition posteriors,
      (f) "spatial_separation" = expected separation under partition posteriors.

    Args:
        step (int): final time index k at which to evaluate.
        b_0 (np.ndarray): initial state belief (shape [M]).
        occ_vectors (List[np.ndarray]): list of occlusion vectors (1=occluded, 0=visible).
        P_caches_by_mode (Dict[int, Dict[int, np.ndarray]]): for each mode c, a cache
              of P_c^p (p=1..k).
        mode_weights (np.ndarray): prior over modes (shape [C]).

    Returns:
        dict: Same unified entropy result schema as _exact_entropy_calc(), with
            a non-zero "mode_entropy" term for multimodal HMMs.
    """
    # Storage for partition contributions
    legacy_entropy_total = 0.0
    mode_entropy_total = 0.0
    spatial_separation_total = 0.0
    p_occ_total = 0.0

    partition_contributions = []

    if _has_sensor_opportunity(
        step,
        next_sensor_step=next_sensor_step,
        sensing_interval=sensing_interval,
        planning_speed=planning_speed,
    ):
        mode_partition_results = []
        for c, P_cache in P_caches_by_mode.items():
            fb = b_0 @ occ_suffixes_by_mode[c][step][1]
            fp = fb.sum()
            if fp < TOLERANCE:
                fb = None
                fp = 0.0
            mode_partition_results.append((mode_weights[c], fb, fp))

        mixture_result = _mixture_probability(
            state_dim=b_0.shape,
            mode_dim=mode_weights.shape,
            mode_partition_results=mode_partition_results,
            spatial_separation_fn=spatial_separation_fn,
            separation_metric=separation_metric,
        )
        legacy_entropy_total += mixture_result["state_mix_entropy_contribution"]
        mode_entropy_total += mixture_result["mode_entropy_contribution"]
        spatial_separation_total += mixture_result["separation_contribution"]
        p_occ_total += mixture_result["p_part"]
        if mixture_result["state_post"] is not None:
            partition_contributions.append(
                (
                    mixture_result["p_part"],
                    mixture_result["state_post"],
                    mixture_result["state_post_entropy"],
                )
            )

    for partition_step in _iter_sensor_partition_steps(
        step,
        next_sensor_step=next_sensor_step,
        sensing_interval=sensing_interval,
        planning_speed=planning_speed,
    ):
        # partition > 0: enumerate visible states at time = partition
        # For each visible state m, construct that partition and accumulate its contribution
        # Get visible states: states where occlusion vector is 0 (not occluded)
        vis_states = np.where(occ_vectors[partition_step] < TOLERANCE)[0]
        if vis_states.size == 0:
            continue

        # precalculate belief at partition for each mode: b_0 @ P_c^partition
        b_at_i_by_mode = {
            c: b_0 @ P_cache[partition_step] for c, P_cache in P_caches_by_mode.items()
        }

        # For each visible state m, compute per-mode final belief & probability then combine
        for m in vis_states:
            mode_partition_results = []

            for c, P_cache in P_caches_by_mode.items():
                # Extract mass through state m (one-hot visible state)
                mass_m = b_at_i_by_mode[c][m]
                fp = 0.0
                if mass_m > TOLERANCE:
                    suffix_matrix = occ_suffixes_by_mode[c][step][partition_step + 1]
                    fb = mass_m * suffix_matrix[m, :]
                    fp = fb.sum()
                if fp < TOLERANCE:
                    fb = None
                    fp = 0.0
                mode_partition_results.append((mode_weights[c], fb, fp))

            # Combine across modes for this specific (partition, m)
            mixture_result = _mixture_probability(
                state_dim=b_0.shape,
                mode_dim=mode_weights.shape,
                mode_partition_results=mode_partition_results,
                spatial_separation_fn=spatial_separation_fn,
                separation_metric=separation_metric,
            )
            legacy_entropy_total += mixture_result["state_mix_entropy_contribution"]
            mode_entropy_total += mixture_result["mode_entropy_contribution"]
            spatial_separation_total += mixture_result["separation_contribution"]
            p_occ_total += mixture_result["p_part"]

            if mixture_result["state_post"] is not None:
                partition_contributions.append(
                    (
                        mixture_result["p_part"],
                        mixture_result["state_post"],
                        mixture_result["state_post_entropy"],
                    )
                )

    if p_occ_total <= TOLERANCE:
        return {
            "entropy": 0.0,
            "state_entropy": 0.0,
            "mode_entropy": 0.0,
            "spatial_separation": 0.0,
            "belief": np.zeros_like(b_0),
            "prob": 0.0,
            "A_state": 0.0,
            "E_state": 0.0,
        }

    # find the within and between partition contributions to state entropy and mode entropy
    # and the expected spatial separation, then normalize by total probability to get expected values
    A_state = 0.0

    sum_state = np.zeros_like(b_0)

    for (
        p_prob,
        p_state,
        p_state_entropy,
    ) in partition_contributions:
        if p_occ_total > TOLERANCE and p_state is not None:
            sum_state += p_state * p_prob / p_occ_total
            A_state += p_state_entropy * p_prob / p_occ_total

    state_entropy = calc_entropy(sum_state)
    raw_E_state = state_entropy - A_state
    assert raw_E_state >= -1e-10, f"Expected state entropy is negative: {raw_E_state}"
    E_state = max(0, raw_E_state)  # Ensure non-negativity

    return {
        # Legacy entropy is the weighted sum of the partition entropies.
        "entropy": float(legacy_entropy_total),
        "state_entropy": float(state_entropy),
        "mode_entropy": float(mode_entropy_total),
        "spatial_separation": float(spatial_separation_total),
        "belief": sum_state,
        "prob": float(p_occ_total),
        "A_state": float(A_state),
        "E_state": float(E_state),
    }


def _combined_partition_approximate_calc(
    step,
    b_0,
    occ_vectors,
    P_caches_by_mode,
    occ_suffixes_by_mode,
    mode_weights,
    spatial_separation_fn=None,
    separation_metric="spatial",
    next_sensor_step=0,
    sensing_interval=1,
    planning_speed=1,
):
    """
    Multi-mode approximate partition walk.

    This uses the same decomposition as _combined_partition_exact_calc(), but
    aggregates all visible states at a sensing step into one approximate
    partition instead of splitting them by visible state.

    Args:
        step (int): final time index k at which to evaluate.
        b_0 (np.ndarray): initial state belief (shape [M]).
        occ_vectors (List[np.ndarray]): list of occlusion vectors (1=occluded, 0=visible).
        P_caches_by_mode (Dict[int, Dict[int, np.ndarray]]): for each mode c, a cache
              of P_c^p (p=1..k).
        mode_weights (np.ndarray): prior over modes (shape [C]).

    Returns:
        dict: Same unified entropy result schema as _combined_partition_exact_calc().
    """
    # Storage for partition contributions
    legacy_entropy_total = 0.0
    mode_entropy_total = 0.0
    spatial_separation_total = 0.0
    p_occ_total = 0.0

    partition_contributions = []

    if _has_sensor_opportunity(
        step,
        next_sensor_step=next_sensor_step,
        sensing_interval=sensing_interval,
        planning_speed=planning_speed,
    ):
        mode_partition_results = []
        for c, P_cache in P_caches_by_mode.items():
            fb = b_0 @ occ_suffixes_by_mode[c][step][1]
            fp = fb.sum()
            if fp < TOLERANCE:
                fb = None
                fp = 0.0
            mode_partition_results.append((mode_weights[c], fb, fp))

        mixture_result = _mixture_probability(
            state_dim=b_0.shape,
            mode_dim=mode_weights.shape,
            mode_partition_results=mode_partition_results,
            spatial_separation_fn=spatial_separation_fn,
            separation_metric=separation_metric,
        )
        legacy_entropy_total += mixture_result["state_mix_entropy_contribution"]
        mode_entropy_total += mixture_result["mode_entropy_contribution"]
        spatial_separation_total += mixture_result["separation_contribution"]

        p_occ_total += mixture_result["p_part"]
        if mixture_result["state_post"] is not None:
            partition_contributions.append(
                (
                    mixture_result["p_part"],
                    mixture_result["state_post"],
                    mixture_result["state_post_entropy"],
                )
            )

    for partition_step in _iter_sensor_partition_steps(
        step,
        next_sensor_step=next_sensor_step,
        sensing_interval=sensing_interval,
        planning_speed=planning_speed,
    ):
        mode_partition_results = []

        for c, P_cache in P_caches_by_mode.items():
            # Apply visibility mask: (1 - occ_vector) gives visible states
            vis_mask = 1.0 - occ_vectors[partition_step]
            b_partition = (b_0 @ P_cache[partition_step]) * vis_mask
            fp = 0.0
            if np.sum(b_partition) > TOLERANCE:
                # Propagate occluded from (partition+1) .. step
                fb = b_partition @ occ_suffixes_by_mode[c][step][partition_step + 1]
                fp = fb.sum()

            if fp < TOLERANCE:
                fb = None
                fp = 0.0
            mode_partition_results.append((mode_weights[c], fb, fp))

        # Combine across modes for this specific (partition, m)
        mixture_result = _mixture_probability(
            state_dim=b_0.shape,
            mode_dim=mode_weights.shape,
            mode_partition_results=mode_partition_results,
            spatial_separation_fn=spatial_separation_fn,
            separation_metric=separation_metric,
        )
        legacy_entropy_total += mixture_result["state_mix_entropy_contribution"]
        mode_entropy_total += mixture_result["mode_entropy_contribution"]
        spatial_separation_total += mixture_result["separation_contribution"]
        p_occ_total += mixture_result["p_part"]

        if mixture_result["state_post"] is not None:
            partition_contributions.append(
                (
                    mixture_result["p_part"],
                    mixture_result["state_post"],
                    mixture_result["state_post_entropy"],
                )
            )

    if p_occ_total <= TOLERANCE:
        return {
            "entropy": 0.0,
            "state_entropy": 0.0,
            "mode_entropy": 0.0,
            "spatial_separation": 0.0,
            "belief": np.zeros_like(b_0),
            "prob": 0.0,
            "A_state": 0.0,
            "E_state": 0.0,
        }

    # find the within and between partition contributions to state entropy and mode entropy
    # and the expected spatial separation, then normalize by total probability to get expected values
    A_state = 0.0

    sum_state = np.zeros_like(b_0)

    for (
        p_prob,
        p_state,
        p_state_entropy,
    ) in partition_contributions:
        if p_occ_total > TOLERANCE and p_state is not None:
            sum_state += p_state * p_prob / p_occ_total
            A_state += p_state_entropy * p_prob / p_occ_total

    state_entropy = calc_entropy(sum_state)
    E_state = max(state_entropy - A_state, 0.0)  # Ensure non-negativity

    return {
        # Legacy entropy is the weighted sum of the partition entropies.
        "entropy": float(legacy_entropy_total),
        "state_entropy": float(state_entropy),
        "mode_entropy": float(mode_entropy_total),
        "spatial_separation": float(spatial_separation_total),
        "belief": sum_state,
        "prob": float(p_occ_total),
        "A_state": float(A_state),
        "E_state": float(E_state),
    }


def _get_reachable_states(hmm: HMM) -> int:
    """
    Calculates the number of states reachable from the initial belief distribution
    under any of the HMM's transition modes. This provides a tighter bound for
    the maximum possible state entropy.
    """
    # Find initial states with non-zero belief
    initial_states = np.where(hmm.state_distribution > 1e-9)[0]
    if not initial_states.size:
        return 0

    # Use BFS to find all reachable states from the initial set
    q = list(initial_states)
    reachable = set(initial_states)

    head = 0
    while head < len(q):
        current_state = q[head]
        head += 1

        # A state can transition according to any mode, so we check all transition matrices
        for mode_idx in range(hmm.num_modes):
            P_mode = hmm.transition_matrices[mode_idx]
            # Find next states with non-zero transition probability from the current state
            next_states = np.where(P_mode[current_state, :] > TOLERANCE)[0]
            for next_state in next_states:
                if next_state not in reachable:
                    reachable.add(next_state)
                    q.append(next_state)

    return len(reachable)


def _combined_entropy(
    trial, k, hmm: HMM, I_s: List[np.ndarray], mode="exact", **kwargs
):
    """
    Compute a weighted combination of legacy partition entropy, mode entropy,
    and spatial separation using OCE-style partitions.

    This implements the partition-based mode estimation discussed previously:
      - For each partition (i, m) denoting "last visible at time i in state m, occluded thereafter",
        we compute the probability of that partition UNDER EACH MODE c (using its transition P_c).
      - Using the prior mode weights, we form the posterior P(c | partition) ∝ w_c*P(partition|c).
      - The *mode* entropy contribution is H( P(c | partition) ), weighted by the marginal
        probability of that partition.
      - The legacy partition entropy contribution is H(b_k | partition), weighted
        by the partition probability.
      - The mixed conditional belief is sum_p P(p | occ) b_k^p, whose entropy is
        reported as "state_entropy". The decomposition uses
        state_entropy = E_state + A_state.

    The final objective is:  alpha * E[H_state] + beta * E[H_mode] + zeta * E[separation],
    where alpha weights the legacy partition entropy, beta is the mode weight,
    and zeta = 1 - alpha - beta is the spatial-separation weight.

    Args:
        trial (int): Trial number for logging.
        k (int): Evaluate at steps 1..k along the provided occlusion schedule.
        hmm (HMM): HMM with per-mode transition matrices and mode/state priors.
        I_s (List[np.ndarray]): List of 1D occlusion masks (1=occluded) for each step index
            (length ≥ k+1).
        mode (str): The mode of entropy calculation ("exact" or "approximate").
        kwargs:
            alpha (float, optional): weight for the state-entropy term in [0, 1].
                If not provided, falls back to 1.0.
            beta (float, optional): weight for the mode-entropy term in [0, 1].
                If not provided, defaults to 0.0.
            zeta is set to 1 - alpha - beta.
            next_sensor_step (int, optional): first future step index at which a
                sensor measurement is available. Must satisfy
                0 <= next_sensor_step < sensing_interval.
            sensing_interval (int, optional): spacing between sensor
                measurements. Must be positive.

    Returns:
        List[dict]: For each step, a dict with keys:
            - "entropy": combined weighted objective,
            - "state_entropy": entropy of the mixed conditional belief,
            - "mode_entropy": expected mode entropy,
            - "spatial_separation": expected spatial separation (if grid info provided),
            - "A_state": within-partition entropy term,
            - "E_state": state-aggregation entropy term,
            - "prob": total occluded probability used (sum of partition masses),
            - "alpha": the provided alpha,
            - "time": wall-clock time.
    """
    # Pull state entropy weight from kwargs or from an argparse-style args object, default 0.5
    alpha = float(kwargs.get("alpha", 1.0))
    beta = float(kwargs.get("beta", 0.0))
    zeta = 1.0 - alpha - beta
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")
    if not (0.0 <= beta <= 1.0):
        raise ValueError("beta must be in [0, 1]")
    if not (0.0 <= zeta <= 1.0):
        raise ValueError("alpha + beta must be in [0, 1]")

    separation_metric = kwargs.get("separation_metric", "spatial")
    if separation_metric not in ("spatial", "jsd"):
        raise ValueError("separation_metric must be 'spatial' or 'jsd'")

    if separation_metric == "spatial":
        grid_height = kwargs.get("grid_height", None)
        grid_width = kwargs.get("grid_width", None)
        if grid_height is None or grid_width is None:
            spatial_separation_fn = None
            max_separation = 1.0
        else:
            spatial_separation_fn = _get_spatial_separation_fn(grid_height, grid_width)
            max_separation = ((grid_height - 1) ** 2 + (grid_width - 1) ** 2) / 4
            max_separation = max(max_separation, TOLERANCE)
    else:
        spatial_separation_fn = None
        max_separation = kwargs.get("max_js_separation", np.sqrt(np.log(2.0)))
        max_separation = max(float(max_separation), TOLERANCE)

    if mode not in ("exact", "approximate"):
        raise ValueError("mode must be 'exact' or 'approximate'")
    if mode == "exact":
        entropy_calc = _combined_partition_exact_calc
    else:
        entropy_calc = _combined_partition_approximate_calc

    tic = time()

    next_sensor_step = kwargs.get("next_sensor_step", 0)
    sensing_interval = kwargs.get("sensing_interval", 1)
    planning_speed = kwargs.get("planning_speed", 1)
    next_sensor_step, sensing_interval, planning_speed = _validate_sensor_schedule(
        next_sensor_step=next_sensor_step,
        sensing_interval=sensing_interval,
        planning_speed=planning_speed,
    )

    # Calculate maximum possible entropies for normalization
    num_reachable_states = _get_reachable_states(hmm)
    max_state_entropy = (
        np.log(num_reachable_states) if num_reachable_states > 1 else 1.0
    )
    # Max mode entropy is log(num_modes) for a uniform distribution.
    max_mode_entropy = np.log(hmm.num_modes) if hmm.num_modes > 1 else 1.0

    # Store occlusion vectors directly (no need for diagonal matrices)
    occ_vectors = [np.asarray(occ, dtype=float) for occ in I_s]

    # Precompute P^p caches for each mode
    P_caches_by_mode = {}
    for c, P_c in enumerate(hmm.transition_matrices):
        P_caches_by_mode[c] = _compute_transition_prefixes(
            P_c, k, planning_speed=planning_speed
        )

    # Precompute promotion under occlusion
    occ_suffixes_by_mode = _compute_occ_suffixes_by_mode(
        occ_vectors,
        hmm.transition_matrices,
        k,
        sensing_interval=sensing_interval,
        next_sensor_step=next_sensor_step,
        planning_speed=planning_speed,
    )

    mode_weights = np.asarray(hmm.mode_distribution, dtype=float)

    b0 = np.asarray(hmm.state_distribution, dtype=float)
    result_name = (
        "Combined (Partition Exact)"
        if mode == "exact"
        else "Combined (Partition Approximate)"
    )

    results = []
    agg_combined = 0.0
    for step in range(1, k + 1):
        entropy_result = entropy_calc(
            step=step,
            b_0=b0,
            occ_vectors=occ_vectors,
            P_caches_by_mode=P_caches_by_mode,
            occ_suffixes_by_mode=occ_suffixes_by_mode,
            mode_weights=mode_weights,
            spatial_separation_fn=spatial_separation_fn,
            separation_metric=separation_metric,
            next_sensor_step=next_sensor_step,
            sensing_interval=sensing_interval,
            planning_speed=planning_speed,
        )

        # Normalize entropies before combining
        norm_state_H = entropy_result["entropy"] / max_state_entropy
        norm_mode_H = entropy_result["mode_entropy"] / max_mode_entropy
        norm_separation = entropy_result["spatial_separation"] / max_separation

        combined = (
            (alpha) * norm_state_H + (beta) * norm_mode_H + (zeta) * norm_separation
        )
        agg_combined += combined
        elapsed = time() - tic

        results.append(
            {
                "trial": trial,
                "step": step,
                "name": result_name,
                "entropy": combined,
                "state_entropy": entropy_result["state_entropy"],
                "mode_entropy": entropy_result["mode_entropy"],
                "spatial_separation": entropy_result["spatial_separation"],
                "belief": entropy_result["belief"],
                "A_state": entropy_result["A_state"],
                "E_state": entropy_result["E_state"],
                "norm_state_entropy": norm_state_H,
                "norm_mode_entropy": norm_mode_H,
                "norm_spatial_separation": norm_separation,
                "separation_metric": separation_metric,
                "prob": entropy_result["prob"],
                "time": elapsed,
                "cumulative_entropy": agg_combined,
                "mean_entropy": agg_combined / step,
            }
        )

    return results


def _mc_noisy_iteration(rng, b0, k, P, I_s, O_vis, **kwargs) -> List[float]:
    """
    Calculate the entropy assuming the target moves from an initial belief over
    k steps.

    Parameters:
        b0 : ndarray
            Initial belief vector (1D, shape [n])
        P : ndarray
            Transition matrix (shape [n, n])
        I_s : list of int
            Occluded state indices
        k : int
            Number of steps to simulate
    Returns:
        List(float) : entropy values at each step
    """
    assert k <= len(
        I_s
    ), "k must be less than or equal to the occlusion schedule length"

    # sample an initial position
    x = rng.choice(len(b0), p=b0)
    entropies = []
    occluded = []

    b = b0.copy()
    for step in range(k):
        x = rng.choice(len(b), p=P[x])  # sample

        b_pred = b @ P

        # sample an observation based on the belief and the observation probabilities
        obs_index = rng.choice(len(b) + 1, p=O_vis[step][x, :])

        # filter the belief based on the observation received
        b = b_pred * O_vis[step][:, obs_index]
        eta = np.sum(b)
        if not eta:
            entropies += [0] * (k - step)
            break
        b = b / eta  # normalize

        occluded.append(obs_index == len(b))  # occ is last state

        # calculate the entropy of the full belief state -- if occluded, entropy grows, if
        # visible, entropy collapses to zero (or the limit of sensor error)
        entropies.append(calc_entropy(b))

    return occluded, entropies


def _mc_entropy(trial, k, P, b, I_s, **kwargs):
    """
    Run Monte Carlo simulation to estimate entropy.

    Parameters:
        b0 : ndarray
            Initial belief vector (1D, shape [n])
        P : ndarray
            Transition matrix (shape [n, n])
        I_s : list of int
            Occluded state indices
        N : int
            Number of steps to simulate
        num_mc_trials : int
            Number of trials to run
        error : float
            Maximum error allowed in the estimate

    Returns:
        List(float) : estimated entropy values at each step
    """

    error = kwargs.get("error", 0.005)
    num_mc_trials = kwargs.get("num_mc_trials", 1000)
    max_time = kwargs.get("max_mc_time", None)
    seed = kwargs.get("seed", None)

    tic = time()

    rng = np.random.default_rng(seed)

    entropy_stats = MonteCarloIntegration(
        delta_abs=error,
        p=0.95,
        min_samples=int(0.1 * num_mc_trials if num_mc_trials else 10),
        model_x=np.zeros(k, dtype=float),
    )
    occluded_stats = MonteCarloIntegration(
        delta_abs=error,
        p=0.95,
        min_samples=int(0.1 * num_mc_trials if num_mc_trials else 10),
        model_x=np.zeros(k, dtype=float),
    )

    # construct the observation probabilities if a noisy observer is used.
    grid_height = kwargs.get("grid_height", None)
    grid_width = kwargs.get("grid_width", None)
    if grid_height is None or grid_width is None:
        raise ValueError(
            "Grid height and width must be specified for noisy naive entropy."
        )
    grid_config = GridConfig(height=grid_height, width=grid_width)

    sensor_config = SensorConfig(  # Populate from kwargs or defaults
        mode=kwargs.get("sensor_mode", "noisy"),
        noise_kernel_size=kwargs.get("noise_kernel_size", 3),
        noise_scale=kwargs.get("noise_scale", 1),
        noise_sigma_max=kwargs.get("noise_sigma_max", 0.3),
        noise_sigma_min=kwargs.get("noise_sigma_min", 0.1),
        noise_sigma_rate=kwargs.get("noise_sigma_rate", 0.01),
    )
    sensor_pos = kwargs.get("sensor_pos", None)
    assert sensor_pos is not None, "Sensor position must be provided for noisy entropy."

    O_vis = _precompute_observation_probabilities_with_variable_noise(
        I_s, sensor_config, grid_config, sensor_pos
    )

    count = 0
    while True:
        iteration_occluded, iteration_entropy = _mc_noisy_iteration(
            rng=rng, k=k, b0=b, P=P, I_s=I_s, O_vis=O_vis, **kwargs
        )
        occluded_stats.update(iteration_occluded)
        entropy_stats.update(iteration_entropy)
        count += 1

        mc_time = time() - tic

        if entropy_stats.stop():
            print(f"Stopping after {count} iterations due to convergence")
            break

        if max_time is not None and mc_time > max_time:
            print(f"Stopping after {count} iterations due to time limit")
            break

        if num_mc_trials is not None and count >= num_mc_trials:
            print(f"Stopping after {count} iterations due to iteration limit")
            break

    result = []
    entropy = entropy_stats.mean()
    occluded = occluded_stats.mean()

    for step in range(k):

        prob = occluded[step]
        result.append(
            {
                "trial": trial,
                "step": step + 1,
                "name": "Monte Carlo Entropy (Noisy)",
                "entropy": entropy[step],
                "mean_entropy": entropy[: step + 1].mean(),
                "cumulative_entropy": entropy[: step + 1].sum(),
                "prob": prob,
                "time": mc_time,
            }
        )

    return result


def _noisy_naive(trial, k, P, b, I_s, **kwargs):
    """
    Estimate the entropy for a noisy observer using a naive belief propagation
    but incorporating the noisy observation model for entropy calculation at each step.

    Args:
        trial (int): The trial number.
        k (int): The number of steps to evaluate.
        P (Matrix): The transition matrix.
        b (Matrix): The initial belief vector.
        I_s (np.array): A list of 1D arrays, each defining the occluded states (1=occluded)
                        at each step.

    Returns:
        A list of dictionaries containing the estimated entropy and calculation time.
    """
    results = []

    overall_tic = time()  # Time for the whole k-step calculation

    grid_height = kwargs.get("grid_height", None)
    grid_width = kwargs.get("grid_width", None)
    if grid_height is None or grid_width is None:
        raise ValueError(
            "Grid height and width must be specified for noisy naive entropy."
        )
    grid_config = GridConfig(height=grid_height, width=grid_width)

    sensor_config = SensorConfig(  # Populate from kwargs or defaults
        mode=kwargs.get("sensor_mode", "noisy"),
        noise_kernel_size=kwargs.get("noise_kernel_size", 3),
        noise_scale=kwargs.get("noise_scale", 1),
        noise_sigma_max=kwargs.get("noise_sigma_max", 0.3),
        noise_sigma_min=kwargs.get("noise_sigma_min", 0.1),
        noise_sigma_rate=kwargs.get("noise_sigma_rate", 0.01),
    )
    sensor_pos = kwargs.get("sensor_pos", None)
    assert sensor_pos is not None, "Sensor position must be provided for noisy entropy."

    O_vis = _precompute_observation_probabilities_with_variable_noise(
        I_s, sensor_config, grid_config, sensor_pos
    )

    entropies = []
    probabilities = []
    current_b_for_pred = b.copy()  # This will be b @ P^s

    num_possible_observations = len(I_s[0]) + 1  # M states + 1 occluded type

    for step in range(k):
        # Predict belief for the current step s_calc: b_pred = b @ P^(s_calc+1)
        if step == 0:
            b_pred = b @ P
        else:
            # current_b_for_pred is b @ P^s_calc from previous iteration
            b_pred = current_b_for_pred @ P

        expected_H = 0

        # Calculate P(Y_s = occluded_obs_type | naive_belief_s)
        # O_vis[step] is P(Y_obs | X_true) for current step
        # b_pred is P(X_true | naive_belief_s)
        # M (num_states) is the index for occluded observation type
        prob_occluded_observation = np.sum(b_pred * O_vis[step][:, -1])
        probabilities.append(prob_occluded_observation)

        for obs_type_idx in range(num_possible_observations):
            likelihood_vec = O_vis[step][
                :, obs_type_idx
            ]  # P(obs_type | X_true) for current step
            joint_unnorm = b_pred * likelihood_vec
            prob_obs_y = np.sum(joint_unnorm)  # P(obs_type | Y_1:s_calc-1)

            if np.isclose(prob_obs_y, 0):
                continue

            b_post_y = joint_unnorm / prob_obs_y  # P(X_true | Y_1:s_calc-1, obs_type)

            # Mask by true map occlusion for entropy calculation (I_s[s_calc] is 1 if occluded)
            b_occ_post_y = b_post_y * I_s[step]
            prob_occ_post_y = np.sum(b_occ_post_y)
            if np.isclose(prob_occ_post_y, 0):
                continue

            expected_H += prob_obs_y * calc_entropy(b_occ_post_y / prob_occ_post_y)

        entropies.append(expected_H)
        current_b_for_pred = b_pred  # Update for next iteration's prediction base

    total_calculation_time = time() - overall_tic

    aggregate_entropy = 0
    for step, entropy in enumerate(entropies):
        aggregate_entropy += entropy
        results.append(
            {
                "trial": trial,
                "step": step + 1,
                "name": "Naive Entropy (Noisy)",
                "entropy": entropy,
                "mean_entropy": aggregate_entropy / (step + 1),
                "cumulative_entropy": aggregate_entropy,
                "prob": probabilities[step],
                "time": total_calculation_time,
            }
        )
    return results


def _mc_noisy_mode_iteration(rng, hmm, k, I_s, O_vis) -> List[float]:
    """
    Calculate the entropy assuming the target moves from an initial belief over
    k steps.

    Parameters:
        b0 : ndarray
            Initial belief vector (1D, shape [n])
        P : ndarray
            Transition matrix (shape [n, n])
        I_s : list of int
            Occluded state indices
        k : int
            Number of steps to simulate
    Returns:
        List(float) : entropy values at each step
    """
    assert k <= len(
        I_s
    ), "k must be less than or equal to the occlusion schedule length"

    # sample the mode
    mode = rng.choice(hmm.num_modes, p=hmm.mode_distribution)

    # sample an initial position
    x = rng.choice(hmm.num_states, p=hmm.state_distribution)

    occluded = []
    mode_entropies = []
    for step in range(k):
        x = rng.choice(
            hmm.num_states, p=hmm.transition_matrices[mode][x]
        )  # sample next state

        # find the observation
        observation = rng.choice(hmm.num_observations, p=O_vis[step][x, :])

        # forward step the belief estimates
        hmm.forward_step(observation=observation, emission_probabilities=O_vis[step])

        occluded.append(float(observation == hmm.num_observations - 1))
        mode_entropies.append(calc_entropy(hmm.mode_distribution))

    return occluded, mode_entropies


def _noisy_mode_mc(trial: int, k: int, hmm: HMM, I_s: List[np.ndarray], **kwargs):
    """
    Run Monte Carlo simulation to estimate entropy.

    Parameters:
        b0 : ndarray
            Initial belief vector (1D, shape [n])
        P : ndarray
            Transition matrix (shape [n, n])
        I_s : list of int
            Occluded state indices
        N : int
            Number of steps to simulate
        num_mc_trials : int
            Number of trials to run
        error : float
            Maximum error allowed in the estimate

    Returns:
        List(float) : estimated entropy values at each step
    """

    error = kwargs.get("error", 0.005)
    num_mc_trials = kwargs.get("num_mc_trials", 1000)
    seed = kwargs.get("seed", None)
    confidence_threshold = kwargs.get("conf_requested")

    entropy_threshold = -confidence_threshold * np.log(confidence_threshold) - (
        1 - confidence_threshold
    ) * np.log((1 - confidence_threshold) / (hmm.num_modes - 1))
    tic = time()

    rng = np.random.default_rng(seed)

    occluded_stats = MonteCarloIntegration(
        delta_abs=error,
        p=0.95,
        min_samples=int(0.1 * num_mc_trials if num_mc_trials else 10),
        model_x=np.zeros(k, dtype=float),
    )
    mode_entropy_stats = MonteCarloIntegration(
        delta_abs=error,
        p=0.95,
        min_samples=int(0.1 * num_mc_trials if num_mc_trials else 10),
        model_x=np.zeros(k, dtype=float),
    )

    # construct the observation probabilities
    grid_height = kwargs.get("grid_height", None)
    grid_width = kwargs.get("grid_width", None)
    if grid_height is None or grid_width is None:
        raise ValueError(
            "Grid height and width must be specified for noisy naive entropy."
        )
    grid_config = GridConfig(height=grid_height, width=grid_width)

    sensor_config = SensorConfig(  # Populate from kwargs or defaults
        mode=kwargs.get("sensor_mode", "noisy"),
        noise_kernel_size=kwargs.get("noise_kernel_size", 3),
        noise_scale=kwargs.get("noise_scale", 1),
        noise_sigma_max=kwargs.get("noise_sigma_max", 0.3),
        noise_sigma_min=kwargs.get("noise_sigma_min", 0.1),
        noise_sigma_rate=kwargs.get("noise_sigma_rate", 0.01),
    )
    sensor_pos = kwargs.get("sensor_pos", None)
    assert sensor_pos is not None, "Sensor position must be provided for noisy entropy."

    O_vis = _precompute_observation_probabilities_with_variable_noise(
        I_s, sensor_config, grid_config, sensor_pos
    )

    predict_hmm = HMM(
        num_states=hmm.num_states,
        num_observations=hmm.num_observations,
        num_modes=hmm.num_modes,
        transitions=hmm.transition_matrices,
        emission_probabilities=None,
        distributions={
            "state": hmm.state_distribution.copy(),
            "mode": hmm.mode_distribution.copy(),
        },
    )

    count = 0
    while True:
        predict_hmm.reset()

        occluded, mode_entropies = _mc_noisy_mode_iteration(
            rng=rng, k=k, hmm=predict_hmm, I_s=I_s, O_vis=O_vis
        )

        if np.any(mode_entropies < entropy_threshold):
            # only record the successful identifications
            mode_entropy_stats.update(mode_entropies)

        occluded_stats.update(occluded)

        count += 1
        if (num_mc_trials is not None and count >= num_mc_trials) or (
            mode_entropy_stats.stop() and occluded_stats.stop()
        ):
            print(f"Stopping after {count} iterations")
            break

    mc_time = time() - tic

    result = []
    entropy = mode_entropy_stats.mean()
    occluded = occluded_stats.mean()
    prob_success = mode_entropy_stats.n() / count

    aggregate_entropy = 0
    for step in range(k):
        aggregate_entropy += entropy[step]
        result.append(
            {
                "trial": trial,
                "step": step + 1,
                "name": "Noisy Mode MC",
                "entropy": entropy,
                "mean_entropy": aggregate_entropy / (step + 1),
                "cumulative_entropy": aggregate_entropy,
                "prob": prob_success,
                "time": mc_time,
            }
        )

    return result


def _mc_noisy_steps_iteration(
    rng,
    hmm: HMM,
    k_max_steps: int,
    I_s: List[np.ndarray],
    O_vis: List[np.ndarray],
    confidence_threshold: float,
) -> int:
    """
    Run a single Monte Carlo iteration to find steps to confidence.
    k_max_steps is the maximum number of simulation steps (0-indexed internally).
    Returns (step_count_1_indexed) or (k_max_steps + 1) if failed.
    """
    hmm.reset()  # Reset HMM to its initial state and mode beliefs for this iteration

    # Sample the true target mode based on the HMM's initial mode distribution
    true_mode = rng.choice(hmm.num_modes, p=hmm.initial_mode_distribution)
    # Sample the initial true state of the target based on HMM's initial state distribution
    current_true_state = rng.choice(hmm.num_states, p=hmm.initial_state_distribution)

    for step_idx in range(k_max_steps):  # step_idx from 0 to k_max_steps-1
        # Simulate true target movement for one step
        current_true_state = rng.choice(
            hmm.num_states, p=hmm.transition_matrices[true_mode, current_true_state, :]
        )

        # Simulate observation based on the new true state and precomputed O_vis for this step
        # O_vis[step_idx] is P(Obs | TrueState) for step_idx along the agent's path
        observation = rng.choice(
            hmm.num_observations, p=O_vis[step_idx][current_true_state, :]
        )

        # Update HMM belief based on the observation
        hmm.forward_step(
            observation=observation, emission_probabilities=O_vis[step_idx]
        )

        # Check if confidence threshold is met for any mode
        if np.max(hmm.mode_distribution) > confidence_threshold:
            return step_idx + 1  # Return 1-indexed number of steps

    # If loop completes, confidence was not reached within k_max_steps
    return k_max_steps + 1


def _noisy_steps_mc(trial: int, k: int, hmm: HMM, I_s: List[np.ndarray], **kwargs):
    """
    Run Monte Carlo simulation to estimate the average number of steps to reach
    a desired confidence level about the target's mode.

    Args:
        trial (int): Trial number for logging.
        k (int): The maximum number of steps to simulate internally for each MC trial.
                 This is the horizon along the agent's path segment.
        hmm (HMM): The HMM object, pre-initialized with P(X_0) and P(M_0) for the
                   start of this path segment.
        I_s (List[np.ndarray]): List of occlusion masks for each step along the path.
        **kwargs: Must include 'conf_requested' (float for confidence threshold),
                  and can include 'num_mc_trials', 'seed', 'error' for MonteCarloIntegration,
                  sensor and grid configuration.

    Returns:
        List[dict]: A list of dictionaries, one for each step up to k, containing
                    the estimated average steps and success probability.
    """
    if k == 0:  # No simulation steps to take along the path
        confidence_threshold = kwargs.get("conf_requested", 0.95)
        current_confidence = np.max(hmm.mode_distribution)  # Check current HMM state
        already_confident = current_confidence > confidence_threshold
        est_s = 0.0 if already_confident else float("inf")
        prob_s = 1.0 if already_confident else 0.0
        return [
            {
                "trial": trial,
                "step": 1,  # Reporting for step 1 (or 0) of a 0-step path
                "name": "Noisy Steps MC",
                "average_steps": est_s,  # Using entropy field for the main metric
                "prob": prob_s,
                "time": 0,
                "est_steps": est_s,
            }
        ]

    error = kwargs.get("error", 0.1)  # Error TOLERANCE for avg steps
    num_mc_trials = kwargs.get("num_mc_trials", 2000)
    seed = kwargs.get("seed", None)
    confidence_threshold = kwargs.get("conf_requested")
    if confidence_threshold is None:
        raise ValueError("conf_requested must be specified for _noisy_steps_mc")

    tic = time()
    rng = np.random.default_rng(seed)

    steps_mc_integrator = MonteCarloIntegration(
        delta_abs=error,
        p=0.95,
        min_samples=max(2, int(0.01 * num_mc_trials)),
        max_samples=num_mc_trials,
        model_x=0.0,
    )

    grid_height = kwargs.get("grid_height")
    grid_width = kwargs.get("grid_width")
    if grid_height is None or grid_width is None:
        raise ValueError("Grid height and width must be specified.")
    grid_config = GridConfig(height=grid_height, width=grid_width)

    sensor_config = SensorConfig(  # Populate from kwargs or defaults
        mode=kwargs.get("sensor_mode", "noisy"),
        noise_kernel_size=kwargs.get("noise_kernel_size", 3),
        noise_scale=kwargs.get("noise_scale", 1),
        noise_sigma_max=kwargs.get("noise_sigma_max", 0.3),
        noise_sigma_min=kwargs.get("noise_sigma_min", 0.1),
        noise_sigma_rate=kwargs.get("noise_sigma_rate", 0.01),
    )
    sensor_pos = kwargs.get("sensor_pos", None)
    assert sensor_pos is not None, "Sensor position must be provided for noisy entropy."

    # O_vis = _precompute_observation_probabilities(k, I_s, sensor_config, grid_config)
    O_vis = _precompute_observation_probabilities_with_variable_noise(
        I_s, sensor_config, grid_config, sensor_pos
    )

    # This HMM instance will be used and reset for each MC iteration
    iter_hmm = HMM(
        num_states=hmm.num_states,
        num_observations=hmm.num_observations,
        num_modes=hmm.num_modes,
        transitions=hmm.transition_matrices,
        emission_probabilities=None,  # Will use O_vis[step]
        distributions={  # Crucially, use the initial beliefs from the passed 'hmm'
            "state": hmm.initial_state_distribution.copy(),
            "mode": hmm.initial_mode_distribution.copy(),
        },
    )

    attempts = 0
    while True:
        steps_this_iter = _mc_noisy_steps_iteration(
            rng, iter_hmm, k, I_s, O_vis, confidence_threshold
        )
        if steps_this_iter <= k:
            # only record the successful identifications
            steps_mc_integrator.update(float(steps_this_iter))

        attempts += 1
        if (attempts >= num_mc_trials) or steps_mc_integrator.stop():
            print(f"NoisyStepsMC: Stopping early after {attempts} iterations.")
            break

    mc_time = time() - tic
    avg_steps = (
        steps_mc_integrator.mean() if steps_mc_integrator.n() > 0 else float("inf")
    )
    success_rate = steps_mc_integrator.n() / attempts

    results = [
        {
            "trial": trial,
            "step": k,
            "name": "Noisy Steps MC",
            "average_steps": avg_steps,
            "prob": success_rate,
            "time": mc_time,
            "est_steps": avg_steps,
        },
    ]
    return results


def _noisy_occlusion_probability(trial, k, hmm: HMM, I_s: List[np.ndarray], **kwargs):
    """
    Estimate the visibility of a target for a noisy observer using a naive belief propagation
    and incorporating the noisy observation model.

    Args:
        trial (int): The trial number.
        k (int): The number of steps to evaluate.
        P (Matrix): The transition matrix.
        b (Matrix): The initial belief vector.
        I_s (np.array): A list of 1D arrays, each defining the occluded states (1=occluded)
                        at each step.

    Returns:
        A list of dictionaries containing the estimated entropy and calculation time.
    """
    results = []

    grid_height = kwargs.get("grid_height", None)
    grid_width = kwargs.get("grid_width", None)
    if grid_height is None or grid_width is None:
        raise ValueError(
            "Grid height and width must be specified for noisy naive entropy."
        )
    grid_config = GridConfig(height=grid_height, width=grid_width)

    sensor_config = SensorConfig(  # Populate from kwargs or defaults
        mode=kwargs.get("sensor_mode", "noisy"),
        noise_kernel_size=kwargs.get("noise_kernel_size", 3),
        noise_scale=kwargs.get("noise_scale", 1),
        noise_sigma_max=kwargs.get("noise_sigma_max", 0.3),
        noise_sigma_min=kwargs.get("noise_sigma_min", 0.1),
        noise_sigma_rate=kwargs.get("noise_sigma_rate", 0.01),
    )
    sensor_pos = kwargs.get("sensor_pos", None)
    assert sensor_pos is not None, "Sensor position must be provided for noisy entropy."

    O_vis = _precompute_observation_probabilities_with_variable_noise(
        I_s, sensor_config, grid_config, sensor_pos
    )

    overall_tic = time()  # Time for the whole k-step calculation

    probabilities = [0 for _ in range(k)]  # Initialize probabilities list
    next_sensor_step = kwargs.get("next_sensor_step", 0)
    sensing_interval = kwargs.get("sensing_interval", 1)
    planning_speed = kwargs.get("planning_speed", 1)
    next_sensor_step, sensing_interval, planning_speed = _validate_sensor_schedule(
        next_sensor_step=next_sensor_step,
        sensing_interval=sensing_interval,
        planning_speed=planning_speed,
    )

    P_mix = np.zeros_like(hmm.transition_matrices[0])
    for P, mode_prob in zip(hmm.transition_matrices, hmm.mode_distribution):
        P_mix += mode_prob * P

    prefixes = _compute_transition_prefixes(P_mix, k, planning_speed=planning_speed)
    for step in range(1, k + 1):
        b = hmm.state_distribution @ prefixes[step]

        # Calculate P(Y_s = occluded_obs_type | naive_belief_s)
        # O_vis[step] is P(Y_obs | X_true) for current step
        # b_pred is P(X_true | naive_belief_s)
        # M (num_states) is the index for occluded observation type

        if _sensor_active_at_step(
            step,
            next_sensor_step=next_sensor_step,
            sensing_interval=sensing_interval,
            planning_speed=planning_speed,
        ):
            # since all of the other functions are minimizing, we need to do the same here.
            # We want to maximize visibility, so we return the probability of occlusion.
            probabilities[step - 1] = np.sum(
                b * O_vis[step][:, -1]
            )  # Last column is occluded obs
        else:
            probabilities[step - 1] = 0.0

    total_calculation_time = time() - overall_tic

    mean_occlusion_probability = 0
    for step, prob in enumerate(probabilities):
        mean_occlusion_probability += prob
        results.append(
            {
                "trial": trial,
                "step": step + 1,
                "name": "Visibility (Noisy)",
                "occlusion_probability": mean_occlusion_probability / (step + 1),
                "prob": 0,
                "time": total_calculation_time,
            }
        )
    return results


def _occlusion_probability(trial, k, hmm: HMM, I_s: List[np.ndarray], **kwargs):
    """
    Estimate the visibility of a target for a noisy observer using a naive belief propagation
    and incorporating the noisy observation model.

    Args:
        trial (int): The trial number.
        k (int): The number of steps to evaluate.
        P (Matrix): The transition matrix.
        b (Matrix): The initial belief vector.
        I_s (np.array): A list of 1D arrays, each defining the occluded states (1=occluded)
                        at each step.

    Returns:
        A list of dictionaries containing the estimated entropy and calculation time.
    """
    results = []

    overall_tic = time()  # Time for the whole k-step calculation

    probabilities = [0 for _ in range(k)]  # Initialize probabilities list

    next_sensor_step = kwargs.get("next_sensor_step", 0)
    sensing_interval = kwargs.get("sensing_interval", 1)
    planning_speed = kwargs.get("planning_speed", 1)
    next_sensor_step, sensing_interval, planning_speed = _validate_sensor_schedule(
        next_sensor_step=next_sensor_step,
        sensing_interval=sensing_interval,
        planning_speed=planning_speed,
    )

    P_mix = np.zeros_like(hmm.transition_matrices[0])
    for P, mode_prob in zip(hmm.transition_matrices, hmm.mode_distribution):
        P_mix += mode_prob * P

    prefixes = _compute_transition_prefixes(P_mix, k, planning_speed=planning_speed)
    for step in range(1, k + 1):
        b = hmm.state_distribution @ prefixes[step]

        if _sensor_active_at_step(
            step,
            next_sensor_step=next_sensor_step,
            sensing_interval=sensing_interval,
            planning_speed=planning_speed,
        ):
            # probability of occlusion only when a sensor measurement occurs
            probabilities[step - 1] = np.sum(b * I_s[step])
        else:
            probabilities[step - 1] = 0.0

    total_calculation_time = time() - overall_tic

    mean_occlusion_probability = 0
    for step, prob in enumerate(probabilities):
        mean_occlusion_probability += prob
        results.append(
            {
                "trial": trial,
                "step": step + 1,
                "name": "Visibility (Noisy)",
                "occlusion_probability": mean_occlusion_probability / (step + 1),
                "prob": 0,
                "time": total_calculation_time,
            }
        )
    return results


def calc_expected_information_reward(
    trial, k, hmm: HMM, I_s: List[np.ndarray], reward_type="L2"
):
    """
    Calculate the expected information reward for a given HMM and occlusion schedule.

    Args:
        trial (int): The trial number.
        k (int): The number of steps to evaluate.
        hmm (HMM): The HMM object with transition matrices and initial distributions.
        I_s (List[np.ndarray]): A list of occlusion masks for each step along the path.
        I_0 (np.ndarray): The initial occlusion mask for the current step.
        reward_type (str): The type of reward to calculate, either "L2" or "KL".

        One caveat: I_s starts with the first step of the trajectory, so we cannot
        calculate the expected information reward for step 0.

    Returns:
        Dictionary containing the expected information reward and the probability of success for
        every step of the trajectory.
    """

    tic = time()

    b = hmm.state_distribution.copy()
    P = hmm.transition_matrices[0]

    result = []
    if reward_type == "L2":
        reward_matrix = hmm.l2_differences
    elif reward_type == "KL":
        reward_matrix = hmm.kl_divergences
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")

    I_step = 1 - I_s[0]  # Initial visible mask for the first step

    cumulative_reward = 0.0
    for step in range(1, k):
        b_vis = b * I_step  # Use the old mask to check the starting point

        I_step = 1 - I_s[step]  # Update visible mask for the current step
        b_next = b @ P
        b_vis_next = b_next * I_step  # Use the updated mask to check the end point

        # Calculate the expected information reward
        expected_reward = 0.0
        relevant_states = np.where(b_vis > TOLERANCE)[0]
        for s in relevant_states:
            raw_sum = np.sum(b_vis_next * reward_matrix[s])
            if raw_sum > TOLERANCE:
                expected_reward += b_vis[s] * np.sqrt(raw_sum)

        b = b_next

        EIR_time = time() - tic
        cumulative_reward += expected_reward
        result.append(
            {
                "trial": trial,
                "step": step + 1,
                "name": "Expected Information",
                "reward": expected_reward,
                "cumulative_reward": expected_reward,  # cumulative reward is same as expected
                "average_reward": expected_reward / (step + 1),
                "prob": 0,
                "time": EIR_time,
            }
        )

    return result
