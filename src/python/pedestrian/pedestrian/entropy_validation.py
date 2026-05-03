from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from scipy import sparse
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from datasources import load_scenario
from entropy import calc_entropy
from hmm import HMM

DEFAULT_SCENE_ID = 7
DEFAULT_DESTINATION_CLASS = 5
DEFAULT_TRACK_ID = 13
DEFAULT_HORIZON = 30
DEFAULT_INTERVAL = 5
DEFAULT_ROLLOUT_STEPS = 180
DEFAULT_ROBOT_START = (7.15, 1.8)
DEFAULT_ROBOT_GOAL = (0.35, 2.45)
DEFAULT_ROBOT_HEADING = float(np.pi)
DEFAULT_OUT_DIR = "outputs/entropy_validation/scene_007_track_013"
DEFAULT_SCAN_RANGE = 10.0
DEFAULT_RTOL = 1.0e-4
DEFAULT_ATOL = 1.0e-5
TOLERANCE = 1.0e-10
ROBOT_MARKER_COLOUR = "#00d5ff"
ROBOT_MARKER_EDGE = "#ffffff"


def load_gpu_backend():
    try:
        from warp_mppi.legacy.discrete_oce_pycuda import (
            PYCUDA_AVAILABLE,
            evaluate_discrete_oce_gpu,
        )
    except Exception as exc:  # pragma: no cover - depends on local CUDA install
        raise RuntimeError(
            f"Unable to import the discrete OCE GPU backend: {exc}"
        ) from exc
    if not PYCUDA_AVAILABLE:
        raise RuntimeError(
            "PyCUDA is not available for required GPU entropy validation"
        )
    return evaluate_discrete_oce_gpu


def pycuda_available() -> bool:
    try:
        load_gpu_backend()
    except RuntimeError:
        return False
    return True


@dataclass
class SDDModelBundle:
    scene_id: int
    scene_root: Path
    state_space: dict[str, np.ndarray]
    state_metadata: dict
    destination_classes: dict
    class_ids: list[int]
    class_priors: np.ndarray
    class_transitions: list[sparse.csr_matrix]
    mixed_transition: sparse.csr_matrix
    state_centers_scene: np.ndarray
    state_centers_sim: np.ndarray
    state_grid_indices: np.ndarray
    static_grid: np.ndarray
    cell_size: float
    bounds: dict


def scene_to_sim_display(points: np.ndarray, bounds: dict) -> np.ndarray:
    display_points = np.asarray(points, dtype=float).copy()
    display_points[:, 1] = bounds["min_y"] + bounds["max_y"] - display_points[:, 1]
    return display_points


def load_sdd_model_bundle(model_root: str | Path, scene_id: int) -> SDDModelBundle:
    scene_root = Path(model_root) / f"scene_{int(scene_id):03d}"
    state_npz_path = scene_root / "state_space.npz"
    state_json_path = scene_root / "state_space.json"
    destination_path = scene_root / "destination_classes.json"
    if not state_npz_path.exists():
        raise FileNotFoundError(f"Missing state space: {state_npz_path}")
    if not state_json_path.exists():
        raise FileNotFoundError(f"Missing state metadata: {state_json_path}")
    if not destination_path.exists():
        raise FileNotFoundError(f"Missing destination classes: {destination_path}")

    with np.load(state_npz_path) as state_data:
        state_space = {key: state_data[key] for key in state_data.files}
    state_metadata = json.loads(state_json_path.read_text(encoding="utf-8"))
    destination_classes = json.loads(destination_path.read_text(encoding="utf-8"))

    class_ids = [int(record["class_id"]) for record in destination_classes["classes"]]
    train_counts = np.asarray(
        [
            float(record.get("train_count", 0.0))
            for record in destination_classes["classes"]
        ],
        dtype=float,
    )
    if float(train_counts.sum()) <= 0.0:
        class_priors = np.full(len(class_ids), 1.0 / len(class_ids), dtype=float)
    else:
        class_priors = train_counts / float(train_counts.sum())

    transitions = []
    for class_id in class_ids:
        transition_path = (
            scene_root / "transitions" / f"class_{class_id:03d}_transition.npz"
        )
        if not transition_path.exists():
            raise FileNotFoundError(
                f"Missing class transition matrix: {transition_path}"
            )
        transitions.append(sparse.load_npz(transition_path).tocsr())

    mixed = None
    for weight, transition in zip(class_priors, transitions):
        weighted = transition.multiply(float(weight))
        mixed = weighted if mixed is None else mixed + weighted
    if mixed is None:
        raise ValueError(f"Scene {scene_id:03d} has no class transition matrices")
    mixed = mixed.tocsr()
    mixed.sum_duplicates()
    mixed.eliminate_zeros()

    bounds = state_metadata["bounds"]
    centers_scene = np.asarray(state_space["centers"], dtype=float)
    centers_sim = scene_to_sim_display(centers_scene, bounds)
    walkable_mask = np.asarray(state_space["walkable_mask"], dtype=bool)
    static_grid = np.asarray(np.flipud(~walkable_mask), dtype=np.uint8)

    return SDDModelBundle(
        scene_id=int(scene_id),
        scene_root=scene_root,
        state_space=state_space,
        state_metadata=state_metadata,
        destination_classes=destination_classes,
        class_ids=class_ids,
        class_priors=class_priors,
        class_transitions=transitions,
        mixed_transition=mixed,
        state_centers_scene=centers_scene,
        state_centers_sim=centers_sim,
        state_grid_indices=np.asarray(state_space["grid_indices"], dtype=int),
        static_grid=static_grid,
        cell_size=float(state_metadata["cell_size"]),
        bounds=bounds,
    )


def validate_track_class(
    bundle: SDDModelBundle, track_id: int, destination_class: int
) -> None:
    track_to_class = bundle.destination_classes.get("track_to_class", {})
    actual = int(track_to_class.get(str(int(track_id)), -1))
    if actual != int(destination_class):
        raise ValueError(
            f"Track {int(track_id)} belongs to destination class {actual}, "
            f"not {int(destination_class)}"
        )


def load_track_state_sequence(
    model_root: str | Path, scene_id: int, track_id: int
) -> np.ndarray:
    path = Path(model_root) / f"scene_{int(scene_id):03d}" / "trajectory_states.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing trajectory states: {path}")
    key = f"track_{int(track_id)}"
    with np.load(path) as trajectory_data:
        if key not in trajectory_data.files:
            raise KeyError(f"Missing {key} in {path}")
        states = np.asarray(trajectory_data[key], dtype=np.int64)
    if states.size == 0:
        raise ValueError(f"Track {int(track_id)} has no mapped states")
    if np.any(states < 0):
        raise ValueError(f"Track {int(track_id)} contains unmapped states")
    return states


def blocking_union(static_polygons) -> Polygon | None:
    polygons = []
    for static_polygon in static_polygons:
        if not getattr(static_polygon, "blocking", True):
            continue
        points = np.asarray(static_polygon.points, dtype=float)
        if points.ndim != 2 or points.shape[0] < 3:
            continue
        polygon = Polygon(points)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_empty:
            polygons.append(polygon)
    if not polygons:
        return None
    return unary_union(polygons)


def point_is_free(point, static_polygons) -> bool:
    candidate = Point(float(point[0]), float(point[1]))
    for static_polygon in static_polygons:
        if not getattr(static_polygon, "blocking", True):
            continue
        polygon = Polygon(np.asarray(static_polygon.points, dtype=float))
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_empty and polygon.covers(candidate):
            return False
    return True


def linear_robot_path(start, goal, steps: int) -> np.ndarray:
    steps = int(steps)
    if steps <= 0:
        raise ValueError("steps must be positive")
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    if steps == 1:
        return start.reshape(1, 2)
    alpha = np.linspace(0.0, 1.0, steps, dtype=float).reshape(-1, 1)
    return start.reshape(1, 2) + (goal - start).reshape(1, 2) * alpha


def shapely_occlusion_schedule(
    robot_path: np.ndarray,
    state_centers: np.ndarray,
    static_union,
    scan_range: float,
) -> list[np.ndarray]:
    schedule = []
    for robot in np.asarray(robot_path, dtype=float):
        deltas = state_centers - robot.reshape(1, 2)
        occluded = np.linalg.norm(deltas, axis=1) > float(scan_range)
        if static_union is not None and not static_union.is_empty:
            candidates = np.where(~occluded)[0]
            for state_idx in candidates:
                if LineString([robot, state_centers[state_idx]]).intersects(
                    static_union
                ):
                    occluded[state_idx] = True
        schedule.append(occluded.astype(float))
    return schedule


def target_visibility_over_rollout(
    robot_path: np.ndarray,
    target_states: np.ndarray,
    state_centers: np.ndarray,
    static_union,
    scan_range: float,
) -> list[bool]:
    visibility = []
    for step, robot in enumerate(np.asarray(robot_path, dtype=float)):
        state = int(target_states[min(step, len(target_states) - 1)])
        target = state_centers[state]
        visible = np.linalg.norm(target - robot) <= float(scan_range)
        if visible and static_union is not None and not static_union.is_empty:
            visible = not LineString([robot, target]).intersects(static_union)
        visibility.append(bool(visible))
    return visibility


def build_hmm(bundle: SDDModelBundle, initial_state: int) -> HMM:
    state_distribution = np.zeros(bundle.state_centers_sim.shape[0], dtype=float)
    state_distribution[int(initial_state)] = 1.0
    dense_transitions = np.stack(
        [transition.toarray() for transition in bundle.class_transitions],
        axis=0,
    )
    return HMM(
        num_states=state_distribution.size,
        num_observations=state_distribution.size,
        num_modes=len(bundle.class_ids),
        transitions=dense_transitions,
        emission_probabilities=np.eye(state_distribution.size, dtype=float),
        distributions={
            "state": state_distribution,
            "mode": bundle.class_priors,
        },
        precompute_diagnostics=False,
        copy_transitions=False,
    )


def sparse_prefix_beliefs(
    initial_belief: np.ndarray,
    transition: sparse.csr_matrix,
    horizon: int,
) -> np.ndarray:
    prefix = np.zeros((int(horizon) + 1, initial_belief.size), dtype=np.float32)
    prefix[0] = np.asarray(initial_belief, dtype=np.float32)
    for step in range(1, int(horizon) + 1):
        prefix[step] = prefix[step - 1] @ transition
    return prefix


def sparse_occ_suffixes(
    occlusion_schedule: list[np.ndarray],
    transition: sparse.csr_matrix,
    horizon: int,
) -> dict[int, dict[int, sparse.csr_matrix]]:
    total_steps = min(int(horizon), len(occlusion_schedule) - 1)
    diag_products = {}
    for step in range(1, total_steps + 1):
        occ_diag = sparse.diags(occlusion_schedule[step], format="csr")
        diag_products[step] = transition @ occ_diag

    suffixes = {}
    for step in range(1, total_steps + 1):
        running = None
        suffix = {}
        for idx in range(step, 0, -1):
            running = (
                diag_products[idx] if running is None else diag_products[idx] @ running
            )
            suffix[idx] = running.tocsr()
        suffixes[step] = suffix
    return suffixes


def row_as_array(matrix, row_idx: int) -> np.ndarray:
    return np.asarray(matrix.getrow(int(row_idx)).toarray()).reshape(-1)


def sparse_exact_entropy(
    initial_belief: np.ndarray,
    transition: sparse.csr_matrix,
    occlusion_schedule: list[np.ndarray],
    horizon: int,
) -> list[dict]:
    horizon = min(int(horizon), len(occlusion_schedule) - 1)
    prefix = sparse_prefix_beliefs(initial_belief, transition, horizon).astype(float)
    suffixes = sparse_occ_suffixes(occlusion_schedule, transition, horizon)

    results = []
    cumulative_entropy = 0.0
    for step in range(1, horizon + 1):
        beliefs = []
        probs = []

        final_belief = np.asarray(initial_belief @ suffixes[step][1]).reshape(-1)
        final_prob = float(final_belief.sum())
        if final_prob > TOLERANCE:
            beliefs.append(final_belief)
            probs.append(final_prob)

        for partition_step in range(1, step):
            b_partition = prefix[partition_step]
            visible_states = np.where(
                (np.asarray(occlusion_schedule[partition_step]) < TOLERANCE)
                & (b_partition > TOLERANCE)
            )[0]
            suffix = suffixes[step].get(partition_step + 1)
            if suffix is None:
                continue
            for state in visible_states:
                state_prob = float(b_partition[state])
                state_suffix = row_as_array(suffix, int(state))
                final_belief = state_prob * state_suffix
                final_prob = float(final_belief.sum())
                if final_prob > TOLERANCE:
                    beliefs.append(final_belief)
                    probs.append(final_prob)

        if not beliefs:
            result = {
                "step": step,
                "entropy": 0.0,
                "state_entropy": 0.0,
                "prob": 0.0,
                "E_state": 0.0,
                "A_state": 0.0,
                "belief": np.zeros_like(initial_belief, dtype=float),
            }
        else:
            total_prob = float(np.sum(probs))
            legacy_entropy = 0.0
            a_state = 0.0
            sum_state = np.zeros_like(initial_belief, dtype=float)
            for belief, prob in zip(beliefs, probs):
                posterior = belief / prob
                posterior_entropy = float(calc_entropy(posterior))
                legacy_entropy += posterior_entropy * prob
                a_state += posterior_entropy * prob / total_prob
                sum_state += belief / total_prob
            state_entropy = float(calc_entropy(sum_state))
            result = {
                "step": step,
                "entropy": float(legacy_entropy),
                "state_entropy": state_entropy,
                "prob": total_prob,
                "E_state": float(max(0.0, state_entropy - a_state)),
                "A_state": float(a_state),
                "belief": sum_state,
            }

        cumulative_entropy += result["entropy"]
        result["cumulative_entropy"] = float(cumulative_entropy)
        result["mean_entropy"] = float(cumulative_entropy / step)
        results.append(result)

    return results


def run_gpu_entropy(
    bundle: SDDModelBundle,
    robot_path: np.ndarray,
    initial_belief: np.ndarray,
    transition: sparse.csr_matrix,
    horizon: int,
    scan_range: float,
    prefix_beliefs: np.ndarray | None = None,
):
    evaluate_discrete_oce_gpu = load_gpu_backend()
    transition = transition.tocsr()
    if prefix_beliefs is None:
        prefix_beliefs = sparse_prefix_beliefs(initial_belief, transition, horizon)
    return evaluate_discrete_oce_gpu(
        paths=np.asarray(robot_path[: int(horizon) + 1], dtype=np.float32)[
            np.newaxis, :, :
        ],
        state_centers=bundle.state_centers_sim.astype(np.float32),
        static_grid=bundle.static_grid,
        grid_origin=(float(bundle.bounds["min_x"]), float(bundle.bounds["min_y"])),
        grid_resolution=float(bundle.cell_size),
        transition_data=transition.data.astype(np.float32, copy=False),
        transition_indices=transition.indices.astype(np.int32, copy=False),
        transition_indptr=transition.indptr.astype(np.int32, copy=False).reshape(1, -1),
        prefix_beliefs=np.asarray(prefix_beliefs, dtype=np.float32).reshape(
            1,
            int(horizon) + 1,
            -1,
        ),
        beliefs=np.asarray(initial_belief, dtype=np.float32).reshape(1, -1),
        horizon=int(horizon),
        scan_range=float(scan_range),
        return_visibility=True,
    )


def cpu_gpu_metric_arrays(
    cpu_results: list[dict], gpu_result
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "entropy": (
            np.asarray([row["entropy"] for row in cpu_results], dtype=float),
            np.asarray(gpu_result.step_entropy[0, 0], dtype=float),
        ),
        "prob": (
            np.asarray([row["prob"] for row in cpu_results], dtype=float),
            np.asarray(gpu_result.step_probability[0, 0], dtype=float),
        ),
        "E_state": (
            np.asarray([row["E_state"] for row in cpu_results], dtype=float),
            np.asarray(gpu_result.step_e_state[0, 0], dtype=float),
        ),
        "A_state": (
            np.asarray([row["A_state"] for row in cpu_results], dtype=float),
            np.asarray(gpu_result.step_a_state[0, 0], dtype=float),
        ),
    }


def compare_cpu_gpu(cpu_results, gpu_result, rtol: float, atol: float) -> dict:
    summary = {}
    failures = []
    for name, (cpu_values, gpu_values) in cpu_gpu_metric_arrays(
        cpu_results, gpu_result
    ).items():
        delta = np.abs(cpu_values - gpu_values)
        max_delta = float(delta.max()) if delta.size else 0.0
        summary[name] = {
            "max_abs_delta": max_delta,
            "cpu": cpu_values.tolist(),
            "gpu": gpu_values.tolist(),
        }
        if not np.allclose(cpu_values, gpu_values, rtol=rtol, atol=atol):
            failures.append(
                f"{name} max_abs_delta={max_delta:.6g} cpu={cpu_values} gpu={gpu_values}"
            )
    if failures:
        print("CPU/GPU entropy mismatch with " f"rtol={rtol:g}, atol={atol:g}: ")
        for f in failures:
            print(f"  {f}")
    return summary


def apply_observed_state(hmm: HMM, observation_state: int) -> None:
    observation_state = int(observation_state)
    mode_scores = np.zeros(hmm.num_modes, dtype=float)
    for mode in range(hmm.num_modes):
        transition = hmm.transition_matrices[mode]
        if hmm.alphas is None:
            mode_scores[mode] = hmm.mode_distribution[mode] * float(
                hmm.state_distribution @ transition[:, observation_state]
            )
        else:
            mode_scores[mode] = float(
                hmm.alphas[mode] @ transition[:, observation_state]
            )

    total = float(mode_scores.sum())
    if total > TOLERANCE:
        hmm.mode_distribution = mode_scores / total

    hmm.state_distribution.fill(0.0)
    hmm.state_distribution[observation_state] = 1.0
    hmm.alphas = np.zeros((hmm.num_modes, hmm.num_states), dtype=float)
    hmm.alphas[:, observation_state] = hmm.mode_distribution


def sparse_vector_dot_column(vector: np.ndarray, column) -> float:
    value = vector @ column
    return float(np.asarray(value).reshape(-1)[0])


def mixed_transition_from_mode_distribution(
    bundle: SDDModelBundle,
    mode_distribution: np.ndarray,
) -> sparse.csr_matrix:
    mixed = None
    for weight, transition in zip(mode_distribution, bundle.class_transitions):
        if float(weight) <= 0.0:
            continue
        weighted = transition.multiply(float(weight))
        mixed = weighted if mixed is None else mixed + weighted
    if mixed is None:
        mixed = bundle.class_transitions[0].multiply(0.0)
    mixed = mixed.tocsr()
    mixed.sum_duplicates()
    mixed.eliminate_zeros()
    return mixed


def update_sparse_belief_step(
    bundle: SDDModelBundle,
    *,
    state_distribution: np.ndarray,
    mode_distribution: np.ndarray,
    alphas: np.ndarray | None,
    observation_state: int,
    visible: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    observation_state = int(observation_state)
    state_distribution = np.asarray(state_distribution, dtype=float).copy()
    mode_distribution = np.asarray(mode_distribution, dtype=float).copy()

    if visible:
        mode_scores = np.zeros(len(bundle.class_ids), dtype=float)
        for mode, transition in enumerate(bundle.class_transitions):
            transition_col = transition[:, observation_state]
            if alphas is None:
                mode_scores[mode] = mode_distribution[mode] * sparse_vector_dot_column(
                    state_distribution,
                    transition_col,
                )
            else:
                mode_scores[mode] = sparse_vector_dot_column(
                    alphas[mode],
                    transition_col,
                )

        total = float(mode_scores.sum())
        if total > TOLERANCE:
            mode_distribution = mode_scores / total
        state_distribution.fill(0.0)
        state_distribution[observation_state] = 1.0
        alphas = np.zeros((len(bundle.class_ids), state_distribution.size), dtype=float)
        alphas[:, observation_state] = mode_distribution
        return state_distribution, mode_distribution, alphas

    if alphas is None:
        next_state = np.zeros_like(state_distribution)
        for mode, transition in enumerate(bundle.class_transitions):
            next_state += (state_distribution @ transition) * mode_distribution[mode]
        return np.asarray(next_state, dtype=float), mode_distribution, None

    next_alphas = np.zeros_like(alphas)
    for mode, transition in enumerate(bundle.class_transitions):
        next_alphas[mode] = alphas[mode] @ transition
    total = float(next_alphas.sum())
    if total > TOLERANCE:
        next_alphas /= total
    mode_distribution = next_alphas.sum(axis=1)
    state_distribution = next_alphas.sum(axis=0)
    return state_distribution, mode_distribution, next_alphas


def realized_class_belief(
    bundle: SDDModelBundle, target_states: np.ndarray, visibility: list[bool]
) -> list[dict]:
    state_distribution = np.zeros(bundle.state_centers_sim.shape[0], dtype=float)
    state_distribution[int(target_states[0])] = 1.0
    mode_distribution = np.asarray(bundle.class_priors, dtype=float).copy()
    alphas = None
    rows = []
    for tick, visible in enumerate(visibility):
        state = int(target_states[min(tick, len(target_states) - 1)])
        state_distribution, mode_distribution, alphas = update_sparse_belief_step(
            bundle,
            state_distribution=state_distribution,
            mode_distribution=mode_distribution,
            alphas=alphas,
            observation_state=state,
            visible=visible,
        )
        rows.append(
            {
                "tick": tick,
                "visible": bool(visible),
                "state": state,
                "mode_distribution": mode_distribution.copy(),
                "state_distribution": state_distribution.copy(),
            }
        )
    return rows


def state_belief_to_grid(bundle: SDDModelBundle, belief: np.ndarray) -> np.ndarray:
    rows = int(bundle.state_metadata["rows"])
    cols = int(bundle.state_metadata["cols"])
    grid = np.full((rows, cols), np.nan, dtype=float)
    for state, prob in enumerate(np.asarray(belief, dtype=float)):
        row, col = bundle.state_grid_indices[state]
        grid[rows - 1 - int(row), int(col)] = prob
    return grid


def add_static_polygons(ax, static_polygons) -> None:
    patches = []
    colours = []
    colour_map = {
        "Building": "#34495e",
        "Obstacle": "#922b21",
        "Object": "#af601a",
        "Offroad": "#58785c",
        "Entrance": "#2980b9",
    }
    for static_polygon in static_polygons:
        points = np.asarray(static_polygon.points, dtype=float)
        if points.shape[0] < 3:
            continue
        patches.append(MplPolygon(points[:, :2], closed=True))
        colours.append(colour_map.get(static_polygon.polygon_class, "#777777"))
    if patches:
        collection = PatchCollection(
            patches,
            facecolor=colours,
            edgecolor="#222222",
            linewidth=0.6,
            alpha=0.28,
        )
        ax.add_collection(collection)


def format_layout_axis(ax, scenario) -> None:
    x0, y0 = scenario.display_offset
    diff = float(scenario.display_diff)
    ax.set_xlim(x0, x0 + diff)
    ax.set_ylim(y0 + diff, y0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def draw_layout(
    ax,
    scenario,
    robot_path,
    target_points,
    step: int,
    visible: bool | None = None,
    goal_point=None,
) -> None:
    add_static_polygons(ax, scenario.static_polygons)
    ax.plot(robot_path[:, 0], robot_path[:, 1], color="#2c3e50", linewidth=1.5)
    ax.scatter(robot_path[0, 0], robot_path[0, 1], s=45, c="#27ae60", label="start")
    goal = np.asarray(
        goal_point if goal_point is not None else robot_path[-1], dtype=float
    )
    ax.scatter(goal[0], goal[1], s=45, c="#c0392b", label="goal")
    step = min(int(step), robot_path.shape[0] - 1, target_points.shape[0] - 1)
    ax.scatter(
        robot_path[step, 0],
        robot_path[step, 1],
        s=85,
        marker=(3, 0, 180),
        c=ROBOT_MARKER_COLOUR,
        edgecolors=ROBOT_MARKER_EDGE,
        linewidths=0.9,
        label="robot",
    )
    ax.plot(target_points[:, 0], target_points[:, 1], color="#8e44ad", linewidth=1.2)
    target_colour = "#e74c3c" if visible is False else "#16a085"
    ax.scatter(
        target_points[step, 0],
        target_points[step, 1],
        s=55,
        c=target_colour,
        label="target",
    )
    format_layout_axis(ax, scenario)


def plot_entropy_step(
    path: Path,
    *,
    scenario,
    bundle: SDDModelBundle,
    robot_path: np.ndarray,
    target_points: np.ndarray,
    goal_point,
    step: int,
    cpu_row: dict,
    gpu_result,
    visible: bool,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    draw_layout(axes[0], scenario, robot_path, target_points, step, visible, goal_point)
    axes[0].set_title(f"Layout at t={step} ({'visible' if visible else 'occluded'})")

    grid = state_belief_to_grid(bundle, cpu_row["belief"])
    max_prob = np.nanmax(grid)
    image = axes[1].imshow(
        grid,
        origin="upper",
        cmap="magma",
        vmin=0.0,
        vmax=max(float(max_prob), 1.0e-9),
        extent=[
            float(bundle.bounds["min_x"]),
            float(bundle.bounds["max_x"]),
            float(bundle.bounds["max_y"]),
            float(bundle.bounds["min_y"]),
        ],
    )
    add_static_polygons(axes[1], scenario.static_polygons)
    axes[1].scatter(
        robot_path[step, 0],
        robot_path[step, 1],
        s=52,
        c=ROBOT_MARKER_COLOUR,
        edgecolors=ROBOT_MARKER_EDGE,
        linewidths=0.8,
    )
    axes[1].scatter(target_points[step, 0], target_points[step, 1], s=35, c="#16a085")
    axes[1].scatter(goal_point[0], goal_point[1], s=35, c="#c0392b")
    format_layout_axis(axes[1], scenario)
    gpu_entropy = float(gpu_result.step_entropy[0, 0, step - 1])
    gpu_prob = float(gpu_result.step_probability[0, 0, step - 1])
    max_belief = float(np.nanmax(grid))
    state_entropy = float(cpu_row["state_entropy"])
    e_state = float(cpu_row["E_state"])
    a_state = float(cpu_row["A_state"])
    axes[0].text(
        0.02,
        0.02,
        (
            f"current entropy: {cpu_row['entropy']:.6f}\n"
            f"state entropy: {state_entropy:.6f}\n"
            f"max belief: {max_belief:.6f}"
        ),
        transform=axes[0].transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85},
    )
    axes[1].set_title(
        "Belief heatmap\n"
        f"CPU H={cpu_row['entropy']:.6f}, GPU H={gpu_entropy:.6f}\n"
        f"CPU p_occ={cpu_row['prob']:.6f}, GPU p_occ={gpu_prob:.6f}\n"
        f"E={e_state:.6f}, A={a_state:.6f}, max belief={max_belief:.6f}"
    )
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_contact_sheet(
    path: Path,
    *,
    scenario,
    bundle: SDDModelBundle,
    robot_path: np.ndarray,
    target_points: np.ndarray,
    goal_point,
    selected_steps: list[int],
    cpu_results: list[dict],
    visibility: list[bool],
) -> None:
    panel_count = len(selected_steps) + 1
    columns = min(6, panel_count)
    rows = int(np.ceil(panel_count / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.0 * columns, 3.8 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    flat_axes = axes.reshape(-1)
    draw_layout(
        flat_axes[0], scenario, robot_path, target_points, 0, visibility[0], goal_point
    )
    flat_axes[0].set_title("Current layout")
    for ax, step in zip(flat_axes[1:], selected_steps):
        row = cpu_results[step - 1]
        grid = state_belief_to_grid(bundle, row["belief"])
        ax.imshow(
            grid,
            origin="upper",
            cmap="magma",
            vmin=0.0,
            vmax=max(float(np.nanmax(grid)), 1.0e-9),
            extent=[
                float(bundle.bounds["min_x"]),
                float(bundle.bounds["max_x"]),
                float(bundle.bounds["max_y"]),
                float(bundle.bounds["min_y"]),
            ],
        )
        add_static_polygons(ax, scenario.static_polygons)
        ax.scatter(
            robot_path[step, 0],
            robot_path[step, 1],
            s=38,
            c=ROBOT_MARKER_COLOUR,
            edgecolors=ROBOT_MARKER_EDGE,
            linewidths=0.7,
        )
        ax.scatter(goal_point[0], goal_point[1], s=24, c="#c0392b")
        ax.scatter(target_points[step, 0], target_points[step, 1], s=25, c="#16a085")
        format_layout_axis(ax, scenario)
        ax.set_title(
            f"t={step}\n"
            f"H={row['entropy']:.4f}, p={row['prob']:.4f}\n"
            f"max b={float(np.nanmax(grid)):.4f}"
        )
    for ax in flat_axes[panel_count:]:
        ax.axis("off")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_class_belief(
    path: Path,
    *,
    class_rows: list[dict],
    class_ids: list[int],
    true_class: int,
) -> None:
    ticks = np.asarray([row["tick"] for row in class_rows], dtype=int)
    beliefs = np.stack([row["mode_distribution"] for row in class_rows], axis=0)
    visible = np.asarray([row["visible"] for row in class_rows], dtype=bool)

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    for idx, class_id in enumerate(class_ids):
        linewidth = 2.4 if int(class_id) == int(true_class) else 1.2
        alpha = 1.0 if int(class_id) == int(true_class) else 0.65
        ax.plot(
            ticks,
            beliefs[:, idx],
            linewidth=linewidth,
            alpha=alpha,
            label=f"class {class_id}",
        )

    start = None
    for tick, is_visible in zip(ticks, visible):
        if is_visible and start is None:
            start = tick
        if not is_visible and start is not None:
            ax.axvspan(start, tick - 1, color="#2ecc71", alpha=0.12)
            start = None
    if start is not None:
        ax.axvspan(start, ticks[-1], color="#2ecc71", alpha=0.12)

    ax.set_xlabel("tick")
    ax.set_ylabel("belief")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Realized target class belief")
    ax.legend(ncol=3, fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def robot_future_path(rollout_path: np.ndarray, tick: int, horizon: int) -> np.ndarray:
    tick = int(tick)
    horizon = int(horizon)
    path = np.asarray(rollout_path, dtype=float)
    future = path[tick : tick + horizon + 1]
    if future.shape[0] == 0:
        future = path[-1:, :]
    if future.shape[0] < horizon + 1:
        pad = np.repeat(future[-1:, :], horizon + 1 - future.shape[0], axis=0)
        future = np.vstack([future, pad])
    return future[: horizon + 1]


def target_future_points(
    bundle: SDDModelBundle,
    target_states: np.ndarray,
    tick: int,
    horizon: int,
) -> np.ndarray:
    indices = [
        int(target_states[min(int(tick) + step, len(target_states) - 1)])
        for step in range(int(horizon) + 1)
    ]
    return bundle.state_centers_sim[np.asarray(indices, dtype=int)]


def plot_current_belief_step(
    path: Path,
    *,
    scenario,
    bundle: SDDModelBundle,
    rollout_path: np.ndarray,
    target_points: np.ndarray,
    goal_point: np.ndarray,
    tick: int,
    state_distribution: np.ndarray,
    mode_distribution: np.ndarray,
    visible: bool,
    true_class: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    current_robot_path = np.asarray(rollout_path[: int(tick) + 1], dtype=float)
    if current_robot_path.shape[0] == 0:
        current_robot_path = np.asarray(rollout_path[:1], dtype=float)
    current_targets = np.asarray(target_points[: int(tick) + 1], dtype=float)
    if current_targets.shape[0] == 0:
        current_targets = np.asarray(target_points[:1], dtype=float)
    draw_layout(
        axes[0],
        scenario,
        current_robot_path,
        current_targets,
        current_robot_path.shape[0] - 1,
        visible,
        goal_point,
    )
    axes[0].set_title(
        f"Simulation step {int(tick)} ({'visible' if visible else 'occluded'})"
    )

    grid = state_belief_to_grid(bundle, state_distribution)
    image = axes[1].imshow(
        grid,
        origin="upper",
        cmap="magma",
        vmin=0.0,
        vmax=max(float(np.nanmax(grid)), 1.0e-9),
        extent=[
            float(bundle.bounds["min_x"]),
            float(bundle.bounds["max_x"]),
            float(bundle.bounds["max_y"]),
            float(bundle.bounds["min_y"]),
        ],
    )
    add_static_polygons(axes[1], scenario.static_polygons)
    robot = rollout_path[int(tick)]
    target = target_points[int(tick)]
    axes[1].scatter(
        robot[0],
        robot[1],
        s=58,
        c=ROBOT_MARKER_COLOUR,
        edgecolors=ROBOT_MARKER_EDGE,
        linewidths=0.9,
    )
    axes[1].scatter(target[0], target[1], s=42, c="#16a085" if visible else "#e74c3c")
    axes[1].scatter(goal_point[0], goal_point[1], s=35, c="#c0392b")
    format_layout_axis(axes[1], scenario)
    true_idx = bundle.class_ids.index(int(true_class))
    axes[1].set_title(
        "Current state belief\n"
        f"H_state={calc_entropy(state_distribution):.6f}, "
        f"H_class={calc_entropy(mode_distribution):.6f}\n"
        f"class {true_class}={mode_distribution[true_idx]:.6f}, "
        f"max belief={float(np.nanmax(grid)):.6f}"
    )
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def gpu_horizon_rows(
    gpu_result,
    tick: int,
    future_target_states: list[int] | np.ndarray | None = None,
) -> list[dict]:
    rows = []
    horizon = int(gpu_result.step_entropy.shape[2])
    cumulative_entropy = 0.0
    for idx in range(horizon):
        entropy = float(gpu_result.step_entropy[0, 0, idx])
        cumulative_entropy += entropy
        visibility = np.asarray(gpu_result.visibility_tensor[0, idx + 1], dtype=float)
        target_visible = None
        if future_target_states is not None:
            target_state = int(
                future_target_states[min(idx + 1, len(future_target_states) - 1)]
            )
            target_visible = float(visibility[target_state])
        rows.append(
            {
                "tick": int(tick),
                "horizon_step": idx + 1,
                "visible_state_count": int(np.count_nonzero(visibility > 0.5)),
                "target_visible": target_visible,
                "gpu_entropy": entropy,
                "gpu_prob": float(gpu_result.step_probability[0, 0, idx]),
                "gpu_E_state": float(gpu_result.step_e_state[0, 0, idx]),
                "gpu_A_state": float(gpu_result.step_a_state[0, 0, idx]),
                "gpu_state_entropy": float(gpu_result.step_state_entropy[0, 0, idx]),
                "gpu_cumulative_entropy": float(cumulative_entropy),
            }
        )
    return rows


def plot_simulation_contact_sheet(
    path: Path,
    *,
    scenario,
    bundle: SDDModelBundle,
    future_robot_path: np.ndarray,
    future_target_points: np.ndarray,
    goal_point: np.ndarray,
    tick: int,
    current_visible: bool,
    prefix_beliefs: np.ndarray,
    gpu_result,
    selected_steps: list[int],
    future_target_states: list[int] | np.ndarray,
) -> None:
    panel_count = len(selected_steps) + 1
    columns = panel_count
    fig, axes = plt.subplots(
        1,
        columns,
        figsize=(4.0 * columns, 4.0),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.reshape(-1)
    draw_layout(
        axes[0],
        scenario,
        future_robot_path[:1],
        future_target_points[:1],
        0,
        current_visible,
        goal_point,
    )
    axes[0].set_title(f"Current layout\nsim step {int(tick)}")
    for ax, horizon_step in zip(axes[1:], selected_steps):
        belief = prefix_beliefs[int(horizon_step)]
        grid = state_belief_to_grid(bundle, belief)
        ax.imshow(
            grid,
            origin="upper",
            cmap="magma",
            vmin=0.0,
            vmax=max(float(np.nanmax(grid)), 1.0e-9),
            extent=[
                float(bundle.bounds["min_x"]),
                float(bundle.bounds["max_x"]),
                float(bundle.bounds["max_y"]),
                float(bundle.bounds["min_y"]),
            ],
        )
        add_static_polygons(ax, scenario.static_polygons)
        robot = future_robot_path[int(horizon_step)]
        target = future_target_points[int(horizon_step)]
        ax.scatter(
            robot[0],
            robot[1],
            s=44,
            c=ROBOT_MARKER_COLOUR,
            edgecolors=ROBOT_MARKER_EDGE,
            linewidths=0.8,
        )
        ax.scatter(target[0], target[1], s=28, c="#16a085")
        ax.scatter(goal_point[0], goal_point[1], s=24, c="#c0392b")
        format_layout_axis(ax, scenario)
        idx = int(horizon_step) - 1
        target_state = int(
            future_target_states[min(int(horizon_step), len(future_target_states) - 1)]
        )
        target_visible = float(
            gpu_result.visibility_tensor[0, int(horizon_step), target_state]
        )
        ax.set_title(
            f"h={int(horizon_step)}\n"
            f"GPU H={float(gpu_result.step_entropy[0, 0, idx]):.4f}, "
            f"p={float(gpu_result.step_probability[0, 0, idx]):.4f}\n"
            f"E={float(gpu_result.step_e_state[0, 0, idx]):.4f}, "
            f"A={float(gpu_result.step_a_state[0, 0, idx]):.4f}\n"
            f"target visible={target_visible:.0f}"
        )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_horizon_entropy_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "tick",
        "horizon_step",
        "visible_state_count",
        "target_visible",
        "gpu_entropy",
        "gpu_prob",
        "gpu_E_state",
        "gpu_A_state",
        "gpu_state_entropy",
        "gpu_cumulative_entropy",
        "cpu_entropy",
        "cpu_prob",
        "cpu_E_state",
        "cpu_A_state",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_entropy_csv(path: Path, cpu_results, gpu_result) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "cpu_entropy",
                "gpu_entropy",
                "cpu_prob",
                "gpu_prob",
                "cpu_E_state",
                "gpu_E_state",
                "cpu_A_state",
                "gpu_A_state",
                "cpu_cumulative_entropy",
            ],
        )
        writer.writeheader()
        for idx, row in enumerate(cpu_results):
            writer.writerow(
                {
                    "step": row["step"],
                    "cpu_entropy": row["entropy"],
                    "gpu_entropy": float(gpu_result.step_entropy[0, 0, idx]),
                    "cpu_prob": row["prob"],
                    "gpu_prob": float(gpu_result.step_probability[0, 0, idx]),
                    "cpu_E_state": row["E_state"],
                    "gpu_E_state": float(gpu_result.step_e_state[0, 0, idx]),
                    "cpu_A_state": row["A_state"],
                    "gpu_A_state": float(gpu_result.step_a_state[0, 0, idx]),
                    "cpu_cumulative_entropy": row["cumulative_entropy"],
                }
            )


def write_class_belief_csv(path: Path, class_rows, class_ids) -> None:
    fieldnames = ["tick", "visible", "state"] + [
        f"class_{class_id}" for class_id in class_ids
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in class_rows:
            record = {
                "tick": row["tick"],
                "visible": int(row["visible"]),
                "state": row["state"],
            }
            for class_id, prob in zip(class_ids, row["mode_distribution"]):
                record[f"class_{class_id}"] = float(prob)
            writer.writerow(record)


def run_validation(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    current_dir = out_dir / "current_belief"
    contact_dir = out_dir / "horizon_contact_sheets"
    current_dir.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)

    scenario = load_scenario(
        "sdd",
        sdd_processed_root=args.sdd_processed_root,
        sdd_scene_id=args.scene_id,
    )
    scenario = scenario.__class__(
        name=scenario.name,
        data_source=scenario.data_source,
        tracks={int(args.track_id): scenario.tracks[int(args.track_id)]},
        display_offset=scenario.display_offset,
        display_diff=scenario.display_diff,
        static_polygons=scenario.static_polygons,
        metadata=scenario.metadata,
    )
    bundle = load_sdd_model_bundle(args.sdd_model_root, args.scene_id)
    validate_track_class(bundle, args.track_id, args.destination_class)

    if not point_is_free(args.robot_start, scenario.static_polygons):
        raise ValueError(
            f"Robot start is inside a blocking polygon: {args.robot_start}"
        )
    if not point_is_free(args.robot_goal, scenario.static_polygons):
        raise ValueError(f"Robot goal is inside a blocking polygon: {args.robot_goal}")

    target_states = load_track_state_sequence(
        args.sdd_model_root, args.scene_id, args.track_id
    )
    static_union = blocking_union(scenario.static_polygons)
    rollout_path = linear_robot_path(
        args.robot_start, args.robot_goal, args.rollout_steps
    )
    target_points = target_future_points(
        bundle,
        target_states,
        tick=0,
        horizon=max(int(args.rollout_steps) - 1, 0),
    )
    rollout_visibility = target_visibility_over_rollout(
        rollout_path,
        target_states,
        bundle.state_centers_sim,
        static_union,
        args.scan_range,
    )

    interval = max(1, int(args.interval))
    selected_steps = [
        step
        for step in range(interval, int(args.horizon) + 1, interval)
        if step <= int(args.horizon)
    ]
    if not selected_steps:
        selected_steps = [int(args.horizon)]

    state_distribution = np.zeros(bundle.state_centers_sim.shape[0], dtype=float)
    state_distribution[int(target_states[0])] = 1.0
    mode_distribution = np.asarray(bundle.class_priors, dtype=float).copy()
    alphas = None

    cpu_compare_stride = int(getattr(args, "cpu_compare_stride", 30))
    class_rows = []
    entropy_rows = []
    current_pngs = []
    contact_sheet_pngs = []
    comparisons = {}
    true_class_idx = bundle.class_ids.index(int(args.destination_class))
    goal_point = np.asarray(args.robot_goal, dtype=float)

    for tick in range(int(args.rollout_steps)):
        target_state = int(target_states[min(tick, len(target_states) - 1)])
        visible = bool(rollout_visibility[tick])
        state_distribution, mode_distribution, alphas = update_sparse_belief_step(
            bundle,
            state_distribution=state_distribution,
            mode_distribution=mode_distribution,
            alphas=alphas,
            observation_state=target_state,
            visible=visible,
        )
        class_rows.append(
            {
                "tick": tick,
                "visible": visible,
                "state": target_state,
                "mode_distribution": mode_distribution.copy(),
                "state_distribution": state_distribution.copy(),
            }
        )

        current_path = current_dir / f"current_step_{tick:03d}.png"
        plot_current_belief_step(
            current_path,
            scenario=scenario,
            bundle=bundle,
            rollout_path=rollout_path,
            target_points=target_points,
            goal_point=goal_point,
            tick=tick,
            state_distribution=state_distribution,
            mode_distribution=mode_distribution,
            visible=visible,
            true_class=args.destination_class,
        )
        current_pngs.append(str(current_path))

        future_path = robot_future_path(rollout_path, tick, args.horizon)
        future_targets = target_future_points(bundle, target_states, tick, args.horizon)
        future_target_states = [
            int(target_states[min(tick + step, len(target_states) - 1)])
            for step in range(int(args.horizon) + 1)
        ]
        mixed_transition = mixed_transition_from_mode_distribution(
            bundle,
            mode_distribution,
        )
        prefix_beliefs = sparse_prefix_beliefs(
            state_distribution.astype(np.float32),
            mixed_transition,
            args.horizon,
        )
        gpu_result = run_gpu_entropy(
            bundle,
            future_path,
            state_distribution.astype(np.float32),
            mixed_transition,
            args.horizon,
            args.scan_range,
            prefix_beliefs=prefix_beliefs,
        )
        tick_entropy_rows = gpu_horizon_rows(
            gpu_result,
            tick,
            future_target_states=future_target_states,
        )

        should_compare_cpu = cpu_compare_stride > 0 and (
            tick % cpu_compare_stride == 0 or tick == int(args.rollout_steps) - 1
        )
        if should_compare_cpu:
            gpu_occlusion = [
                (1.0 - gpu_result.visibility_tensor[0, step]).astype(float)
                for step in range(int(args.horizon) + 1)
            ]
            cpu_results = sparse_exact_entropy(
                state_distribution.astype(np.float32),
                mixed_transition,
                gpu_occlusion,
                args.horizon,
            )
            comparisons[str(tick)] = compare_cpu_gpu(
                cpu_results,
                gpu_result,
                args.rtol,
                args.atol,
            )
            for row, cpu_row in zip(tick_entropy_rows, cpu_results):
                row["cpu_entropy"] = float(cpu_row["entropy"])
                row["cpu_prob"] = float(cpu_row["prob"])
                row["cpu_E_state"] = float(cpu_row["E_state"])
                row["cpu_A_state"] = float(cpu_row["A_state"])
        entropy_rows.extend(tick_entropy_rows)

        contact_path = contact_dir / f"contact_sheet_step_{tick:03d}.png"
        plot_simulation_contact_sheet(
            contact_path,
            scenario=scenario,
            bundle=bundle,
            future_robot_path=future_path,
            future_target_points=future_targets,
            goal_point=goal_point,
            tick=tick,
            current_visible=visible,
            prefix_beliefs=prefix_beliefs,
            gpu_result=gpu_result,
            selected_steps=selected_steps,
            future_target_states=future_target_states,
        )
        contact_sheet_pngs.append(str(contact_path))

    class_belief_path = out_dir / "class_belief.png"
    plot_class_belief(
        class_belief_path,
        class_rows=class_rows,
        class_ids=bundle.class_ids,
        true_class=args.destination_class,
    )

    entropy_csv_path = out_dir / "entropy_cpu_gpu.csv"
    class_csv_path = out_dir / "class_belief.csv"
    write_horizon_entropy_csv(entropy_csv_path, entropy_rows)
    write_class_belief_csv(class_csv_path, class_rows, bundle.class_ids)

    final_distribution = class_rows[-1]["mode_distribution"]
    summary = {
        "scene_id": int(args.scene_id),
        "destination_class": int(args.destination_class),
        "track_id": int(args.track_id),
        "robot_start": [float(v) for v in args.robot_start],
        "robot_goal": [float(v) for v in args.robot_goal],
        "robot_heading": float(args.robot_heading),
        "horizon": int(args.horizon),
        "rollout_steps": int(args.rollout_steps),
        "scan_range": float(args.scan_range),
        "rtol": float(args.rtol),
        "atol": float(args.atol),
        "gpu_execution_path": "cuda_discrete_exact_csr",
        "cpu_compare_stride": cpu_compare_stride,
        "cpu_gpu_comparison": comparisons,
        "visibility": {
            "rollout_visible_count": int(np.sum(rollout_visibility)),
            "first_rollout_visible_tick": next(
                (int(idx) for idx, visible in enumerate(rollout_visibility) if visible),
                None,
            ),
            "last_rollout_visible_tick": next(
                (
                    int(idx)
                    for idx in range(len(rollout_visibility) - 1, -1, -1)
                    if rollout_visibility[idx]
                ),
                None,
            ),
        },
        "final_class_belief": {
            str(class_id): float(prob)
            for class_id, prob in zip(bundle.class_ids, final_distribution)
        },
        "final_true_class_probability": float(final_distribution[true_class_idx]),
        "outputs": {
            "current_belief_pngs": current_pngs,
            "horizon_contact_sheet_pngs": contact_sheet_pngs,
            "current_belief_dir": str(current_dir),
            "horizon_contact_sheet_dir": str(contact_dir),
            "class_belief": str(class_belief_path),
            "entropy_cpu_gpu_csv": str(entropy_csv_path),
            "class_belief_csv": str(class_csv_path),
            "summary": str(out_dir / "summary.json"),
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate SDD discrete entropy growth and target class belief."
    )
    parser.add_argument("--sdd-processed-root", default="outputs/sdd_processed")
    parser.add_argument("--sdd-model-root", default="outputs/sdd_models")
    parser.add_argument("--scene-id", type=int, default=DEFAULT_SCENE_ID)
    parser.add_argument(
        "--destination-class", type=int, default=DEFAULT_DESTINATION_CLASS
    )
    parser.add_argument("--track-id", type=int, default=DEFAULT_TRACK_ID)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    parser.add_argument("--rollout-steps", type=int, default=DEFAULT_ROLLOUT_STEPS)
    parser.add_argument(
        "--robot-start", nargs=2, type=float, default=list(DEFAULT_ROBOT_START)
    )
    parser.add_argument(
        "--robot-goal", nargs=2, type=float, default=list(DEFAULT_ROBOT_GOAL)
    )
    parser.add_argument("--robot-heading", type=float, default=DEFAULT_ROBOT_HEADING)
    parser.add_argument("--scan-range", type=float, default=DEFAULT_SCAN_RANGE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument(
        "--cpu-compare-stride",
        type=int,
        default=30,
        help=(
            "Run the sparse CPU entropy oracle every N simulation steps for "
            "CPU/GPU validation. Use 1 to compare every step; use 0 to disable."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        summary = run_validation(args)
    except (AssertionError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        "Wrote entropy validation outputs to "
        f"{summary['outputs']['summary']}; "
        f"final class {summary['destination_class']} probability="
        f"{summary['final_true_class_probability']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
