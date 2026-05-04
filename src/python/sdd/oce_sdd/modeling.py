from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from sklearn.cluster import AgglomerativeClustering
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

NON_WALKABLE_CLASSES = ("Building", "Obstacle", "Object", "Offroad")


@dataclass(frozen=True)
class ProcessedSceneRecord:
    scene_id: int
    root: Path
    metadata: dict[str, Any]
    trajectories: dict[int, np.ndarray]
    polygons: dict[str, list[np.ndarray]]

    @property
    def bounds(self) -> dict[str, float]:
        return self.metadata["transform"]["scene_bounds"]

    @property
    def units(self) -> str:
        return self.metadata["coordinate_frames"]["scene"]["units"]


@dataclass(frozen=True)
class GridStateSpace:
    scene_id: int
    cell_size: float
    cell_size_meters: float
    scene_scale: float
    bounds: dict[str, float]
    rows: int
    cols: int
    state_ids: np.ndarray
    centers: np.ndarray
    grid_indices: np.ndarray
    grid_to_state: np.ndarray
    walkable_mask: np.ndarray
    non_walkable_classes: tuple[str, ...]


@dataclass(frozen=True)
class DestinationClasses:
    radius: float
    centers: np.ndarray
    train_counts: np.ndarray
    member_endpoints: tuple[np.ndarray, ...]
    member_state_ids: tuple[np.ndarray, ...]
    track_to_class: dict[int, int]
    class_radii: np.ndarray | None = None
    merge_epsilon: float | None = None
    merge_epsilon_meters: float | None = None
    pre_merge_class_count: int | None = None


def main() -> None:
    args = parse_args()
    processed_root = Path(args.processed_root)
    scene_ids = args.scene_id if args.scene_id else available_scene_ids(processed_root)
    output_root = Path(args.out)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for scene_id in scene_ids:
        scene = load_processed_scene(processed_root, scene_id)
        scene_out = output_root / f"scene_{scene_id:03d}"
        scene_out.mkdir(parents=True, exist_ok=True)
        scale = scene_scale(scene)
        cell_size = meters_to_scene_units(args.grid_size, scale)
        warn_if_grid_too_large(
            scene,
            cell_size=cell_size,
            cell_size_meters=args.grid_size,
            max_cells=args.max_grid_cells_warning,
        )
        destination_radius = destination_radius_scene_units(scene, args, scale)
        rows, cols = grid_shape_for_cell_size(scene, cell_size=cell_size)
        print(
            f"Building scene {scene_id:03d}: "
            f"scene_scale={scale:.6g}, grid_size_meters={args.grid_size:.6g}, "
            f"cell_size={cell_size:.6g}, rows={rows}, cols={cols}, "
            f"destination_radius={destination_radius:.6g}",
            flush=True,
        )

        state_space = build_grid_state_space(
            scene,
            cell_size=cell_size,
            cell_size_meters=args.grid_size,
            scene_scale=scale,
            non_walkable_classes=tuple(args.non_walkable_class or NON_WALKABLE_CLASSES),
        )
        trajectory_states = map_trajectories_to_states(scene, state_space)
        splits = split_tracks(
            sorted(scene.trajectories),
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        destination_classes = fit_destination_classes(
            scene,
            state_space,
            splits["train"],
            radius=destination_radius,
            min_samples=args.destination_min_samples,
            snap_distance=args.endpoint_snap_distance,
        )
        pre_merge_class_count = len(destination_classes.centers)
        destination_merge_epsilon = None
        track_to_class = assign_destination_classes(
            scene,
            state_space,
            destination_classes,
            max_distance=destination_radius,
            snap_distance=args.endpoint_snap_distance,
        )
        destination_classes = DestinationClasses(
            radius=destination_classes.radius,
            centers=destination_classes.centers,
            train_counts=destination_classes.train_counts,
            member_endpoints=destination_classes.member_endpoints,
            member_state_ids=destination_classes.member_state_ids,
            class_radii=destination_classes.class_radii,
            track_to_class=track_to_class,
            merge_epsilon=destination_classes.merge_epsilon,
            merge_epsilon_meters=destination_classes.merge_epsilon_meters,
            pre_merge_class_count=destination_classes.pre_merge_class_count,
        )
        if args.merge_adjacent_destination_classes:
            destination_merge_epsilon = destination_merge_epsilon_scene_units(args, scale)
            destination_classes = merge_adjacent_destination_classes(
                destination_classes,
                epsilon=destination_merge_epsilon,
                epsilon_meters=(
                    destination_merge_epsilon / scale
                    if scale > 0
                    else args.destination_merge_epsilon_meters
                ),
            )
            track_to_class = destination_classes.track_to_class

        transition_model = learn_transition_model(
            state_space=state_space,
            destination_classes=destination_classes,
            trajectory_states=trajectory_states,
            train_track_ids=splits["train"],
            track_to_class=track_to_class,
            trajectory_stride=args.trajectory_stride,
            transition_min_support=args.transition_min_support,
            global_goal_tau_meters=args.global_goal_tau_meters,
            global_goal_tau=meters_to_scene_units(args.global_goal_tau_meters, scale),
            map_goal_tau_meters=args.map_goal_tau_meters,
            map_goal_tau=meters_to_scene_units(args.map_goal_tau_meters, scale),
            endpoint_snap_distance=args.endpoint_snap_distance,
        )

        write_state_space(state_space, scene_out)
        write_trajectory_states(trajectory_states, scene_out / "trajectory_states.npz")
        write_splits(splits, scene_out / "splits.json")
        write_destination_classes(destination_classes, scene_out / "destination_classes.json")
        write_transition_model(transition_model, scene_out)
        write_model_metadata(
            scene,
            state_space,
            destination_classes,
            transition_model,
            splits,
            scene_out / "model_metadata.json",
            trajectory_stride=args.trajectory_stride,
        )

        summary_rows.append(
            {
                "scene_id": scene_id,
                "scene_scale": scale,
                "grid_size_meters": args.grid_size,
                "cell_size": cell_size,
                "destination_radius_meters": args.destination_radius_meters,
                "destination_radius": destination_radius,
                "destination_merge_enabled": args.merge_adjacent_destination_classes,
                "destination_merge_epsilon_meters": (
                    destination_merge_epsilon / scale
                    if destination_merge_epsilon is not None and scale > 0
                    else None
                ),
                "destination_merge_epsilon": destination_merge_epsilon,
                "pre_merge_destination_classes": pre_merge_class_count,
                "state_count": len(state_space.state_ids),
                "rows": state_space.rows,
                "cols": state_space.cols,
                "train_tracks": len(splits["train"]),
                "val_tracks": len(splits["val"]),
                "test_tracks": len(splits["test"]),
                "destination_classes": len(destination_classes.centers),
                "global_nonzero": int(transition_model["global_counts"].nnz),
                "transition_min_support": args.transition_min_support,
                "global_goal_tau_meters": args.global_goal_tau_meters,
                "global_goal_tau": transition_model["global_goal_tau"],
                "map_goal_tau_meters": args.map_goal_tau_meters,
                "map_goal_tau": transition_model["map_goal_tau"],
                "unassigned_tracks": sum(1 for c in track_to_class.values() if c < 0),
            }
        )

    write_summary(summary_rows, output_root / "model_summary.csv")
    print(f"Wrote grid state spaces and transition models for {len(summary_rows)} scenes to {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build uniform-grid state spaces and class-conditioned pedestrian transitions."
    )
    parser.add_argument("--processed-root", default="outputs/sdd_processed")
    parser.add_argument("--out", default="outputs/sdd_models")
    parser.add_argument(
        "--scene-id",
        type=int,
        action="append",
        help="Scene ID to process. Repeat for several scenes. Defaults to all processed scenes.",
    )
    parser.add_argument(
        "--grid-size",
        type=float,
        default=0.25,
        help="Uniform grid cell size in meters.",
    )
    radius_group = parser.add_mutually_exclusive_group()
    radius_group.add_argument(
        "--destination-radius",
        type=float,
        default=None,
        help="Maximum destination-cluster complete-link diameter in normalized scene units.",
    )
    radius_group.add_argument(
        "--destination-radius-meters",
        type=float,
        default=None,
        help="Maximum destination-cluster complete-link diameter in meters.",
    )
    parser.add_argument(
        "--destination-min-samples",
        type=int,
        default=3,
        help="Minimum train endpoints required to keep a destination class.",
    )
    parser.add_argument(
        "--merge-adjacent-destination-classes",
        action="store_true",
        help=(
            "Merge destination classes whose endpoint goal regions are adjacent under "
            "the class-merge graph."
        ),
    )
    merge_group = parser.add_mutually_exclusive_group()
    merge_group.add_argument(
        "--destination-merge-epsilon",
        type=float,
        default=None,
        help="Adjacent destination-class merge tolerance epsilon_c in scene units.",
    )
    merge_group.add_argument(
        "--destination-merge-epsilon-meters",
        type=float,
        default=0.5,
        help="Adjacent destination-class merge tolerance epsilon_c in meters.",
    )
    parser.add_argument(
        "--endpoint-snap-distance",
        type=float,
        default=None,
        help=(
            "Maximum distance for snapping destination endpoints that land just outside "
            "the walkable grid, in scene units. Defaults to the grid cell size."
        ),
    )
    parser.add_argument(
        "--max-grid-cells-warning",
        type=int,
        default=1_000_000,
        help="Warn when a scene's resolved grid rows*cols exceeds this value.",
    )
    parser.add_argument(
        "--non-walkable-class",
        action="append",
        default=None,
        help="Polygon class to remove from walkable space. Repeat to specify several.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--trajectory-stride",
        type=int,
        default=1,
        help="Use every Nth state sample when counting transitions.",
    )
    parser.add_argument(
        "--transition-min-support",
        type=int,
        default=10,
        help="Visit count N_min required for stable transition estimates.",
    )
    parser.add_argument(
        "--global-goal-tau-meters",
        type=float,
        default=1.0,
        help="Goal-distance penalty temperature tau_g for class-conditioned global transitions, in meters.",
    )
    parser.add_argument(
        "--map-goal-tau-meters",
        type=float,
        default=1.0,
        help="Goal-distance temperature tau_m for class-conditioned map priors, in meters.",
    )
    args = parser.parse_args()
    if args.destination_radius is None and args.destination_radius_meters is None:
        args.destination_radius_meters = 2.0
    if args.destination_merge_epsilon is not None:
        if not np.isfinite(args.destination_merge_epsilon) or args.destination_merge_epsilon < 0:
            parser.error("--destination-merge-epsilon must be finite and non-negative")
    elif (
        not np.isfinite(args.destination_merge_epsilon_meters)
        or args.destination_merge_epsilon_meters < 0
    ):
        parser.error("--destination-merge-epsilon-meters must be finite and non-negative")
    if args.transition_min_support <= 0:
        parser.error("--transition-min-support must be positive")
    if not np.isfinite(args.global_goal_tau_meters) or args.global_goal_tau_meters <= 0:
        parser.error("--global-goal-tau-meters must be finite and positive")
    if not np.isfinite(args.map_goal_tau_meters) or args.map_goal_tau_meters <= 0:
        parser.error("--map-goal-tau-meters must be finite and positive")
    return args


def scene_scale(scene: ProcessedSceneRecord) -> float:
    scale = float(scene.metadata.get("scene_scale", 0.0))
    if np.isfinite(scale) and scale > 0:
        return scale
    if not scene.trajectories:
        print(
            f"WARNING: scene {scene.scene_id:03d} has no trajectories and invalid "
            f"scene_scale={scale}; using 1.0 for empty-scene model artifacts."
        )
        return 1.0
    raise ValueError(
        f"Scene {scene.scene_id} metadata is missing a finite positive scene_scale. "
        "Regenerate processed SDD data with oce_sdd.preprocess."
    )


def meters_to_scene_units(value_meters: float, scale: float) -> float:
    value_meters = float(value_meters)
    scale = float(scale)
    if not np.isfinite(value_meters) or value_meters <= 0:
        raise ValueError("meter value must be finite and positive")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scene scale must be finite and positive")
    return value_meters * scale


def destination_radius_scene_units(
    scene: ProcessedSceneRecord,
    args: argparse.Namespace,
    scale: float,
) -> float:
    if args.destination_radius is not None:
        radius = float(args.destination_radius)
    else:
        radius = meters_to_scene_units(args.destination_radius_meters, scale)

    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("destination radius must be finite and positive")
    return radius


def destination_merge_epsilon_scene_units(args: argparse.Namespace, scale: float) -> float:
    if args.destination_merge_epsilon is not None:
        epsilon = float(args.destination_merge_epsilon)
    else:
        value_meters = float(args.destination_merge_epsilon_meters)
        scale = float(scale)
        if not np.isfinite(value_meters) or value_meters < 0:
            raise ValueError("destination merge epsilon meters must be finite and non-negative")
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("scene scale must be finite and positive")
        epsilon = value_meters * scale

    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("destination merge epsilon must be finite and non-negative")
    return epsilon


def grid_shape_for_cell_size(
    scene: ProcessedSceneRecord,
    *,
    cell_size: float,
) -> tuple[int, int]:
    bounds = scene.bounds
    width = bounds["max_x"] - bounds["min_x"]
    height = bounds["max_y"] - bounds["min_y"]
    return int(np.ceil(height / cell_size)), int(np.ceil(width / cell_size))


def warn_if_grid_too_large(
    scene: ProcessedSceneRecord,
    *,
    cell_size: float,
    cell_size_meters: float,
    max_cells: int,
) -> None:
    if max_cells <= 0:
        return
    rows, cols = grid_shape_for_cell_size(scene, cell_size=cell_size)
    total = rows * cols
    if total > max_cells:
        print(
            "WARNING: "
            f"scene {scene.scene_id:03d} grid has {rows} rows x {cols} cols "
            f"= {total} cells, exceeding --max-grid-cells-warning={max_cells}; "
            f"grid_size_meters={cell_size_meters:.6g}, cell_size={cell_size:.6g}"
        )


def available_scene_ids(processed_root: Path) -> list[int]:
    scene_ids = []
    for path in processed_root.glob("scene_*"):
        if path.is_dir():
            try:
                scene_ids.append(int(path.name.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
    return sorted(scene_ids)


def load_processed_scene(processed_root: Path, scene_id: int) -> ProcessedSceneRecord:
    scene_root = processed_root / f"scene_{scene_id:03d}"
    metadata_path = scene_root / "metadata.json"
    trajectories_path = scene_root / "trajectories_scene.npz"
    polygons_path = scene_root / "polygons_scene.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing scene metadata: {metadata_path}")

    metadata = json.loads(metadata_path.read_text())
    trajectory_data = np.load(trajectories_path)
    trajectories = {
        int(track_id): np.asarray(trajectory_data[f"track_{int(track_id)}"], dtype=float)
        for track_id in trajectory_data["track_ids"]
    }
    raw_polygons = json.loads(polygons_path.read_text())["polygons"]
    polygons: dict[str, list[np.ndarray]] = {}
    for record in raw_polygons:
        polygons.setdefault(record["polygon_class"], []).append(
            np.asarray(record["vertices"], dtype=float)
        )

    return ProcessedSceneRecord(
        scene_id=scene_id,
        root=scene_root,
        metadata=metadata,
        trajectories=trajectories,
        polygons=polygons,
    )


def build_grid_state_space(
    scene: ProcessedSceneRecord,
    *,
    cell_size: float,
    cell_size_meters: float,
    scene_scale: float,
    non_walkable_classes: tuple[str, ...] = NON_WALKABLE_CLASSES,
) -> GridStateSpace:
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")

    bounds = scene.bounds
    rows, cols = grid_shape_for_cell_size(scene, cell_size=cell_size)
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Invalid scene bounds for scene {scene.scene_id}: {bounds}")

    walkable_region = build_walkable_region(scene, non_walkable_classes)
    walkable_mask = np.zeros((rows, cols), dtype=bool)
    state_ids = []
    centers = []
    grid_indices = []
    grid_to_state = np.full((rows, cols), -1, dtype=np.int64)

    for row in range(rows):
        y = bounds["min_y"] + (row + 0.5) * cell_size
        if y > bounds["max_y"]:
            continue
        for col in range(cols):
            x = bounds["min_x"] + (col + 0.5) * cell_size
            if x > bounds["max_x"]:
                continue
            if walkable_region.covers(Point(float(x), float(y))):
                state_id = len(state_ids)
                walkable_mask[row, col] = True
                grid_to_state[row, col] = state_id
                state_ids.append(state_id)
                centers.append((x, y))
                grid_indices.append((row, col))

    return GridStateSpace(
        scene_id=scene.scene_id,
        cell_size=float(cell_size),
        cell_size_meters=float(cell_size_meters),
        scene_scale=float(scene_scale),
        bounds=bounds,
        rows=rows,
        cols=cols,
        state_ids=np.asarray(state_ids, dtype=np.int64),
        centers=np.asarray(centers, dtype=float).reshape((-1, 2)),
        grid_indices=np.asarray(grid_indices, dtype=np.int64).reshape((-1, 2)),
        grid_to_state=grid_to_state,
        walkable_mask=walkable_mask,
        non_walkable_classes=non_walkable_classes,
    )


def build_walkable_region(
    scene: ProcessedSceneRecord,
    non_walkable_classes: tuple[str, ...] = NON_WALKABLE_CLASSES,
) -> Polygon | MultiPolygon:
    bounds = scene.bounds
    scene_area = box(bounds["min_x"], bounds["min_y"], bounds["max_x"], bounds["max_y"])
    non_walkable_polygons = []
    for polygon_class in non_walkable_classes:
        for vertices in scene.polygons.get(polygon_class, []):
            if len(vertices) < 3:
                continue
            polygon = Polygon(vertices)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if not polygon.is_empty:
                non_walkable_polygons.append(polygon)

    if not non_walkable_polygons:
        return scene_area

    non_walkable = unary_union(non_walkable_polygons)
    if not non_walkable.is_valid:
        non_walkable = non_walkable.buffer(0)
    walkable = scene_area.difference(non_walkable)
    if not walkable.is_valid:
        walkable = walkable.buffer(0)
    return walkable


def map_trajectories_to_states(
    scene: ProcessedSceneRecord,
    state_space: GridStateSpace,
) -> dict[int, np.ndarray]:
    return {
        track_id: map_points_to_states(points, state_space)
        for track_id, points in scene.trajectories.items()
    }


def map_points_to_states(
    points: np.ndarray,
    state_space: GridStateSpace,
    *,
    snap_distance: float | None = None,
) -> np.ndarray:
    bounds = state_space.bounds
    col = np.floor((points[:, 0] - bounds["min_x"]) / state_space.cell_size).astype(np.int64)
    row = np.floor((points[:, 1] - bounds["min_y"]) / state_space.cell_size).astype(np.int64)
    valid = (
        (row >= 0)
        & (row < state_space.rows)
        & (col >= 0)
        & (col < state_space.cols)
    )
    states = np.full(points.shape[0], -1, dtype=np.int64)
    states[valid] = state_space.grid_to_state[row[valid], col[valid]]
    if snap_distance is not None:
        if not np.isfinite(snap_distance) or snap_distance < 0:
            raise ValueError("snap_distance must be finite and non-negative")
        if snap_distance > 0 and state_space.centers.shape[0] > 0:
            invalid_indices = np.flatnonzero(states < 0)
            max_distance_sq = snap_distance * snap_distance
            for point_index in invalid_indices:
                distances_sq = np.sum((state_space.centers - points[point_index]) ** 2, axis=1)
                nearest_state = int(np.argmin(distances_sq))
                if distances_sq[nearest_state] <= max_distance_sq:
                    states[point_index] = nearest_state
    return states


def split_tracks(
    track_ids: list[int],
    *,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> dict[str, list[int]]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be less than 1")

    rng = np.random.default_rng(seed)
    shuffled = np.asarray(track_ids, dtype=np.int64)
    rng.shuffle(shuffled)
    train_end = int(round(len(shuffled) * train_fraction))
    val_end = train_end + int(round(len(shuffled) * val_fraction))
    return {
        "train": sorted(int(v) for v in shuffled[:train_end]),
        "val": sorted(int(v) for v in shuffled[train_end:val_end]),
        "test": sorted(int(v) for v in shuffled[val_end:]),
    }


def build_8way_distance_graph(state_space: GridStateSpace) -> sparse.csr_matrix:
    rows, cols = state_space.rows, state_space.cols
    state_count = len(state_space.state_ids)

    row_indices = []
    col_indices = []
    values = []

    grid_to_state = state_space.grid_to_state
    cell_size = state_space.cell_size
    diag_dist = cell_size * np.sqrt(2)

    directions = [
        (-1, 0, cell_size),
        (1, 0, cell_size),
        (0, -1, cell_size),
        (0, 1, cell_size),
        (-1, -1, diag_dist),
        (-1, 1, diag_dist),
        (1, -1, diag_dist),
        (1, 1, diag_dist),
    ]

    for state_i, (r, c) in enumerate(state_space.grid_indices):
        for dr, dc, weight in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                state_j = grid_to_state[nr, nc]
                if state_j >= 0:
                    row_indices.append(state_i)
                    col_indices.append(state_j)
                    values.append(weight)

    return sparse.coo_matrix(
        (values, (row_indices, col_indices)),
        shape=(state_count, state_count),
    ).tocsr()


def fit_destination_classes(
    scene: ProcessedSceneRecord,
    state_space: GridStateSpace,
    train_track_ids: list[int],
    *,
    radius: float,
    min_samples: int,
    snap_distance: float | None,
) -> DestinationClasses:
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("destination radius must be finite and positive")
    if min_samples < 1:
        raise ValueError("destination min samples must be >= 1")

    endpoints = []
    valid_track_ids = []
    for track_id in train_track_ids:
        trajectory = scene.trajectories[track_id]
        if trajectory.shape[0] == 0:
            continue
        endpoints.append(trajectory[-1])
        valid_track_ids.append(track_id)

    track_to_class: dict[int, int] = {}
    endpoints_array = np.asarray(endpoints, dtype=float)
    if endpoints_array.size == 0:
        return DestinationClasses(
            radius=float(radius),
            centers=np.zeros((0, 2), dtype=float),
            train_counts=np.zeros(0, dtype=np.int64),
            member_endpoints=tuple(),
            member_state_ids=tuple(),
            class_radii=np.zeros(0, dtype=float),
            track_to_class=track_to_class,
        )

    if snap_distance is None:
        snap_distance = state_space.cell_size

    endpoint_states = map_points_to_states(
        endpoints_array,
        state_space,
        snap_distance=snap_distance,
    )
    valid_mask = endpoint_states >= 0
    valid_endpoint_states = endpoint_states[valid_mask]

    for i, track_id in enumerate(valid_track_ids):
        if not valid_mask[i]:
            track_to_class[track_id] = -1

    if valid_endpoint_states.size == 0:
        return DestinationClasses(
            radius=float(radius),
            centers=np.zeros((0, 2), dtype=float),
            train_counts=np.zeros(0, dtype=np.int64),
            member_endpoints=tuple(),
            member_state_ids=tuple(),
            class_radii=np.zeros(0, dtype=float),
            track_to_class=track_to_class,
        )

    graph = build_8way_distance_graph(state_space)

    distances_from_endpoints = dijkstra(
        csgraph=graph,
        directed=False,
        indices=valid_endpoint_states,
        limit=radius,
    )
    pairwise_distances = distances_from_endpoints[:, valid_endpoint_states]
    if not np.all(np.isfinite(pairwise_distances)):
        disconnected_distance = radius + max(
            state_space.cell_size,
            np.finfo(np.float64).eps * max(1.0, radius),
        )
        pairwise_distances = np.nan_to_num(
            pairwise_distances,
            nan=disconnected_distance,
            posinf=disconnected_distance,
            neginf=disconnected_distance,
        )
    endpoint_distances = np.linalg.norm(
        endpoints_array[valid_mask][:, np.newaxis, :] - endpoints_array[valid_mask][np.newaxis, :, :],
        axis=2,
    )
    pairwise_distances = np.maximum(pairwise_distances, endpoint_distances)

    if valid_endpoint_states.size == 1:
        labels = np.zeros(1, dtype=np.int64)
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=radius,
            metric="precomputed",
            linkage="complete",
        )
        labels = clustering.fit_predict(pairwise_distances)

    label_to_endpoints: dict[int, list[np.ndarray]] = {}
    label_to_states: dict[int, list[int]] = {}
    label_counts: dict[int, int] = {}
    for label in labels:
        label = int(label)
        label_counts[label] = label_counts.get(label, 0) + 1

    valid_idx = 0
    for i, track_id in enumerate(valid_track_ids):
        if not valid_mask[i]:
            continue

        label = int(labels[valid_idx])
        state_id = int(valid_endpoint_states[valid_idx])
        valid_idx += 1

        if label_counts[label] < min_samples:
            track_to_class[track_id] = -1
        else:
            label_to_endpoints.setdefault(label, []).append(endpoints_array[i])
            label_to_states.setdefault(label, []).append(state_id)
            track_to_class[track_id] = label

    centers = []
    ordered = sorted(
        label_to_endpoints.items(),
        key=lambda item: (np.mean(item[1], axis=0)[0], np.mean(item[1], axis=0)[1]),
    )

    remap = {old_label: new_label for new_label, (old_label, _) in enumerate(ordered)}

    counts = np.zeros(len(ordered), dtype=np.int64)
    member_endpoints = []
    member_state_ids = []
    class_radii = np.zeros(len(ordered), dtype=float)
    for old_label, pts in ordered:
        new_label = remap[old_label]
        endpoints_for_label = np.asarray(pts, dtype=float).reshape((-1, 2))
        center = np.mean(endpoints_for_label, axis=0)
        centers.append(center)
        counts[new_label] = len(pts)
        class_radii[new_label] = endpoint_radius(endpoints_for_label, center)
        member_endpoints.append(endpoints_for_label)
        member_state_ids.append(np.asarray(label_to_states[old_label], dtype=np.int64))

    for track_id, old_label in track_to_class.items():
        if old_label != -1:
            track_to_class[track_id] = remap[old_label]

    ordered_centers = np.asarray(centers, dtype=float).reshape((-1, 2))

    return DestinationClasses(
        radius=float(radius),
        centers=ordered_centers,
        train_counts=counts,
        member_endpoints=tuple(member_endpoints),
        member_state_ids=tuple(member_state_ids),
        class_radii=class_radii,
        track_to_class=track_to_class,
        pre_merge_class_count=len(ordered_centers),
    )


def endpoint_radius(endpoints: np.ndarray, center: np.ndarray) -> float:
    endpoints = np.asarray(endpoints, dtype=float).reshape((-1, 2))
    if endpoints.shape[0] == 0:
        return 0.0
    center = np.asarray(center, dtype=float).reshape((1, 2))
    return float(np.max(np.linalg.norm(endpoints - center, axis=1)))


def merge_adjacent_destination_classes(
    destination_classes: DestinationClasses,
    *,
    epsilon: float,
    epsilon_meters: float | None,
) -> DestinationClasses:
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("destination merge epsilon must be finite and non-negative")

    class_count = len(destination_classes.centers)
    pre_merge_class_count = (
        destination_classes.pre_merge_class_count
        if destination_classes.pre_merge_class_count is not None
        else class_count
    )
    class_radii = destination_class_radii(destination_classes)
    if class_count <= 1:
        return DestinationClasses(
            radius=destination_classes.radius,
            centers=destination_classes.centers,
            train_counts=destination_classes.train_counts,
            member_endpoints=destination_classes.member_endpoints,
            member_state_ids=destination_classes.member_state_ids,
            track_to_class=dict(destination_classes.track_to_class),
            class_radii=class_radii,
            merge_epsilon=float(epsilon),
            merge_epsilon_meters=epsilon_meters,
            pre_merge_class_count=pre_merge_class_count,
        )

    components = destination_class_merge_components(
        centers=destination_classes.centers,
        radii=class_radii,
        max_radius=destination_classes.radius,
        epsilon=epsilon,
    )

    if all(len(component) == 1 and component[0] == i for i, component in enumerate(components)):
        return DestinationClasses(
            radius=destination_classes.radius,
            centers=destination_classes.centers,
            train_counts=destination_classes.train_counts,
            member_endpoints=destination_classes.member_endpoints,
            member_state_ids=destination_classes.member_state_ids,
            class_radii=class_radii,
            track_to_class=dict(destination_classes.track_to_class),
            merge_epsilon=float(epsilon),
            merge_epsilon_meters=epsilon_meters,
            pre_merge_class_count=pre_merge_class_count,
        )

    merged = []
    old_to_component: dict[int, int] = {}
    for component_id, component in enumerate(components):
        endpoints = np.vstack([destination_classes.member_endpoints[i] for i in component])
        states = np.concatenate([destination_classes.member_state_ids[i] for i in component])
        center = np.mean(endpoints, axis=0)
        merged.append(
            {
                "component_id": component_id,
                "old_class_ids": component,
                "center": center,
                "radius": endpoint_radius(endpoints, center),
                "count": int(sum(destination_classes.train_counts[i] for i in component)),
                "endpoints": endpoints,
                "states": states.astype(np.int64, copy=False),
            }
        )
        for old_class_id in component:
            old_to_component[old_class_id] = component_id

    merged.sort(key=lambda item: (item["center"][0], item["center"][1]))
    component_to_new = {
        int(item["component_id"]): new_class_id
        for new_class_id, item in enumerate(merged)
    }
    track_to_class = {}
    for track_id, old_class_id in destination_classes.track_to_class.items():
        if old_class_id < 0:
            track_to_class[track_id] = -1
        else:
            track_to_class[track_id] = component_to_new[old_to_component[int(old_class_id)]]

    return DestinationClasses(
        radius=destination_classes.radius,
        centers=np.asarray([item["center"] for item in merged], dtype=float).reshape((-1, 2)),
        train_counts=np.asarray([item["count"] for item in merged], dtype=np.int64),
        member_endpoints=tuple(item["endpoints"] for item in merged),
        member_state_ids=tuple(item["states"] for item in merged),
        class_radii=np.asarray([item["radius"] for item in merged], dtype=float),
        track_to_class=track_to_class,
        merge_epsilon=float(epsilon),
        merge_epsilon_meters=epsilon_meters,
        pre_merge_class_count=pre_merge_class_count,
    )


def destination_class_radii(destination_classes: DestinationClasses) -> np.ndarray:
    if (
        destination_classes.class_radii is not None
        and destination_classes.class_radii.shape == (len(destination_classes.centers),)
    ):
        return destination_classes.class_radii.astype(float, copy=True)
    return np.asarray(
        [
            endpoint_radius(endpoints, center)
            for endpoints, center in zip(
                destination_classes.member_endpoints,
                destination_classes.centers,
            )
        ],
        dtype=float,
    )


def destination_class_merge_components(
    *,
    centers: np.ndarray,
    radii: np.ndarray,
    max_radius: float,
    epsilon: float,
) -> list[list[int]]:
    if not np.isfinite(max_radius) or max_radius < 0:
        raise ValueError("destination merge max_radius must be finite and non-negative")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("destination merge epsilon must be finite and non-negative")
    class_count = len(centers)
    parent = list(range(class_count))
    capped_radii = np.minimum(np.asarray(radii, dtype=float), float(max_radius))
    if capped_radii.shape != (class_count,) or not np.all(np.isfinite(capped_radii)):
        raise ValueError("destination merge radii must be finite and match centers")

    def find(class_id: int) -> int:
        while parent[class_id] != class_id:
            parent[class_id] = parent[parent[class_id]]
            class_id = parent[class_id]
        return class_id

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    centers = np.asarray(centers, dtype=float).reshape((-1, 2))
    for i in range(class_count):
        for j in range(i + 1, class_count):
            center_distance = float(np.linalg.norm(centers[i] - centers[j]))
            region_gap = center_distance - capped_radii[i] - capped_radii[j]
            tolerance = np.finfo(np.float64).eps * max(
                1.0,
                abs(center_distance),
                abs(float(capped_radii[i])),
                abs(float(capped_radii[j])),
                abs(float(epsilon)),
            )
            if region_gap <= epsilon + tolerance:
                union(i, j)

    root_to_members: dict[int, list[int]] = {}
    for class_id in range(class_count):
        root_to_members.setdefault(find(class_id), []).append(class_id)

    return sorted(
        (members for members in root_to_members.values()),
        key=lambda members: (centers[members].mean(axis=0)[0], centers[members].mean(axis=0)[1]),
    )


def assign_destination_classes(
    scene: ProcessedSceneRecord,
    state_space: GridStateSpace,
    destination_classes: DestinationClasses,
    *,
    max_distance: float,
    snap_distance: float | None,
) -> dict[int, int]:
    track_to_class: dict[int, int] = dict(destination_classes.track_to_class)
    if not np.isfinite(max_distance) or max_distance <= 0:
        raise ValueError("destination assignment distance must be finite and positive")

    if destination_classes.centers.shape[0] == 0:
        return {
            track_id: track_to_class.get(track_id, -1)
            for track_id in scene.trajectories
        }

    endpoints = []
    valid_track_ids = []
    for track_id, trajectory in scene.trajectories.items():
        if track_id in track_to_class:
            continue
        if trajectory.shape[0] == 0:
            track_to_class[track_id] = -1
            continue
        endpoints.append(trajectory[-1])
        valid_track_ids.append(track_id)

    if not endpoints:
        return track_to_class

    if snap_distance is None:
        snap_distance = state_space.cell_size

    endpoint_states = map_points_to_states(
        np.asarray(endpoints, dtype=float),
        state_space,
        snap_distance=snap_distance,
    )
    valid_mask = endpoint_states >= 0
    valid_endpoint_states = endpoint_states[valid_mask]
    for i, track_id in enumerate(valid_track_ids):
        if not valid_mask[i]:
            track_to_class[track_id] = -1

    if valid_endpoint_states.size == 0:
        return track_to_class

    graph = build_8way_distance_graph(state_space)
    distances_from_endpoints = dijkstra(
        csgraph=graph,
        directed=False,
        indices=valid_endpoint_states,
        limit=max_distance,
    )

    working_member_endpoints = [points.copy() for points in destination_classes.member_endpoints]
    working_member_states = [states.copy() for states in destination_classes.member_state_ids]

    valid_idx = 0
    for i, track_id in enumerate(valid_track_ids):
        if not valid_mask[i]:
            continue

        distances = distances_from_endpoints[valid_idx]
        endpoint = endpoints[i]
        valid_idx += 1
        best_class = -1
        best_distance = np.inf
        for class_id, (member_endpoints, member_states) in enumerate(
            zip(working_member_endpoints, working_member_states)
        ):
            if member_states.size == 0:
                continue
            graph_distances = distances[member_states]
            euclidean_distances = np.linalg.norm(member_endpoints - endpoint, axis=1)
            member_distances = np.maximum(graph_distances, euclidean_distances)
            if not np.all(np.isfinite(member_distances)):
                continue
            class_distance = float(np.max(member_distances))
            if class_distance <= max_distance and class_distance < best_distance:
                best_class = class_id
                best_distance = class_distance
        track_to_class[track_id] = best_class
        if best_class >= 0:
            working_member_endpoints[best_class] = np.vstack(
                [working_member_endpoints[best_class], endpoint.reshape(1, 2)]
            )
            working_member_states[best_class] = np.append(
                working_member_states[best_class],
                valid_endpoint_states[valid_idx - 1],
            )

    return track_to_class


def learn_transition_model(
    *,
    state_space: GridStateSpace,
    destination_classes: DestinationClasses,
    trajectory_states: dict[int, np.ndarray],
    train_track_ids: list[int],
    track_to_class: dict[int, int],
    transition_min_support: int,
    global_goal_tau_meters: float,
    global_goal_tau: float,
    map_goal_tau_meters: float,
    map_goal_tau: float,
    endpoint_snap_distance: float | None,
    trajectory_stride: int = 1,
) -> dict[str, Any]:
    if trajectory_stride < 1:
        raise ValueError("trajectory_stride must be >= 1")
    if transition_min_support <= 0:
        raise ValueError("transition_min_support must be positive")
    if not np.isfinite(global_goal_tau) or global_goal_tau <= 0:
        raise ValueError("global_goal_tau must be finite and positive")
    if not np.isfinite(map_goal_tau) or map_goal_tau <= 0:
        raise ValueError("map_goal_tau must be finite and positive")

    state_count = len(state_space.state_ids)
    class_count = len(destination_classes.centers)
    global_counts = sparse.dok_matrix((state_count, state_count), dtype=np.float64)
    global_visits = np.zeros(state_count, dtype=np.float64)
    class_counts = [
        sparse.dok_matrix((state_count, state_count), dtype=np.float64)
        for _ in range(class_count)
    ]
    class_visits = np.zeros((class_count, state_count), dtype=np.float64)

    for track_id in train_track_ids:
        states = trajectory_states.get(track_id)
        if states is None:
            continue
        states = states[::trajectory_stride]
        destination_class = track_to_class.get(track_id, -1)
        for source, target in zip(states[:-1], states[1:]):
            if source < 0 or target < 0:
                continue
            source = int(source)
            target = int(target)
            global_counts[source, target] += 1.0
            global_visits[source] += 1.0
            if 0 <= destination_class < class_count:
                class_counts[destination_class][source, target] += 1.0
                class_visits[destination_class, source] += 1.0

    global_counts_csr = global_counts.tocsr()
    global_empirical_transition = row_normalize_zero(global_counts_csr)
    global_transition = row_normalize_self_loop(global_counts_csr)
    class_counts_csr = [counts.tocsr() for counts in class_counts]
    class_empirical_transitions = [row_normalize_zero(counts) for counts in class_counts_csr]

    graph = build_8way_distance_graph(state_space)
    goal_state_ids = destination_goal_state_ids(
        state_space,
        destination_classes,
        endpoint_snap_distance=endpoint_snap_distance,
    )
    goal_distance_fields = goal_distance_fields_for_classes(
        graph,
        state_count=state_count,
        goal_state_ids=goal_state_ids,
    )
    class_global_transitions = [
        class_conditioned_global_transition(
            global_empirical_transition,
            goal_distances=goal_distance_fields[class_id],
            tau=global_goal_tau,
        )
        for class_id in range(class_count)
    ]
    class_map_transitions = [
        class_topological_transition(
            graph,
            goal_distances=goal_distance_fields[class_id],
            tau=map_goal_tau,
        )
        for class_id in range(class_count)
    ]
    class_transitions = [
        combine_transition_layers(
            class_transition=class_empirical_transitions[class_id],
            global_transition=class_global_transitions[class_id],
            map_transition=class_map_transitions[class_id],
            class_visits=class_visits[class_id],
            global_visits=global_visits,
            transition_min_support=transition_min_support,
        )
        for class_id in range(class_count)
    ]
    for class_id, transition in enumerate(class_transitions):
        validate_row_stochastic(transition, label=f"class {class_id} transition")

    return {
        "global_counts": global_counts_csr,
        "global_visits": global_visits,
        "global_empirical_transition": global_empirical_transition,
        "global_transition": global_transition,
        "class_counts": class_counts_csr,
        "class_visits": class_visits,
        "class_empirical_transitions": class_empirical_transitions,
        "class_global_transitions": class_global_transitions,
        "class_map_transitions": class_map_transitions,
        "class_transitions": class_transitions,
        "goal_state_ids": goal_state_ids,
        "transition_min_support": int(transition_min_support),
        "global_goal_tau_meters": float(global_goal_tau_meters),
        "global_goal_tau": float(global_goal_tau),
        "map_goal_tau_meters": float(map_goal_tau_meters),
        "map_goal_tau": float(map_goal_tau),
    }


def row_normalize_zero(counts: sparse.csr_matrix) -> sparse.csr_matrix:
    counts = counts.tocsr()
    row_sums = np.asarray(counts.sum(axis=1)).ravel()
    nonzero_rows = row_sums > 0
    inv = np.zeros_like(row_sums, dtype=np.float64)
    inv[nonzero_rows] = 1.0 / row_sums[nonzero_rows]
    return sparse.diags(inv).dot(counts).tocsr()


def row_normalize_self_loop(counts: sparse.csr_matrix) -> sparse.csr_matrix:
    transition = row_normalize_zero(counts).tolil()
    row_sums = np.asarray(counts.sum(axis=1)).ravel()
    zero_rows = np.flatnonzero(row_sums <= 0)
    for row in zero_rows:
        transition[row, row] = 1.0
    return transition.tocsr()


def destination_goal_state_ids(
    state_space: GridStateSpace,
    destination_classes: DestinationClasses,
    *,
    endpoint_snap_distance: float | None,
) -> np.ndarray:
    if len(destination_classes.centers) == 0:
        return np.zeros(0, dtype=np.int64)
    snap_distance = state_space.cell_size if endpoint_snap_distance is None else endpoint_snap_distance
    goal_state_ids = map_points_to_states(
        destination_classes.centers,
        state_space,
        snap_distance=snap_distance,
    )
    invalid = np.flatnonzero(goal_state_ids < 0)
    if invalid.size:
        raise ValueError(
            f"Could not map destination class centers to walkable states for "
            f"scene {state_space.scene_id}: classes {invalid.tolist()}"
        )
    return goal_state_ids.astype(np.int64, copy=False)


def goal_distance_fields_for_classes(
    graph: sparse.csr_matrix,
    *,
    state_count: int,
    goal_state_ids: np.ndarray,
) -> list[np.ndarray]:
    if goal_state_ids.size == 0:
        return []
    distances = dijkstra(
        csgraph=graph,
        directed=False,
        indices=goal_state_ids,
    )
    distances = np.asarray(distances, dtype=np.float64).reshape((goal_state_ids.size, state_count))
    return [distances[class_id] for class_id in range(goal_state_ids.size)]


def class_conditioned_global_transition(
    global_transition: sparse.csr_matrix,
    *,
    goal_distances: np.ndarray,
    tau: float,
) -> sparse.csr_matrix:
    global_transition = global_transition.tocsr()
    data = []
    indices = []
    indptr = [0]
    for source in range(global_transition.shape[0]):
        row_start = global_transition.indptr[source]
        row_end = global_transition.indptr[source + 1]
        targets = global_transition.indices[row_start:row_end]
        probabilities = global_transition.data[row_start:row_end]
        source_distance = goal_distances[source]
        if targets.size == 0 or not np.isfinite(source_distance):
            indptr.append(len(data))
            continue

        target_distances = goal_distances[targets]
        penalties = np.exp(-np.maximum(0.0, target_distances - source_distance) / tau)
        penalties[~np.isfinite(target_distances)] = 0.0
        weighted = probabilities * penalties
        total = float(weighted.sum())
        if total > 0.0:
            keep = weighted > 0.0
            indices.extend(int(target) for target in targets[keep])
            data.extend(float(value) for value in weighted[keep] / total)
        indptr.append(len(data))
    return sparse.csr_matrix(
        (
            np.asarray(data, dtype=np.float64),
            np.asarray(indices, dtype=np.int64),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=global_transition.shape,
    )


def class_topological_transition(
    graph: sparse.csr_matrix,
    *,
    goal_distances: np.ndarray,
    tau: float,
) -> sparse.csr_matrix:
    graph = graph.tocsr()
    data = []
    indices = []
    indptr = [0]
    for source in range(graph.shape[0]):
        row_start = graph.indptr[source]
        row_end = graph.indptr[source + 1]
        neighbors = graph.indices[row_start:row_end]
        neighbor_distances = goal_distances[neighbors]
        finite = np.isfinite(neighbor_distances)
        if np.any(finite):
            finite_neighbors = neighbors[finite]
            finite_distances = neighbor_distances[finite]
            min_distance = float(finite_distances.min())
            weights = np.exp(-(finite_distances - min_distance) / tau)
            total = float(weights.sum())
            indices.extend(int(target) for target in finite_neighbors)
            data.extend(float(value) for value in weights / total)
        else:
            indices.append(source)
            data.append(1.0)
        indptr.append(len(data))
    return sparse.csr_matrix(
        (
            np.asarray(data, dtype=np.float64),
            np.asarray(indices, dtype=np.int64),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=graph.shape,
    )


def combine_transition_layers(
    *,
    class_transition: sparse.csr_matrix,
    global_transition: sparse.csr_matrix,
    map_transition: sparse.csr_matrix,
    class_visits: np.ndarray,
    global_visits: np.ndarray,
    transition_min_support: int,
) -> sparse.csr_matrix:
    class_transition = class_transition.tocsr()
    global_transition = global_transition.tocsr()
    map_transition = map_transition.tocsr()
    class_lambda = class_visits / (class_visits + float(transition_min_support))
    global_lambda = global_visits / (global_visits + float(transition_min_support))
    global_row_sums = np.asarray(global_transition.sum(axis=1)).ravel()
    global_available = global_row_sums > 0.0

    global_weight = (1.0 - class_lambda) * global_lambda * global_available
    map_weight = (1.0 - class_lambda) * (1.0 - global_lambda)
    map_weight += (1.0 - class_lambda) * global_lambda * (~global_available)

    return (
        sparse.diags(class_lambda).dot(class_transition)
        + sparse.diags(global_weight).dot(global_transition)
        + sparse.diags(map_weight).dot(map_transition)
    ).tocsr()


def validate_row_stochastic(
    transition: sparse.csr_matrix,
    *,
    label: str,
    atol: float = 1.0e-9,
) -> None:
    row_sums = np.asarray(transition.sum(axis=1)).ravel()
    if row_sums.size and not np.allclose(row_sums, 1.0, atol=atol):
        bad_rows = np.flatnonzero(~np.isclose(row_sums, 1.0, atol=atol))
        raise ValueError(
            f"{label} has {bad_rows.size} non-stochastic rows; "
            f"first_bad_row={int(bad_rows[0])}, row_sum={row_sums[bad_rows[0]]:.12g}"
        )


def write_state_space(state_space: GridStateSpace, output_root: Path) -> None:
    np.savez_compressed(
        output_root / "state_space.npz",
        state_ids=state_space.state_ids,
        centers=state_space.centers,
        grid_indices=state_space.grid_indices,
        grid_to_state=state_space.grid_to_state,
        walkable_mask=state_space.walkable_mask,
    )
    metadata = {
        "scene_id": state_space.scene_id,
        "cell_size": state_space.cell_size,
        "grid_size_meters": state_space.cell_size_meters,
        "scene_scale": state_space.scene_scale,
        "bounds": state_space.bounds,
        "rows": state_space.rows,
        "cols": state_space.cols,
        "state_count": int(len(state_space.state_ids)),
        "non_walkable_classes": list(state_space.non_walkable_classes),
    }
    (output_root / "state_space.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def write_trajectory_states(trajectory_states: dict[int, np.ndarray], path: Path) -> None:
    arrays = {f"track_{track_id}": states for track_id, states in sorted(trajectory_states.items())}
    track_ids = np.array(sorted(trajectory_states), dtype=np.int64)
    np.savez_compressed(path, track_ids=track_ids, **arrays)


def write_splits(splits: dict[str, list[int]], path: Path) -> None:
    path.write_text(json.dumps(splits, indent=2), encoding="utf-8")


def write_destination_classes(destination_classes: DestinationClasses, path: Path) -> None:
    records = []
    class_radii = destination_class_radii(destination_classes)
    for class_id, center in enumerate(destination_classes.centers):
        records.append(
            {
                "class_id": class_id,
                "center": center.tolist(),
                "radius": float(class_radii[class_id]),
                "train_count": int(destination_classes.train_counts[class_id]),
            }
        )
    payload = {
        "radius": destination_classes.radius,
        "merge": {
            "enabled": destination_classes.merge_epsilon is not None,
            "epsilon": destination_classes.merge_epsilon,
            "epsilon_meters": destination_classes.merge_epsilon_meters,
            "pre_merge_class_count": destination_classes.pre_merge_class_count,
            "post_merge_class_count": int(len(destination_classes.centers)),
        },
        "classes": records,
        "track_to_class": {
            str(track_id): int(class_id)
            for track_id, class_id in sorted(destination_classes.track_to_class.items())
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_transition_model(transition_model: dict[str, Any], output_root: Path) -> None:
    transitions_root = output_root / "transitions"
    transitions_root.mkdir(exist_ok=True)
    for stale_path in transitions_root.glob("class_*.npz"):
        stale_path.unlink()
    sparse.save_npz(transitions_root / "global_counts.npz", transition_model["global_counts"])
    sparse.save_npz(transitions_root / "global_transition.npz", transition_model["global_transition"])
    np.savez_compressed(
        transitions_root / "visits.npz",
        global_visits=transition_model["global_visits"],
        class_visits=transition_model["class_visits"],
        goal_state_ids=transition_model["goal_state_ids"],
    )
    for class_id, counts in enumerate(transition_model["class_counts"]):
        sparse.save_npz(transitions_root / f"class_{class_id:03d}_counts.npz", counts)
    for class_id, transition in enumerate(transition_model["class_empirical_transitions"]):
        sparse.save_npz(
            transitions_root / f"class_{class_id:03d}_class_transition.npz",
            transition,
        )
    for class_id, transition in enumerate(transition_model["class_global_transitions"]):
        sparse.save_npz(
            transitions_root / f"class_{class_id:03d}_global_conditioned_transition.npz",
            transition,
        )
    for class_id, transition in enumerate(transition_model["class_map_transitions"]):
        sparse.save_npz(transitions_root / f"class_{class_id:03d}_map_transition.npz", transition)
    for class_id, transition in enumerate(transition_model["class_transitions"]):
        sparse.save_npz(transitions_root / f"class_{class_id:03d}_transition.npz", transition)


def write_model_metadata(
    scene: ProcessedSceneRecord,
    state_space: GridStateSpace,
    destination_classes: DestinationClasses,
    transition_model: dict[str, Any],
    splits: dict[str, list[int]],
    path: Path,
    *,
    trajectory_stride: int,
) -> None:
    global_transition = transition_model["global_transition"]
    row_sums = np.asarray(global_transition.sum(axis=1)).ravel()
    metadata = {
        "scene_id": scene.scene_id,
        "scene_units": scene.units,
        "scene_scale": state_space.scene_scale,
        "grid": {
            "grid_size_meters": state_space.cell_size_meters,
            "cell_size": state_space.cell_size,
            "rows": state_space.rows,
            "cols": state_space.cols,
            "state_count": int(len(state_space.state_ids)),
            "non_walkable_classes": list(state_space.non_walkable_classes),
        },
        "destination_classes": {
            "radius_meters": (
                destination_classes.radius / state_space.scene_scale
                if state_space.scene_scale > 0
                else None
            ),
            "radius": destination_classes.radius,
            "merge_enabled": destination_classes.merge_epsilon is not None,
            "merge_epsilon_meters": destination_classes.merge_epsilon_meters,
            "merge_epsilon": destination_classes.merge_epsilon,
            "pre_merge_class_count": destination_classes.pre_merge_class_count,
            "class_count": int(len(destination_classes.centers)),
            "unassigned_track_count": int(
                sum(1 for class_id in destination_classes.track_to_class.values() if class_id < 0)
            ),
        },
        "splits": {key: len(value) for key, value in splits.items()},
        "transitions": {
            "trajectory_stride": trajectory_stride,
            "transition_min_support": int(transition_model["transition_min_support"]),
            "global_goal_tau_meters": transition_model["global_goal_tau_meters"],
            "global_goal_tau": transition_model["global_goal_tau"],
            "map_goal_tau_meters": transition_model["map_goal_tau_meters"],
            "map_goal_tau": transition_model["map_goal_tau"],
            "global_nonzero_count": int(transition_model["global_counts"].nnz),
            "global_visit_count": float(transition_model["global_visits"].sum()),
            "global_visited_state_count": int(np.count_nonzero(transition_model["global_visits"])),
            "class_nonzero_counts": [int(counts.nnz) for counts in transition_model["class_counts"]],
            "class_visit_counts": [
                float(transition_model["class_visits"][class_id].sum())
                for class_id in range(transition_model["class_visits"].shape[0])
            ],
            "goal_state_ids": [int(value) for value in transition_model["goal_state_ids"]],
            "min_global_row_sum": float(row_sums.min()) if row_sums.size else 0.0,
            "max_global_row_sum": float(row_sums.max()) if row_sums.size else 0.0,
        },
        "files": {
            "state_space": "state_space.npz",
            "state_space_metadata": "state_space.json",
            "trajectory_states": "trajectory_states.npz",
            "splits": "splits.json",
            "destination_classes": "destination_classes.json",
            "transitions": "transitions/",
            "transition_visits": "transitions/visits.npz",
        },
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "scene_id",
        "scene_scale",
        "grid_size_meters",
        "cell_size",
        "destination_radius_meters",
        "destination_radius",
        "destination_merge_enabled",
        "destination_merge_epsilon_meters",
        "destination_merge_epsilon",
        "pre_merge_destination_classes",
        "state_count",
        "rows",
        "cols",
        "train_tracks",
        "val_tracks",
        "test_tracks",
        "destination_classes",
        "global_nonzero",
        "transition_min_support",
        "global_goal_tau_meters",
        "global_goal_tau",
        "map_goal_tau_meters",
        "map_goal_tau",
        "unassigned_tracks",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
