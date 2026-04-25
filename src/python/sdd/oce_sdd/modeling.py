from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
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
    track_to_class: dict[int, int]


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

        state_space = build_grid_state_space(
            scene,
            cell_size=args.grid_size,
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
            splits["train"],
            radius=args.destination_radius,
        )
        track_to_class = assign_destination_classes(
            scene,
            destination_classes,
            max_distance=args.destination_radius,
        )
        destination_classes = DestinationClasses(
            radius=destination_classes.radius,
            centers=destination_classes.centers,
            train_counts=destination_classes.train_counts,
            track_to_class=track_to_class,
        )

        transition_model = learn_transition_model(
            state_count=len(state_space.state_ids),
            trajectory_states=trajectory_states,
            train_track_ids=splits["train"],
            track_to_class=track_to_class,
            class_count=len(destination_classes.centers),
            trajectory_stride=args.trajectory_stride,
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
                "cell_size": args.grid_size,
                "state_count": len(state_space.state_ids),
                "rows": state_space.rows,
                "cols": state_space.cols,
                "train_tracks": len(splits["train"]),
                "val_tracks": len(splits["val"]),
                "test_tracks": len(splits["test"]),
                "destination_classes": len(destination_classes.centers),
                "global_nonzero": int(transition_model["global_counts"].nnz),
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
        help="Uniform grid cell size in normalized scene units.",
    )
    parser.add_argument(
        "--destination-radius",
        type=float,
        default=2.0,
        help="Maximum destination-cluster radius in scene units.",
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
    return parser.parse_args()


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
    non_walkable_classes: tuple[str, ...] = NON_WALKABLE_CLASSES,
) -> GridStateSpace:
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")

    bounds = scene.bounds
    width = bounds["max_x"] - bounds["min_x"]
    height = bounds["max_y"] - bounds["min_y"]
    cols = int(np.ceil(width / cell_size))
    rows = int(np.ceil(height / cell_size))
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


def map_points_to_states(points: np.ndarray, state_space: GridStateSpace) -> np.ndarray:
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


def fit_destination_classes(
    scene: ProcessedSceneRecord,
    train_track_ids: list[int],
    *,
    radius: float,
) -> DestinationClasses:
    if radius <= 0:
        raise ValueError("destination radius must be positive")

    endpoints = []
    valid_track_ids = []
    for track_id in train_track_ids:
        trajectory = scene.trajectories[track_id]
        if trajectory.shape[0] == 0:
            continue
        endpoints.append(trajectory[-1])
        valid_track_ids.append(track_id)

    centers: list[np.ndarray] = []
    member_endpoints: list[list[np.ndarray]] = []
    members: list[list[int]] = []
    for track_id, endpoint in sorted(zip(valid_track_ids, endpoints), key=lambda item: (item[1][0], item[1][1])):
        best_index = -1
        best_distance = np.inf
        for index, center in enumerate(centers):
            candidate_endpoints = [*member_endpoints[index], np.asarray(endpoint, dtype=float)]
            candidate_center = np.asarray(candidate_endpoints).mean(axis=0)
            candidate_distances = np.linalg.norm(
                np.asarray(candidate_endpoints) - candidate_center,
                axis=1,
            )
            distance = float(np.linalg.norm(endpoint - candidate_center))
            if float(np.max(candidate_distances)) <= radius and distance < best_distance:
                best_index = index
                best_distance = distance
        if best_index < 0:
            centers.append(np.asarray(endpoint, dtype=float).copy())
            member_endpoints.append([np.asarray(endpoint, dtype=float).copy()])
            members.append([track_id])
        else:
            members[best_index].append(track_id)
            member_endpoints[best_index].append(np.asarray(endpoint, dtype=float).copy())
            centers[best_index] = np.asarray(member_endpoints[best_index]).mean(axis=0)

    ordered = sorted(
        enumerate(centers),
        key=lambda item: (float(item[1][0]), float(item[1][1])),
    )
    remap = {old_index: new_index for new_index, (old_index, _) in enumerate(ordered)}
    ordered_centers = np.asarray([center for _, center in ordered], dtype=float).reshape((-1, 2))
    counts = np.zeros(len(ordered_centers), dtype=np.int64)
    track_to_class: dict[int, int] = {}
    for old_index, track_ids in enumerate(members):
        new_index = remap[old_index]
        counts[new_index] = len(track_ids)
        for track_id in track_ids:
            track_to_class[track_id] = new_index

    return DestinationClasses(
        radius=float(radius),
        centers=ordered_centers,
        train_counts=counts,
        track_to_class=track_to_class,
    )


def assign_destination_classes(
    scene: ProcessedSceneRecord,
    destination_classes: DestinationClasses,
    *,
    max_distance: float,
) -> dict[int, int]:
    track_to_class: dict[int, int] = {}
    centers = destination_classes.centers
    for track_id, trajectory in scene.trajectories.items():
        if trajectory.shape[0] == 0 or centers.shape[0] == 0:
            track_to_class[track_id] = -1
            continue
        distances = np.linalg.norm(centers - trajectory[-1], axis=1)
        best_index = int(np.argmin(distances))
        track_to_class[track_id] = best_index if distances[best_index] <= max_distance else -1
    return track_to_class


def learn_transition_model(
    *,
    state_count: int,
    trajectory_states: dict[int, np.ndarray],
    train_track_ids: list[int],
    track_to_class: dict[int, int],
    class_count: int,
    trajectory_stride: int = 1,
) -> dict[str, Any]:
    if trajectory_stride < 1:
        raise ValueError("trajectory_stride must be >= 1")

    global_counts = sparse.dok_matrix((state_count, state_count), dtype=np.float64)
    class_counts = [
        sparse.dok_matrix((state_count, state_count), dtype=np.float64)
        for _ in range(class_count)
    ]

    for track_id in train_track_ids:
        states = trajectory_states.get(track_id)
        if states is None:
            continue
        states = states[::trajectory_stride]
        destination_class = track_to_class.get(track_id, -1)
        for source, target in zip(states[:-1], states[1:]):
            if source < 0 or target < 0:
                continue
            global_counts[int(source), int(target)] += 1.0
            if destination_class >= 0:
                class_counts[destination_class][int(source), int(target)] += 1.0

    global_counts_csr = global_counts.tocsr()
    global_transition = row_normalize_with_fallback(global_counts_csr)
    class_counts_csr = [counts.tocsr() for counts in class_counts]
    class_transitions = [
        row_normalize_with_fallback(counts, fallback=global_transition)
        for counts in class_counts_csr
    ]
    return {
        "global_counts": global_counts_csr,
        "global_transition": global_transition,
        "class_counts": class_counts_csr,
        "class_transitions": class_transitions,
    }


def row_normalize_with_fallback(
    counts: sparse.csr_matrix,
    *,
    fallback: sparse.csr_matrix | None = None,
) -> sparse.csr_matrix:
    counts = counts.tocsr()
    row_sums = np.asarray(counts.sum(axis=1)).ravel()
    nonzero_rows = row_sums > 0
    inv = np.zeros_like(row_sums, dtype=np.float64)
    inv[nonzero_rows] = 1.0 / row_sums[nonzero_rows]
    transition = sparse.diags(inv).dot(counts).tolil()

    zero_rows = np.flatnonzero(~nonzero_rows)
    if fallback is not None:
        fallback = fallback.tocsr()
    for row in zero_rows:
        if fallback is not None and fallback[row].nnz:
            transition[row, :] = fallback[row]
        else:
            transition[row, row] = 1.0

    return transition.tocsr()


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
    for class_id, center in enumerate(destination_classes.centers):
        records.append(
            {
                "class_id": class_id,
                "center": center.tolist(),
                "train_count": int(destination_classes.train_counts[class_id]),
            }
        )
    payload = {
        "radius": destination_classes.radius,
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
    sparse.save_npz(transitions_root / "global_counts.npz", transition_model["global_counts"])
    sparse.save_npz(transitions_root / "global_transition.npz", transition_model["global_transition"])
    for class_id, counts in enumerate(transition_model["class_counts"]):
        sparse.save_npz(transitions_root / f"class_{class_id:03d}_counts.npz", counts)
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
        "grid": {
            "cell_size": state_space.cell_size,
            "rows": state_space.rows,
            "cols": state_space.cols,
            "state_count": int(len(state_space.state_ids)),
            "non_walkable_classes": list(state_space.non_walkable_classes),
        },
        "destination_classes": {
            "radius": destination_classes.radius,
            "class_count": int(len(destination_classes.centers)),
            "unassigned_track_count": int(
                sum(1 for class_id in destination_classes.track_to_class.values() if class_id < 0)
            ),
        },
        "splits": {key: len(value) for key, value in splits.items()},
        "transitions": {
            "trajectory_stride": trajectory_stride,
            "global_nonzero_count": int(transition_model["global_counts"].nnz),
            "class_nonzero_counts": [int(counts.nnz) for counts in transition_model["class_counts"]],
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
        },
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "scene_id",
        "cell_size",
        "state_count",
        "rows",
        "cols",
        "train_tracks",
        "val_tracks",
        "test_tracks",
        "destination_classes",
        "global_nonzero",
        "unassigned_tracks",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
