from __future__ import annotations

import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


POLYGON_CLASS_ORDER = ("Building", "Obstacle", "Object", "Offroad", "Entrance")


@dataclass(frozen=True)
class SDDData:
    data_root: Path
    images: dict[int, np.ndarray]
    trajectories: dict[int, dict[int, np.ndarray]]
    polygons: dict[int, dict[str, list[np.ndarray]]]

    @property
    def scene_ids(self) -> list[int]:
        return sorted(set(self.images) & set(self.trajectories) & set(self.polygons))


def load_constrained_sdd_data(data_root: str | Path, *, dequantized: bool = True) -> SDDData:
    """Load the local constrained-SDD pickle artifacts without modifying third-party code."""
    root = Path(data_root)
    image_path = root / "all_images.pkl"
    trajectory_path = root / (
        "trajectories_dequantized.pkl" if dequantized else "trajectories.pkl"
    )
    polygon_path = root / "polygons.pkl"

    missing = [path for path in (image_path, trajectory_path, polygon_path) if not path.exists()]
    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing constrained-SDD artifact(s): {missing_str}")

    images = _load_pickle(image_path)
    raw_trajectories = _load_pickle(trajectory_path)
    raw_polygons = _load_pickle(polygon_path)

    trajectories: dict[int, dict[int, np.ndarray]] = {
        int(scene_id): {
            int(track_id): np.asarray(points, dtype=float)
            for track_id, points in scene_tracks.items()
        }
        for scene_id, scene_tracks in raw_trajectories.items()
    }
    polygons: dict[int, dict[str, list[np.ndarray]]] = {
        int(scene_id): {
            str(poly_class): [np.asarray(vertices, dtype=float) for vertices in poly_list]
            for poly_class, poly_list in scene_polygons.items()
        }
        for scene_id, scene_polygons in raw_polygons.items()
    }
    images = {int(scene_id): np.asarray(image) for scene_id, image in images.items()}

    return SDDData(data_root=root, images=images, trajectories=trajectories, polygons=polygons)


def load_agent_class_map(path: str | Path | None) -> dict[tuple[int, int], str]:
    """Load optional scene/track class labels from CSV or JSON."""
    if path is None:
        return {}

    label_path = Path(path)
    if not label_path.exists():
        raise FileNotFoundError(f"Agent class map does not exist: {label_path}")

    if label_path.suffix.lower() == ".csv":
        return _load_agent_class_csv(label_path)
    if label_path.suffix.lower() == ".json":
        return _load_agent_class_json(label_path)

    raise ValueError("Agent class map must be a .csv or .json file")


def scene_summary(
    sdd_data: SDDData,
    scene_id: int,
    class_map: dict[tuple[int, int], str],
    default_agent_class: str,
) -> dict[str, Any]:
    image = sdd_data.images[scene_id]
    tracks = sdd_data.trajectories[scene_id]
    polygons = sdd_data.polygons[scene_id]
    height, width = image.shape[:2]

    class_counts: dict[str, int] = {}
    lengths: list[int] = []
    for track_id, points in tracks.items():
        agent_class = class_map.get((scene_id, track_id), default_agent_class)
        class_counts[agent_class] = class_counts.get(agent_class, 0) + 1
        lengths.append(int(points.shape[0]))

    polygon_counts = {poly_class: len(poly_list) for poly_class, poly_list in polygons.items()}
    return {
        "scene_id": scene_id,
        "image_width": int(width),
        "image_height": int(height),
        "track_count": len(tracks),
        "max_track_length": max(lengths, default=0),
        "median_track_length": float(np.median(lengths)) if lengths else 0.0,
        "agent_class_counts": class_counts,
        "polygon_counts": polygon_counts,
    }


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _load_agent_class_csv(path: Path) -> dict[tuple[int, int], str]:
    class_map: dict[tuple[int, int], str] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"scene_id", "track_id", "agent_class"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")
        for row in reader:
            class_map[(int(row["scene_id"]), int(row["track_id"]))] = row["agent_class"]
    return class_map


def _load_agent_class_json(path: Path) -> dict[tuple[int, int], str]:
    with path.open("r") as f:
        raw = json.load(f)

    class_map: dict[tuple[int, int], str] = {}
    if isinstance(raw, list):
        for row in raw:
            class_map[(int(row["scene_id"]), int(row["track_id"]))] = str(row["agent_class"])
        return class_map

    if isinstance(raw, dict):
        for scene_id, scene_tracks in raw.items():
            for track_id, agent_class in scene_tracks.items():
                class_map[(int(scene_id), int(track_id))] = str(agent_class)
        return class_map

    raise ValueError("JSON class map must be a list of records or scene->track mapping")
