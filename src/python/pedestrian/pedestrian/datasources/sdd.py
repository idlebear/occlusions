from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import GRID_RESOLUTION
from datasources.scenario import Scenario, StaticPolygon


BLOCKING_POLYGON_CLASSES = {"Building", "Obstacle", "Object", "Offroad"}


def load_sdd_scenario(
    processed_root: str | Path,
    scene_id: int,
) -> Scenario:
    processed_root = Path(processed_root)
    scene_root = processed_root / f"scene_{scene_id:03d}"
    metadata_path = scene_root / "metadata.json"
    trajectories_path = scene_root / "trajectories_scene.npz"
    polygons_path = scene_root / "polygons_scene.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing SDD scene metadata: {metadata_path}")
    if not trajectories_path.exists():
        raise FileNotFoundError(f"Missing SDD scene trajectories: {trajectories_path}")
    if not polygons_path.exists():
        raise FileNotFoundError(f"Missing SDD scene polygons: {polygons_path}")

    metadata = json.loads(metadata_path.read_text())
    metric_calibration = dict(metadata.get("metric_calibration", {}))
    scale = scene_scale_from_metadata(metadata)
    bounds = metadata["transform"]["scene_bounds"]
    width = bounds["max_x"] - bounds["min_x"]
    height = bounds["max_y"] - bounds["min_y"]
    display_diff = float(max(width, height))
    display_offset = [
        float(bounds["min_x"] - (display_diff - width) / 2),
        float(bounds["min_y"] - (display_diff - height) / 2),
    ]

    with np.load(trajectories_path) as trajectory_data:
        trajectory_arrays = {
            int(track_id): np.asarray(
                trajectory_data[f"track_{int(track_id)}"],
                dtype=float,
            )
            for track_id in trajectory_data["track_ids"]
        }
    tracks = {}
    for track_id, points in trajectory_arrays.items():
        points = scene_to_sim_display(points, bounds)
        tracks[track_id] = [
            [float(point[0]), float(point[1]), float(frame)]
            for frame, point in enumerate(points)
        ]

    polygon_data = json.loads(polygons_path.read_text())
    static_polygons = []
    for record in polygon_data.get("polygons", []):
        polygon_class = record["polygon_class"]
        points = scene_to_sim_display(np.asarray(record["vertices"], dtype=float), bounds)
        static_polygons.append(
            StaticPolygon(
                polygon_class=polygon_class,
                points=points,
                blocking=polygon_class in BLOCKING_POLYGON_CLASSES,
                metadata={"source": "sdd_processed"},
            )
        )

    return Scenario(
        name=f"sdd_scene_{scene_id:03d}",
        data_source="sdd",
        tracks=tracks,
        display_offset=display_offset,
        display_diff=display_diff,
        static_polygons=static_polygons,
        metadata={
            "scene_id": int(scene_id),
            "processed_root": str(processed_root),
            "scene_root": str(scene_root),
            "coordinate_units": metadata["coordinate_frames"]["scene"]["units"],
            "coordinate_transform": "sdd_scene_y_up_to_simulator_y_down",
            "bounds": bounds,
            "dt": metadata["source"]["dt"],
            "frame_rate_hz": metadata["source"]["frame_rate_hz"],
            "metric_calibration": metric_calibration,
            "scene_scale": scale,
            "grid": {
                "width": display_diff,
                "height": display_diff,
                "resolution": GRID_RESOLUTION * scale,
                "resolution_meters": GRID_RESOLUTION,
            },
        },
    )


def scene_scale_from_metadata(metadata: dict) -> float:
    try:
        scale = float(metadata["scene_scale"])
    except KeyError as exc:
        raise KeyError(
            "SDD scene metadata is missing scene_scale. Regenerate processed "
            "SDD data with oce_sdd.preprocess."
        ) from exc
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"SDD scene metadata has invalid scene_scale: {scale}")
    return scale


def scene_to_sim_display(points: np.ndarray, bounds: dict[str, float]) -> np.ndarray:
    display_points = np.asarray(points, dtype=float).copy()
    display_points[:, 1] = bounds["min_y"] + bounds["max_y"] - display_points[:, 1]
    return display_points
