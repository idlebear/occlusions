from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from datasources.scenario import Scenario, StaticPolygon


BLOCKING_POLYGON_CLASSES = {"Building", "Obstacle", "Object", "Offroad"}


DEFAULT_ACTOR_SCALE_PERCENTILE = 75.0


def load_sdd_scenario(
    processed_root: str | Path,
    scene_id: int,
    *,
    actor_scale_percentile: float | None = DEFAULT_ACTOR_SCALE_PERCENTILE,
) -> Scenario:
    if actor_scale_percentile is not None and actor_scale_percentile < 0:
        actor_scale_percentile = None

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
    actor_dimension_multiplier = actor_dimension_multiplier_from_metadata(
        metric_calibration,
        trajectory_arrays,
        dt=float(metadata.get("source", {}).get("dt", 1.0 / 30.0)),
    )
    local_actor_dimension_multiplier = actor_dimension_multiplier
    dataset_actor_dimension_multiplier = None
    if actor_scale_percentile is not None:
        dataset_actor_dimension_multiplier = dataset_actor_scale_percentile(
            str(processed_root),
            float(actor_scale_percentile),
        )
        if dataset_actor_dimension_multiplier > 0:
            actor_dimension_multiplier = dataset_actor_dimension_multiplier

    metadata_actor_dimension_multiplier = float(
        metric_calibration.get("actor_dimension_multiplier")
        or metric_calibration.get("scene_units_per_meter")
        or 1.0
    )
    if abs(actor_dimension_multiplier - metadata_actor_dimension_multiplier) > 1e-12:
        metric_calibration["loader_method"] = (
            "mean_track_median_speed_matches_assumed_walking_speed"
        )
        metric_calibration["loader_actor_dimension_multiplier"] = (
            local_actor_dimension_multiplier
        )
    if dataset_actor_dimension_multiplier is not None:
        metric_calibration["dataset_actor_scale_percentile"] = float(
            actor_scale_percentile
        )
        metric_calibration["dataset_actor_dimension_multiplier"] = (
            dataset_actor_dimension_multiplier
        )
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
            "local_actor_dimension_multiplier": local_actor_dimension_multiplier,
            "actor_dimension_multiplier": actor_dimension_multiplier,
        },
    )


@lru_cache(maxsize=16)
def dataset_actor_scale_percentile(processed_root: str, percentile: float) -> float:
    if percentile < 0.0 or percentile > 100.0:
        raise ValueError("SDD actor scale percentile must be in [0, 100]")

    root = Path(processed_root)
    multipliers = []
    for metadata_path in sorted(root.glob("scene_*/metadata.json")):
        scene_root = metadata_path.parent
        trajectories_path = scene_root / "trajectories_scene.npz"
        if not trajectories_path.exists():
            continue

        metadata = json.loads(metadata_path.read_text())
        metric_calibration = dict(metadata.get("metric_calibration", {}))
        with np.load(trajectories_path) as trajectory_data:
            trajectories = {
                int(track_id): np.asarray(
                    trajectory_data[f"track_{int(track_id)}"],
                    dtype=float,
                )
                for track_id in trajectory_data["track_ids"]
            }
        multiplier = actor_dimension_multiplier_from_metadata(
            metric_calibration,
            trajectories,
            dt=float(metadata.get("source", {}).get("dt", 1.0 / 30.0)),
        )
        if multiplier > 0:
            multipliers.append(multiplier)

    if not multipliers:
        return 0.0
    return float(np.percentile(np.asarray(multipliers, dtype=float), percentile))


def actor_dimension_multiplier_from_metadata(
    metric_calibration: dict,
    trajectories: dict[int, np.ndarray],
    *,
    dt: float,
) -> float:
    multiplier = float(
        metric_calibration.get("actor_dimension_multiplier")
        or metric_calibration.get("scene_units_per_meter")
        or 1.0
    )
    if (
        metric_calibration.get("method")
        == "mean_track_median_speed_matches_assumed_walking_speed"
    ):
        return multiplier

    assumed_walking_speed = float(metric_calibration.get("assumed_walking_speed_mps") or 1.4)
    recalibrated = estimate_actor_dimension_multiplier(
        trajectories,
        dt=dt,
        assumed_walking_speed=assumed_walking_speed,
    )
    return recalibrated if recalibrated > 0 else multiplier


def estimate_actor_dimension_multiplier(
    trajectories: dict[int, np.ndarray],
    *,
    dt: float,
    assumed_walking_speed: float,
) -> float:
    if dt <= 0 or assumed_walking_speed <= 0:
        return 0.0

    track_median_speeds = []
    for points in trajectories.values():
        points = np.asarray(points, dtype=float)
        if points.shape[0] < 2:
            continue
        speeds = np.linalg.norm(np.diff(points, axis=0), axis=1) / dt
        if speeds.size:
            track_median_speeds.append(float(np.median(speeds)))

    if not track_median_speeds:
        return 0.0
    return float(np.mean(track_median_speeds) / assumed_walking_speed)


def scene_to_sim_display(points: np.ndarray, bounds: dict[str, float]) -> np.ndarray:
    display_points = np.asarray(points, dtype=float).copy()
    display_points[:, 1] = bounds["min_y"] + bounds["max_y"] - display_points[:, 1]
    return display_points
