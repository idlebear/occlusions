from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import GRID_RESOLUTION
from datasources.scenario import Scenario, StaticPolygon

BLOCKING_POLYGON_CLASSES = {"Building", "Obstacle", "Object", "Offroad"}
SCENE_COORDINATE_FRAMES = {
    "normalized_scene_coordinates",
    "normalized_scene_units",
    "scene_coordinates",
}
SOURCE_PIXEL_COORDINATE_FRAMES = {
    "constrained_sdd_image_pixels",
    "image_pixels",
    "source_pixels",
}


def load_sdd_scenario(
    processed_root: str | Path,
    scene_id: int,
    scenario_config: str | None = None,
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

    if scenario_config is not None:
        scenario_config_path = (
            Path(scene_root) / "scenarios" / f"{scenario_config}.json"
        )
        if not scenario_config_path.exists():
            raise FileNotFoundError(
                f"SDD scenario config not found: {scenario_config_path}"
            )
        with scenario_config_path.open() as f:
            scenario_design = json.load(f)
        selected_track_ids = {
            int(track_id) for track_id in scenario_design.get("selectedTrackIds", [])
        }
        coordinate_frame = scenario_design.get("coordinateFrame")
        robot_start_zone = scenario_design.get("robotStartZone")
        if robot_start_zone is not None:
            robot_start_zone = scenario_design_points_to_sim_display(
                robot_start_zone["vertices"],
                bounds=bounds,
                metadata=metadata,
                coordinate_frame=coordinate_frame,
            )
        target_goal_zones = [
            scenario_design_zone_to_sim_display(
                zone,
                bounds=bounds,
                metadata=metadata,
                coordinate_frame=coordinate_frame,
            )
            for zone in scenario_design.get("targetGoalZones", [])
        ]
        robot_goal = scenario_design.get("robotGoal")
        if robot_goal is not None:
            robot_goal = scenario_design_point_to_sim_display(
                robot_goal,
                bounds=bounds,
                metadata=metadata,
                coordinate_frame=coordinate_frame,
            )
    else:
        scenario_design = {}
        coordinate_frame = None
        selected_track_ids = None
        robot_start_zone = None
        target_goal_zones = []
        robot_goal = None

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
        if selected_track_ids is not None and track_id not in selected_track_ids:
            continue
        points = scene_to_sim_display(points, bounds)
        tracks[track_id] = [
            [float(point[0]), float(point[1]), float(frame)]
            for frame, point in enumerate(points)
        ]

    polygon_data = json.loads(polygons_path.read_text())
    static_polygons = []
    for record in polygon_data.get("polygons", []):
        polygon_class = record["polygon_class"]
        points = scene_to_sim_display(
            np.asarray(record["vertices"], dtype=float), bounds
        )
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
            "scenario_config": scenario_design,
            "scenario_coordinate_frame": coordinate_frame,
            "robot_start_zone": robot_start_zone,
            "target_goal_zones": target_goal_zones,
            "robot_goal": robot_goal,
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


def scenario_design_point_to_sim_display(
    point: dict[str, float] | list[float] | tuple[float, float],
    *,
    bounds: dict[str, float],
    metadata: dict,
    coordinate_frame: str | None,
) -> np.ndarray:
    if isinstance(point, dict):
        point = [point["x"], point["y"]]
    return scenario_design_points_to_sim_display(
        [point],
        bounds=bounds,
        metadata=metadata,
        coordinate_frame=coordinate_frame,
    ).squeeze()


def scenario_design_zone_to_sim_display(
    zone: dict,
    *,
    bounds: dict[str, float],
    metadata: dict,
    coordinate_frame: str | None,
) -> dict:
    return {
        **zone,
        "vertices": scenario_design_points_to_sim_display(
            zone["vertices"],
            bounds=bounds,
            metadata=metadata,
            coordinate_frame=coordinate_frame,
        ),
    }


def scenario_design_points_to_sim_display(
    points: np.ndarray | list,
    *,
    bounds: dict[str, float],
    metadata: dict,
    coordinate_frame: str | None,
) -> np.ndarray:
    scene_points = scenario_design_points_to_scene(
        points,
        bounds=bounds,
        metadata=metadata,
        coordinate_frame=coordinate_frame,
    )
    return scene_to_sim_display(scene_points, bounds)


def scenario_design_points_to_scene(
    points: np.ndarray | list,
    *,
    bounds: dict[str, float],
    metadata: dict,
    coordinate_frame: str | None,
) -> np.ndarray:
    points = as_xy_points(points)
    frame = normalize_scenario_coordinate_frame(coordinate_frame, points, bounds)
    if frame in SCENE_COORDINATE_FRAMES:
        return points
    if frame in SOURCE_PIXEL_COORDINATE_FRAMES:
        return source_pixels_to_scene(points, metadata["transform"])
    raise ValueError(f"Unsupported SDD scenario coordinateFrame: {coordinate_frame}")


def normalize_scenario_coordinate_frame(
    coordinate_frame: str | None,
    points: np.ndarray,
    bounds: dict[str, float],
) -> str:
    if coordinate_frame is None:
        return infer_scenario_coordinate_frame(points, bounds)
    return str(coordinate_frame).strip().lower()


def infer_scenario_coordinate_frame(
    points: np.ndarray,
    bounds: dict[str, float],
) -> str:
    max_scene_x = float(bounds["max_x"])
    max_scene_y = float(bounds["max_y"])
    tolerance = 1e-6
    if (
        np.nanmax(points[:, 0]) > max_scene_x + tolerance
        or np.nanmax(points[:, 1]) > max_scene_y + tolerance
    ):
        return "constrained_sdd_image_pixels"
    return "normalized_scene_coordinates"


def source_pixels_to_scene(points: np.ndarray, transform: dict) -> np.ndarray:
    origin_x, origin_y = transform["source_origin_xy"]
    scale = float(transform["source_to_scene_scale"])
    scene_x = (points[:, 0] - float(origin_x)) * scale
    scene_y = (float(origin_y) - points[:, 1]) * scale
    return np.column_stack((scene_x, scene_y))


def as_xy_points(points: np.ndarray | list) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected an Nx2 point array, got shape {points.shape}")
    return points
