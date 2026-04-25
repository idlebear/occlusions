from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from oce_sdd.coordinates import SceneTransform, make_scene_transform
from oce_sdd.data import POLYGON_CLASS_ORDER, SDDData, load_constrained_sdd_data


@dataclass(frozen=True)
class ProcessedScene:
    scene_id: int
    transform: SceneTransform
    trajectories: dict[int, np.ndarray]
    polygons: dict[str, list[np.ndarray]]
    filtering: dict[str, Any]
    validation: dict[str, Any]


def main() -> None:
    args = parse_args()
    data = load_constrained_sdd_data(args.data_root, dequantized=not args.quantized)
    scene_ids = args.scene_id if args.scene_id else data.scene_ids
    write_processed_dataset(
        data,
        output_root=Path(args.out),
        scene_ids=scene_ids,
        source_units=args.source_units,
        scene_units=args.scene_units,
        target_long_axis=args.target_long_axis,
        fps=args.fps,
        speed_threshold=args.speed_threshold,
        min_track_displacement=args.min_track_displacement,
        write_overlays=not args.no_overlays,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert constrained-SDD data into normalized OCE scene coordinates and bounds."
    )
    parser.add_argument(
        "--data-root",
        default="src/thirdParty/sdd/data",
        help="Path containing constrained-SDD pickle artifacts.",
    )
    parser.add_argument(
        "--out",
        default="outputs/sdd_processed",
        help="Output directory for processed scene records.",
    )
    parser.add_argument(
        "--scene-id",
        type=int,
        action="append",
        help="Scene ID to process. Repeat to process several scenes. Defaults to all scenes.",
    )
    parser.add_argument("--source-units", default="pixels", help="Units of the source coordinates.")
    parser.add_argument(
        "--scene-units",
        default="normalized_scene_units",
        help="Units assigned to the normalized scene coordinates.",
    )
    parser.add_argument(
        "--target-long-axis",
        type=float,
        default=10.0,
        help="Normalized size assigned to each scene image's longest side.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Trajectory sample rate in frames per second.",
    )
    parser.add_argument(
        "--speed-threshold",
        type=float,
        default=2.0,
        help="Flag tracks whose maximum speed exceeds this value in scene units per second.",
    )
    parser.add_argument(
        "--min-track-displacement",
        type=float,
        default=10.0,
        help=(
            "Remove tracks whose endpoint-to-endpoint Euclidean displacement is shorter "
            "than this value in source pixels."
        ),
    )
    parser.add_argument(
        "--quantized",
        action="store_true",
        help="Use trajectories.pkl instead of trajectories_dequantized.pkl.",
    )
    parser.add_argument(
        "--no-overlays",
        action="store_true",
        help="Skip validation overlay image generation.",
    )
    return parser.parse_args()


def process_scene(
    data: SDDData,
    scene_id: int,
    *,
    source_units: str = "pixels",
    scene_units: str = "normalized_scene_units",
    target_long_axis: float = 10.0,
    dt: float = 1.0 / 30.0,
    speed_threshold: float = 2.0,
    min_track_displacement: float = 10.0,
) -> ProcessedScene:
    image = data.images[scene_id]
    transform = make_scene_transform(
        scene_id,
        image.shape,
        source_units=source_units,
        scene_units=scene_units,
        target_long_axis=target_long_axis,
    )

    raw_trajectories, filtering = filter_short_tracks(
        data.trajectories[scene_id],
        min_displacement=min_track_displacement,
        units=source_units,
    )

    scene_trajectories = {
        track_id: transform.source_to_scene(points)
        for track_id, points in raw_trajectories.items()
    }
    scene_polygons = {
        polygon_class: [transform.source_to_scene(vertices) for vertices in polygon_list]
        for polygon_class, polygon_list in data.polygons[scene_id].items()
    }
    validation = validate_scene_transform(
        transform,
        raw_trajectories=raw_trajectories,
        scene_trajectories=scene_trajectories,
        raw_polygons=data.polygons[scene_id],
        scene_polygons=scene_polygons,
        dt=dt,
        speed_threshold=speed_threshold,
    )

    return ProcessedScene(
        scene_id=scene_id,
        transform=transform,
        trajectories=scene_trajectories,
        polygons=scene_polygons,
        filtering=filtering,
        validation=validation,
    )


def write_processed_dataset(
    data: SDDData,
    *,
    output_root: Path,
    scene_ids: list[int],
    source_units: str,
    scene_units: str,
    target_long_axis: float,
    fps: float,
    speed_threshold: float,
    min_track_displacement: float,
    write_overlays: bool,
) -> None:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if speed_threshold < 0:
        raise ValueError("speed_threshold must be non-negative")
    if min_track_displacement < 0:
        raise ValueError("min_track_displacement must be non-negative")
    dt = 1.0 / fps
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []

    for scene_id in sorted(scene_ids):
        if scene_id not in data.scene_ids:
            raise ValueError(f"Scene {scene_id} is not available in the constrained-SDD data")

        scene = process_scene(
            data,
            scene_id,
            source_units=source_units,
            scene_units=scene_units,
            target_long_axis=target_long_axis,
            dt=dt,
            speed_threshold=speed_threshold,
            min_track_displacement=min_track_displacement,
        )
        scene_root = output_root / f"scene_{scene_id:03d}"
        scene_root.mkdir(parents=True, exist_ok=True)

        write_scene_metadata(
            scene,
            scene_root / "metadata.json",
            data_root=data.data_root,
            fps=fps,
        )
        write_scene_trajectories(scene.trajectories, scene_root / "trajectories_scene.npz")
        write_scene_polygons(scene.polygons, scene_root / "polygons_scene.json")
        if write_overlays:
            write_validation_overlay(data, scene, scene_root / "alignment_overlay.png")

        summary_rows.append(scene_summary_row(scene))

    write_processing_summary(summary_rows, output_root / "scene_summary.csv")
    print(f"Wrote processed data for {len(summary_rows)} scenes to {output_root}")


def filter_short_tracks(
    trajectories: dict[int, np.ndarray],
    *,
    min_displacement: float,
    units: str,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    if min_displacement < 0:
        raise ValueError("min_displacement must be non-negative")

    kept: dict[int, np.ndarray] = {}
    removed_track_ids: list[int] = []
    displacements: dict[str, float] = {}

    for track_id, points in sorted(trajectories.items()):
        endpoint_displacement = endpoint_distance(points)
        displacements[str(track_id)] = endpoint_displacement
        if endpoint_displacement < min_displacement:
            removed_track_ids.append(int(track_id))
            continue
        kept[int(track_id)] = points

    values = list(displacements.values())
    return kept, {
        "min_endpoint_displacement": min_displacement,
        "units": units,
        "method": "euclidean_distance_between_first_and_last_source_points",
        "input_trajectory_count": len(trajectories),
        "output_trajectory_count": len(kept),
        "removed_trajectory_count": len(removed_track_ids),
        "removed_track_ids": removed_track_ids,
        "endpoint_displacement_summary": {
            "min": float(np.min(values)) if values else 0.0,
            "median": float(np.median(values)) if values else 0.0,
            "max": float(np.max(values)) if values else 0.0,
        },
    }


def endpoint_distance(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(points[-1] - points[0]))


def validate_scene_transform(
    transform: SceneTransform,
    *,
    raw_trajectories: dict[int, np.ndarray],
    scene_trajectories: dict[int, np.ndarray],
    raw_polygons: dict[str, list[np.ndarray]],
    scene_polygons: dict[str, list[np.ndarray]],
    dt: float,
    speed_threshold: float,
) -> dict[str, Any]:
    trajectory_errors = []
    for track_id, raw_points in raw_trajectories.items():
        round_trip = transform.scene_to_source(scene_trajectories[track_id])
        trajectory_errors.append(float(np.max(np.abs(round_trip - raw_points))))

    polygon_errors = []
    for polygon_class, raw_polygon_list in raw_polygons.items():
        scene_polygon_list = scene_polygons[polygon_class]
        for raw_vertices, scene_vertices in zip(raw_polygon_list, scene_polygon_list):
            round_trip = transform.scene_to_source(scene_vertices)
            polygon_errors.append(float(np.max(np.abs(round_trip - raw_vertices))))

    all_points = (
        np.concatenate(list(scene_trajectories.values()), axis=0)
        if scene_trajectories
        else np.empty((0, 2))
    )
    bounds = transform.scene_bounds
    if all_points.size:
        x = all_points[:, 0]
        y = all_points[:, 1]
        inside = (
            (x >= bounds["min_x"])
            & (x <= bounds["max_x"])
            & (y >= bounds["min_y"])
            & (y <= bounds["max_y"])
        )
        trajectory_points_outside_scene = int(np.count_nonzero(~inside))
        trajectory_bounds = {
            "min_x": float(np.min(x)),
            "min_y": float(np.min(y)),
            "max_x": float(np.max(x)),
            "max_y": float(np.max(y)),
        }
    else:
        trajectory_points_outside_scene = 0
        trajectory_bounds = {"min_x": 0.0, "min_y": 0.0, "max_x": 0.0, "max_y": 0.0}

    speeds = []
    max_track_speeds: dict[str, float] = {}
    for track_id, points in scene_trajectories.items():
        if points.shape[0] < 2:
            max_track_speeds[str(track_id)] = 0.0
            continue
        track_speeds = np.linalg.norm(np.diff(points, axis=0), axis=1) / dt
        speeds.extend(track_speeds.tolist())
        max_track_speeds[str(track_id)] = float(np.max(track_speeds))

    return {
        "max_trajectory_round_trip_error_source_units": max(trajectory_errors, default=0.0),
        "max_polygon_round_trip_error_source_units": max(polygon_errors, default=0.0),
        "trajectory_points_outside_scene": trajectory_points_outside_scene,
        "trajectory_bounds": trajectory_bounds,
        "speed_summary": {
            "units": f"{transform.scene_units}/s",
            "mean": float(np.mean(speeds)) if speeds else 0.0,
            "median": float(np.median(speeds)) if speeds else 0.0,
            "p95": float(np.percentile(speeds, 95)) if speeds else 0.0,
            "max": float(np.max(speeds)) if speeds else 0.0,
            "threshold": speed_threshold,
            "tracks_over_threshold": [
                int(track_id) for track_id, speed in max_track_speeds.items() if speed > speed_threshold
            ],
        },
    }


def write_scene_metadata(
    scene: ProcessedScene,
    path: Path,
    *,
    data_root: Path,
    fps: float,
) -> None:
    dt = 1.0 / fps
    metadata = {
        "scene_id": scene.scene_id,
        "source": {
            "dataset": "constrained_stanford_drone_dataset",
            "data_root": str(data_root),
            "frame_rate_hz": fps,
            "dt": dt,
            "dt_units": "seconds",
        },
        "coordinate_frames": {
            "source": {
                "origin": "top_left",
                "x_axis": "right",
                "y_axis": "down",
                "units": scene.transform.source_units,
            },
            "scene": {
                "origin": "bottom_left_image_corner",
                "x_axis": "right",
                "y_axis": "up",
                "units": scene.transform.scene_units,
            },
            "display": {
                "origin": "bottom_left",
                "x_axis": "right",
                "y_axis": "up",
                "units": "unit_square",
                "note": "Display coordinates use one scalar factor to preserve aspect ratio on a square surface.",
            },
        },
        "transform": scene.transform.to_metadata(),
        "filtering": scene.filtering,
        "validation": scene.validation,
        "files": {
            "trajectories": "trajectories_scene.npz",
            "polygons": "polygons_scene.json",
            "overlay": "alignment_overlay.png",
        },
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def write_scene_trajectories(trajectories: dict[int, np.ndarray], path: Path) -> None:
    arrays = {f"track_{track_id}": points for track_id, points in sorted(trajectories.items())}
    track_ids = np.array(sorted(trajectories), dtype=np.int64)
    np.savez_compressed(path, track_ids=track_ids, **arrays)


def write_scene_polygons(polygons: dict[str, list[np.ndarray]], path: Path) -> None:
    records = []
    for polygon_class in ordered_polygon_classes(polygons):
        for vertices in polygons.get(polygon_class, []):
            records.append(
                {
                    "polygon_class": polygon_class,
                    "vertices": np.round(vertices, 8).tolist(),
                }
            )
    path.write_text(json.dumps({"polygons": records}, indent=2), encoding="utf-8")


def write_validation_overlay(data: SDDData, scene: ProcessedScene, path: Path) -> None:
    image = data.images[scene.scene_id]
    if image.ndim == 3 and image.shape[2] == 3:
        canvas = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    draw_polygons(canvas, scene)
    draw_trajectories(canvas, scene)
    draw_origin(canvas, scene.transform)
    if not cv2.imwrite(str(path), canvas):
        raise IOError(f"Could not write validation overlay: {path}")


def draw_polygons(canvas: np.ndarray, scene: ProcessedScene) -> None:
    colors = {
        "Building": (66, 66, 66),
        "Obstacle": (40, 40, 210),
        "Object": (0, 140, 220),
        "Offroad": (60, 170, 60),
        "Entrance": (220, 120, 20),
    }
    for polygon_class in ordered_polygon_classes(scene.polygons):
        color = colors.get(polygon_class, (140, 140, 140))
        for vertices in scene.polygons[polygon_class]:
            pixel_vertices = scene.transform.scene_to_source(vertices)
            pts = np.round(pixel_vertices).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)


def draw_trajectories(canvas: np.ndarray, scene: ProcessedScene) -> None:
    for index, (_, points) in enumerate(sorted(scene.trajectories.items())):
        if points.shape[0] < 2:
            continue
        color = _track_color(index)
        pixel_points = scene.transform.scene_to_source(points)
        pts = np.round(pixel_points[:: max(1, points.shape[0] // 250)]).astype(np.int32)
        cv2.polylines(canvas, [pts.reshape((-1, 1, 2))], isClosed=False, color=color, thickness=1)
        cv2.circle(canvas, tuple(pts[0]), 2, (255, 255, 255), -1)
        cv2.circle(canvas, tuple(pts[-1]), 2, color, -1)


def draw_origin(canvas: np.ndarray, transform: SceneTransform) -> None:
    x = int(round(transform.source_origin_xy[0]))
    y = int(round(transform.source_origin_xy[1])) - 1
    y = max(0, min(canvas.shape[0] - 1, y))
    cv2.drawMarker(
        canvas,
        (x, y),
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=16,
        thickness=2,
    )
    cv2.putText(
        canvas,
        "scene origin",
        (x + 6, max(14, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )


def ordered_polygon_classes(polygons: dict[str, list[np.ndarray]]) -> list[str]:
    ordered = [polygon_class for polygon_class in POLYGON_CLASS_ORDER if polygon_class in polygons]
    ordered.extend(sorted(set(polygons) - set(ordered)))
    return ordered


def scene_summary_row(scene: ProcessedScene) -> dict[str, Any]:
    validation = scene.validation
    return {
        "scene_id": scene.scene_id,
        "source_width": scene.transform.source_width,
        "source_height": scene.transform.source_height,
        "source_units": scene.transform.source_units,
        "scene_units": scene.transform.scene_units,
        "target_long_axis": scene.transform.target_long_axis,
        "source_to_scene_scale": scene.transform.source_to_scene_scale,
        "scene_width": scene.transform.scene_width,
        "scene_height": scene.transform.scene_height,
        "display_normalization_factor": scene.transform.display_normalization_factor,
        "input_trajectory_count": scene.filtering["input_trajectory_count"],
        "trajectory_count": len(scene.trajectories),
        "removed_trajectory_count": scene.filtering["removed_trajectory_count"],
        "min_endpoint_displacement_source_units": scene.filtering["min_endpoint_displacement"],
        "polygon_count": sum(len(items) for items in scene.polygons.values()),
        "max_trajectory_round_trip_error_source_units": validation[
            "max_trajectory_round_trip_error_source_units"
        ],
        "max_polygon_round_trip_error_source_units": validation[
            "max_polygon_round_trip_error_source_units"
        ],
        "trajectory_points_outside_scene": validation["trajectory_points_outside_scene"],
        "max_speed": validation["speed_summary"]["max"],
        "tracks_over_speed_threshold": len(validation["speed_summary"]["tracks_over_threshold"]),
    }


def write_processing_summary(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "scene_id",
        "source_width",
        "source_height",
        "source_units",
        "scene_units",
        "target_long_axis",
        "source_to_scene_scale",
        "scene_width",
        "scene_height",
        "display_normalization_factor",
        "input_trajectory_count",
        "trajectory_count",
        "removed_trajectory_count",
        "min_endpoint_displacement_source_units",
        "polygon_count",
        "max_trajectory_round_trip_error_source_units",
        "max_polygon_round_trip_error_source_units",
        "trajectory_points_outside_scene",
        "max_speed",
        "tracks_over_speed_threshold",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _track_color(index: int) -> tuple[int, int, int]:
    palette = [
        (50, 80, 220),
        (60, 160, 60),
        (220, 120, 20),
        (200, 60, 160),
        (180, 170, 40),
        (40, 170, 200),
    ]
    return palette[index % len(palette)]


if __name__ == "__main__":
    main()
