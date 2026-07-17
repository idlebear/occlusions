from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SceneTransform:
    """Source-to-scene transform for one SDD scene.

    The constrained-SDD artifacts use image-like source coordinates: origin at
    the top-left, x right, y down. Scene coordinates use a Cartesian convention
    with lower-left origin, x right, y up, and a per-scene scale where the
    longest scene side is 10 normalized units.
    """

    scene_id: int
    source_width: float
    source_height: float
    source_units: str
    scene_units: str
    target_long_axis: float
    source_to_scene_scale: float
    source_origin_xy: tuple[float, float]
    scene_origin_xy: tuple[float, float]
    source_axes: dict[str, str]
    scene_axes: dict[str, str]

    @property
    def scene_width(self) -> float:
        return self.source_width * self.source_to_scene_scale

    @property
    def scene_height(self) -> float:
        return self.source_height * self.source_to_scene_scale

    @property
    def scene_bounds(self) -> dict[str, float]:
        return {
            "min_x": 0.0,
            "min_y": 0.0,
            "max_x": self.scene_width,
            "max_y": self.scene_height,
        }

    @property
    def display_normalization_factor(self) -> float:
        long_axis = max(self.scene_width, self.scene_height)
        return 1.0 / long_axis if long_axis else 0.0

    @property
    def display_bounds(self) -> dict[str, float]:
        factor = self.display_normalization_factor
        return {
            "min_x": 0.0,
            "min_y": 0.0,
            "max_x": self.scene_width * factor,
            "max_y": self.scene_height * factor,
        }

    def source_to_scene(self, points_xy: np.ndarray) -> np.ndarray:
        points = _as_points(points_xy)
        origin_x, origin_y = self.source_origin_xy
        scene_x = (points[:, 0] - origin_x) * self.source_to_scene_scale
        scene_y = (origin_y - points[:, 1]) * self.source_to_scene_scale
        return np.column_stack((scene_x, scene_y))

    def scene_to_source(self, points_xy: np.ndarray) -> np.ndarray:
        points = _as_points(points_xy)
        origin_x, origin_y = self.source_origin_xy
        source_x = (points[:, 0] / self.source_to_scene_scale) + origin_x
        source_y = origin_y - (points[:, 1] / self.source_to_scene_scale)
        return np.column_stack((source_x, source_y))

    def scene_to_display(self, points_xy: np.ndarray) -> np.ndarray:
        points = _as_points(points_xy)
        factor = self.display_normalization_factor
        display_x = points[:, 0] * factor
        display_y = points[:, 1] * factor
        return np.column_stack((display_x, display_y))

    def to_metadata(self) -> dict[str, Any]:
        metadata = asdict(self)
        metadata["scene_width"] = self.scene_width
        metadata["scene_height"] = self.scene_height
        metadata["scene_bounds"] = self.scene_bounds
        metadata["display_normalization_factor"] = self.display_normalization_factor
        metadata["display_bounds"] = self.display_bounds
        metadata["transform_equations"] = {
            "source_to_scene": [
                "scene_x = (source_x - source_origin_x) * source_to_scene_scale",
                "scene_y = (source_origin_y - source_y) * source_to_scene_scale",
            ],
            "scene_to_source": [
                "source_x = scene_x / source_to_scene_scale + source_origin_x",
                "source_y = source_origin_y - scene_y / source_to_scene_scale",
            ],
            "scene_to_display": [
                "display_x = (scene_x - scene_bounds.min_x) * display_normalization_factor",
                "display_y = (scene_y - scene_bounds.min_y) * display_normalization_factor",
            ],
        }
        return metadata


def make_scene_transform(
    scene_id: int,
    image_shape: tuple[int, ...],
    *,
    source_units: str = "pixels",
    scene_units: str = "normalized_scene_units",
    target_long_axis: float = 10.0,
) -> SceneTransform:
    if target_long_axis <= 0:
        raise ValueError("target_long_axis must be positive")
    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    scale = float(target_long_axis) / max(image_width, image_height)
    return SceneTransform(
        scene_id=int(scene_id),
        source_width=float(image_width),
        source_height=float(image_height),
        source_units=source_units,
        scene_units=scene_units,
        target_long_axis=float(target_long_axis),
        source_to_scene_scale=scale,
        source_origin_xy=(0.0, float(image_height)),
        scene_origin_xy=(0.0, 0.0),
        source_axes={"x": "right", "y": "down"},
        scene_axes={"x": "right", "y": "up"},
    )


def _as_points(points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected an Nx2 point array, got shape {points.shape}")
    return points
