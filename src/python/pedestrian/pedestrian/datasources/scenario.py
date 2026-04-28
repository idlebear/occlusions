from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StaticPolygon:
    polygon_class: str
    points: np.ndarray
    blocking: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Scenario:
    name: str
    data_source: str
    tracks: dict[int | float | str, list[list[float]]]
    display_offset: list[float]
    display_diff: float
    static_polygons: list[StaticPolygon] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
