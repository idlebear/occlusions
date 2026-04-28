from __future__ import annotations

from pathlib import Path

import numpy as np

from datasources.scenario import Scenario


def load_eth_scenario(tracks_path: str | Path) -> Scenario:
    tracks_path = Path(tracks_path)
    objects: dict[float, list[list[float]]] = {}
    parsed_lines = []

    min_x = np.inf
    max_x = -np.inf
    min_y = np.inf
    max_y = -np.inf

    with tracks_path.open("rb") as f:
        lines = f.readlines()

    for line in lines:
        frame, track_id, x, y = [float(value) for value in line.split()]
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        parsed_lines.append([frame, track_id, x, y])

    diff_y = max_y - min_y
    diff_x = max_x - min_x

    if diff_x > diff_y:
        max_diff = diff_x
        min_y = min_y - (diff_x - diff_y) / 2
    else:
        max_diff = diff_y
        min_x = min_x - (diff_y - diff_x) / 2

    display_offset = [float(min_x), float(min_y)]

    for line in parsed_lines:
        frame, track_id, x, y = line

        if track_id not in objects:
            objects[track_id] = [[x, y, frame]]
        else:
            last_frame = objects[track_id][-1][2]
            step_x = (x - objects[track_id][-1][0]) / (frame - last_frame)
            step_y = (y - objects[track_id][-1][1]) / (frame - last_frame)
            for i in range(1, int(frame - last_frame) + 1):
                objects[track_id].append(
                    [
                        objects[track_id][-1][0] + step_x,
                        objects[track_id][-1][1] + step_y,
                        last_frame + i,
                    ]
                )

    return Scenario(
        name=tracks_path.stem,
        data_source="eth",
        tracks=objects,
        display_offset=display_offset,
        display_diff=float(max_diff),
        static_polygons=[],
        metadata={
            "tracks_path": str(tracks_path),
            "raw_track_count": len(objects),
            "coordinate_units": "source_track_units",
        },
    )
