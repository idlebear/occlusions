from dataclasses import dataclass
from math import floor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    from Grid.bresenham import bresenham
except Exception:  # pragma: no cover - package-relative fallback
    from .bresenham import bresenham


OWNER_THRESHOLD = 0.05
PLANNING_HARD_THRESHOLD = 0.65
VISIBILITY_THRESHOLD = 0.25


@dataclass
class OccupancyHorizon:
    probability_grids: np.ndarray
    owner_mask_grids: np.ndarray
    overlap_count_grids: np.ndarray
    collision_grids: np.ndarray
    origin: tuple[float, float]
    resolution: float
    agent_bit_indices: dict
    owner_threshold: float = OWNER_THRESHOLD
    planning_hard_threshold: float = PLANNING_HARD_THRESHOLD
    visibility_threshold: float = VISIBILITY_THRESHOLD

    @property
    def horizon(self):
        return max(0, int(self.probability_grids.shape[0]) - 1)

    @property
    def current_blocked(self):
        return self.collision_grids[0]


def _probability_from_grid(base_grid, rows, cols):
    if base_grid is None:
        return np.zeros((rows, cols), dtype=np.float32)
    if hasattr(base_grid, "probabilityMap"):
        values = np.asarray(base_grid.probabilityMap(), dtype=np.float32)
        neutral_probability = float(getattr(base_grid, "pUnk", 0.5))
        values = np.where(
            values <= neutral_probability + 1.0e-6,
            0.0,
            values,
        )
    else:
        values = np.asarray(base_grid, dtype=np.float32)
    if values.shape != (rows, cols):
        raise ValueError(
            f"base grid shape {values.shape} does not match requested {(rows, cols)}"
        )
    return np.clip(values, 0.0, 1.0).astype(np.float32, copy=False)


def _state_cells(state_centers, origin, resolution, rows, cols):
    centers = np.asarray(state_centers, dtype=float)
    col = np.floor((centers[:, 0] - float(origin[0])) / float(resolution)).astype(
        np.int64
    )
    row = np.floor((centers[:, 1] - float(origin[1])) / float(resolution)).astype(
        np.int64
    )
    valid = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
    return row, col, valid


def _state_occupancy_slices(tracker, origin, resolution, rows, cols):
    if not all(
        hasattr(tracker, attr)
        for attr in ("state_grid_indices", "cell_size", "bounds", "state_centers_sim")
    ):
        state_rows, state_cols, valid_states = _state_cells(
            tracker.state_centers_sim,
            origin,
            resolution,
            rows,
            cols,
        )
        return [
            (
                slice(int(row), int(row) + 1),
                slice(int(col), int(col) + 1),
                bool(valid),
            )
            for row, col, valid in zip(state_rows, state_cols, valid_states)
        ]

    state_grid_indices = np.asarray(tracker.state_grid_indices, dtype=float)
    bounds = tracker.bounds
    cell_size = float(tracker.cell_size)
    min_x = float(bounds["min_x"])
    min_y = float(bounds["min_y"])
    max_y = float(bounds["max_y"])
    origin_x = float(origin[0])
    origin_y = float(origin[1])
    resolution = float(resolution)

    slices = []
    for row_col in state_grid_indices:
        scene_row = int(row_col[0])
        scene_col = int(row_col[1])
        x0 = min_x + scene_col * cell_size
        x1 = x0 + cell_size
        scene_y0 = min_y + scene_row * cell_size
        scene_y1 = scene_y0 + cell_size
        sim_y0 = min_y + max_y - scene_y1
        sim_y1 = min_y + max_y - scene_y0

        col0 = int(np.floor((min(x0, x1) - origin_x) / resolution))
        col1 = int(np.ceil((max(x0, x1) - origin_x) / resolution))
        row0 = int(np.floor((min(sim_y0, sim_y1) - origin_y) / resolution))
        row1 = int(np.ceil((max(sim_y0, sim_y1) - origin_y) / resolution))

        col0 = max(0, min(cols, col0))
        col1 = max(0, min(cols, col1))
        row0 = max(0, min(rows, row0))
        row1 = max(0, min(rows, row1))
        valid = row1 > row0 and col1 > col0
        slices.append((slice(row0, row1), slice(col0, col1), valid))
    return slices


def _agent_prefix_beliefs(tracker, agent_ids, horizon):
    if not agent_ids:
        return np.zeros((0, int(horizon) + 1, 0), dtype=np.float32)
    _data, _indices, _indptr, prefix = tracker.agent_mixed_transition_csr(
        agent_ids,
        int(horizon),
    )
    return np.asarray(prefix, dtype=np.float32)


def build_transition_occupancy_horizon(
    *,
    tracker,
    active_agent_ids=None,
    horizon,
    origin,
    resolution,
    rows,
    cols,
    base_grid=None,
    owner_threshold=OWNER_THRESHOLD,
    planning_hard_threshold=PLANNING_HARD_THRESHOLD,
    visibility_threshold=VISIBILITY_THRESHOLD,
):
    horizon = max(0, int(horizon))
    rows = int(rows)
    cols = int(cols)
    base_probability = _probability_from_grid(base_grid, rows, cols)

    agent_ids = list(tracker.agent_hmms.keys()) if tracker is not None else []
    if active_agent_ids is not None:
        active = set(active_agent_ids)
        agent_ids = [agent_id for agent_id in agent_ids if agent_id in active]
    if len(agent_ids) > 64:
        raise ValueError("Occupancy owner masks support at most 64 active agents")

    probability_grids = np.empty((horizon + 1, rows, cols), dtype=np.float32)
    owner_mask_grids = np.zeros((horizon + 1, rows, cols), dtype=np.uint64)
    overlap_count_grids = np.zeros((horizon + 1, rows, cols), dtype=np.uint8)

    free_probability = 1.0 - base_probability.astype(np.float64)
    for step in range(horizon + 1):
        probability_grids[step] = base_probability

    if tracker is not None and agent_ids:
        state_slices = _state_occupancy_slices(tracker, origin, resolution, rows, cols)
        prefixes = _agent_prefix_beliefs(tracker, agent_ids, horizon)
        dynamic_free = np.ones((horizon + 1, rows, cols), dtype=np.float64)
        agent_bit_indices = {}

        for agent_index, agent_id in enumerate(agent_ids):
            bit = np.uint64(1) << np.uint64(agent_index)
            agent_bit_indices[agent_id] = agent_index
            for step in range(horizon + 1):
                belief = np.asarray(prefixes[agent_index, step], dtype=np.float64)
                agent_grid = np.zeros((rows, cols), dtype=np.float64)
                for state_index, probability in enumerate(belief):
                    if probability <= 0.0:
                        continue
                    row_slice, col_slice, valid = state_slices[state_index]
                    if not valid:
                        continue
                    agent_grid[row_slice, col_slice] += float(probability)
                np.clip(agent_grid, 0.0, 1.0, out=agent_grid)
                dynamic_free[step] *= 1.0 - agent_grid
                owns_cell = agent_grid >= float(owner_threshold)
                owner_mask_grids[step, owns_cell] |= bit
                overlap_count_grids[step, owns_cell] += np.uint8(1)

        for step in range(horizon + 1):
            probability_grids[step] = (
                1.0 - free_probability * dynamic_free[step]
            ).astype(np.float32)
    else:
        agent_bit_indices = {}

    collision_grids = (probability_grids >= float(planning_hard_threshold)) | (
        overlap_count_grids >= 2
    )
    return OccupancyHorizon(
        probability_grids=probability_grids,
        owner_mask_grids=owner_mask_grids,
        overlap_count_grids=overlap_count_grids,
        collision_grids=collision_grids,
        origin=(float(origin[0]), float(origin[1])),
        resolution=float(resolution),
        agent_bit_indices=agent_bit_indices,
        owner_threshold=float(owner_threshold),
        planning_hard_threshold=float(planning_hard_threshold),
        visibility_threshold=float(visibility_threshold),
    )


def world_to_cell(point, origin, resolution):
    return (
        int(floor((float(point[0]) - float(origin[0])) / float(resolution))),
        int(floor((float(point[1]) - float(origin[1])) / float(resolution))),
    )


def line_cells(start, end, origin, resolution):
    sx, sy = world_to_cell(start, origin, resolution)
    ex, ey = world_to_cell(end, origin, resolution)
    cells_x, cells_y = bresenham(sx, sy, ex, ey)
    return zip(cells_x.astype(np.int64), cells_y.astype(np.int64))


def line_is_occluded_by_occupancy(
    *,
    start,
    end,
    probability_grid,
    owner_mask_grid=None,
    target_owner_bit=None,
    origin,
    resolution,
    threshold=VISIBILITY_THRESHOLD,
):
    rows, cols = probability_grid.shape
    target_owner_bit = np.uint64(0 if target_owner_bit is None else target_owner_bit)
    for col, row in line_cells(start, end, origin, resolution):
        if row < 0 or row >= rows or col < 0 or col >= cols:
            return True
        if float(probability_grid[row, col]) < float(threshold):
            continue
        if owner_mask_grid is not None and target_owner_bit:
            owners = np.uint64(owner_mask_grid[row, col])
            if owners != 0 and (owners & ~target_owner_bit) == 0:
                continue
        return True
    return False


def save_occupancy_belief_png(occupancy_horizon, path, *, step=0):
    step = int(np.clip(step, 0, occupancy_horizon.horizon))
    probability = np.asarray(occupancy_horizon.probability_grids[step], dtype=float)
    collision = np.asarray(occupancy_horizon.collision_grids[step], dtype=bool)
    overlap = np.asarray(occupancy_horizon.overlap_count_grids[step] >= 2, dtype=bool)

    base = np.clip(probability, 0.0, 1.0)
    image = np.zeros((*base.shape, 3), dtype=np.uint8)
    image[..., 0] = (255.0 * base).astype(np.uint8)
    image[..., 1] = (255.0 * (1.0 - base)).astype(np.uint8)
    image[..., 2] = (255.0 * (1.0 - base)).astype(np.uint8)
    image[collision] = np.asarray([220, 0, 0], dtype=np.uint8)
    image[overlap] = np.asarray([140, 0, 220], dtype=np.uint8)

    scale = max(1, int(np.ceil(500.0 / max(base.shape))))
    pil_image = Image.fromarray(np.flipud(image), mode="RGB")
    if scale > 1:
        nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST", Image.NEAREST)
        pil_image = pil_image.resize(
            (pil_image.width * scale, pil_image.height * scale),
            resample=nearest,
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(path)


def save_oce_debug_review_png(
    occupancy_horizon,
    path,
    *,
    robot_state=None,
    targets=None,
    static_polygons=None,
    steps=None,
    step_stride=5,
):
    rows, cols = occupancy_horizon.probability_grids.shape[1:]
    if steps is None:
        steps = [
            0,
            *range(int(step_stride), occupancy_horizon.horizon + 1, int(step_stride)),
        ]
    steps = [int(np.clip(step, 0, occupancy_horizon.horizon)) for step in steps]
    steps = list(dict.fromkeys(steps))
    if not steps:
        steps = [0]

    scale = max(1, int(np.floor(1000.0 / max(rows, cols))))
    tile_w = cols * scale
    tile_h = rows * scale
    label_h = 24
    pad = 10
    panel_cols = len(steps)
    panel_rows = 1
    image = Image.new(
        "RGB",
        (
            panel_cols * tile_w + (panel_cols + 1) * pad,
            panel_rows * (tile_h + label_h) + (panel_rows + 1) * pad,
        ),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(image)

    def world_to_pixel(point, panel_x, panel_y):
        col, row = world_to_cell(
            point, occupancy_horizon.origin, occupancy_horizon.resolution
        )
        return panel_x + col * scale + scale // 2, panel_y + row * scale + scale // 2

    def polygon_points(points, panel_x, panel_y):
        result = []
        for point in np.asarray(points, dtype=float):
            result.append(world_to_pixel(point[:2], panel_x, panel_y))
        return result

    def draw_static(panel_x, panel_y):
        for static_polygon in static_polygons or []:
            if hasattr(static_polygon, "blocking") and not getattr(
                static_polygon, "blocking", True
            ):
                continue
            points = getattr(static_polygon, "points", None)
            if isinstance(static_polygon, dict):
                if not bool(static_polygon.get("blocking", True)):
                    continue
                points = static_polygon.get("points")
                if points is None:
                    points = static_polygon.get("polygon")
                if points is None:
                    points = static_polygon.get("vertices")
            if points is None:
                continue
            pts = np.asarray(points, dtype=float)
            if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
                continue
            pix = polygon_points(pts[:, :2], panel_x, panel_y)
            draw.polygon(pix, outline=(20, 20, 20))

    def draw_robot(panel_x, panel_y):
        if robot_state is None:
            return
        state = np.asarray(robot_state, dtype=float).reshape(-1)
        if state.size < 2:
            return
        x, y = world_to_pixel(state[:2], panel_x, panel_y)
        radius = max(4, int(0.35 / max(occupancy_horizon.resolution, 1.0e-6) * scale))
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=(40, 90, 230),
            outline=(0, 0, 0),
        )
        if state.size >= 4:
            heading = float(state[3])
            hx = x + int(radius * 1.8 * np.cos(heading))
            hy = y + int(radius * 1.8 * np.sin(heading))
            draw.line([x, y, hx, hy], fill=(0, 0, 0), width=max(1, scale // 2))

    def draw_targets(panel_x, panel_y):
        for target in targets or []:
            if "pos" not in target:
                continue
            pos = np.asarray(target["pos"], dtype=float).reshape(-1)
            if pos.size < 2:
                continue
            x, y = world_to_pixel(pos[:2], panel_x, panel_y)
            extent = float(target.get("extent", occupancy_horizon.resolution))
            radius = max(
                3,
                int(
                    max(extent, occupancy_horizon.resolution * 0.5)
                    / occupancy_horizon.resolution
                    * scale
                ),
            )
            fill = (
                (20, 170, 80) if bool(target.get("visible", True)) else (245, 155, 20)
            )
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=fill,
                outline=(0, 0, 0),
            )
            # if "id" in target:
            #     draw.text((x + radius + 2, y - radius), str(target["id"]), fill=(0, 0, 0))

    for panel_idx, step in enumerate(steps):
        row_idx = panel_idx // panel_cols
        col_idx = panel_idx % panel_cols
        panel_x = pad + col_idx * (tile_w + pad)
        panel_y = pad + row_idx * (tile_h + label_h + pad) + label_h

        probability = np.clip(
            np.asarray(occupancy_horizon.probability_grids[step], dtype=float),
            0.0,
            1.0,
        )
        collision = np.asarray(occupancy_horizon.collision_grids[step], dtype=bool)
        overlap = np.asarray(
            occupancy_horizon.overlap_count_grids[step] >= 2, dtype=bool
        )
        rgb = np.empty((rows, cols, 3), dtype=np.uint8)
        rgb[..., 0] = (255.0 * probability).astype(np.uint8)
        rgb[..., 1] = (255.0 * (1.0 - probability)).astype(np.uint8)
        rgb[..., 2] = (255.0 * (1.0 - probability)).astype(np.uint8)
        rgb[collision] = np.asarray([220, 0, 0], dtype=np.uint8)
        rgb[overlap] = np.asarray([140, 0, 220], dtype=np.uint8)
        tile = Image.fromarray(rgb, mode="RGB")
        if scale > 1:
            nearest = getattr(
                getattr(Image, "Resampling", Image), "NEAREST", Image.NEAREST
            )
            tile = tile.resize((tile_w, tile_h), resample=nearest)
        image.paste(tile, (panel_x, panel_y))
        draw.rectangle(
            [panel_x, panel_y, panel_x + tile_w - 1, panel_y + tile_h - 1],
            outline=(0, 0, 0),
        )
        draw.text(
            (panel_x, panel_y - label_h + 4), f"belief step {step}", fill=(0, 0, 0)
        )
        draw_static(panel_x, panel_y)
        draw_robot(panel_x, panel_y)
        draw_targets(panel_x, panel_y)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
