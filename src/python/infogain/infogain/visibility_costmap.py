#
# Planning/Policy for visibility
#
from scipy import ndimage
import numpy as np
from time import time

from config import *
from trajectory import generate_trajectory
from polygpu import visibility_from_region

from Grid.VisibilityGrid import VisibilityGrid


def dump_grid(grid):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(grid)
    plt.show(block=False)


def update_visibility_costmap(costmap, obs, map, origin, resolution, obs_trajectory, target_trajectory, v_des, dt):
    # move the costmap to reflect the updated position of the car
    costmap.move_origin(origin)
    costmap.decay(FORGET_FACTOR)

    distance_grid = np.ones([GRID_SIZE, GRID_SIZE])

    x0 = origin[0]
    y0 = origin[1]

    locs = [
        [
            int((x - x0) / resolution),
            int((y - y0) / resolution),
        ]
        for (x, y) in zip(obs_trajectory.x, obs_trajectory.y)
    ]
    for loc in locs:
        if loc[0] < GRID_SIZE and loc[1] < GRID_SIZE:
            distance_grid[loc[1], loc[0]] = 0
    distance_grid = ndimage.distance_transform_cdt(distance_grid, metric="chessboard").astype(np.float64)

    obs_pts = np.where(distance_grid < (LANE_WIDTH + 1) / resolution)

    # place the obs points in the larger map
    obs_pts = [[x + GRID_SIZE // 2, y + GRID_SIZE // 2] for y, x in zip(*obs_pts) if y > -LANE_WIDTH + GRID_SIZE // 2]

    target_pts = []
    start_index = 0  # int(self._s0 / (self._target_speed))
    t0 = target_trajectory.t[start_index]

    # the region of interest is strictly left/right of the intended trajectory
    for index in range(start_index + 1, len(target_trajectory.x)):
        t = target_trajectory.t[index] - t0
        dy = max(1, np.ceil(t * MAX_PEDESTRIAN_SPEED / resolution))

        _x = int((target_trajectory.x[index] - target_trajectory.x[0]) / resolution + GRID_SIZE)
        if _x >= 2 * GRID_SIZE:
            break  # reached edge of the map

        _y = int((-LANE_WIDTH / 2.0 - target_trajectory.y[0]) / resolution + GRID_SIZE)

        _low_y = max(0, _y - dy)
        _high_y = min(GRID_SIZE * 2 - 1, _y + dy)

        out_of_obs = _x < GRID_SIZE / 2 or _y < GRID_SIZE / 2 or _x >= 3 * GRID_SIZE / 2 or _y >= 3 * GRID_SIZE / 2
        target_pts.extend(
            [
                (int(_x), int(y))
                for y in np.arange(_low_y, _high_y, 1)
                if (out_of_obs and map[int(y), int(_x)] < 0.9)
                or (
                    not out_of_obs
                    and obs[int(y - GRID_SIZE / 2), int(_x - GRID_SIZE / 2)] > OCCUPANCY_THRESHOLD
                    and obs[int(y - GRID_SIZE / 2), int(_x - GRID_SIZE / 2)] < 0.9
                )
            ]
        )
    target_pts = list(set(target_pts))

    def dump_targets(map, target_pts, results):
        grid = np.array(map)

        for (x, y), val in zip(target_pts, results):
            grid[y, x] = val

        dump_grid(grid)

    if len(target_pts):
        # results are num observation points rows by num region of interest points columns
        result = np.zeros((len(obs_pts), len(target_pts)))
        result = visibility_from_region(map, obs_pts, target_pts).reshape((len(obs_pts), -1))
        # log_result = -np.log(result + 0.00000001) * result
        # summed_log_result = np.sum(log_result, axis=1)

        def draw_vis(map, pts, src, result):
            visibility_map = np.array(map)
            for pt, val in zip(pts, result):
                visibility_map[pt[1], pt[0]] = max(val / 2 + 0.1, visibility_map[pt[1], pt[0]])
            visibility_map[src[1], src[0]] = 2

            dump_grid(visibility_map)

        summed_result = np.sum(result, axis=1)

        # to normalize the results, convert the visibility into a proportion of the region requested and convert to
        # non-info so we penalize locations that give nothing
        # summed_result /= len(target_pts)
        # assert np.max(summed_result) <= 1
        min_sum = np.min(summed_result)
        summed_result -= min_sum
        max_sum = np.max(summed_result)
        if max_sum:
            summed_result /= max_sum

        for pt in obs_pts:
            pt[0] = int(pt[0] - GRID_SIZE // 2)
            pt[1] = int(pt[1] - GRID_SIZE // 2)

        costmap.update(obs_pts, summed_result)
