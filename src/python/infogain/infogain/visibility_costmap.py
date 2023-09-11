#
# Planning/Policy for visibility
#
from scipy import ndimage
import numpy as np
from time import time

from config import *
from trajectory import generate_trajectory
from polycheck import visibility_from_region


def dump_grid(grid):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(grid)
    plt.show(block=False)


def build_visibility_costmap(
    obs, map, origin, resolution, obs_trajectory, target_trajectory, v_des, dt
):
    tic = time()

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
    distance_grid = ndimage.distance_transform_cdt(
        distance_grid, metric="chessboard"
    ).astype(np.float64)

    obs_pts = np.where(distance_grid < LANE_WIDTH / resolution)
    obs_pts = [[x + GRID_SIZE // 2, y + GRID_SIZE // 2] for y, x in zip(*obs_pts)]

    target_pts = []
    start_index = 0  # int(self._s0 / (self._target_speed))
    t0 = target_trajectory.t[start_index]

    # the region of interest is strictly left/right of the intended trajectory
    for index in range(start_index + 1, len(target_trajectory.x)):
        t = target_trajectory.t[index] - t0
        dy = max(1, int(t * OPPONENT_CAR_SPEED / resolution))

        _x = int(
            (target_trajectory.x[index] - target_trajectory.x[0]) / resolution
            + GRID_SIZE
        )
        _y = int(
            (target_trajectory.y[index] - target_trajectory.y[0]) / resolution
            + GRID_SIZE
        )

        _low_y = max(0, _y - dy)
        _high_y = min(GRID_SIZE * 2 - 1, _y + dy)

        out_of_obs = (
            _x < GRID_SIZE / 2
            or _y < GRID_SIZE / 2
            or _x > 3 * GRID_SIZE / 2
            or _y > 3 * GRID_SIZE / 2
        )
        target_pts.extend(
            [
                (_x, y)
                for y in np.arange(_low_y, _high_y, 1)
                if out_of_obs
                or obs[int(y - GRID_SIZE / 2), int(_x - GRID_SIZE / 2)]
                > OCCUPANCY_THRESHOLD
            ]
        )

    # start the visibility grid based on the supplied map, ensuring any occupied locations are toxic
    visibility_grid = np.ones([GRID_SIZE, GRID_SIZE])

    if len(target_pts):
        # results are num observation points rows by num region of interest points columns
        result = np.zeros((len(obs_pts), len(target_pts)))
        visibility_from_region(map, obs_pts, target_pts, result)
        # log_result = -np.log(result + 0.00000001) * result
        # summed_log_result = np.sum(log_result, axis=1)

        # visibility_map1 = np.zeros_like(map)
        # vis_pt = obs_pts[-1]
        # vis_data = result[-1]
        # for pt, val in zip(target_pts, vis_data):
        #     visibility_map1[pt[1], pt[0]] = val
        # visibility_map1[vis_pt[1], vis_pt[0]] = 2

        # visibility_map2 = np.zeros_like(map)
        # vis_pt = obs_pts[0]
        # vis_data = result[0]
        # for pt, val in zip(target_pts, vis_data):
        #     visibility_map2[pt[1], pt[0]] = val
        # visibility_map2[vis_pt[1], vis_pt[0]] = 2

        summed_result = np.sum(result, axis=1)
        min_result = np.min(summed_result)
        if min_result > 0:
            summed_result -= min_result
        max_result = np.max(summed_result)
        if max_result > 0:
            # we want the trajectories that have more visibility to be rewarded, so reverse the
            # sense -- no information is 'bad'
            summed_result = 1 - np.square(summed_result / max_result)

            # add in the visibility values - making sure to account for the offset to the larger target map
            for pt, val in zip(obs_pts, summed_result):
                visibility_grid[
                    int(pt[1] - GRID_SIZE // 2), int(pt[0] - GRID_SIZE // 2)
                ] = val
        else:
            # nothing to do
            pass

    return visibility_grid
