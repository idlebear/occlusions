#
# Planning/Policy for visibility
#
from scipy import ndimage
import numpy as np

from config import *
from trajectory import generate_trajectory
from polycheck import visibility_from_region


def build_visibility_costmap(obs, map, obs_trajectory, target_trajectory, v_des, dt):
    distance_grid = np.ones([GRID_SIZE, GRID_SIZE])

    x0 = obs_trajectory.x[0]
    y0 = obs_trajectory.y[0]

    locs = [
        [int(GRID_SIZE // 2 + (x - x0) / GRID_RESOLUTION), int(GRID_SIZE // 2 + (y - y0) / GRID_RESOLUTION)]
        for (x, y) in zip(obs_trajectory.x, obs_trajectory.y)
    ]
    for loc in locs:
        distance_grid[loc[1], loc[0]] = 0
    distance_grid = ndimage.distance_transform_cdt(distance_grid, metric="chessboard")

    obs_pts = np.where(distance_grid < LANE_WIDTH / GRID_RESOLUTION)
    obs_pts = [[x + GRID_SIZE // 2, y + GRID_SIZE // 2] for y, x in zip(*obs_pts)]

    target_pts = []
    start_index = 0  # int(self._s0 / (self._target_speed))
    t0 = target_trajectory.t[start_index]

    # the region of interest is strictly left/right of the intended trajectory
    target_grid = np.zeros([GRID_SIZE * 2, GRID_SIZE * 2])
    for index in range(start_index + 1, len(target_trajectory.x)):
        t = target_trajectory.t[index] - t0
        dy = max(1, int(t * OPPONENT_CAR_SPEED / GRID_RESOLUTION))

        _x = int((target_trajectory.x[index] - target_trajectory.x[0]) / GRID_RESOLUTION + GRID_SIZE)
        _y = int((target_trajectory.y[index] - target_trajectory.y[0]) / GRID_RESOLUTION + GRID_SIZE)

        _low_y = max(0, _y - dy)
        _high_y = min(GRID_SIZE * 2 - 1, _y + dy)

        target_pts.extend([(_x, y) for y in np.arange(_low_y, _high_y, 1)])

    # results are num observation points rows by num region of interest points columns
    result = np.zeros((len(obs_pts), len(target_pts)))
    visibility_from_region(map, obs_pts, target_pts, result)
    log_result = -np.log(result + 0.00000001) * result

    visibility_map1 = np.zeros_like(map)
    vis_pt = obs_pts[-1]
    vis_data = result[-1]
    for pt, val in zip(target_pts, vis_data):
        visibility_map1[pt[1], pt[0]] = val
    visibility_map1[vis_pt[1], vis_pt[0]] = 2

    visibility_map2 = np.zeros_like(map)
    vis_pt = obs_pts[0]
    vis_data = result[0]
    for pt, val in zip(target_pts, vis_data):
        visibility_map2[pt[1], pt[0]] = val
    visibility_map2[vis_pt[1], vis_pt[0]] = 2

    summed_result = np.sum(result, axis=1)

    summed_log_result = np.sum(log_result, axis=1)

    for pt, val in zip(obs_pts, summed_result):
        target_grid[pt[1], pt[0]] = val

    return obs_pts, target_pts
