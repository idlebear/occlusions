#
# Planning/Policy for visibility
#
from scipy import ndimage
import numpy as np
from time import time

from config import *
from polygpu import visibility_from_region

from Grid.VisibilityGrid import VisibilityGrid


def dump_grid(grid):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(grid)
    plt.show(block=False)


def _get_observation_points(
    origin,
    obs_trajectory,
    lane_width,
    costmap_dim=GRID_SIZE,
    resolution=GRID_RESOLUTION,
    map_offset_x=0,
    map_offset_y=0,
):
    # get the points where we want to observe the visibility

    distance_grid = np.ones([costmap_dim, costmap_dim])

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
        if loc[0] >= 0 and loc[0] < costmap_dim and loc[1] >= 0 and loc[1] < costmap_dim:
            distance_grid[loc[1], loc[0]] = 0
    distance_grid = ndimage.distance_transform_cdt(distance_grid, metric="chessboard").astype(np.float64)

    obs_pts = np.where(distance_grid < (lane_width / resolution))

    # place the obs points in the larger map
    obs_pts = [[x + map_offset_x, y + map_offset_y] for y, x in zip(*obs_pts)]

    return obs_pts


def update_visibility_costmap(costmap, map, centre, obs_trajectory, lane_width, target_pts):
    # move the costmap to reflect the updated position of the car - origin is the bottom left corner of the grid
    map_height, map_width = map.shape

    origin = [centre[0] - (costmap.dim / 2) * costmap.grid_size, centre[1] - (costmap.dim / 2) * costmap.grid_size]
    costmap.move_origin(origin)

    map_offset_x = (map_width - costmap.dim) // 2
    map_offset_y = (map_height - costmap.dim) // 2
    obs_pts = _get_observation_points(origin, obs_trajectory, lane_width, map_offset_x, map_offset_y)

    def dump_targets(map, target_pts, results):
        grid = np.array(map)

        for (x, y), val in zip(target_pts, results):
            grid[y, x] = val

        dump_grid(grid)

    if len(target_pts):
        # results are num observation points rows by num region of interest points columns
        result = visibility_from_region(map, obs_pts, target_pts).reshape((len(obs_pts), -1))

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
            pt[0] = int(pt[0] - map_offset_x)
            pt[1] = int(pt[1] - map_offset_y)

        costmap.update(obs_pts, summed_result)

    return obs_pts


def get_visibility_dictionary(map, obs_pts, target_pts):

    map_height, map_width = map.shape

    result = visibility_from_region(map, obs_pts, target_pts).reshape((len(obs_pts), -1))

    # probability of perception is sum of all cells from each observation point
    result = np.sum(result, axis=1).squeeze() / len(target_pts)

    visibility_dict = {}
    for pt, val in zip(obs_pts, result):
        visibility_dict[tuple(pt)] = val

    return visibility_dict
