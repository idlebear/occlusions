# Using the APCM, evaluate the probable accupancy/uncertainty of the trajectory

import numpy as np
from math import ceil, sqrt
import pandas as pd
import time
import matplotlib.pyplot as plt

from config import GRID_CELL_WIDTH, GRID_SIZE, GRID_WIDTH, DEBUG_INFORMATION_GAIN, FIG_VISIBILITY

from controller.validate import draw_agent_in_occupancy
from Grid.visibility_costmap import get_visibility_dictionary, update_visibility_costmap

from tracker.agent_track import AgentTrack


def get_agent_footprint(agent, trajectory_index=0, origin=(0, 0), resolution=0.1, prediction_num=1):
    """
    Given an agent, return the locations that the agent can observe at the current time step for the given trajectory
    """
    predictions = agent["predictions"]
    if len(predictions) == 0:
        return []

    # get the current position of the agent
    center = predictions[trajectory_index, prediction_num, :2]
    yaw = predictions[trajectory_index, prediction_num, 2]
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    x = int((center[0] - origin[0]) / resolution)
    y = int((center[1] - origin[1]) / resolution)

    # calculate the size of the rectangle in grid cells
    size = agent["size"]
    half_x = int(np.ceil(size[0] / (2.0 * resolution)))
    half_y = int(np.ceil(size[1] / (2.0 * resolution)))

    rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])

    # Create a grid of points around the agent
    dx = np.arange(-half_x, half_x + 1)
    dy = np.arange(-half_y, half_y + 1)
    grid_x, grid_y = np.meshgrid(dx, dy)
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    # Apply the rotation matrix to each point
    rotated_points = np.dot(grid_points, rotation_matrix.T)

    # Translate the rotated points to the correct position
    translated_points = rotated_points + np.array([x, y])

    return translated_points.astype(int).tolist()

def populate_grid( grid, origin, resolution, agents, target, prediction_num, belief ):
    # make a copy of the grid
    grid = np.copy(grid)

    # draw in the agents
    for agent in agents:
        if agent == target:
            continue

        N, K, D = agent["predictions"].shape
        for n in range(N):
            draw_agent_in_occupancy(
                grid,
                origin=origin,
                resolution=resolution,
                centre=agent["predictions"][n, prediction_num, :AgentTrack.DataColumm.DX],
                size=agent["size"],
                yaw=agent["predictions"][n, prediction_num, AgentTrack.DataColumm.HEADING],
                visibility=belief[agent["id"][n]],
            )
    return grid


def get_perception_dictionary(
    grid,
    origin,
    resolution,
    trajectory,
    target_agent,
    target_agent_trajectory,
    prediction_num,
):
    """
    Update the APCM based on the current occupancy grid, visibility grid, and trajectory

    Arguments:
    grid: a grid map of the environment
    origin: the origin of (0,0) corner of the grid, relative to the data
    resolution: the representative size of each cell in the map
    trajectory: the current trajectory
    visibility: the visibility grid
    map: the current map
    agents: the list of agent predictions
    prediction_num: the index number of the prediction
    """

    # get the target agent's footprint -- there are going to be duplicates, so use a set and then convert to a list
    agent_target = get_agent_footprint(
        target_agent,
        trajectory_index=target_agent_trajectory,
        origin=origin,
        resolution=resolution,
        prediction_num=prediction_num,
    )

    # TODO: we could reduce the number of observations made by limiting the observation points to those on the trajectory
    #       in the interval in question.  For now, get them all and sort it out later.
    #
    # TODO: make sure that the time-step is used instead of the prediction number
    obs_pts = [
        [
            int((pt[0] - origin[0]) / resolution),
            int((pt[1] - origin[1]) / resolution),
        ]
        for (x,y) in zip( trajectory.x, trajectory.y ))
    ]

    tic = time.time()
    visibility_dictionary = get_visibility_dictionary(
        grid,
        origin=origin,
        resolution=resolution,
        obs_trajectory=obs_pts,
        target_pts=agent_target,
    )
    costmap_update_time = time.time() - tic
    print(f"    Perception update time: {costmap_update_time:.3f} seconds")

    return visibility_dictionary


def calculate_similarity(trajectories, prediction_idx):
    """
    Calculate the similarity of each trajectory to all of the other trajectories at the requested index
    """

    similarity = []
    for t1 in trajectories:
        obs = t1[prediction_idx, :2]

        trajectory_similarity = []
        for t2 in trajectories:
            pose = t2[prediction_idx, :2]
            diff = np.linalg.norm(obs - pose)
            trajectory_similarity.append(np.exp(-diff))

        similarity.append(trajectory_similarity)

    # normalize the similarity
    similarity = np.array(similarity)
    similarity = similarity / np.sum(similarity, axis=1)[:, np.newaxis]

    return similarity


def get_collision_centers(pos, size):
    """
    Get the collision centers for the agent.   We are covering the vehicle with three circles, sized by the
    maximum of the width and one third the length of the vehicle.   The radius of each circle is sqrt(2) times
    the maximum of the width and 1/3 the length of the vehicle and centred on a 1/3 length block of the vehicle.
    """

    sep = size[0] / 3.0
    rad = sqrt(2) * max(sep, size[1] / 2.0)
    cos_yaw = np.cos(pos[2])
    sin_yaw = np.sin(pos[2])
    points = np.array(
        [
            pos[:2] + np.array([-sep * cos_yaw, -sep * sin_yaw]),
            pos[:2],
            pos[:2] + np.array([sep * cos_yaw, sep * sin_yaw]),
        ]
    )

    return points, rad


def get_min_distance(av_pos, av_size, agent_pos, agent_size):
    """
    Get the minimum distance between the av and the agent
    """
    av_points, av_rad = get_collision_centers(av_pos, av_size)
    agent_points, agent_rad = get_collision_centers(agent_pos, agent_size)

    min_distance = np.inf
    for av_pt in av_points:
        for agent_pt in agent_points:
            dist = np.linalg.norm(av_pt - agent_pt)
            if dist < min_distance:
                min_distance = dist

    return min_distance - av_rad - agent_rad


def get_relative_velocity(av_velocity, agent_velocity):
    """
    Get the relative velocity between the AV and the agent
    """
    return np.linalg.norm(av_velocity[:2] - agent_velocity[:2])


def calculate_collision_probabilities(av_pos, av_size, av_velocity, agents, prediction_num, beliefs, alpha=0.2, beta=1.0, dt=0.1):
    """
    Each agent is represented by three collision circles of radius root2*(max(length/3,width)), separated by
    length/3 along the length of the agent (left, centered, and right).

    The relative velocity is calculated by taking the difference of the two velocities and the difference of the
    two headings.  The relative velocity is then the magnitude of the relative velocity vector.

    """

    collision_probs = np.zeros(len(agents))
    for ai, agent in enumerate(agents):
        collision_count = 0
        for ti, trajectory in enumerate(agent["predictions"]):
            if beliefs[agent["id"][ti]] > alpha:
                min_distance = get_min_distance(
                    av_pos, av_size, agent["predictions"][:, prediction_num, :3], agent["size"]
                )
                relative_velocity = get_relative_velocity(
                    av_velocity,
                    agent["predictions"][:, prediction_num, AgentTrack.DataColumn.DX : AgentTrack.DataColumm.DY + 1],
                )
                ttc = min_distance / relative_velocity if relative_velocity > 0 else np.inf
                if ttc < beta:
                    collision_count += 1
        collision_probs[ai] = float(collision_count) / float(len(agent["predictions"]))

    return collision_probs


def evaluate_trajectory(grid, origin, resolution, av_trajectory, av_size, agents, prediction_iterval=0.1, dt=0.1):
    """
    Evaluate the trajectory using the APCM

    Arguments:
    grid: grid representation of the environment
    origin: location in space of grid
    resolution: size of each grid cell
    trajectory: the AV trajectory to evaluate
    agents: the list of agent predictions
    prediction_interval: time between predictions
    min_ttc: time threshold for stopping
    max_occ_prob: occupancy threshold for a collision to be considered
    dt: time interval between AV trajectory positions

    """

    beliefs = {}
    visibility = {}

    steps_per_prediction = int(prediction_iterval / dt)

    for agent in agents:
        prediction = agent.get_predictions()
        N, K, D = prediction.shape

        # initialize trajectory beliefs for each agent
        belief = np.zeros([K, N])
        belief[0, :] = 1.0 / N

        # for each prediction
        for k in range(1, K):

            tic = time.time()

            # Draw all of the other agents on the grid
            current_grid = populate_grid( grid=grid, origin=origin, resolution=resolution, agents=agents, target=agent, prediction_num=k, belief=belief[k] )

            # We need to evaluate the probability of viewing each footprint of the agent at the current time step
            visibility = {}
            for traj_num in range(N):
                visibility[traj_num] = get_perception_dictionary(
                    grid=current_grid,
                    origin=origin,
                    resolution=resolution,
                    trajectory=av_trajectory,
                    target_agent=agent,
                    target_agent_trajectory=traj_num,
                    prediction_num=k
                )

            perception_time = time.time() - tic
            print(f"Perception time: {perception_time:.3f} seconds")

            similarity = calculate_similarity(agent, agent["predictions"], k)

            time_step = k * steps_per_prediction

            obs_loc = [av_trajectory.x[time_step], av_trajectory.x[time_step]]  # get the position of the AV at the current time step

            for candidate_num in range(N):

                belief[k, candidate_num] = 0
                for traj_num in range(N):
                    target_vis = visibility[traj_num][obs_loc]

                    occ = 0
                    for alt_idx in range(N):
                        alt_vis = visibility[alt_idx][obs_loc]
                        occ += (1 - alt_vis) * beliefs[k - 1, alt_idx]

                    belief[k, candidate_num] += (
                        target_vis * similarity[candidate_num, traj_num] * belief[k - 1, traj_num]
                        + (1 - target_vis) * occ
                    )

            # normalize the belief
            belief[k, :] = belief[k, :] / np.sum(belief[k, :])

    # now evaluate each time step for the probability of stopping based on TTC > beta and occupancy/belief > alpha
    collision_probs = np.zeros([K, len(agents)])
    for k in range(K):
        time_step = k * steps_per_prediction
        av_pos = [av_trajectory.x[time_step], av_trajectory.x[time_step]]  # get the position of the AV at the current time step
        av_velocity = [av_trajectory.s_d[time_step]*np.cos(av_trajectory.yaw[time_step]),av_trajectory.s_d[time_step]*np.sin(av_trajectory.yaw[time_step])]
        collision_probs[k, :] = calculate_collision_probabilities(av_pos=av_pos, av_size=av_size, av_velocity=av_velocity, agents=agents, k=k, belief=belief, alpha=0.2, beta=1.0, dt=dt)

    # return the mean probability of stopping over all agents and the min stopping time
    return np.mean(collision_probs, axis=1), np.min(collision_probs, axis=1), np.sum(collision_probs, axis=1)


def evaluate(grid, origin, resolution, trajectories, av_size, agents, prediction_interval, stopping_threshold=1.0, dt=0.1):
    # Evaluate each of the trajectories for visibility.  The result for each trajectory is a vector with the
    # probability (mean, min, sum) of stopping at each time step.  Then, the cumulative stopping probability
    # is (1 - prob(not stopping)), which can be expressed for each time stop as
    #  1 - PROD( ( 1 - P_stop[k] ) ), k = 0...K

    trajectory_stops = np.zeros(len(trajectories))

    for traj_index, trajectory in enumerate(trajectories):
        stop_mean, stop_min, stop_sum = evaluate_trajectory(
            grid=grid,
            origin=origin,
            resolution=resolution,
            av_trajectory=trajectory,
            av_size=av_size,
            agents=agents,
            prediction_iterval=prediction_interval,
            dt=dt,
        )

        # force reasonable limits on stopping probability
        np.clip(stop_sum, 0.0, 1.0)

        # initialize, assuming we will not stop
        trajectory_stops[traj_index] = len(stop_sum)

        # Assume, for now, that the sum of probabilities is the most reasonable option
        prob_not_stop = 1
        for k in range(len(stop_sum)):
            prob_not_stop = prob_not_stop * (1.0 - stop_sum[k])
            if (1 - prob_not_stop) > stopping_threshold:
                trajectory_stops[traj_index] = k

    return np.argmin(trajectory_stops)