import argparse
import cProfile
import csv
import io
import pstats
import re
from heapq import heappop, heappush
from pathlib import Path
from random import seed
from simulation import Simulation
from config import *
import pygame
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from scipy import sparse
import json

from os import mkdir
from time import time, perf_counter
from pickle import load, dump
from math import floor, sqrt
import numpy as np
from PIL import Image, ImageDraw

from controller.ModelParameters.Ackermann import Ackermann4
from warp_mppi.legacy import PyCudaMPPI as MPPI
from warp_mppi.legacy import evaluate_trajectories_by_entropy_gpu
from warp_mppi.legacy import evaluate_discrete_oce_gpu

from controller.validate import (
    visualize_variations,
    run_trajectory,
    rollout_trajectories,
    validate_controls,
)

from controller.velocity_prediction import ControlVariations
from controller.ModelParameters.Unicycle import Unicycle

# from tracker.tracker import Tracker
from trajectory_planner.trajectory_planner import TrajectoryPlanner
from trajectory_planner.frenet_optimal_trajectory import (
    generate_target_course,
    frenet_optimal_planning,
    PlannerArgs,
    Frenet_path,
)
from trajectory_planner.trajectory_eval import evaluate

from Actor import STATE as ActorStateEnum
from entropy import evaluate_method
from hmm import HMM
from specialk import generate_specialk_trajectories


def blocking_static_polygon_union(static_polygons):
    polygons = []
    for static_polygon in static_polygons or []:
        if not getattr(static_polygon, "blocking", True):
            continue
        points = np.asarray(getattr(static_polygon, "points", []), dtype=float)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
            continue
        polygon = Polygon(points[:, :2])
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_empty:
            polygons.append(polygon)
    return unary_union(polygons) if polygons else None


def vehicle_footprint_polygon(state, length, width):
    x, y, _v, theta = np.asarray(state, dtype=float)[:4]
    half_length = float(length) / 2.0
    half_width = float(width) / 2.0
    corners = np.asarray(
        [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width],
        ],
        dtype=float,
    )
    c = np.cos(theta)
    s = np.sin(theta)
    rotation = np.asarray([[c, -s], [s, c]], dtype=float)
    return Polygon(corners @ rotation.T + np.asarray([x, y], dtype=float))


def trajectory_collides_with_static(
    trajectory,
    static_union,
    *,
    vehicle_length,
    vehicle_width,
    clearance=0.0,
):
    if static_union is None or static_union.is_empty:
        return False
    collision_region = static_union.buffer(clearance) if clearance > 0 else static_union
    for state in trajectory:
        footprint = vehicle_footprint_polygon(state, vehicle_length, vehicle_width)
        if footprint.intersects(collision_region):
            return True
    return False


def filter_static_collision_samples(
    *,
    robot_model,
    initial_state,
    u_nom,
    u_variations,
    weights,
    static_polygons,
    dt,
    clearance=0.0,
):
    static_union = blocking_static_polygon_union(static_polygons)
    if static_union is None or static_union.is_empty:
        return weights, None, None

    input_weights = np.asarray(weights, dtype=np.float32).copy()
    filtered_weights = input_weights.copy()
    sampled_controls = np.asarray(u_nom, dtype=np.float32)[
        np.newaxis, :, :
    ] + np.asarray(
        u_variations,
        dtype=np.float32,
    )
    collision_free = np.zeros(sampled_controls.shape[0], dtype=bool)
    for sample_idx, controls in enumerate(sampled_controls):
        trajectory = run_trajectory(
            vehicle=robot_model,
            initial_state=initial_state,
            controls=controls,
            dt=dt,
        )
        collision_free[sample_idx] = not trajectory_collides_with_static(
            trajectory,
            static_union,
            vehicle_length=robot_model.L,
            vehicle_width=robot_model.W,
            clearance=clearance,
        )

    filtered_weights[~collision_free] = 0.0
    if np.any(collision_free) and float(np.sum(filtered_weights)) <= 0.0:
        filtered_weights[collision_free] = 1.0

    debug = {
        "sampled_controls": sampled_controls,
        "collision_free": collision_free,
        "input_weights": input_weights,
        "filtered_weights": filtered_weights,
    }
    return filtered_weights, static_union, debug


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def deg(angle):
    return float(np.degrees(angle))


def expected_yaw_delta(state, control, dt, vehicle_length):
    state = np.asarray(state, dtype=float)
    control = np.asarray(control, dtype=float)
    if control.size < 2:
        return 0.0
    velocity_midpoint = state[ActorStateEnum.VELOCITY] + 0.5 * control[0] * dt
    return velocity_midpoint * np.tan(control[1]) / max(vehicle_length, EPSILON) * dt


def bounded_weighted_control_average(u_nom, sampled_controls, weights, limits):
    """Average bounded controls in tanh space to avoid steering boundary bias."""
    weights = np.asarray(weights, dtype=float)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return np.asarray(u_nom, dtype=float).copy()

    limits = np.asarray(limits, dtype=float)
    u_nom = np.asarray(u_nom, dtype=float)
    sampled_controls = np.asarray(sampled_controls, dtype=float)
    eps = 1.0e-5

    def to_z(controls):
        unit_controls = np.clip(
            controls / np.maximum(limits, np.finfo(float).eps),
            -1.0 + eps,
            1.0 - eps,
        )
        return np.arctanh(unit_controls)

    z_nom = to_z(u_nom)
    z_samples = to_z(sampled_controls)
    normalized_weights = weights / weight_sum
    weighted_delta_z = np.sum(
        (z_samples - z_nom[np.newaxis, :, :])
        * normalized_weights[:, np.newaxis, np.newaxis],
        axis=0,
    )
    return limits * np.tanh(z_nom + weighted_delta_z)


def _agent_prediction_sequence(agent, agent_predictions, horizon):
    """Return horizon future states for an agent in MPPI state layout."""
    horizon = int(horizon)
    current = np.asarray(agent.get("pos", np.zeros(4)), dtype=float).reshape(-1)
    if current.size < 4:
        current = np.pad(current, (0, 4 - current.size), mode="constant")

    agent_id = agent.get("id")
    prediction = None
    if agent_predictions is not None and agent_id in agent_predictions:
        prediction = agent_predictions[agent_id]
    elif agent.get("future") is not None:
        prediction = agent.get("future")

    if prediction is None:
        states = np.repeat(current[:4][np.newaxis, :], horizon, axis=0)
        return states.astype(np.float32), "current"

    states = np.asarray(prediction, dtype=float)
    if states.ndim == 3:
        states = np.nanmean(states, axis=0)
    if states.ndim == 1:
        states = states.reshape(1, -1)
    if states.ndim != 2 or states.shape[0] == 0 or states.shape[1] < 2:
        states = np.repeat(current[:4][np.newaxis, :], horizon, axis=0)
        return states.astype(np.float32), "current"

    if states.shape[0] > 1 and np.linalg.norm(states[0, :2] - current[:2]) < 1.0e-4:
        states = states[1:]

    normalized = np.repeat(current[:4][np.newaxis, :], max(horizon, 1), axis=0)
    valid_steps = min(horizon, states.shape[0])
    state_cols = min(4, states.shape[1])
    normalized[:valid_steps, :state_cols] = states[:valid_steps, :state_cols]
    if valid_steps > 0 and valid_steps < horizon:
        normalized[valid_steps:, :] = normalized[valid_steps - 1, :]
    return normalized[:horizon].astype(np.float32), "prediction"


def scene_scale(robot_model=None):
    if robot_model is None:
        return 1.0
    return float(
        getattr(
            robot_model,
            "scene_scale",
            getattr(robot_model, "size_scale", getattr(robot_model, "scale", 1.0)),
        )
    )


def mppi_dynamic_collision_buffer(robot_model=None, hard_clearance_margin=0.0):
    # CUDA checks the ego footprint separately using vehicle_width/vehicle_length,
    # so this buffer is only for hard collision inflation. Desired social
    # clearance is handled separately by the soft dynamic clearance cost.
    return float(hard_clearance_margin) * scene_scale(robot_model)


def mppi_dynamic_clearance_margin(robot_model, clearance_margin):
    return float(clearance_margin) * scene_scale(robot_model)


def mppi_static_clearance_margin(robot_model, clearance_margin):
    return float(clearance_margin) * scene_scale(robot_model)


def build_dynamic_obstacles_for_mppi(
    agents, agent_predictions, horizon, robot_model=None, hard_clearance_margin=0.0
):
    obstacles = []
    debug = []
    collision_buffer = mppi_dynamic_collision_buffer(
        robot_model,
        hard_clearance_margin,
    )
    for agent in agents:
        states, source = _agent_prediction_sequence(agent, agent_predictions, horizon)
        extent = float(agent.get("extent", 0.0))
        obstacles.append(
            {
                "id": agent.get("id"),
                "states": states,
                "extent": [extent, extent],
                "collision_buffer": collision_buffer,
            }
        )
        debug.append(
            {
                "id": agent.get("id"),
                "source": source,
                "extent": extent,
                "collision_buffer": collision_buffer,
                "steps": int(states.shape[0]),
                "start": states[0, :2].astype(float).tolist() if len(states) else None,
                "end": states[-1, :2].astype(float).tolist() if len(states) else None,
            }
        )
    return obstacles, debug


def dynamic_agent_collision_threshold(agent, robot_model, clearance):
    robot_radius = float(max(robot_model.W, robot_model.L) / 2.0)
    return float(agent.get("extent", 0.0)) + max(0.0, float(clearance)) + robot_radius


def collision_geometry_from_extent(extent, buffer=0.0):
    extent_arr = np.asarray(extent, dtype=float).reshape(-1)
    if extent_arr.size <= 0:
        return np.asarray([0.0, 0.0, 0.0], dtype=float)
    half_length = abs(float(extent_arr[0]))
    half_width = abs(float(extent_arr[1])) if extent_arr.size > 1 else half_length
    margin = max(0.0, float(buffer))
    half_length += margin
    half_width += margin
    radius = max(min(half_length, half_width), 1.0e-3)
    if half_length >= half_width:
        return np.asarray([radius, max(half_length - radius, 0.0), 0.0], dtype=float)
    return np.asarray([radius, 0.0, max(half_width - radius, 0.0)], dtype=float)


def collision_circle_centers(state, geometry):
    x, y, _v, theta = np.asarray(state, dtype=float).reshape(-1)[:4]
    radius, offset_x, offset_y = np.asarray(geometry, dtype=float).reshape(3)
    c = np.cos(theta)
    s = np.sin(theta)
    centers = []
    for local_x, local_y in (
        (-offset_x, -offset_y),
        (0.0, 0.0),
        (offset_x, offset_y),
    ):
        centers.append([x + local_x * c - local_y * s, y + local_x * s + local_y * c])
    return radius, np.asarray(centers, dtype=float)


def three_circle_min_clearance_host(
    ego_state, ego_geometry, actor_state, actor_geometry
):
    ego_radius, ego_centers = collision_circle_centers(ego_state, ego_geometry)
    actor_radius, actor_centers = collision_circle_centers(actor_state, actor_geometry)
    deltas = ego_centers[:, np.newaxis, :] - actor_centers[np.newaxis, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    min_distance = float(np.min(distances))
    min_clearance = min_distance - float(ego_radius + actor_radius)
    return min_clearance, min_distance, float(ego_radius + actor_radius)


def trajectory_dynamic_collision_summary(
    trajectory,
    agents,
    agent_predictions,
    horizon,
    robot_model,
    clearance,
):
    min_distance = float("inf")
    min_clearance = float("inf")
    closest = None
    collision = False
    ego_geometry = collision_geometry_from_extent(
        (float(robot_model.L) / 2.0, float(robot_model.W) / 2.0)
    )
    for agent in agents:
        states, source = _agent_prediction_sequence(agent, agent_predictions, horizon)
        extent = float(agent.get("extent", 0.0))
        actor_geometry = collision_geometry_from_extent(
            (extent, extent),
            buffer=clearance,
        )
        steps = min(states.shape[0], max(0, trajectory.shape[0] - 1))
        for step_idx in range(steps):
            clearance_value, dist, threshold = three_circle_min_clearance_host(
                trajectory[step_idx + 1, :4],
                ego_geometry,
                states[step_idx, :4],
                actor_geometry,
            )
            if clearance_value < min_clearance:
                min_clearance = clearance_value
            if dist < min_distance:
                min_distance = dist
                closest = {
                    "agent_id": agent.get("id"),
                    "step": step_idx,
                    "time": None,
                    "threshold": threshold,
                    "clearance": clearance_value,
                    "source": source,
                }
            if clearance_value < 0.0:
                collision = True
    return {
        "collision": collision,
        "min_distance": min_distance,
        "min_clearance": min_clearance,
        "closest": closest,
    }


def filter_dynamic_collision_samples(
    *,
    robot_model,
    initial_state,
    u_nom,
    u_variations,
    weights,
    agents,
    agent_predictions,
    horizon,
    dt,
    clearance=None,
):
    if clearance is None:
        clearance = MIN_SEPARATION * scene_scale(robot_model)
    input_weights = np.asarray(weights, dtype=np.float32).copy()
    filtered_weights = input_weights.copy()
    sampled_controls = np.asarray(u_nom, dtype=np.float32)[
        np.newaxis, :, :
    ] + np.asarray(u_variations, dtype=np.float32)

    collision_free = np.ones(sampled_controls.shape[0], dtype=bool)
    min_distances = np.full(sampled_controls.shape[0], np.inf, dtype=np.float32)
    for sample_idx, controls in enumerate(sampled_controls):
        trajectory = run_trajectory(
            vehicle=robot_model,
            initial_state=initial_state,
            controls=controls,
            dt=dt,
        )
        summary = trajectory_dynamic_collision_summary(
            trajectory=trajectory,
            agents=agents,
            agent_predictions=agent_predictions,
            horizon=horizon,
            robot_model=robot_model,
            clearance=clearance,
        )
        collision_free[sample_idx] = not summary["collision"]
        min_distances[sample_idx] = summary["min_distance"]

    if np.any(collision_free):
        filtered_weights[~collision_free] = 0.0
    else:
        # If every sample violates desired clearance, preserving MPPI's relative
        # weights avoids deadlocking into a zero-weight emergency stop. The final
        # hard-overlap check below still catches true selected-trajectory collisions.
        filtered_weights = input_weights.copy()

    if np.any(collision_free) and float(np.sum(filtered_weights)) <= 0.0:
        filtered_weights[collision_free] = 1.0

    debug = {
        "sampled_controls": sampled_controls,
        "collision_free": collision_free,
        "input_weights": input_weights,
        "filtered_weights": filtered_weights,
        "min_distances": min_distances,
        "all_samples_violate_clearance": not bool(np.any(collision_free)),
        "clearance": float(clearance),
    }
    return filtered_weights, debug


def control_sequence_collision_summary(
    *,
    robot_model,
    initial_state,
    controls,
    dt,
    static_union=None,
    static_clearance=0.0,
    agents=None,
    agent_predictions=None,
    horizon=None,
    dynamic_clearance=0.0,
):
    trajectory = run_trajectory(
        vehicle=robot_model,
        initial_state=initial_state,
        controls=controls,
        dt=dt,
    )
    static_collision = False
    if static_union is not None:
        static_collision = trajectory_collides_with_static(
            trajectory,
            static_union,
            vehicle_length=robot_model.L,
            vehicle_width=robot_model.W,
            clearance=static_clearance,
        )

    dynamic_summary = {
        "collision": False,
        "min_distance": float("inf"),
        "closest": None,
    }
    if agents:
        dynamic_summary = trajectory_dynamic_collision_summary(
            trajectory=trajectory,
            agents=agents,
            agent_predictions=agent_predictions,
            horizon=horizon if horizon is not None else len(controls),
            robot_model=robot_model,
            clearance=dynamic_clearance,
        )

    return {
        "trajectory": trajectory,
        "static_collision": static_collision,
        "dynamic_collision": bool(dynamic_summary["collision"]),
        "dynamic_summary": dynamic_summary,
        "safe": not static_collision and not bool(dynamic_summary["collision"]),
    }


def find_safe_control_candidate(
    *,
    robot_model,
    initial_state,
    u_nom,
    u_variations,
    weights,
    dt,
    static_union=None,
    static_clearance=0.0,
    agents=None,
    agent_predictions=None,
    horizon=None,
    dynamic_clearance=0.0,
    max_candidates=200,
):
    candidates = [("nominal", None, np.asarray(u_nom, dtype=float))]
    if u_variations is not None and len(u_variations):
        sampled_controls = np.asarray(u_nom, dtype=float)[
            np.newaxis, :, :
        ] + np.asarray(
            u_variations,
            dtype=float,
        )
        weights_arr = np.asarray(weights, dtype=float).reshape(-1)
        if weights_arr.shape[0] == sampled_controls.shape[0]:
            order = np.argsort(weights_arr)[::-1]
        else:
            order = np.arange(sampled_controls.shape[0])
        for sample_idx in order[: min(int(max_candidates), sampled_controls.shape[0])]:
            candidates.append(
                (
                    "sample",
                    int(sample_idx),
                    sampled_controls[int(sample_idx)],
                )
            )

    best_collision = None
    for source, sample_idx, controls in candidates:
        summary = control_sequence_collision_summary(
            robot_model=robot_model,
            initial_state=initial_state,
            controls=controls,
            dt=dt,
            static_union=static_union,
            static_clearance=static_clearance,
            agents=agents,
            agent_predictions=agent_predictions,
            horizon=horizon,
            dynamic_clearance=dynamic_clearance,
        )
        if summary["safe"]:
            summary.update(
                {
                    "source": source,
                    "sample_idx": sample_idx,
                    "controls": controls,
                }
            )
            return summary
        if best_collision is None:
            best_collision = summary
    return best_collision


def path_state(path, index):
    index = min(max(index, 0), len(path.x) - 1)
    return np.asarray(
        [path.x[index], path.y[index], path.s_d[index], path.yaw[index]],
        dtype=float,
    )


def path_index_at_lookahead(path, state, start_index, lookahead_distance):
    start_index = min(max(start_index, 0), len(path.x) - 1)
    state_xy = np.asarray(state[:2], dtype=float)
    fallback_index = len(path.x) - 1
    for index in range(start_index, len(path.x)):
        point_xy = np.asarray([path.x[index], path.y[index]], dtype=float)
        if np.linalg.norm(point_xy - state_xy) >= lookahead_distance:
            return index
    return fallback_index


def path_state_at_lookahead(path, state, start_index, lookahead_distance):
    return path_state(
        path,
        path_index_at_lookahead(path, state, start_index, lookahead_distance),
    )


def closest_path_index(path, state):
    points = np.column_stack([path.x, path.y])
    state_xy = np.asarray(state[:2], dtype=float)
    distances = np.linalg.norm(points - state_xy, axis=1)
    return int(np.argmin(distances))


def nominal_steering_components(robot_model, state, target, args, lookahead_distance):
    dx = target[0] - state[ActorStateEnum.X]
    dy = target[1] - state[ActorStateEnum.Y]
    distance = float(np.hypot(dx, dy))

    if distance > 1.0e-6:
        target_heading = np.arctan2(dy, dx)
    else:
        target_heading = target[3]

    bearing_heading_error = wrap_angle(target_heading - state[ActorStateEnum.THETA])
    path_heading_error = wrap_angle(target[3] - state[ActorStateEnum.THETA])

    lookahead = max(distance, lookahead_distance)
    pure_pursuit_delta = np.arctan2(
        2.0 * robot_model.L * np.sin(bearing_heading_error),
        lookahead,
    )

    path_direction = np.asarray(
        [np.cos(target[3]), np.sin(target[3])],
        dtype=float,
    )
    target_vector = np.asarray([dx, dy], dtype=float)
    cross_track_error = (
        path_direction[0] * target_vector[1] - path_direction[1] * target_vector[0]
    )
    speed_for_heading = max(abs(state[ActorStateEnum.VELOCITY]), 0.15)
    steering_cross_track_gain = getattr(args, "steering_cross_track_gain", 0.75)
    steering_heading_gain = getattr(args, "steering_heading_gain", 2.0)
    steering_opposing_cross_track_scale = getattr(
        args,
        "steering_opposing_cross_track_scale",
        0.25,
    )
    cross_track_delta = np.arctan2(
        steering_cross_track_gain * cross_track_error,
        speed_for_heading,
    )
    heading_delta = steering_heading_gain * path_heading_error

    effective_cross_track_delta = cross_track_delta
    if (
        abs(path_heading_error) > np.deg2rad(2.0)
        and np.sign(path_heading_error) != 0
        and np.sign(cross_track_delta) != 0
        and np.sign(path_heading_error) != np.sign(cross_track_delta)
    ):
        effective_cross_track_delta *= steering_opposing_cross_track_scale

    stanley_delta = heading_delta + effective_cross_track_delta

    if np.sign(pure_pursuit_delta) == np.sign(stanley_delta):
        delta = (
            stanley_delta
            if abs(stanley_delta) > abs(pure_pursuit_delta)
            else pure_pursuit_delta
        )
    elif abs(path_heading_error) > np.deg2rad(2.0):
        delta = stanley_delta
    else:
        delta = 0.7 * stanley_delta + 0.3 * pure_pursuit_delta

    return {
        "target_heading": target_heading,
        "bearing_heading_error": bearing_heading_error,
        "path_heading_error": path_heading_error,
        "distance": distance,
        "lookahead": lookahead,
        "cross_track_error": cross_track_error,
        "pure_pursuit_delta": pure_pursuit_delta,
        "heading_delta": heading_delta,
        "cross_track_delta": cross_track_delta,
        "effective_cross_track_delta": effective_cross_track_delta,
        "stanley_delta": stanley_delta,
        "delta": delta,
    }


def nominal_controls_to_path(robot_model, initial_state, path, args):
    u_nom = np.zeros((args.horizon, 2), dtype=float)
    state = np.asarray(initial_state[:4], dtype=float).copy()
    lookahead_distance = max(
        1.5 * float(robot_model.L),
        args.robot_speed * args.tick_time * 4.0,
        0.05,
    )

    for i in range(args.horizon):
        target = path_state_at_lookahead(
            path,
            state,
            start_index=min(i + 1, len(path.x) - 1),
            lookahead_distance=lookahead_distance,
        )
        steering = nominal_steering_components(
            robot_model,
            state,
            target,
            args,
            lookahead_distance,
        )
        delta = steering["delta"]

        turn_severity = min(
            max(
                abs(steering["path_heading_error"]),
                abs(steering["bearing_heading_error"]),
            )
            / (np.pi / 2.0),
            1.0,
        )
        target_speed = args.robot_speed * (1.0 - 0.75 * turn_severity)
        target_speed = max(0.15, target_speed)
        accel = (target_speed - state[2]) / args.tick_time

        accel = np.clip(accel, -CONTROL_LIMITS[0], CONTROL_LIMITS[0])
        delta = np.clip(delta, -CONTROL_LIMITS[1], CONTROL_LIMITS[1])
        u_nom[i] = [accel, delta]

        step = robot_model.ode(state.copy(), u_nom[i].copy())
        state = state + step * args.tick_time

    return u_nom


def print_planned_steering_debug(
    *,
    tick,
    robot_model,
    initial_state,
    path,
    u_nom,
    u_final,
    args,
    total_weight,
    u_weights,
    static_union,
    emergency_stop,
):
    state = np.asarray(initial_state[:4], dtype=float)
    lookahead_distance = max(
        1.5 * float(robot_model.L),
        args.robot_speed * args.tick_time * 4.0,
        0.05,
    )
    target_index = path_index_at_lookahead(
        path,
        state,
        start_index=min(1, len(path.x) - 1),
        lookahead_distance=lookahead_distance,
    )
    target = path_state(path, target_index)
    closest_index = closest_path_index(path, state)
    closest = path_state(path, closest_index)
    closest_distance = float(np.linalg.norm(closest[:2] - state[:2]))
    closest_heading_error = wrap_angle(closest[3] - state[ActorStateEnum.THETA])
    steering = nominal_steering_components(
        robot_model,
        state,
        target,
        args,
        lookahead_distance,
    )
    nom_delta_theta = expected_yaw_delta(
        state,
        u_nom[0],
        args.tick_time,
        robot_model.L,
    )
    final_delta_theta = expected_yaw_delta(
        state,
        u_final[0],
        args.tick_time,
        robot_model.L,
    )
    projected_target_error = wrap_angle(
        steering["path_heading_error"] - final_delta_theta
    )
    if abs(final_delta_theta) > np.deg2rad(0.05):
        ticks_to_align = abs(steering["path_heading_error"] / final_delta_theta)
    else:
        ticks_to_align = float("inf")
    weight_sum = float(np.sum(u_weights))
    ess = (weight_sum**2) / (float(np.sum(np.asarray(u_weights) ** 2)) + 1.0e-12)
    static_label = "yes" if static_union is not None else "no"

    print(
        "[steering-debug plan] "
        f"tick={tick} pos=({state[0]:.3f},{state[1]:.3f}) "
        f"theta={deg(state[3]):.1f}deg v={state[2]:.3f} "
        f"target=({target[0]:.3f},{target[1]:.3f}) "
        f"bearing={deg(steering['target_heading']):.1f}deg "
        f"path_yaw={deg(target[3]):.1f}deg "
        f"bearing_err={deg(steering['bearing_heading_error']):.1f}deg "
        f"path_err={deg(steering['path_heading_error']):.1f}deg"
    )
    print(
        "[steering-debug tracking] "
        f"tick={tick} closest_idx={closest_index} "
        f"closest_dist={closest_distance:.3f} "
        f"closest_yaw={deg(closest[3]):.1f}deg "
        f"closest_err={deg(closest_heading_error):.1f}deg "
        f"target_idx={target_index} "
        f"target_dist={steering['distance']:.3f} "
        f"projected_target_err_next={deg(projected_target_error):.1f}deg "
        f"ticks_to_align={ticks_to_align:.1f}"
    )
    print(
        "[steering-debug terms] "
        f"tick={tick} "
        f"pure={deg(steering['pure_pursuit_delta']):.1f}deg "
        f"heading={deg(steering['heading_delta']):.1f}deg "
        f"cross={deg(steering['cross_track_delta']):.1f}deg "
        f"cross_eff={deg(steering['effective_cross_track_delta']):.1f}deg "
        f"stanley={deg(steering['stanley_delta']):.1f}deg "
        f"selected={deg(steering['delta']):.1f}deg "
        f"cte={steering['cross_track_error']:.3f}"
    )
    print(
        "[steering-debug control] "
        f"tick={tick} "
        f"u_nom=[a={u_nom[0,0]:.3f}, steer={deg(u_nom[0,1]):.1f}deg] "
        f"u_final=[a={u_final[0,0]:.3f}, steer={deg(u_final[0,1]):.1f}deg] "
        f"expected_dtheta_nom={deg(nom_delta_theta):.2f}deg/tick "
        f"expected_dtheta_final={deg(final_delta_theta):.2f}deg/tick "
        f"limits=[a={CONTROL_LIMITS[0]:.2f}, steer={deg(CONTROL_LIMITS[1]):.1f}deg] "
        f"model=[L={robot_model.L:.3f}, W={robot_model.W:.3f}, "
        f"max_steer={deg(robot_model.max_delta):.1f}deg]"
    )
    print(
        "[steering-debug weights] "
        f"tick={tick} total_weight={total_weight:.3e} ess={ess:.2f} "
        f"static_filter={static_label} emergency_stop={emergency_stop}"
    )


def print_mppi_filter_debug(
    *,
    tick,
    u_nom,
    u_mppi,
    u_final,
    pre_filter_weights,
    post_filter_weights,
    static_debug,
):
    pre_filter_weights = np.asarray(pre_filter_weights, dtype=float)
    post_filter_weights = np.asarray(post_filter_weights, dtype=float)
    pre_sum = float(np.sum(pre_filter_weights))
    post_sum = float(np.sum(post_filter_weights))
    pre_ess = (pre_sum**2) / (float(np.sum(pre_filter_weights**2)) + 1.0e-12)
    post_ess = (post_sum**2) / (float(np.sum(post_filter_weights**2)) + 1.0e-12)

    print(
        "[steering-debug mppi] "
        f"tick={tick} "
        f"u_nom=[a={u_nom[0,0]:.3f}, steer={deg(u_nom[0,1]):.1f}deg] "
        f"u_mppi=[a={u_mppi[0,0]:.3f}, steer={deg(u_mppi[0,1]):.1f}deg] "
        f"u_final=[a={u_final[0,0]:.3f}, steer={deg(u_final[0,1]):.1f}deg] "
        f"pre_weight={pre_sum:.3e} pre_ess={pre_ess:.2f} "
        f"post_weight={post_sum:.3e} post_ess={post_ess:.2f}"
    )

    if static_debug is None:
        print(f"[steering-debug static] tick={tick} no static filter applied")
        return

    sampled_controls = np.asarray(static_debug["sampled_controls"], dtype=float)
    collision_free = np.asarray(static_debug["collision_free"], dtype=bool)
    safe_controls = sampled_controls[collision_free]
    safe_count = int(np.count_nonzero(collision_free))
    total_count = int(collision_free.size)

    if safe_controls.size:
        first_safe_steer = safe_controls[:, 0, 1]
        first_safe_accel = safe_controls[:, 0, 0]
        steer_min = deg(np.min(first_safe_steer))
        steer_median = deg(np.median(first_safe_steer))
        steer_max = deg(np.max(first_safe_steer))
        accel_min = float(np.min(first_safe_accel))
        accel_median = float(np.median(first_safe_accel))
        accel_max = float(np.max(first_safe_accel))
    else:
        steer_min = steer_median = steer_max = float("nan")
        accel_min = accel_median = accel_max = float("nan")

    print(
        "[steering-debug static] "
        f"tick={tick} safe_samples={safe_count}/{total_count} "
        f"safe_steer_deg[min/median/max]="
        f"{steer_min:.1f}/{steer_median:.1f}/{steer_max:.1f} "
        f"safe_accel[min/median/max]={accel_min:.3f}/{accel_median:.3f}/{accel_max:.3f}"
    )

    top_count = min(5, post_filter_weights.size)
    if top_count == 0:
        return
    top_indices = np.argsort(post_filter_weights)[-top_count:][::-1]
    top_parts = []
    for index in top_indices:
        top_parts.append(
            f"#{int(index)} w={post_filter_weights[index]:.2e} "
            f"a={sampled_controls[index,0,0]:.3f} "
            f"steer={deg(sampled_controls[index,0,1]):.1f}deg "
            f"safe={bool(collision_free[index])}"
        )
    print(f"[steering-debug top-samples] tick={tick} " + " | ".join(top_parts))


def print_applied_steering_debug(
    *,
    tick,
    previous_state,
    current_state,
    action,
    dt,
    vehicle_length,
):
    previous_state = np.asarray(previous_state[:4], dtype=float)
    current_state = np.asarray(current_state[:4], dtype=float)
    action = np.asarray(action, dtype=float)
    if action.size < 2:
        action = np.asarray([float(action[0]) if action.size else 0.0, 0.0])

    actual_delta_theta = wrap_angle(
        current_state[ActorStateEnum.THETA] - previous_state[ActorStateEnum.THETA]
    )
    expected_delta_theta = expected_yaw_delta(
        previous_state,
        action,
        dt,
        vehicle_length,
    )
    displacement = current_state[:2] - previous_state[:2]
    actual_direction = (
        np.arctan2(displacement[1], displacement[0])
        if np.linalg.norm(displacement) > 1.0e-9
        else previous_state[ActorStateEnum.THETA]
    )

    print(
        "[steering-debug applied] "
        f"tick={tick} action=[a={action[0]:.3f}, steer={deg(action[1]):.1f}deg] "
        f"theta={deg(previous_state[3]):.1f}->{deg(current_state[3]):.1f}deg "
        f"actual_dtheta={deg(actual_delta_theta):.2f}deg/tick "
        f"expected_dtheta={deg(expected_delta_theta):.2f}deg/tick "
        f"v={previous_state[2]:.3f}->{current_state[2]:.3f} "
        f"move_dir={deg(actual_direction):.1f}deg "
        f"pos=({previous_state[0]:.3f},{previous_state[1]:.3f})"
        f"->({current_state[0]:.3f},{current_state[1]:.3f})"
    )


def print_loop_timing_debug(tick, timing):
    ordered_keys = [
        "tick",
        "predict",
        "hmm_update",
        "route",
        "trajectories",
        "filter_paths",
        "oce_eval",
        "control",
        "render",
        "total",
    ]
    parts = [f"{key}={timing.get(key, 0.0) * 1000.0:.2f}ms" for key in ordered_keys]
    print(f"[timing] tick={tick} " + " ".join(parts))


def print_sim_tick_timing_debug(tick, timing):
    if not timing:
        return
    ordered_keys = [
        "advance_time",
        "ego",
        "generate_agents",
        "static_collision",
        "actors",
        "cleanup",
        "decay",
        "scan",
        "observation",
        "info",
        "done",
        "total",
    ]
    parts = [f"{key}={timing.get(key, 0.0) * 1000.0:.2f}ms" for key in ordered_keys]
    counts = (
        f"actors={int(timing.get('actor_count', 0))} "
        f"visible={int(timing.get('visible_actors', 0))} "
        f"finished={int(timing.get('finished_actors', 0))} "
        f"collisions={int(timing.get('collisions', 0))} "
        f"scan_enabled={bool(timing.get('scan_enabled', False))}"
    )
    print(f"[sim-tick] tick={tick} {counts} " + " ".join(parts))


def print_sim_scan_timing_debug(tick, timing):
    if not timing or "scan_total" not in timing:
        return
    ordered_keys = [
        "scan_load_polycheck",
        "scan_polygons",
        "scan_faux_scan",
        "scan_visible_update",
        "scan_postprocess",
        "scan_total",
    ]
    parts = [f"{key}={timing.get(key, 0.0) * 1000.0:.2f}ms" for key in ordered_keys]
    counts = (
        f"polygons={int(timing.get('scan_polygon_count', 0))} "
        f"vertices={int(timing.get('scan_vertex_count', 0))} "
        f"rays={int(timing.get('scan_ray_count', 0))} "
        f"fallback={bool(timing.get('scan_fallback', False))}"
    )
    print(f"[sim-scan] tick={tick} {counts} " + " ".join(parts))


def print_sim_observation_timing_debug(tick, timing):
    if not timing or "observation_total" not in timing:
        return
    ordered_keys = [
        "observation_move_origin",
        "observation_update",
        "observation_probability_map",
        "observation_total",
    ]
    parts = [f"{key}={timing.get(key, 0.0) * 1000.0:.2f}ms" for key in ordered_keys]
    print(f"[sim-observation] tick={tick} " + " ".join(parts))


TIMING_CSV_FIELDS = [
    "tick",
    "actors",
    "visible_actors",
    "tick_ms",
    "predict_ms",
    "hmm_update_ms",
    "route_ms",
    "trajectories_ms",
    "filter_paths_ms",
    "oce_eval_ms",
    "control_ms",
    "render_ms",
    "total_ms",
    "control_nominal_ms",
    "control_mppi_ms",
    "control_static_sample_filter_ms",
    "control_static_final_check_ms",
    "control_rollout_display_ms",
    "control_trajectory_agent_check_ms",
    "control_total_ms",
]


def append_timing_csv(
    path, tick, timing, control_timing=None, *, actors=0, visible_actors=0
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    control_timing = control_timing or {}
    row = {
        "tick": int(tick),
        "actors": int(actors),
        "visible_actors": int(visible_actors),
        "tick_ms": timing.get("tick", 0.0) * 1000.0,
        "predict_ms": timing.get("predict", 0.0) * 1000.0,
        "hmm_update_ms": timing.get("hmm_update", 0.0) * 1000.0,
        "route_ms": timing.get("route", 0.0) * 1000.0,
        "trajectories_ms": timing.get("trajectories", 0.0) * 1000.0,
        "filter_paths_ms": timing.get("filter_paths", 0.0) * 1000.0,
        "oce_eval_ms": timing.get("oce_eval", 0.0) * 1000.0,
        "control_ms": timing.get("control", 0.0) * 1000.0,
        "render_ms": timing.get("render", 0.0) * 1000.0,
        "total_ms": timing.get("total", 0.0) * 1000.0,
        "control_nominal_ms": control_timing.get("nominal", 0.0) * 1000.0,
        "control_mppi_ms": control_timing.get("mppi", 0.0) * 1000.0,
        "control_static_sample_filter_ms": control_timing.get(
            "static_sample_filter",
            0.0,
        )
        * 1000.0,
        "control_static_final_check_ms": control_timing.get(
            "static_final_check",
            0.0,
        )
        * 1000.0,
        "control_rollout_display_ms": control_timing.get("rollout_display", 0.0)
        * 1000.0,
        "control_trajectory_agent_check_ms": control_timing.get(
            "trajectory_agent_check",
            0.0,
        )
        * 1000.0,
        "control_total_ms": control_timing.get("total", 0.0) * 1000.0,
    }

    with path.open("a", newline="") as timing_file:
        writer = csv.DictWriter(timing_file, fieldnames=TIMING_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def uniform_prediction_probabilities(predictions):
    probabilities = {}
    for agent_id, prediction in predictions.items():
        prediction = np.asarray(prediction)
        if prediction.ndim < 2 or prediction.shape[0] <= 0:
            continue
        num_modes = int(prediction.shape[0])
        probabilities[agent_id] = np.full(
            (num_modes,),
            1.0 / float(num_modes),
            dtype=np.float32,
        )
    return probabilities


def agent_records_by_id(agents):
    return {agent["id"]: agent for agent in agents if "id" in agent}


def evaluate_candidate_paths_by_oce(
    *,
    time_step,
    grid,
    origin,
    resolution,
    paths,
    agents,
    predictions,
    prediction_interval,
    dt,
    debug=False,
):
    if not predictions:
        return 0, None, None

    probabilities = uniform_prediction_probabilities(predictions)
    if not probabilities:
        return 0, None, None

    return evaluate_trajectories_by_entropy_gpu(
        time_step=time_step,
        grid=grid,
        origin=origin,
        resolution=resolution,
        trajectories=paths,
        agents=agent_records_by_id(agents),
        predictions=predictions,
        probabilities=probabilities,
        prediction_interval=prediction_interval,
        dt=dt,
        debug=debug,
    )


def sdd_scene_points_to_sim_display(points, bounds):
    display_points = np.asarray(points, dtype=float).copy()
    display_points[:, 1] = bounds["min_y"] + bounds["max_y"] - display_points[:, 1]
    return display_points


def sdd_sim_display_point_to_scene(point, bounds):
    point = np.asarray(point, dtype=float)
    return np.asarray(
        [
            point[0],
            bounds["min_y"] + bounds["max_y"] - point[1],
        ],
        dtype=float,
    )


def frenet_path_xy(path):
    return np.column_stack(
        [np.asarray(path.x, dtype=float), np.asarray(path.y, dtype=float)]
    )


class DiscreteOCETracker:
    def __init__(
        self,
        sdd_models,
        *,
        observation_epsilon=1.0e-6,
        debug_dir=None,
        debug_top_k=8,
    ):
        self.sdd_models = sdd_models
        self.metadata = sdd_models["state_space_metadata"]
        self.state_space = sdd_models["state_space"]
        self.bounds = self.metadata["bounds"]
        self.rows = int(self.metadata["rows"])
        self.cols = int(self.metadata["cols"])
        self.num_states = int(self.metadata["state_count"])
        self.grid_to_state = np.asarray(
            self.state_space["grid_to_state"], dtype=np.int64
        )
        self.walkable_mask = np.asarray(self.state_space["walkable_mask"], dtype=bool)
        self.static_occupancy_grid = np.asarray(
            np.flipud(~self.walkable_mask),
            dtype=np.uint8,
        )
        self.state_grid_indices = np.asarray(
            self.state_space["grid_indices"], dtype=float
        )
        self.state_centers_scene = np.asarray(self.state_space["centers"], dtype=float)
        self.state_centers_sim = sdd_scene_points_to_sim_display(
            self.state_centers_scene,
            self.bounds,
        )
        self.cell_size = float(self.metadata["cell_size"])
        self.observation_epsilon = float(observation_epsilon)
        (
            self.class_ids,
            self.mode_distribution,
            self.transitions,
            self.sparse_transitions,
        ) = self._build_mode_model(sdd_models)
        self.emission = self._build_emission_matrix(observation_epsilon)
        self.agent_hmms = {}
        self.agent_last_observed = {}
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.debug_top_k = int(debug_top_k)
        if self.debug_dir is not None:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def _build_mode_model(self, sdd_models):
        class_ids = []
        class_probs = []
        transitions = []
        sparse_transitions = []
        for record in sdd_models["classes"]:
            class_id = int(record["id"])
            model = sdd_models["models"].get(class_id)
            if model is None:
                continue
            class_ids.append(class_id)
            class_probs.append(float(record.get("prob", 0.0)))
            P_sparse = model["transitions"].tocsr()
            sparse_transitions.append(P_sparse)
            transitions.append(P_sparse.toarray())

        if not transitions:
            class_ids = ["global"]
            class_probs = [1.0]
            P_sparse = sdd_models["models"]["global"]["transitions"].tocsr()
            sparse_transitions = [P_sparse]
            transitions = [P_sparse.toarray()]

        mode_distribution = np.asarray(class_probs, dtype=float)
        total = float(mode_distribution.sum())
        if total <= 0:
            mode_distribution[:] = 1.0 / len(mode_distribution)
        else:
            mode_distribution /= total

        return (
            class_ids,
            mode_distribution,
            np.stack(transitions, axis=0),
            sparse_transitions,
        )

    def _build_emission_matrix(self, epsilon):
        epsilon = float(epsilon)
        if self.num_states <= 1 or epsilon <= 0:
            return np.eye(self.num_states, dtype=float)
        off_diag = epsilon / float(self.num_states - 1)
        emission = np.full((self.num_states, self.num_states), off_diag, dtype=float)
        np.fill_diagonal(emission, 1.0 - epsilon)
        return emission

    def position_to_state(self, position):
        scene_point = sdd_sim_display_point_to_scene(position[:2], self.bounds)
        col = int(np.floor((scene_point[0] - self.bounds["min_x"]) / self.cell_size))
        row = int(np.floor((scene_point[1] - self.bounds["min_y"]) / self.cell_size))
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return None
        state = int(self.grid_to_state[row, col])
        return state if state >= 0 else None

    def _new_hmm(self, initial_state):
        state_distribution = np.zeros(self.num_states, dtype=float)
        state_distribution[int(initial_state)] = 1.0
        return HMM(
            num_states=self.num_states,
            num_observations=self.num_states,
            num_modes=len(self.class_ids),
            transitions=self.transitions,
            emission_probabilities=self.emission,
            distributions={
                "state": state_distribution,
                "mode": self.mode_distribution,
            },
            precompute_diagnostics=False,
            copy_transitions=False,
        )

    def _apply_observed_state(self, hmm, observation_state):
        observation_state = int(observation_state)
        mode_scores = np.zeros(hmm.num_modes, dtype=float)
        for mode in range(hmm.num_modes):
            P_mode = hmm.transition_matrices[mode]
            if hmm.alphas is None:
                mode_scores[mode] = hmm.mode_distribution[mode] * float(
                    hmm.state_distribution @ P_mode[:, observation_state]
                )
            else:
                mode_scores[mode] = float(
                    hmm.alphas[mode] @ P_mode[:, observation_state]
                )

        total = float(mode_scores.sum())
        if total > 1.0e-12:
            hmm.mode_distribution = mode_scores / total

        hmm.state_distribution.fill(0.0)
        hmm.state_distribution[observation_state] = 1.0
        hmm.alphas = np.zeros((hmm.num_modes, hmm.num_states), dtype=float)
        hmm.alphas[:, observation_state] = hmm.mode_distribution

    def update(self, actors, tick):
        visible_observations = {}
        for actor in actors:
            if not actor.get("visible", False) or "id" not in actor:
                continue
            state = self.position_to_state(actor["pos"])
            if state is not None:
                visible_observations[actor["id"]] = state

        for agent_id, observation_state in visible_observations.items():
            if agent_id not in self.agent_hmms:
                self.agent_hmms[agent_id] = self._new_hmm(observation_state)
            else:
                self._apply_observed_state(
                    self.agent_hmms[agent_id],
                    observation_state,
                )
            self.agent_last_observed[agent_id] = int(tick)

        for agent_id, hmm in self.agent_hmms.items():
            if agent_id not in visible_observations:
                hmm.forward_step(observation=None, steps=1)

        if self.debug_dir is not None:
            self.write_debug(tick)

    def agent_belief_matrix(self):
        agent_ids = list(self.agent_hmms.keys())
        if not agent_ids:
            return agent_ids, np.zeros((0, self.num_states), dtype=np.float32)
        beliefs = np.stack(
            [
                np.asarray(
                    self.agent_hmms[agent_id].state_distribution, dtype=np.float32
                )
                for agent_id in agent_ids
            ],
            axis=0,
        )
        return agent_ids, beliefs

    def agent_mixed_transition_matrices(self, agent_ids):
        if not agent_ids:
            return np.zeros((0, self.num_states, self.num_states), dtype=np.float32)
        transitions = np.empty(
            (len(agent_ids), self.num_states, self.num_states),
            dtype=np.float32,
        )
        for idx, agent_id in enumerate(agent_ids):
            mode_distribution = np.asarray(
                self.agent_hmms[agent_id].mode_distribution,
                dtype=np.float32,
            )
            transitions[idx] = np.tensordot(
                mode_distribution,
                self.transitions.astype(np.float32, copy=False),
                axes=(0, 0),
            )
        return transitions

    def agent_mixed_transition_csr(self, agent_ids, horizon):
        if not agent_ids:
            empty_indptr = np.zeros((0, self.num_states + 1), dtype=np.int32)
            empty_prefix = np.zeros(
                (0, int(horizon) + 1, self.num_states), dtype=np.float32
            )
            return (
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
                empty_indptr,
                empty_prefix,
            )

        data_chunks = []
        index_chunks = []
        indptr = np.empty((len(agent_ids), self.num_states + 1), dtype=np.int32)
        prefix = np.empty(
            (len(agent_ids), int(horizon) + 1, self.num_states),
            dtype=np.float32,
        )
        offset = 0
        for agent_idx, agent_id in enumerate(agent_ids):
            mode_distribution = np.asarray(
                self.agent_hmms[agent_id].mode_distribution,
                dtype=float,
            )
            mixed = None
            for weight, P_sparse in zip(mode_distribution, self.sparse_transitions):
                if weight <= 0.0:
                    continue
                weighted = P_sparse.multiply(float(weight))
                mixed = weighted if mixed is None else mixed + weighted
            if mixed is None:
                mixed = self.sparse_transitions[0].multiply(0.0)
            mixed = mixed.tocsr()
            mixed.sum_duplicates()
            mixed.eliminate_zeros()

            data = mixed.data.astype(np.float32, copy=False)
            indices = mixed.indices.astype(np.int32, copy=False)
            data_chunks.append(data)
            index_chunks.append(indices)
            indptr[agent_idx] = mixed.indptr.astype(np.int32, copy=False) + offset
            offset += int(data.shape[0])

            prefix[agent_idx, 0] = np.asarray(
                self.agent_hmms[agent_id].state_distribution,
                dtype=np.float32,
            )
            for step in range(1, int(horizon) + 1):
                prefix[agent_idx, step] = prefix[agent_idx, step - 1] @ mixed

        if data_chunks:
            data = np.concatenate(data_chunks).astype(np.float32, copy=False)
            indices = np.concatenate(index_chunks).astype(np.int32, copy=False)
        else:
            data = np.zeros((0,), dtype=np.float32)
            indices = np.zeros((0,), dtype=np.int32)
        return data, indices, indptr, prefix

    def write_debug(self, tick):
        records = {}
        for agent_id, hmm in self.agent_hmms.items():
            top_states = np.argsort(hmm.state_distribution)[-self.debug_top_k :][::-1]
            records[str(agent_id)] = {
                "last_observed_tick": self.agent_last_observed.get(agent_id),
                "class_ids": self.class_ids,
                "mode_distribution": hmm.mode_distribution.tolist(),
                "top_states": [
                    {
                        "state": int(state),
                        "prob": float(hmm.state_distribution[state]),
                        "center": self.state_centers_sim[state].tolist(),
                    }
                    for state in top_states
                    if hmm.state_distribution[state] > 0
                ],
            }

        json_path = self.debug_dir / f"discrete_oce_{int(tick):05d}.json"
        json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        self.write_debug_image(tick)

    def write_debug_image(self, tick):
        if not self.agent_hmms:
            return

        scale = max(1, int(np.ceil(500.0 / max(self.rows, self.cols))))
        tile_w = self.cols * scale
        tile_h = self.rows * scale
        label_h = 22
        image = Image.new(
            "RGB",
            (tile_w * len(self.agent_hmms), tile_h + label_h),
            (255, 255, 255),
        )
        draw = ImageDraw.Draw(image)
        for panel, (agent_id, hmm) in enumerate(sorted(self.agent_hmms.items())):
            values = np.zeros((self.rows, self.cols), dtype=float)
            for state, prob in enumerate(hmm.state_distribution):
                row, col = self.state_grid_indices[state].astype(int)
                values[row, col] = prob
            max_prob = float(values.max())
            if max_prob > 0:
                values /= max_prob
            x0 = panel * tile_w
            for row in range(self.rows):
                for col in range(self.cols):
                    v = values[row, col]
                    if v <= 0:
                        colour = (245, 245, 245)
                    else:
                        colour = (255, int(255 * (1.0 - v)), int(255 * (1.0 - v)))
                    draw.rectangle(
                        [
                            x0 + col * scale,
                            label_h + row * scale,
                            x0 + (col + 1) * scale - 1,
                            label_h + (row + 1) * scale - 1,
                        ],
                        fill=colour,
                    )
            mode = int(np.argmax(hmm.mode_distribution))
            draw.text(
                (x0 + 4, 4),
                f"agent {agent_id} class {self.class_ids[mode]} p={hmm.mode_distribution[mode]:.2f}",
                fill=(0, 0, 0),
            )
        image.save(self.debug_dir / f"discrete_oce_{int(tick):05d}.png")


def build_path_occlusion_schedule(
    *,
    path,
    state_centers,
    static_union,
    horizon,
    scan_range,
):
    path_points = frenet_path_xy(path)
    if path_points.shape[0] == 0:
        return []

    schedule = []
    max_step = min(int(horizon), max(0, path_points.shape[0] - 1))
    for step in range(max_step + 1):
        observer = path_points[step]
        deltas = state_centers - observer.reshape(1, 2)
        distances = np.linalg.norm(deltas, axis=1)
        occluded = distances > float(scan_range)
        if static_union is not None and not static_union.is_empty:
            for state_idx in np.where(~occluded)[0]:
                line = LineString([observer, state_centers[state_idx]])
                if line.intersects(static_union):
                    occluded[state_idx] = True
        schedule.append(occluded.astype(float))

    return schedule


def evaluate_candidate_paths_by_discrete_oce(
    *,
    time_step,
    paths,
    tracker,
    static_polygons,
    horizon,
    method="discrete_exact_entropy",
    scan_range=SCAN_RANGE,
    backend="auto",
    return_debug_tensors=False,
    debug=False,
):
    if tracker is None or not tracker.agent_hmms or not paths:
        return 0, None, None

    timing = {
        "backend": "cpu",
        "pack": 0.0,
        "visibility": 0.0,
        "entropy": 0.0,
        "gpu_total": 0.0,
    }
    eval_horizon = max(1, int(horizon))
    backend = str(backend or "auto").lower()

    if backend in {"auto", "gpu"} and evaluate_discrete_oce_gpu is not None:
        gpu_start = perf_counter()
        try:
            pack_start = perf_counter()
            path_arrays = []
            for candidate in paths:
                xy = frenet_path_xy(candidate).astype(np.float32, copy=False)
                if xy.shape[0] == 0:
                    xy = np.zeros((1, 2), dtype=np.float32)
                if xy.shape[0] < eval_horizon + 1:
                    pad = np.repeat(xy[-1:, :], eval_horizon + 1 - xy.shape[0], axis=0)
                    xy = np.vstack([xy, pad])
                else:
                    xy = xy[: eval_horizon + 1]
                path_arrays.append(xy)
            packed_paths = np.stack(path_arrays, axis=0).astype(np.float32)
            agent_ids, beliefs = tracker.agent_belief_matrix()
            (
                transition_data,
                transition_indices,
                transition_indptr,
                prefix_beliefs,
            ) = tracker.agent_mixed_transition_csr(agent_ids, eval_horizon)
            timing["pack"] = perf_counter() - pack_start

            result = evaluate_discrete_oce_gpu(
                paths=packed_paths,
                state_centers=tracker.state_centers_sim,
                static_grid=tracker.static_occupancy_grid,
                grid_origin=(
                    float(tracker.bounds["min_x"]),
                    float(tracker.bounds["min_y"]),
                ),
                grid_resolution=tracker.cell_size,
                transition_data=transition_data,
                transition_indices=transition_indices,
                transition_indptr=transition_indptr,
                prefix_beliefs=prefix_beliefs,
                beliefs=beliefs,
                horizon=eval_horizon,
                scan_range=scan_range,
                return_visibility=return_debug_tensors,
            )
            timing["backend"] = getattr(result, "execution_path", "cuda_discrete_exact")
            timing["gpu_total"] = perf_counter() - gpu_start
            gpu_results = {
                "timing": timing,
                "agent_ids": agent_ids,
                "step_entropy": getattr(result, "step_entropy", None),
                "step_probability": getattr(result, "step_probability", None),
                "step_e_state": getattr(result, "step_e_state", None),
                "step_a_state": getattr(result, "step_a_state", None),
                "step_state_entropy": getattr(result, "step_state_entropy", None),
                "score_components": getattr(result, "score_components", None),
                "visibility_tensor": getattr(result, "visibility_tensor", None),
                "metadata": getattr(result, "metadata", None),
            }
            if debug:
                component_shape = (
                    None
                    if getattr(result, "score_components", None) is None
                    else result.score_components.shape
                )
                print(
                    "[discrete-oce-gpu] "
                    f"scores={np.asarray(result.scores).tolist()} "
                    f"timing={timing} "
                    f"component_shape={component_shape} "
                    f"metadata={getattr(result, 'metadata', None)}"
                )
            return int(result.best_trajectory), np.asarray(result.scores), gpu_results
        except Exception as exc:
            if backend == "gpu":
                raise
            if not getattr(
                evaluate_candidate_paths_by_discrete_oce, "_warned_gpu", False
            ):
                print(
                    f"WARNING: discrete OCE GPU unavailable, falling back to CPU: {exc}"
                )
                evaluate_candidate_paths_by_discrete_oce._warned_gpu = True

    static_union = blocking_static_polygon_union(static_polygons)
    path_scores = np.zeros(len(paths), dtype=float)
    path_results = []

    for path_idx, candidate in enumerate(paths):
        visibility_start = perf_counter()
        I_s = build_path_occlusion_schedule(
            path=candidate,
            state_centers=tracker.state_centers_sim,
            static_union=static_union,
            horizon=eval_horizon,
            scan_range=scan_range,
        )
        timing["visibility"] += perf_counter() - visibility_start
        if len(I_s) <= 1:
            path_results.append({})
            continue

        k = min(eval_horizon, len(I_s) - 1)
        agent_results = {}
        for agent_id, hmm in tracker.agent_hmms.items():
            entropy_start = perf_counter()
            result = evaluate_method(
                method,
                trial=int(time_step),
                k=k,
                hmm=hmm,
                I_s=I_s,
                state_coords=tracker.state_grid_indices,
                alpha=1.0,
            )
            timing["entropy"] += perf_counter() - entropy_start
            score = float(result[-1]["cumulative_entropy"]) if result else 0.0
            path_scores[path_idx] += score
            agent_results[str(agent_id)] = result
        path_results.append(agent_results)

    best_trajectory = int(np.argmin(path_scores))
    if debug:
        print(
            "[discrete-oce] "
            f"tick={time_step} best={best_trajectory} scores={path_scores.tolist()} "
            f"timing={timing}"
        )

    return best_trajectory, path_scores, {"timing": timing, "agents": path_results}


def frenet_path_to_states(path):
    return np.column_stack([path.x, path.y, path.s_d, path.yaw])


def sanitize_reference_path(path, min_spacing=1.0e-6):
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[0] == 0:
        return path

    keep = [0]
    for idx in range(1, path.shape[0]):
        if np.linalg.norm(path[idx, :2] - path[keep[-1], :2]) > min_spacing:
            keep.append(idx)
    return path[keep, :]


def filter_static_safe_paths(paths, static_polygons, vehicle_length, vehicle_width):
    static_union = blocking_static_polygon_union(static_polygons)
    if static_union is None or static_union.is_empty:
        return paths

    safe_paths = []
    for path in paths:
        if not trajectory_collides_with_static(
            frenet_path_to_states(path),
            static_union,
            vehicle_length=vehicle_length,
            vehicle_width=vehicle_width,
            clearance=MIN_SEPARATION,
        ):
            safe_paths.append(path)
    return safe_paths or paths


def find_nearest_free_cell(start_cell, is_blocked, cols, rows):
    sx, sy = start_cell
    if 0 <= sx < cols and 0 <= sy < rows and not is_blocked[sy, sx]:
        return start_cell

    max_radius = max(cols, rows)
    for radius in range(1, max_radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:
                    continue
                nx = sx + dx
                ny = sy + dy
                if 0 <= nx < cols and 0 <= ny < rows and not is_blocked[ny, nx]:
                    return nx, ny
    return None


def build_static_planning_grid(
    static_polygons,
    *,
    display_offset,
    display_diff,
    vehicle_length,
    vehicle_width,
    resolution=None,
):
    obstacle_union = buffered_static_obstacle_union(
        static_polygons,
        vehicle_length,
        vehicle_width,
    )
    resolution = float(resolution or GRID_RESOLUTION)
    min_x = float(display_offset[0])
    min_y = float(display_offset[1])
    max_x = min_x + float(display_diff)
    max_y = min_y + float(display_diff)
    cols = int(np.ceil((max_x - min_x) / resolution))
    rows = int(np.ceil((max_y - min_y) / resolution))

    is_blocked = np.zeros((rows, cols), dtype=bool)
    if obstacle_union is not None and not obstacle_union.is_empty:
        for row in range(rows):
            y = min_y + (row + 0.5) * resolution
            for col in range(cols):
                x = min_x + (col + 0.5) * resolution
                is_blocked[row, col] = obstacle_union.covers(Point(x, y))

    def point_to_cell(point):
        col = int(np.clip(np.floor((point[0] - min_x) / resolution), 0, cols - 1))
        row = int(np.clip(np.floor((point[1] - min_y) / resolution), 0, rows - 1))
        return col, row

    def cell_to_point(cell):
        col, row = cell
        return [
            min(max(min_x + (col + 0.5) * resolution, min_x), max_x),
            min(max(min_y + (row + 0.5) * resolution, min_y), max_y),
        ]

    return {
        "obstacle_union": obstacle_union,
        "resolution": resolution,
        "rows": rows,
        "cols": cols,
        "is_blocked": is_blocked,
        "point_to_cell": point_to_cell,
        "cell_to_point": cell_to_point,
    }


def reconstruct_grid_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def reconstruct_oriented_grid_path(came_from, current):
    path = []
    while current is not None:
        path.append(current)
        current = came_from.get(current)
    path.reverse()
    return path


def astar_grid(start_cell, goal_cell, is_blocked):
    rows, cols = is_blocked.shape
    neighbors = (
        (-1, -1, np.sqrt(2.0)),
        (0, -1, 1.0),
        (1, -1, np.sqrt(2.0)),
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (-1, 1, np.sqrt(2.0)),
        (0, 1, 1.0),
        (1, 1, np.sqrt(2.0)),
    )

    frontier = []
    heappush(frontier, (0.0, start_cell))
    came_from = {}
    cost_so_far = {start_cell: 0.0}

    def heuristic(cell):
        return np.hypot(cell[0] - goal_cell[0], cell[1] - goal_cell[1])

    while frontier:
        _priority, current = heappop(frontier)
        if current == goal_cell:
            return reconstruct_grid_path(came_from, current)

        for dx, dy, move_cost in neighbors:
            nx = current[0] + dx
            ny = current[1] + dy
            if nx < 0 or ny < 0 or nx >= cols or ny >= rows or is_blocked[ny, nx]:
                continue

            if dx != 0 and dy != 0:
                if is_blocked[current[1], nx] or is_blocked[ny, current[0]]:
                    continue

            next_cell = (nx, ny)
            new_cost = cost_so_far[current] + move_cost
            if next_cell not in cost_so_far or new_cost < cost_so_far[next_cell]:
                cost_so_far[next_cell] = new_cost
                heappush(frontier, (new_cost + heuristic(next_cell), next_cell))
                came_from[next_cell] = current

    return None


def heading_bin(angle, heading_bins):
    angle = float(angle) % (2.0 * np.pi)
    return int(np.round(angle / (2.0 * np.pi / heading_bins))) % int(heading_bins)


def heading_from_bin(index, heading_bins):
    return float(index) * 2.0 * np.pi / float(heading_bins)


def astar_kinematic_grid(
    start_cell,
    goal_cell,
    start_heading,
    is_blocked,
    *,
    resolution,
    vehicle_length,
    max_steer,
    heading_bins=16,
    cell_penalty=None,
    edge_penalty=None,
    turn_penalty=0.15,
    start_xy=None,
    cell_to_point=None,
):
    rows, cols = is_blocked.shape
    heading_bins = max(8, int(heading_bins))
    min_turn_radius = float(vehicle_length) / max(np.tan(float(max_steer)), 1.0e-6)
    start_state = (
        int(start_cell[0]),
        int(start_cell[1]),
        heading_bin(start_heading, heading_bins),
    )
    goal_cell = (int(goal_cell[0]), int(goal_cell[1]))
    cell_penalty = cell_penalty or {}
    edge_penalty = edge_penalty or {}

    neighbors = (
        (-1, -1, np.sqrt(2.0)),
        (0, -1, 1.0),
        (1, -1, np.sqrt(2.0)),
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (-1, 1, np.sqrt(2.0)),
        (0, 1, 1.0),
        (1, 1, np.sqrt(2.0)),
    )

    frontier = []
    heappush(frontier, (0.0, 0.0, start_state))
    came_from = {start_state: None}
    cost_so_far = {start_state: 0.0}
    expansions = 0
    max_expansions = rows * cols * heading_bins

    def heuristic(cell):
        return resolution * np.hypot(cell[0] - goal_cell[0], cell[1] - goal_cell[1])

    while frontier and expansions < max_expansions:
        _priority, current_cost, current = heappop(frontier)
        if current_cost > cost_so_far.get(current, float("inf")) + 1.0e-9:
            continue
        expansions += 1
        c, r, h_idx = current
        if (c, r) == goal_cell:
            return reconstruct_oriented_grid_path(came_from, current)

        current_heading = heading_from_bin(h_idx, heading_bins)
        for dx, dy, move_units in neighbors:
            nc = c + dx
            nr = r + dy
            if nc < 0 or nr < 0 or nc >= cols or nr >= rows or is_blocked[nr, nc]:
                continue

            if dx != 0 and dy != 0:
                if is_blocked[r, nc] or is_blocked[nr, c]:
                    continue

            if current == start_state and start_xy is not None and cell_to_point:
                move_delta = (
                    np.asarray(cell_to_point((nc, nr)), dtype=float)
                    - np.asarray(start_xy, dtype=float)[:2]
                )
                move_distance = float(np.linalg.norm(move_delta))
                if move_distance <= 1.0e-6:
                    continue
                heading_vector = np.asarray(
                    [np.cos(start_heading), np.sin(start_heading)],
                    dtype=float,
                )
                if float(move_delta @ heading_vector) <= 1.0e-6:
                    continue
                move_heading = float(np.arctan2(move_delta[1], move_delta[0]))
            else:
                move_heading = float(np.arctan2(dy, dx))
                move_distance = float(move_units) * float(resolution)
            heading_delta = abs(wrap_angle(move_heading - current_heading))
            max_heading_delta = move_distance / max(min_turn_radius, 1.0e-6)
            max_heading_delta += np.pi / float(heading_bins)
            if heading_delta > max_heading_delta:
                continue

            nh = heading_bin(move_heading, heading_bins)
            next_state = (nc, nr, nh)
            edge = tuple(sorted(((c, r), (nc, nr))))
            penalty = float(cell_penalty.get((nc, nr), 0.0)) + float(
                edge_penalty.get(edge, 0.0)
            )
            turn_cost = float(turn_penalty) * heading_delta
            new_cost = current_cost + move_distance + turn_cost + penalty
            if new_cost < cost_so_far.get(next_state, float("inf")):
                cost_so_far[next_state] = new_cost
                priority = new_cost + heuristic((nc, nr))
                came_from[next_state] = current
                heappush(frontier, (priority, new_cost, next_state))

    return None


def oriented_path_cells(oriented_path):
    return [(int(c), int(r)) for c, r, _h in oriented_path]


def grid_path_length(cells, resolution):
    if len(cells) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(cells[:-1], cells[1:]):
        total += float(resolution) * np.hypot(b[0] - a[0], b[1] - a[1])
    return float(total)


def grid_path_has_loop(cells):
    return len(set(cells)) != len(cells)


def grid_path_overlap(candidate, accepted):
    if not candidate or not accepted:
        return 0.0
    candidate_set = set(candidate)
    overlaps = []
    for route in accepted:
        route_set = set(route)
        denom = max(1, min(len(candidate_set), len(route_set)))
        overlaps.append(len(candidate_set & route_set) / float(denom))
    return float(max(overlaps)) if overlaps else 0.0


def waypoint_route_length(route):
    route = np.asarray(route, dtype=float)
    if route.ndim != 2 or route.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1)))


def route_respects_initial_kinematics(
    route,
    start_state,
    *,
    vehicle_length,
    max_steer,
    heading_bins=16,
):
    route = dedupe_waypoints(route)
    if len(route) < 2:
        return False, "short"

    start = np.asarray(start_state[:2], dtype=float)
    heading = float(start_state[ActorStateEnum.THETA])
    heading_vector = np.asarray([np.cos(heading), np.sin(heading)], dtype=float)
    min_turn_radius = float(vehicle_length) / max(np.tan(float(max_steer)), 1.0e-6)
    heading_slack = np.pi / max(float(heading_bins), 8.0)

    for waypoint in route[1:]:
        delta = np.asarray(waypoint, dtype=float) - start
        distance = float(np.linalg.norm(delta))
        if distance <= 1.0e-6:
            continue
        if float(delta @ heading_vector) <= 1.0e-6:
            return False, "behind"
        route_heading = float(np.arctan2(delta[1], delta[0]))
        heading_delta = abs(wrap_angle(route_heading - heading))
        max_heading_delta = distance / max(min_turn_radius, 1.0e-6) + heading_slack
        if heading_delta > max_heading_delta:
            return False, "turn"
        return True, ""

    return False, "short"


def hybrid_state_key(state, *, display_offset, resolution, heading_bins):
    x, y, _v, theta = np.asarray(state, dtype=float)[:4]
    min_x = float(display_offset[0])
    min_y = float(display_offset[1])
    col = int(np.floor((x - min_x) / float(resolution)))
    row = int(np.floor((y - min_y) / float(resolution)))
    return col, row, heading_bin(theta, heading_bins)


def hybrid_state_cell(state, *, display_offset, resolution):
    x, y = np.asarray(state, dtype=float)[:2]
    min_x = float(display_offset[0])
    min_y = float(display_offset[1])
    return (
        int(np.floor((x - min_x) / float(resolution))),
        int(np.floor((y - min_y) / float(resolution))),
    )


def hybrid_state_in_bounds(state, *, display_offset, display_diff):
    x, y = np.asarray(state, dtype=float)[:2]
    min_x = float(display_offset[0])
    min_y = float(display_offset[1])
    return min_x <= x <= min_x + float(display_diff) and min_y <= y <= min_y + float(
        display_diff
    )


def hybrid_state_collision_free(
    state,
    collision_region,
    *,
    display_offset,
    display_diff,
    vehicle_length,
    vehicle_width,
):
    if not hybrid_state_in_bounds(
        state,
        display_offset=display_offset,
        display_diff=display_diff,
    ):
        return False
    if collision_region is None or collision_region.is_empty:
        return True
    footprint = vehicle_footprint_polygon(state, vehicle_length, vehicle_width)
    return not footprint.intersects(collision_region)


def goal_position_collision_free(
    goal_xy,
    collision_region,
    *,
    display_offset,
    display_diff,
    vehicle_length,
    vehicle_width,
    heading_bins=16,
):
    goal = np.asarray(goal_xy, dtype=float)[:2]
    if not point_within_display(goal, display_offset, display_diff):
        return False
    if collision_region is None or collision_region.is_empty:
        return True

    for heading_index in range(max(8, int(heading_bins))):
        heading = heading_from_bin(heading_index, heading_bins)
        goal_state = np.asarray([goal[0], goal[1], 0.0, heading], dtype=float)
        if hybrid_state_collision_free(
            goal_state,
            collision_region,
            display_offset=display_offset,
            display_diff=display_diff,
            vehicle_length=vehicle_length,
            vehicle_width=vehicle_width,
        ):
            return True
    return False


def rollout_ackermann_distance(
    state,
    *,
    steer,
    distance,
    sample_distance,
    vehicle_length,
    speed,
):
    state = np.asarray(state, dtype=float)[:4].copy()
    state[ActorStateEnum.VELOCITY] = float(speed)
    distance = max(0.0, float(distance))
    sample_distance = max(float(sample_distance), 1.0e-3)
    steps = max(1, int(np.ceil(distance / sample_distance)))
    ds = distance / float(steps)
    curvature = np.tan(float(steer)) / max(float(vehicle_length), 1.0e-6)
    states = []
    for _ in range(steps):
        theta_mid = state[ActorStateEnum.THETA] + 0.5 * curvature * ds
        state[ActorStateEnum.X] += ds * np.cos(theta_mid)
        state[ActorStateEnum.Y] += ds * np.sin(theta_mid)
        state[ActorStateEnum.THETA] = wrap_angle(
            state[ActorStateEnum.THETA] + curvature * ds
        )
        state[ActorStateEnum.VELOCITY] = float(speed)
        states.append(state.copy())
    return states


def hybrid_primitive_is_safe(
    states,
    collision_region,
    *,
    display_offset,
    display_diff,
    vehicle_length,
    vehicle_width,
):
    return all(
        hybrid_state_collision_free(
            state,
            collision_region,
            display_offset=display_offset,
            display_diff=display_diff,
            vehicle_length=vehicle_length,
            vehicle_width=vehicle_width,
        )
        for state in states
    )


def hybrid_connect_to_goal(
    state,
    goal_xy,
    collision_region,
    *,
    display_offset,
    display_diff,
    vehicle_length,
    vehicle_width,
    max_steer,
    speed,
    sample_distance,
    max_distance,
    goal_tolerance,
):
    goal = np.asarray(goal_xy, dtype=float)[:2]
    current = np.asarray(state, dtype=float)[:4].copy()
    states = []
    traveled = 0.0
    max_steps = max(
        1, int(np.ceil(float(max_distance) / max(float(sample_distance), 1.0e-3)))
    )
    for _ in range(max_steps):
        delta = goal - current[:2]
        distance = float(np.linalg.norm(delta))
        if distance <= float(goal_tolerance):
            return states

        bearing = float(np.arctan2(delta[1], delta[0]))
        heading_error = wrap_angle(bearing - current[ActorStateEnum.THETA])
        if np.cos(heading_error) <= -0.05:
            return None

        lookahead = max(distance, float(vehicle_length), float(sample_distance))
        steer = np.arctan2(
            2.0 * float(vehicle_length) * np.sin(heading_error), lookahead
        )
        steer = float(np.clip(steer, -float(max_steer), float(max_steer)))
        step_distance = min(float(sample_distance), distance)
        step_states = rollout_ackermann_distance(
            current,
            steer=steer,
            distance=step_distance,
            sample_distance=step_distance,
            vehicle_length=vehicle_length,
            speed=speed,
        )
        if not hybrid_primitive_is_safe(
            step_states,
            collision_region,
            display_offset=display_offset,
            display_diff=display_diff,
            vehicle_length=vehicle_length,
            vehicle_width=vehicle_width,
        ):
            return None
        current = step_states[-1]
        states.extend(step_states)
        traveled += step_distance
        if traveled > float(max_distance):
            return None

    delta = goal - current[:2]
    if float(np.linalg.norm(delta)) <= float(goal_tolerance):
        return states
    return None


def reconstruct_hybrid_states(parent, node_states, goal_key):
    keys = []
    key = goal_key
    while key is not None:
        keys.append(key)
        key = parent[key][0]
    keys.reverse()

    states = [node_states[keys[0]].copy()]
    for key in keys[1:]:
        states.extend(state.copy() for state in parent[key][1])
    return states


def hybrid_route_cells(states, *, display_offset, resolution):
    cells = []
    for state in states:
        cell = hybrid_state_cell(
            state,
            display_offset=display_offset,
            resolution=resolution,
        )
        if not cells or cells[-1] != cell:
            cells.append(cell)
    return cells


def hybrid_states_path_length(states):
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1)))


def hybrid_states_to_route(states, *, min_spacing=0.35):
    if not states:
        return []
    route = [np.asarray(states[0], dtype=float)[:2].tolist()]
    last = np.asarray(route[-1], dtype=float)
    for state in states[1:]:
        point = np.asarray(state, dtype=float)[:2]
        if np.linalg.norm(point - last) >= float(min_spacing):
            route.append(point.tolist())
            last = point
    final = np.asarray(states[-1], dtype=float)[:2]
    if np.linalg.norm(final - np.asarray(route[-1], dtype=float)) > 1.0e-6:
        route.append(final.tolist())
    return dedupe_waypoints(route)


def hybrid_states_to_frenet_path(states, *, speed, dt, max_points=None):
    states = np.asarray(states, dtype=float)
    path = Frenet_path()
    if states.ndim != 2 or states.shape[0] == 0:
        return path

    if max_points is not None:
        states = states[: max(1, int(max_points))]

    path.x = states[:, ActorStateEnum.X].astype(float).tolist()
    path.y = states[:, ActorStateEnum.Y].astype(float).tolist()
    path.yaw = states[:, ActorStateEnum.THETA].astype(float).tolist()
    path.s_d = [float(speed)] * states.shape[0]
    path.t = [idx * float(dt) for idx in range(states.shape[0])]
    if states.shape[0] >= 2:
        deltas = np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1)
        path.ds = deltas.astype(float).tolist()
        path.s = [0.0]
        path.s.extend(np.cumsum(deltas).astype(float).tolist())
        curvatures = []
        for idx, ds in enumerate(deltas):
            if ds <= 1.0e-6:
                curvatures.append(0.0)
            else:
                curvatures.append(
                    wrap_angle(path.yaw[idx + 1] - path.yaw[idx]) / float(ds)
                )
        path.c = curvatures
    else:
        path.ds = []
        path.s = [0.0]
        path.c = []
    path.s_dd = [0.0] * states.shape[0]
    path.s_ddd = [0.0] * states.shape[0]
    path.d = [0.0] * states.shape[0]
    path.d_d = [0.0] * states.shape[0]
    path.d_dd = [0.0] * states.shape[0]
    path.d_ddd = [0.0] * states.shape[0]
    return path


def astar_grid_penalized(
    start_cell,
    goal_cell,
    is_blocked,
    *,
    cell_penalty=None,
    edge_penalty=None,
    turn_penalty=0.0,
    start_heading=None,
    resolution=1.0,
    min_turn_radius=None,
):
    rows, cols = is_blocked.shape
    cell_penalty = cell_penalty or {}
    edge_penalty = edge_penalty or {}
    neighbors = (
        (-1, -1, np.sqrt(2.0)),
        (0, -1, 1.0),
        (1, -1, np.sqrt(2.0)),
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (-1, 1, np.sqrt(2.0)),
        (0, 1, 1.0),
        (1, 1, np.sqrt(2.0)),
    )

    frontier = []
    heappush(frontier, (0.0, 0.0, start_cell))
    came_from = {}
    cost_so_far = {start_cell: 0.0}

    def heuristic(cell):
        return np.hypot(cell[0] - goal_cell[0], cell[1] - goal_cell[1])

    def heading_for_current_cell(cell):
        previous = came_from.get(cell)
        if previous is not None:
            return float(np.arctan2(cell[1] - previous[1], cell[0] - previous[0]))
        if start_heading is not None:
            return float(start_heading)
        return None

    while frontier:
        _priority, current_cost, current = heappop(frontier)
        if current_cost > cost_so_far.get(current, float("inf")) + 1.0e-9:
            continue
        if current == goal_cell:
            return reconstruct_grid_path(came_from, current)

        current_heading = heading_for_current_cell(current)
        for dx, dy, move_cost in neighbors:
            move_heading = float(np.arctan2(dy, dx))
            if current_heading is not None:
                heading_delta = abs(wrap_angle(move_heading - current_heading))
                if heading_delta > np.pi / 2.0 + 1.0e-9:
                    continue
                if min_turn_radius is not None and float(min_turn_radius) > 1.0e-9:
                    move_distance = float(move_cost) * float(resolution)
                    max_heading_delta = move_distance / float(min_turn_radius)
                    max_heading_delta += np.deg2rad(5.0)
                    if heading_delta > max_heading_delta:
                        continue

            nx = current[0] + dx
            ny = current[1] + dy
            if nx < 0 or ny < 0 or nx >= cols or ny >= rows or is_blocked[ny, nx]:
                continue
            if dx != 0 and dy != 0:
                if is_blocked[current[1], nx] or is_blocked[ny, current[0]]:
                    continue

            next_cell = (nx, ny)
            edge = tuple(sorted((current, next_cell)))
            penalty = float(cell_penalty.get(next_cell, 0.0)) + float(
                edge_penalty.get(edge, 0.0)
            )
            turn_cost = 0.0
            previous = came_from.get(current)
            if previous is not None:
                previous_heading = np.arctan2(
                    current[1] - previous[1],
                    current[0] - previous[0],
                )
                turn_cost = float(turn_penalty) * abs(
                    wrap_angle(move_heading - previous_heading)
                )
            new_cost = current_cost + float(move_cost) + penalty + turn_cost
            if new_cost < cost_so_far.get(next_cell, float("inf")):
                cost_so_far[next_cell] = new_cost
                came_from[next_cell] = current
                heappush(
                    frontier, (new_cost + heuristic(next_cell), new_cost, next_cell)
                )

    return None


def simplify_route_collinear(points, angle_tolerance=np.deg2rad(4.0)):
    points = dedupe_waypoints(points)
    if len(points) <= 2:
        return points

    simplified = [points[0]]
    for idx in range(1, len(points) - 1):
        a = np.asarray(simplified[-1], dtype=float)
        b = np.asarray(points[idx], dtype=float)
        c = np.asarray(points[idx + 1], dtype=float)
        ab = b - a
        bc = c - b
        if np.linalg.norm(ab) <= 1.0e-9 or np.linalg.norm(bc) <= 1.0e-9:
            continue
        heading_a = np.arctan2(ab[1], ab[0])
        heading_b = np.arctan2(bc[1], bc[0])
        if abs(wrap_angle(heading_b - heading_a)) > float(angle_tolerance):
            simplified.append(points[idx])
    simplified.append(points[-1])
    return dedupe_waypoints(simplified)


def arc_points_for_turn(
    tangent_start,
    tangent_end,
    incoming_heading,
    turn_angle,
    turn_radius,
    sample_distance,
):
    sign = 1.0 if turn_angle > 0.0 else -1.0
    normal = np.asarray(
        [-np.sin(incoming_heading), np.cos(incoming_heading)],
        dtype=float,
    )
    center = np.asarray(tangent_start, dtype=float) + sign * float(turn_radius) * normal
    start_angle = np.arctan2(
        tangent_start[1] - center[1],
        tangent_start[0] - center[0],
    )
    steps = max(
        2,
        int(
            np.ceil(
                abs(float(turn_angle))
                * float(turn_radius)
                / max(float(sample_distance), 1.0e-3)
            )
        ),
    )
    arc = []
    for step in range(1, steps + 1):
        angle = start_angle + sign * abs(float(turn_angle)) * step / float(steps)
        point = center + float(turn_radius) * np.asarray(
            [np.cos(angle), np.sin(angle)],
            dtype=float,
        )
        arc.append(point.tolist())
    if arc:
        arc[-1] = np.asarray(tangent_end, dtype=float).tolist()
    return arc


def ackermann_smooth_route(
    route,
    *,
    start_heading,
    vehicle_length,
    max_steer,
    sample_distance,
):
    route = dedupe_waypoints(route)
    if len(route) < 2:
        return None, "short"

    min_turn_radius = float(vehicle_length) / max(np.tan(float(max_steer)), 1.0e-6)
    sample_distance = max(float(sample_distance), 0.05)
    points = [np.asarray(point, dtype=float)[:2] for point in route]
    smoothed = [points[0].tolist()]

    first_delta = points[1] - points[0]
    if np.linalg.norm(first_delta) <= 1.0e-6:
        return None, "short"
    first_heading = float(np.arctan2(first_delta[1], first_delta[0]))
    if abs(wrap_angle(first_heading - float(start_heading))) > np.pi / 2.0:
        return None, "behind"

    for idx in range(1, len(points) - 1):
        prev_point = points[idx - 1]
        corner = points[idx]
        next_point = points[idx + 1]
        incoming = corner - prev_point
        outgoing = next_point - corner
        incoming_len = float(np.linalg.norm(incoming))
        outgoing_len = float(np.linalg.norm(outgoing))
        if incoming_len <= 1.0e-6 or outgoing_len <= 1.0e-6:
            continue

        incoming_unit = incoming / incoming_len
        outgoing_unit = outgoing / outgoing_len
        incoming_heading = float(np.arctan2(incoming_unit[1], incoming_unit[0]))
        outgoing_heading = float(np.arctan2(outgoing_unit[1], outgoing_unit[0]))
        turn_angle = wrap_angle(outgoing_heading - incoming_heading)
        if abs(turn_angle) < np.deg2rad(3.0):
            smoothed.append(corner.tolist())
            continue

        tangent_distance = min_turn_radius * np.tan(abs(turn_angle) / 2.0)
        max_tangent = 0.8 * min(incoming_len, outgoing_len)
        if tangent_distance > max_tangent:
            return None, "turn_radius"

        tangent_start = corner - incoming_unit * tangent_distance
        tangent_end = corner + outgoing_unit * tangent_distance
        if np.linalg.norm(np.asarray(smoothed[-1]) - tangent_start) > 1.0e-6:
            smoothed.append(tangent_start.tolist())
        smoothed.extend(
            arc_points_for_turn(
                tangent_start,
                tangent_end,
                incoming_heading,
                turn_angle,
                min_turn_radius,
                sample_distance,
            )
        )

    if np.linalg.norm(np.asarray(smoothed[-1]) - points[-1]) > 1.0e-6:
        smoothed.append(points[-1].tolist())
    return dedupe_waypoints(smoothed), ""


def sample_polyline_points(points, *, spacing, max_points):
    points = np.asarray(dedupe_waypoints(points), dtype=float)
    if points.ndim != 2 or points.shape[0] < 2:
        return points

    lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    cumulative = np.zeros(points.shape[0], dtype=float)
    cumulative[1:] = np.cumsum(lengths)
    total = float(cumulative[-1])
    if total <= 1.0e-9:
        return points[:1]

    spacing = max(float(spacing), 0.02)
    max_points = max(2, int(max_points))
    distances = np.arange(0.0, total + 0.5 * spacing, spacing, dtype=float)
    if distances.size < max_points:
        distances = np.concatenate(
            [distances, np.full(max_points - distances.size, total, dtype=float)]
        )
    distances = distances[:max_points]

    sampled = []
    segment = 0
    for distance in distances:
        while segment < len(lengths) - 1 and cumulative[segment + 1] < distance:
            segment += 1
        segment_length = lengths[segment]
        if segment_length <= 1.0e-9:
            sampled.append(points[segment, :2].tolist())
            continue
        u = (distance - cumulative[segment]) / segment_length
        point = points[segment, :2] + np.clip(u, 0.0, 1.0) * (
            points[segment + 1, :2] - points[segment, :2]
        )
        sampled.append(point.tolist())
    return np.asarray(sampled, dtype=float)


def points_to_frenet_path(points, *, start_heading, speed, dt, max_points):
    sampled = sample_polyline_points(
        points,
        spacing=max(float(speed) * float(dt), 0.03),
        max_points=max_points,
    )
    path = Frenet_path()
    if sampled.ndim != 2 or sampled.shape[0] == 0:
        return path

    yaws = []
    for idx in range(sampled.shape[0]):
        if idx < sampled.shape[0] - 1:
            delta = sampled[idx + 1] - sampled[idx]
            if np.linalg.norm(delta) > 1.0e-9:
                yaws.append(float(np.arctan2(delta[1], delta[0])))
                continue
        yaws.append(yaws[-1] if yaws else float(start_heading))

    if sampled.shape[0] >= 2:
        deltas = np.linalg.norm(np.diff(sampled[:, :2], axis=0), axis=1)
        curvatures = [
            0.0 if ds <= 1.0e-9 else wrap_angle(yaws[idx + 1] - yaws[idx]) / float(ds)
            for idx, ds in enumerate(deltas)
        ]
    else:
        deltas = np.asarray([], dtype=float)
        curvatures = []

    if yaws:
        yaws[0] = float(start_heading)

    path.x = sampled[:, 0].astype(float).tolist()
    path.y = sampled[:, 1].astype(float).tolist()
    path.yaw = yaws
    path.s_d = [float(speed)] * sampled.shape[0]
    path.t = [idx * float(dt) for idx in range(sampled.shape[0])]
    if sampled.shape[0] >= 2:
        path.ds = deltas.astype(float).tolist()
        path.s = [0.0]
        path.s.extend(np.cumsum(deltas).astype(float).tolist())
        path.c = curvatures
    else:
        path.ds = []
        path.s = [0.0]
        path.c = []
    path.s_dd = [0.0] * sampled.shape[0]
    path.s_ddd = [0.0] * sampled.shape[0]
    path.d = [0.0] * sampled.shape[0]
    path.d_d = [0.0] * sampled.shape[0]
    path.d_dd = [0.0] * sampled.shape[0]
    path.d_ddd = [0.0] * sampled.shape[0]
    return path


def frenet_path_static_collision_free(
    path,
    collision_region,
    *,
    vehicle_length,
    vehicle_width,
):
    if collision_region is None or collision_region.is_empty:
        return True
    for x, y, yaw in zip(path.x, path.y, path.yaw):
        state = np.asarray([x, y, 0.0, yaw], dtype=float)
        footprint = vehicle_footprint_polygon(state, vehicle_length, vehicle_width)
        if footprint.intersects(collision_region):
            return False
    return True


def select_best_kpath_candidates(candidates, *, k, max_overlap):
    if not candidates:
        return []

    cell_counts = {}
    for candidate in candidates:
        for cell in set(candidate["cells"]):
            cell_counts[cell] = cell_counts.get(cell, 0) + 1
    common_threshold = max(2, int(np.ceil(0.5 * len(candidates))))
    common_cells = {
        cell for cell, count in cell_counts.items() if count >= common_threshold
    }

    def diversity_cells(candidate):
        cells = [cell for cell in candidate["cells"] if cell not in common_cells]
        return cells or candidate["cells"]

    def candidate_overlap(candidate, selected):
        if not selected:
            return 0.0
        cells = set(diversity_cells(candidate))
        if not cells:
            return 0.0
        overlaps = []
        for selected_candidate in selected:
            selected_cells = set(diversity_cells(selected_candidate))
            denom = max(1, min(len(cells), len(selected_cells)))
            overlaps.append(len(cells & selected_cells) / float(denom))
        return float(max(overlaps)) if overlaps else 0.0

    ordered = sorted(candidates, key=lambda item: item["length"])
    selected = []
    for candidate in ordered:
        overlap = candidate_overlap(candidate, selected)
        if overlap <= float(max_overlap) or not selected:
            selected.append(candidate)
            if len(selected) >= int(k):
                return selected

    selected_ids = {id(candidate) for candidate in selected}
    for candidate in ordered:
        if id(candidate) in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(id(candidate))
        if len(selected) >= int(k):
            break
    return selected


def grid_path_terminal_heading(cells, fallback_heading):
    if len(cells) < 2:
        return float(fallback_heading)
    start = cells[-2]
    end = cells[-1]
    return float(np.arctan2(end[1] - start[1], end[0] - start[0]))


def point_to_segment_distance_cells(cell, start_cell, goal_cell):
    point = np.asarray(cell, dtype=float)
    start = np.asarray(start_cell, dtype=float)
    goal = np.asarray(goal_cell, dtype=float)
    segment = goal - start
    length2 = float(segment @ segment)
    if length2 <= 1.0e-9:
        return float(np.linalg.norm(point - start)), 0.0
    t = float(np.clip(((point - start) @ segment) / length2, 0.0, 1.0))
    projection = start + t * segment
    return float(np.linalg.norm(point - projection)), t


def build_kpath_anchor_cells(grid, start_cell, goal_cell, *, max_anchors):
    is_blocked = grid["is_blocked"]
    rows = int(grid["rows"])
    cols = int(grid["cols"])
    candidates = []
    min_endpoint_distance = 2.0
    for row in range(rows):
        for col in range(cols):
            if is_blocked[row, col]:
                continue
            cell = (col, row)
            if cell == start_cell or cell == goal_cell:
                continue
            start_distance = np.hypot(col - start_cell[0], row - start_cell[1])
            goal_distance = np.hypot(col - goal_cell[0], row - goal_cell[1])
            if min(start_distance, goal_distance) < min_endpoint_distance:
                continue
            perpendicular, progress = point_to_segment_distance_cells(
                cell,
                start_cell,
                goal_cell,
            )
            if progress <= 0.05 or progress >= 0.95:
                continue
            score = perpendicular + 0.15 * min(start_distance, goal_distance)
            candidates.append((score, cell))

    candidates.sort(reverse=True)
    selected = []
    spacing = max(3.0, 0.12 * max(rows, cols))
    for _score, cell in candidates:
        if all(
            np.hypot(cell[0] - other[0], cell[1] - other[1]) >= spacing
            for other in selected
        ):
            selected.append(cell)
            if len(selected) >= int(max_anchors):
                break
    return selected


def astar_grid_penalized_via(
    start_cell,
    goal_cell,
    anchor_cell,
    is_blocked,
    *,
    start_heading,
    resolution,
    min_turn_radius,
    cell_penalty=None,
    edge_penalty=None,
    turn_penalty=0.0,
):
    if anchor_cell is None:
        return astar_grid_penalized(
            start_cell,
            goal_cell,
            is_blocked,
            cell_penalty=cell_penalty,
            edge_penalty=edge_penalty,
            turn_penalty=turn_penalty,
            start_heading=start_heading,
            resolution=resolution,
            min_turn_radius=min_turn_radius,
        )

    first = astar_grid_penalized(
        start_cell,
        anchor_cell,
        is_blocked,
        cell_penalty=cell_penalty,
        edge_penalty=edge_penalty,
        turn_penalty=turn_penalty,
        start_heading=start_heading,
        resolution=resolution,
        min_turn_radius=min_turn_radius,
    )
    if not first:
        return None
    second_heading = grid_path_terminal_heading(first, start_heading)
    second = astar_grid_penalized(
        anchor_cell,
        goal_cell,
        is_blocked,
        cell_penalty=cell_penalty,
        edge_penalty=edge_penalty,
        turn_penalty=turn_penalty,
        start_heading=second_heading,
        resolution=resolution,
        min_turn_radius=min_turn_radius,
    )
    if not second:
        return None
    return first + second[1:]


def hybrid_astar_search(
    start_state,
    goal_xy,
    collision_region,
    *,
    display_offset,
    display_diff,
    vehicle_length,
    vehicle_width,
    max_steer,
    speed,
    resolution,
    heading_bins,
    cell_penalty=None,
    edge_penalty=None,
    turn_penalty=0.15,
    motion_step=None,
    sample_distance=None,
    goal_tolerance=None,
    connect_distance=None,
    max_expansions=None,
):
    resolution = float(resolution or GRID_RESOLUTION)
    heading_bins = max(8, int(heading_bins))
    motion_step = float(motion_step or max(2.0 * resolution, 0.45))
    sample_distance = float(
        sample_distance or max(min(resolution, motion_step / 4.0), 0.05)
    )
    goal_tolerance = float(goal_tolerance or max(2.0 * resolution, 0.45))
    connect_distance = float(connect_distance or max(8.0 * resolution, 2.5))
    speed = max(float(speed), 0.05)
    cell_penalty = cell_penalty or {}
    edge_penalty = edge_penalty or {}
    max_expansions = int(
        max_expansions
        or max(2000, (float(display_diff) / resolution) ** 2 * heading_bins * 0.25)
    )

    start = np.asarray(start_state, dtype=float)[:4].copy()
    start[ActorStateEnum.VELOCITY] = speed
    if not hybrid_state_collision_free(
        start,
        collision_region,
        display_offset=display_offset,
        display_diff=display_diff,
        vehicle_length=vehicle_length,
        vehicle_width=vehicle_width,
    ):
        return None

    goal = np.asarray(goal_xy, dtype=float)[:2]
    min_turn_radius = float(vehicle_length) / max(np.tan(float(max_steer)), 1.0e-6)
    steer_values = np.asarray(
        [
            -float(max_steer),
            -0.65 * float(max_steer),
            -0.35 * float(max_steer),
            0.0,
            0.35 * float(max_steer),
            0.65 * float(max_steer),
            float(max_steer),
        ],
        dtype=float,
    )

    def heuristic(state):
        delta = goal - np.asarray(state, dtype=float)[:2]
        distance = float(np.linalg.norm(delta))
        if distance <= 1.0e-9:
            return 0.0
        goal_heading = float(np.arctan2(delta[1], delta[0]))
        heading_error = abs(wrap_angle(goal_heading - state[ActorStateEnum.THETA]))
        return distance + 0.15 * min_turn_radius * heading_error

    start_key = hybrid_state_key(
        start,
        display_offset=display_offset,
        resolution=resolution,
        heading_bins=heading_bins,
    )
    node_states = {start_key: start}
    parent = {start_key: (None, [])}
    cost_so_far = {start_key: 0.0}
    frontier = []
    counter = 0
    heappush(frontier, (heuristic(start), counter, start_key))
    expansions = 0

    while frontier and expansions < max_expansions:
        _priority, _counter, current_key = heappop(frontier)
        current = node_states[current_key]
        expansions += 1

        distance_to_goal = float(np.linalg.norm(goal - current[:2]))
        if distance_to_goal <= goal_tolerance:
            return reconstruct_hybrid_states(parent, node_states, current_key)

        if distance_to_goal <= connect_distance:
            connector = hybrid_connect_to_goal(
                current,
                goal,
                collision_region,
                display_offset=display_offset,
                display_diff=display_diff,
                vehicle_length=vehicle_length,
                vehicle_width=vehicle_width,
                max_steer=max_steer,
                speed=speed,
                sample_distance=sample_distance,
                max_distance=max(connect_distance, distance_to_goal + motion_step),
                goal_tolerance=goal_tolerance,
            )
            if connector is not None:
                goal_key = ("goal", expansions)
                node_states[goal_key] = connector[-1] if connector else current.copy()
                parent[goal_key] = (current_key, connector)
                return reconstruct_hybrid_states(parent, node_states, goal_key)

        current_cell = hybrid_state_cell(
            current,
            display_offset=display_offset,
            resolution=resolution,
        )
        for steer in steer_values:
            primitive_states = rollout_ackermann_distance(
                current,
                steer=steer,
                distance=motion_step,
                sample_distance=sample_distance,
                vehicle_length=vehicle_length,
                speed=speed,
            )
            if not hybrid_primitive_is_safe(
                primitive_states,
                collision_region,
                display_offset=display_offset,
                display_diff=display_diff,
                vehicle_length=vehicle_length,
                vehicle_width=vehicle_width,
            ):
                continue

            next_state = primitive_states[-1]
            next_key = hybrid_state_key(
                next_state,
                display_offset=display_offset,
                resolution=resolution,
                heading_bins=heading_bins,
            )
            if next_key == current_key:
                continue
            next_cell = hybrid_state_cell(
                next_state,
                display_offset=display_offset,
                resolution=resolution,
            )
            edge = tuple(sorted((current_cell, next_cell)))
            penalty = float(cell_penalty.get(next_cell, 0.0)) + float(
                edge_penalty.get(edge, 0.0)
            )
            steer_cost = (
                float(turn_penalty) * abs(float(steer)) / max(float(max_steer), 1.0e-6)
            )
            new_cost = (
                cost_so_far[current_key] + motion_step * (1.0 + steer_cost) + penalty
            )
            if new_cost >= cost_so_far.get(next_key, float("inf")):
                continue

            cost_so_far[next_key] = new_cost
            node_states[next_key] = next_state
            parent[next_key] = (current_key, primitive_states)
            counter += 1
            heappush(frontier, (new_cost + heuristic(next_state), counter, next_key))

    return None


def k_diverse_hybrid_astar_paths(
    start_state,
    goal_xy,
    static_polygons,
    *,
    display_offset,
    display_diff,
    vehicle_length,
    vehicle_width,
    max_steer,
    speed,
    dt,
    horizon,
    resolution=None,
    search_resolution=None,
    k=3,
    heading_bins=16,
    near_shortest_factor=1.8,
    max_overlap=0.65,
    max_attempts=30,
    diversity_penalty=1.0,
    turn_penalty=0.15,
    motion_step=None,
    goal_tolerance=None,
    connect_distance=None,
    debug=False,
):
    base_resolution = float(resolution or GRID_RESOLUTION)
    search_resolution = float(search_resolution or max(0.5, 2.5 * base_resolution))
    static_union = blocking_static_polygon_union(static_polygons)
    collision_region = (
        static_union.buffer(MIN_SEPARATION)
        if static_union is not None and not static_union.is_empty
        else None
    )
    if not goal_position_collision_free(
        goal_xy,
        collision_region,
        display_offset=display_offset,
        display_diff=display_diff,
        vehicle_length=vehicle_length,
        vehicle_width=vehicle_width,
        heading_bins=heading_bins,
    ):
        if debug:
            print(
                "[kpaths] goal rejected before hybrid search: "
                "no collision-free vehicle orientation at goal"
            )
        return []

    accepted = []
    accepted_cells = []
    cell_penalty = {}
    edge_penalty = {}
    best_length = None
    reject_counts = {
        "missing": 0,
        "duplicate": 0,
        "long": 0,
        "overlap": 0,
        "loop": 0,
    }
    max_attempts = max(int(max_attempts), int(k))

    for attempt in range(max_attempts):
        states = hybrid_astar_search(
            start_state,
            goal_xy,
            collision_region,
            display_offset=display_offset,
            display_diff=display_diff,
            vehicle_length=vehicle_length,
            vehicle_width=vehicle_width,
            max_steer=max_steer,
            speed=speed,
            resolution=search_resolution,
            heading_bins=heading_bins,
            cell_penalty=cell_penalty,
            edge_penalty=edge_penalty,
            turn_penalty=turn_penalty,
            motion_step=motion_step,
            goal_tolerance=goal_tolerance,
            connect_distance=connect_distance,
        )
        if states is None or len(states) < 2:
            reject_counts["missing"] += 1
            break

        cells = hybrid_route_cells(
            states,
            display_offset=display_offset,
            resolution=search_resolution,
        )
        length = hybrid_states_path_length(states)
        if best_length is None:
            best_length = max(length, search_resolution)
        duplicate = cells in accepted_cells
        overlap = grid_path_overlap(cells, accepted_cells)
        near_shortest = length <= best_length * float(near_shortest_factor)
        diverse = overlap <= float(max_overlap) or len(accepted_cells) == 0
        loop = grid_path_has_loop(cells)

        if not duplicate and not loop and near_shortest and diverse:
            max_points = max(int(horizon) + 1, 2)
            path = hybrid_states_to_frenet_path(
                states,
                speed=speed,
                dt=dt,
                max_points=max_points,
            )
            route = hybrid_states_to_route(
                states,
                min_spacing=max(search_resolution, 0.35),
            )
            accepted.append({"states": states, "path": path, "route": route})
            accepted_cells.append(cells)
            if len(accepted) >= int(k):
                break
        else:
            if duplicate:
                reject_counts["duplicate"] += 1
            if loop:
                reject_counts["loop"] += 1
            if not near_shortest:
                reject_counts["long"] += 1
            if not diverse:
                reject_counts["overlap"] += 1
            if debug:
                print(
                    "[kpaths] rejected hybrid route "
                    f"attempt={attempt} len={length:.3f} overlap={overlap:.2f} "
                    f"loop={loop} duplicate={duplicate}"
                )

        penalty_scale = float(diversity_penalty) * (1.0 + 0.25 * attempt)
        for cell in cells:
            cell_penalty[cell] = cell_penalty.get(cell, 0.0) + penalty_scale
        for a, b in zip(cells[:-1], cells[1:]):
            edge = tuple(sorted((a, b)))
            edge_penalty[edge] = edge_penalty.get(edge, 0.0) + 2.0 * penalty_scale

    if debug:
        lengths = [
            round(hybrid_states_path_length(item["states"]), 3) for item in accepted
        ]
        print(
            "[kpaths] hybrid accepted "
            f"{len(accepted)}/{int(k)} routes lengths={lengths} "
            f"search_resolution={search_resolution:.3f} "
            f"rejects={reject_counts}"
        )

    return accepted


def simplify_grid_route_points(points):
    points = dedupe_waypoints(points)
    if len(points) <= 2:
        return points

    simplified = [points[0]]
    prev_delta = None
    for idx in range(1, len(points) - 1):
        a = np.asarray(points[idx - 1], dtype=float)
        b = np.asarray(points[idx], dtype=float)
        c = np.asarray(points[idx + 1], dtype=float)
        delta1 = b - a
        delta2 = c - b
        norm1 = np.linalg.norm(delta1)
        norm2 = np.linalg.norm(delta2)
        if norm1 <= 1.0e-9 or norm2 <= 1.0e-9:
            continue
        direction = delta2 / norm2
        previous_direction = delta1 / norm1
        cross = (
            previous_direction[0] * direction[1] - previous_direction[1] * direction[0]
        )
        if prev_delta is None or abs(float(cross)) > 1.0e-3:
            simplified.append(points[idx])
        prev_delta = direction
    simplified.append(points[-1])
    return dedupe_waypoints(simplified)


def k_diverse_kinematic_routes(
    start_state,
    goal_xy,
    static_polygons,
    *,
    display_offset,
    display_diff,
    vehicle_length,
    vehicle_width,
    max_steer,
    resolution=None,
    k=3,
    heading_bins=16,
    near_shortest_factor=1.8,
    max_overlap=0.65,
    max_attempts=30,
    diversity_penalty=1.0,
    turn_penalty=0.15,
    debug=False,
):
    grid = build_static_planning_grid(
        static_polygons,
        display_offset=display_offset,
        display_diff=display_diff,
        vehicle_length=vehicle_length,
        vehicle_width=vehicle_width,
        resolution=resolution,
    )
    is_blocked = grid["is_blocked"]
    rows = int(grid["rows"])
    cols = int(grid["cols"])
    start_cell = find_nearest_free_cell(
        grid["point_to_cell"](start_state[:2]),
        is_blocked,
        cols,
        rows,
    )
    goal_cell = find_nearest_free_cell(
        grid["point_to_cell"](goal_xy),
        is_blocked,
        cols,
        rows,
    )
    if start_cell is None or goal_cell is None:
        return [[list(start_state[:2]), list(goal_xy)]]

    accepted_cells = []
    accepted_routes = []
    cell_penalty = {}
    edge_penalty = {}
    best_length = None
    max_attempts = max(int(max_attempts), int(k))
    reject_counts = {
        "duplicate": 0,
        "loop": 0,
        "long": 0,
        "overlap": 0,
        "initial_kinematic": 0,
        "obstacle": 0,
    }

    for attempt in range(max_attempts):
        oriented_path = astar_kinematic_grid(
            start_cell,
            goal_cell,
            start_state[ActorStateEnum.THETA],
            is_blocked,
            resolution=grid["resolution"],
            vehicle_length=vehicle_length,
            max_steer=max_steer,
            heading_bins=heading_bins,
            cell_penalty=cell_penalty,
            edge_penalty=edge_penalty,
            turn_penalty=turn_penalty,
            start_xy=start_state[:2],
            cell_to_point=grid["cell_to_point"],
        )
        if oriented_path is None:
            break

        cells = oriented_path_cells(oriented_path)
        path_length = grid_path_length(cells, grid["resolution"])
        if best_length is None:
            best_length = max(path_length, grid["resolution"])

        overlap = grid_path_overlap(cells, accepted_cells)
        near_shortest = path_length <= best_length * float(near_shortest_factor)
        diverse = overlap <= float(max_overlap) or len(accepted_cells) == 0
        loop = grid_path_has_loop(cells)
        duplicate = cells in accepted_cells
        points = [list(start_state[:2])]
        points.extend(grid["cell_to_point"](cell) for cell in cells[1:])
        goal = np.asarray(goal_xy, dtype=float).tolist()
        if (
            np.linalg.norm(np.asarray(points[-1], dtype=float) - np.asarray(goal))
            > 1.0e-6
        ):
            if not route_intersects_obstacle(
                [points[-1], goal],
                grid["obstacle_union"],
            ):
                points.append(goal)
        points = dedupe_waypoints(points)
        initial_ok, initial_reason = route_respects_initial_kinematics(
            points,
            start_state,
            vehicle_length=vehicle_length,
            max_steer=max_steer,
            heading_bins=heading_bins,
        )
        obstacle_ok = not route_intersects_obstacle(points, grid["obstacle_union"])
        if (
            not loop
            and near_shortest
            and diverse
            and not duplicate
            and initial_ok
            and obstacle_ok
        ):
            accepted_cells.append(cells)
            accepted_routes.append(points)
            if len(accepted_routes) >= int(k):
                break
        else:
            if duplicate:
                reject_counts["duplicate"] += 1
            if loop:
                reject_counts["loop"] += 1
            if not near_shortest:
                reject_counts["long"] += 1
            if not diverse:
                reject_counts["overlap"] += 1
            if not initial_ok:
                reject_counts["initial_kinematic"] += 1
            if not obstacle_ok:
                reject_counts["obstacle"] += 1
            if debug:
                first_error = route_initial_heading_error(
                    points,
                    start_state[ActorStateEnum.THETA],
                )
                print(
                    "[kpaths] rejected route "
                    f"attempt={attempt} "
                    f"reason={initial_reason if not initial_ok else 'constraints'} "
                    f"len={path_length:.3f} overlap={overlap:.2f} "
                    f"first_heading_error={deg(first_error):.1f}deg "
                    f"obstacle={not obstacle_ok}"
                )

        penalty_scale = float(diversity_penalty) * (1.0 + 0.25 * attempt)
        for cell in cells:
            cell_penalty[cell] = cell_penalty.get(cell, 0.0) + penalty_scale
        for a, b in zip(cells[:-1], cells[1:]):
            edge = tuple(sorted((a, b)))
            edge_penalty[edge] = edge_penalty.get(edge, 0.0) + 2.0 * penalty_scale

    if accepted_routes:
        if debug:
            print(
                "[kpaths] accepted "
                f"{len(accepted_routes)}/{int(k)} routes "
                f"rejects={reject_counts}"
            )
        return accepted_routes

    fallback = plan_static_route(
        start_state[:2],
        goal_xy,
        static_polygons,
        display_offset=display_offset,
        display_diff=display_diff,
        vehicle_length=vehicle_length,
        vehicle_width=vehicle_width,
        resolution=resolution,
        start_heading=start_state[ActorStateEnum.THETA],
    )
    if debug:
        print(
            "[kpaths] no valid diverse route found; using static-route fallback "
            f"rejects={reject_counts}"
        )
    return [fallback]


def prune_line_of_sight_route(points, obstacle_union):
    if obstacle_union is None or obstacle_union.is_empty or len(points) <= 2:
        return points

    pruned = [points[0]]
    anchor = 0
    while anchor < len(points) - 1:
        next_anchor = anchor + 1
        for candidate in range(len(points) - 1, anchor, -1):
            if not LineString([points[anchor], points[candidate]]).intersects(
                obstacle_union
            ):
                next_anchor = candidate
                break
        pruned.append(points[next_anchor])
        anchor = next_anchor
    return pruned


def dedupe_waypoints(waypoints, tolerance=1.0e-6):
    deduped = []
    for waypoint in waypoints:
        if (
            not deduped
            or np.linalg.norm(np.asarray(waypoint) - np.asarray(deduped[-1]))
            > tolerance
        ):
            deduped.append(list(waypoint))
    return deduped


def waypoint_segment_heading(waypoints, fallback_heading):
    if len(waypoints) < 2:
        return float(fallback_heading)

    for index in range(len(waypoints) - 1, 0, -1):
        start = np.asarray(waypoints[index - 1], dtype=float)
        end = np.asarray(waypoints[index], dtype=float)
        delta = end - start
        if np.linalg.norm(delta) > 1.0e-6:
            return float(np.arctan2(delta[1], delta[0]))
    return float(fallback_heading)


def point_within_display(point, display_offset, display_diff):
    if display_offset is None or display_diff is None:
        return True
    x, y = np.asarray(point, dtype=float)[:2]
    min_x = float(display_offset[0])
    min_y = float(display_offset[1])
    max_x = min_x + float(display_diff)
    max_y = min_y + float(display_diff)
    return min_x <= x <= max_x and min_y <= y <= max_y


def apply_initial_heading_constraint(
    route,
    *,
    start_heading=None,
    obstacle_union=None,
    display_offset=None,
    display_diff=None,
    vehicle_length=0.7,
    resolution=None,
    heading_lookahead=None,
    max_heading_error=np.deg2rad(35.0),
):
    route = dedupe_waypoints(route)
    if start_heading is None or len(route) < 2:
        return route

    start = np.asarray(route[0], dtype=float)
    first = np.asarray(route[1], dtype=float)
    first_delta = first - start
    first_distance = float(np.linalg.norm(first_delta))
    if first_distance <= 1.0e-6:
        return route

    first_heading = float(np.arctan2(first_delta[1], first_delta[0]))
    heading_error = abs(wrap_angle(first_heading - float(start_heading)))
    if heading_error <= float(max_heading_error):
        return route

    resolution = float(resolution or GRID_RESOLUTION)
    lookahead = float(
        heading_lookahead
        if heading_lookahead is not None
        else max(1.5 * float(vehicle_length), 2.0 * resolution, 0.25)
    )
    lookahead = min(lookahead, max(first_distance * 0.75, resolution))
    heading_vector = np.asarray(
        [np.cos(start_heading), np.sin(start_heading)],
        dtype=float,
    )
    anchor = start + lookahead * heading_vector
    if np.linalg.norm(anchor - start) <= 1.0e-6:
        return route
    if not point_within_display(anchor, display_offset, display_diff):
        return route
    if obstacle_union is not None and not obstacle_union.is_empty:
        if LineString([start, anchor, first]).intersects(obstacle_union):
            return route

    return dedupe_waypoints([route[0], anchor.tolist(), *route[1:]])


def polyline_cumulative_lengths(points):
    points = np.asarray(points, dtype=float)
    cumulative = np.zeros(points.shape[0], dtype=float)
    if points.shape[0] < 2:
        return cumulative
    deltas = np.diff(points[:, :2], axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    cumulative[1:] = np.cumsum(lengths)
    return cumulative


def project_point_to_polyline(point, waypoints):
    points = np.asarray(waypoints, dtype=float)
    point = np.asarray(point, dtype=float)[:2]
    if points.shape[0] == 0:
        return point, 0, 0.0, 0.0
    if points.shape[0] == 1:
        return points[0, :2], 0, 0.0, float(np.linalg.norm(point - points[0, :2]))

    cumulative = polyline_cumulative_lengths(points)
    best_projection = points[0, :2]
    best_segment = 0
    best_s = 0.0
    best_distance = float("inf")
    for segment in range(points.shape[0] - 1):
        a = points[segment, :2]
        b = points[segment + 1, :2]
        delta = b - a
        length2 = float(delta @ delta)
        if length2 <= 1.0e-12:
            continue
        u = float(np.clip(((point - a) @ delta) / length2, 0.0, 1.0))
        projection = a + u * delta
        distance = float(np.linalg.norm(point - projection))
        if distance < best_distance:
            best_projection = projection
            best_segment = segment
            best_s = cumulative[segment] + u * np.sqrt(length2)
            best_distance = distance

    return best_projection, best_segment, best_s, best_distance


def point_at_polyline_distance(waypoints, distance):
    points = np.asarray(waypoints, dtype=float)
    if points.shape[0] == 0:
        return np.zeros(2, dtype=float), 0
    if points.shape[0] == 1:
        return points[0, :2], 0

    cumulative = polyline_cumulative_lengths(points)
    distance = float(np.clip(distance, 0.0, cumulative[-1]))
    segment = int(np.searchsorted(cumulative, distance, side="right") - 1)
    segment = min(max(segment, 0), points.shape[0] - 2)
    segment_length = cumulative[segment + 1] - cumulative[segment]
    if segment_length <= 1.0e-12:
        return points[segment, :2], segment
    u = (distance - cumulative[segment]) / segment_length
    point = points[segment, :2] + u * (points[segment + 1, :2] - points[segment, :2])
    return point, segment


def buffered_static_obstacle_union(static_polygons, vehicle_length, vehicle_width):
    obstacle_union = blocking_static_polygon_union(static_polygons)
    if obstacle_union is None or obstacle_union.is_empty:
        return None
    robot_clearance = (
        max(float(vehicle_length), float(vehicle_width)) / 2.0 + MIN_SEPARATION
    )
    return obstacle_union.buffer(robot_clearance)


def route_initial_heading_error(route, start_heading):
    route = dedupe_waypoints(route)
    if start_heading is None or len(route) < 2:
        return 0.0

    start = np.asarray(route[0], dtype=float)
    for waypoint in route[1:]:
        waypoint = np.asarray(waypoint, dtype=float)
        delta = waypoint - start
        if np.linalg.norm(delta) > 1.0e-6:
            heading = float(np.arctan2(delta[1], delta[0]))
            return abs(wrap_angle(heading - float(start_heading)))
    return 0.0


def route_intersects_obstacle(route, obstacle_union):
    route = dedupe_waypoints(route)
    if obstacle_union is None or obstacle_union.is_empty or len(route) < 2:
        return False
    return LineString(route).intersects(obstacle_union)


def route_is_initially_feasible(
    route,
    *,
    start_heading,
    max_heading_error,
    obstacle_union=None,
):
    if route_initial_heading_error(route, start_heading) > float(max_heading_error):
        return False
    return not route_intersects_obstacle(route, obstacle_union)


def path_initial_heading_error(path, start_heading):
    points = frenet_path_xy(path)
    if points.shape[0] < 2:
        return 0.0
    start = points[0]
    for point in points[1:]:
        delta = point - start
        if np.linalg.norm(delta) > 1.0e-6:
            heading = float(np.arctan2(delta[1], delta[0]))
            return abs(wrap_angle(heading - float(start_heading)))
    return 0.0


def build_forward_recovery_route(
    start_state,
    static_polygons,
    *,
    display_offset,
    display_diff,
    vehicle_length,
    vehicle_width,
    resolution=None,
    recovery_lookahead=1.0,
    max_heading_error=np.deg2rad(35.0),
):
    resolution = float(resolution or GRID_RESOLUTION)
    start = np.asarray(start_state[:2], dtype=float)
    heading = float(start_state[ActorStateEnum.THETA])
    obstacle_union = buffered_static_obstacle_union(
        static_polygons,
        vehicle_length,
        vehicle_width,
    )
    distances = [
        max(float(recovery_lookahead), resolution),
        max(float(recovery_lookahead) * 0.6, resolution),
        max(float(recovery_lookahead) * 0.35, resolution),
        resolution,
    ]
    angle_offsets = [0.0]
    for fraction in (0.5, 1.0):
        angle_offsets.extend(
            [
                -float(max_heading_error) * fraction,
                float(max_heading_error) * fraction,
            ]
        )

    best = None
    for distance in distances:
        for offset in angle_offsets:
            candidate_heading = heading + offset
            candidate = start + distance * np.asarray(
                [np.cos(candidate_heading), np.sin(candidate_heading)],
                dtype=float,
            )
            route = [start.tolist(), candidate.tolist()]
            if not point_within_display(candidate, display_offset, display_diff):
                continue
            best = route
            if route_intersects_obstacle(route, obstacle_union):
                continue
            return route

    if best is not None:
        return best

    candidate = start + resolution * np.asarray(
        [np.cos(heading), np.sin(heading)],
        dtype=float,
    )
    return [start.tolist(), candidate.tolist()]


def build_route_rejoining_nominal(
    start_state,
    goal_xy,
    nominal_route,
    static_polygons,
    *,
    display_offset,
    display_diff,
    vehicle_length,
    vehicle_width,
    resolution=None,
    rejoin_lookahead=1.5,
    rejoin_search_distance=5.0,
    heading_lookahead=None,
    recovery_lookahead=1.0,
    max_heading_error=np.deg2rad(35.0),
):
    obstacle_union = buffered_static_obstacle_union(
        static_polygons,
        vehicle_length,
        vehicle_width,
    )
    nominal_route = dedupe_waypoints(nominal_route)
    if len(nominal_route) < 2:
        return plan_static_route(
            start_state[:2],
            goal_xy,
            static_polygons,
            display_offset=display_offset,
            display_diff=display_diff,
            vehicle_length=vehicle_length,
            vehicle_width=vehicle_width,
            resolution=resolution,
            start_heading=start_state[3],
            heading_lookahead=heading_lookahead,
            max_heading_error=max_heading_error,
        )

    points = np.asarray(nominal_route, dtype=float)
    cumulative = polyline_cumulative_lengths(points)
    if cumulative[-1] <= 1.0e-6:
        return build_forward_recovery_route(
            start_state,
            static_polygons,
            display_offset=display_offset,
            display_diff=display_diff,
            vehicle_length=vehicle_length,
            vehicle_width=vehicle_width,
            resolution=resolution,
            recovery_lookahead=recovery_lookahead,
            max_heading_error=max_heading_error,
        )

    _, _, s_current, _ = project_point_to_polyline(start_state[:2], nominal_route)
    resolution = float(resolution or GRID_RESOLUTION)
    start_offset = max(float(rejoin_lookahead), resolution)
    max_offset = max(start_offset, float(rejoin_search_distance))
    search_end = min(cumulative[-1], s_current + max_offset)
    rejoin_candidates = np.arange(
        s_current + start_offset,
        search_end + 0.5 * resolution,
        max(resolution, start_offset * 0.5),
    )
    if rejoin_candidates.size == 0 or rejoin_candidates[-1] < cumulative[-1]:
        rejoin_candidates = np.concatenate(
            [rejoin_candidates, np.asarray([cumulative[-1]], dtype=float)]
        )

    for rejoin_s in rejoin_candidates:
        rejoin_s = min(float(rejoin_s), cumulative[-1])
        rejoin_point, rejoin_segment = point_at_polyline_distance(
            nominal_route,
            rejoin_s,
        )

        if (
            np.linalg.norm(rejoin_point - np.asarray(start_state[:2], dtype=float))
            <= 1.0e-3
        ):
            continue

        connector = plan_static_route(
            start_state[:2],
            rejoin_point,
            static_polygons,
            display_offset=display_offset,
            display_diff=display_diff,
            vehicle_length=vehicle_length,
            vehicle_width=vehicle_width,
            resolution=resolution,
            start_heading=start_state[3],
            heading_lookahead=heading_lookahead,
            max_heading_error=max_heading_error,
        )

        tail = [rejoin_point.tolist()]
        tail.extend(
            points[index, :2].tolist()
            for index in range(rejoin_segment + 1, points.shape[0])
            if cumulative[index] > rejoin_s + 1.0e-6
        )
        if (
            np.linalg.norm(np.asarray(tail[-1]) - np.asarray(goal_xy, dtype=float))
            > 1.0e-6
        ):
            tail.append(list(goal_xy))

        route = dedupe_waypoints([*connector[:-1], *tail])
        if not route_intersects_obstacle(route, obstacle_union):
            return route

    return build_forward_recovery_route(
        start_state,
        static_polygons,
        display_offset=display_offset,
        display_diff=display_diff,
        vehicle_length=vehicle_length,
        vehicle_width=vehicle_width,
        resolution=resolution,
        recovery_lookahead=recovery_lookahead,
        max_heading_error=max_heading_error,
    )


def path_deviation_from_waypoints(path, waypoints):
    if not waypoints:
        return 0.0
    points = frenet_path_xy(path)
    if points.size == 0:
        return 0.0
    distances = [project_point_to_polyline(point, waypoints)[3] for point in points]
    return float(max(distances)) if distances else 0.0


def trajectory_respects_kinematic_model(
    path,
    *,
    vehicle_length,
    max_steer,
    curvature_tolerance=0.25,
):
    max_curvature = np.tan(float(max_steer)) / max(float(vehicle_length), 1.0e-6)
    max_allowed = max_curvature * (1.0 + float(curvature_tolerance))
    curvatures = np.asarray(getattr(path, "c", []), dtype=float)
    if curvatures.size == 0:
        points = frenet_path_xy(path)
        yaws = np.asarray(getattr(path, "yaw", []), dtype=float)
        if points.shape[0] < 3 or yaws.size < 2:
            return True
        ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
        dyaw = np.asarray(
            [wrap_angle(yaws[idx + 1] - yaws[idx]) for idx in range(yaws.size - 1)],
            dtype=float,
        )
        valid = ds > 1.0e-6
        if not np.any(valid):
            return True
        curvatures = np.zeros_like(ds)
        curvatures[valid] = dyaw[: ds.size][valid] / ds[valid]

    return bool(np.all(np.abs(curvatures[np.isfinite(curvatures)]) <= max_allowed))


def constrain_selected_path_to_nominal(
    paths,
    selected_index,
    scores,
    nominal_route,
    max_deviation,
    *,
    debug=False,
):
    if not paths or max_deviation is None or float(max_deviation) <= 0.0:
        return selected_index

    deviations = np.asarray(
        [path_deviation_from_waypoints(path, nominal_route) for path in paths],
        dtype=float,
    )
    allowed = deviations <= float(max_deviation)
    if selected_index < len(allowed) and allowed[selected_index]:
        return selected_index

    if np.any(allowed):
        if scores is not None:
            score_values = np.asarray(scores, dtype=float).copy()
            score_values[~allowed] = np.inf
            replacement = int(np.argmin(score_values))
        else:
            replacement = int(np.flatnonzero(allowed)[0])
    else:
        replacement = 0

    if debug:
        print(
            "[path-selection] "
            f"replaced={selected_index}->{replacement} "
            f"max_deviation={float(max_deviation):.3f} "
            f"deviations={deviations.tolist()}"
        )
    return replacement


def choose_control_path_index(
    paths,
    selected_index,
    *,
    start_heading,
    max_heading_error=np.deg2rad(90.0),
    debug=False,
):
    if not paths:
        return 0

    selected_index = int(np.clip(selected_index, 0, len(paths) - 1))
    heading_error = path_initial_heading_error(
        paths[selected_index]["path"], start_heading
    )
    if heading_error <= float(max_heading_error):
        return selected_index

    if debug:
        print(
            "[path-selection] "
            f"control path reset to nominal selected={selected_index} "
            f"heading_error={deg(heading_error):.1f}deg"
        )
    return 0


def plan_static_route(
    start_xy,
    goal_xy,
    static_polygons,
    *,
    display_offset,
    display_diff,
    vehicle_length,
    vehicle_width,
    resolution=None,
    start_heading=None,
    heading_lookahead=None,
    max_heading_error=np.deg2rad(35.0),
):
    obstacle_union = blocking_static_polygon_union(static_polygons)
    if obstacle_union is None or obstacle_union.is_empty:
        return apply_initial_heading_constraint(
            [list(start_xy), list(goal_xy)],
            start_heading=start_heading,
            display_offset=display_offset,
            display_diff=display_diff,
            vehicle_length=vehicle_length,
            resolution=resolution,
            heading_lookahead=heading_lookahead,
            max_heading_error=max_heading_error,
        )

    robot_clearance = (
        max(float(vehicle_length), float(vehicle_width)) / 2.0 + MIN_SEPARATION
    )
    obstacle_union = obstacle_union.buffer(robot_clearance)
    if not LineString([start_xy, goal_xy]).intersects(obstacle_union):
        return apply_initial_heading_constraint(
            [list(start_xy), list(goal_xy)],
            start_heading=start_heading,
            obstacle_union=obstacle_union,
            display_offset=display_offset,
            display_diff=display_diff,
            vehicle_length=vehicle_length,
            resolution=resolution,
            heading_lookahead=heading_lookahead,
            max_heading_error=max_heading_error,
        )

    resolution = float(resolution or GRID_RESOLUTION)
    min_x = float(display_offset[0])
    min_y = float(display_offset[1])
    max_x = min_x + float(display_diff)
    max_y = min_y + float(display_diff)
    cols = int(np.ceil((max_x - min_x) / resolution))
    rows = int(np.ceil((max_y - min_y) / resolution))

    is_blocked = np.zeros((rows, cols), dtype=bool)
    for row in range(rows):
        y = min_y + (row + 0.5) * resolution
        for col in range(cols):
            x = min_x + (col + 0.5) * resolution
            is_blocked[row, col] = obstacle_union.covers(Point(x, y))

    def point_to_cell(point):
        col = int(np.clip(np.floor((point[0] - min_x) / resolution), 0, cols - 1))
        row = int(np.clip(np.floor((point[1] - min_y) / resolution), 0, rows - 1))
        return col, row

    def cell_to_point(cell):
        col, row = cell
        return [
            min(max(min_x + (col + 0.5) * resolution, min_x), max_x),
            min(max(min_y + (row + 0.5) * resolution, min_y), max_y),
        ]

    start_cell = find_nearest_free_cell(point_to_cell(start_xy), is_blocked, cols, rows)
    goal_cell = find_nearest_free_cell(point_to_cell(goal_xy), is_blocked, cols, rows)
    if start_cell is None or goal_cell is None:
        print(
            "WARNING: No free cell found for static route planning; using direct route."
        )
        return apply_initial_heading_constraint(
            [list(start_xy), list(goal_xy)],
            start_heading=start_heading,
            obstacle_union=obstacle_union,
            display_offset=display_offset,
            display_diff=display_diff,
            vehicle_length=vehicle_length,
            resolution=resolution,
            heading_lookahead=heading_lookahead,
            max_heading_error=max_heading_error,
        )

    grid_path = astar_grid(start_cell, goal_cell, is_blocked)
    if grid_path is None:
        print("WARNING: No static-obstacle route found; using direct route.")
        return apply_initial_heading_constraint(
            [list(start_xy), list(goal_xy)],
            start_heading=start_heading,
            obstacle_union=obstacle_union,
            display_offset=display_offset,
            display_diff=display_diff,
            vehicle_length=vehicle_length,
            resolution=resolution,
            heading_lookahead=heading_lookahead,
            max_heading_error=max_heading_error,
        )

    route = [list(start_xy)]
    route.extend(cell_to_point(cell) for cell in grid_path[1:-1])
    route.append(list(goal_xy))
    route = prune_line_of_sight_route(route, obstacle_union)
    return apply_initial_heading_constraint(
        route,
        start_heading=start_heading,
        obstacle_union=obstacle_union,
        display_offset=display_offset,
        display_diff=display_diff,
        vehicle_length=vehicle_length,
        resolution=resolution,
        heading_lookahead=heading_lookahead,
        max_heading_error=max_heading_error,
    )


def generate_trajectory_for_waypoints(
    start,
    waypoints,
    args,
    *,
    static_polygons=None,
    display_offset=None,
    display_diff=None,
    vehicle_length=0.7,
    vehicle_width=0.7,
    resolution=None,
    trajectories_requested=1,
    allow_recovery=True,
):
    from trajectory_planner.trajectory import Cubic

    initial_v = max(0, start[2])
    start_xy = list(start[:2])
    waypoints = dedupe_waypoints(waypoints)
    if len(waypoints) < 2:
        return []
    obstacle_union = buffered_static_obstacle_union(
        static_polygons,
        vehicle_length,
        vehicle_width,
    )
    max_heading_error = np.deg2rad(
        getattr(args, "max_initial_route_heading_error_deg", 35.0)
    )
    if route_intersects_obstacle(waypoints, obstacle_union):
        if getattr(args, "debug_steering", False):
            print(
                "[route-recovery] "
                "trajectory reference route intersects static obstacle "
                f"heading_error={deg(route_initial_heading_error(waypoints, start[3])):.1f}deg"
            )
        if not allow_recovery:
            return []
        waypoints = build_forward_recovery_route(
            start,
            static_polygons,
            display_offset=display_offset,
            display_diff=display_diff,
            vehicle_length=vehicle_length,
            vehicle_width=vehicle_width,
            resolution=resolution,
            recovery_lookahead=getattr(args, "recovery_route_lookahead", 1.0),
            max_heading_error=max_heading_error,
        )

    heading_error = route_initial_heading_error(waypoints, start[3])
    reference_initial_v = initial_v
    if heading_error > max_heading_error:
        reference_initial_v = min(
            initial_v,
            getattr(args, "heading_mismatch_reference_speed", 0.05),
        )
        if getattr(args, "debug_steering", False):
            print(
                "[route-heading] "
                f"heading_error={deg(heading_error):.1f}deg "
                f"reference_initial_v={reference_initial_v:.3f}"
            )

    path = Cubic(waypoints=waypoints, dt=args.tick_time)
    initial_heading = start[3]
    final_heading = waypoint_segment_heading(waypoints, initial_heading)
    path.np_trajectory(
        [reference_initial_v, args.robot_speed],
        initial_heading,
        final_heading,
        cubic_fn=Cubic.np_polynomial_time_scaling_3rd_order,
    )

    path = path.get_np_trajectory()
    path = sanitize_reference_path(path)
    if path.shape[0] < 2:
        fallback = np.asarray([start_xy, waypoints[-1]], dtype=float)
        headings = np.full((2, 1), float(start[3]))
        speeds = np.asarray([[initial_v], [args.robot_speed]], dtype=float)
        path = np.hstack([fallback, speeds, headings])

    from trajectory_planner.trajectory_planner import TrajectoryPlanner

    planner = TrajectoryPlanner(path, dt=args.tick_time)

    trajectories = planner.generate_trajectories(
        pos=start_xy,
        initial_v=initial_v,
        target_v=args.robot_speed,
        trajectories_requested=max(1, int(trajectories_requested)),
        planning_horizon=args.horizon,
    )
    return trajectories


def generate_k_path_trajectories(
    start,
    end,
    args,
    *,
    static_polygons=None,
    display_offset=None,
    display_diff=None,
    vehicle_length=0.7,
    vehicle_width=0.7,
    vehicle_scale=1.0,
    max_steer=np.deg2rad(30.0),
    resolution=None,
):

    speed = args.robot_speed
    dt = args.tick_time
    horizon = args.horizon
    k = max(1, int(args.trajectory_count))
    heading_bins = args.kpaths_heading_bins
    near_shortest_factor = args.kpaths_near_shortest_factor
    max_overlap = args.kpaths_max_overlap
    max_attempts = args.kpaths_max_attempts
    diversity_penalty = args.kpaths_diversity_penalty
    turn_penalty = args.kpaths_turn_penalty
    debug = (
        getattr(args, "debug_paths", False)
        or getattr(args, "debug_kpaths", False)
        or getattr(args, "debug_steering", False)
    )
    base_resolution = float(resolution or GRID_RESOLUTION)
    min_turn_radius = float(vehicle_length) / max(np.tan(float(max_steer)), 1.0e-6)
    search_resolution = float(
        args.kpaths_search_resolution or max(min_turn_radius, base_resolution)
    )

    grid = build_static_planning_grid(
        static_polygons,
        display_offset=display_offset,
        display_diff=display_diff,
        vehicle_length=vehicle_length,
        vehicle_width=vehicle_width,
        resolution=search_resolution,
    )
    static_union = blocking_static_polygon_union(static_polygons)
    collision_region = (
        static_union.buffer(MIN_SEPARATION * float(vehicle_scale))
        if static_union is not None and not static_union.is_empty
        else None
    )

    start_cell = find_nearest_free_cell(
        grid["point_to_cell"](start[:2]),
        grid["is_blocked"],
        grid["cols"],
        grid["rows"],
    )
    goal_cell = find_nearest_free_cell(
        grid["point_to_cell"](end[:2]),
        grid["is_blocked"],
        grid["cols"],
        grid["rows"],
    )
    if start_cell is None or goal_cell is None:
        if debug:
            print("[kpaths] no free start or goal cell in coarse search grid")
        return []

    anchor_cells = build_kpath_anchor_cells(
        grid,
        start_cell,
        goal_cell,
        max_anchors=max(int(max_attempts), int(k)),
    )
    candidates = []
    candidate_cells = []
    cell_penalty = {}
    edge_penalty = {}
    best_length = None
    reject_counts = {
        "missing": 0,
        "duplicate": 0,
        "loop": 0,
        "long": 0,
        "overlap": 0,
        "initial": 0,
        "turn_radius": 0,
        "collision": 0,
        "kinematics": 0,
    }

    for attempt in range(max(int(max_attempts), int(k))):
        anchor_cell = None
        if anchor_cells and attempt > 0:
            anchor_cell = anchor_cells[(attempt - 1) % len(anchor_cells)]

        cells = astar_grid_penalized_via(
            start_cell,
            goal_cell,
            anchor_cell,
            grid["is_blocked"],
            cell_penalty=cell_penalty,
            edge_penalty=edge_penalty,
            turn_penalty=turn_penalty,
            start_heading=start[ActorStateEnum.THETA],
            resolution=search_resolution,
            min_turn_radius=min_turn_radius,
        )
        if not cells:
            reject_counts["missing"] += 1
            continue

        length = grid_path_length(cells, search_resolution)
        if best_length is None:
            best_length = max(length, search_resolution)
        duplicate = cells in candidate_cells
        loop = grid_path_has_loop(cells)
        overlap = grid_path_overlap(cells, candidate_cells)
        near_shortest = length <= best_length * float(near_shortest_factor)

        route = [list(start[:2])]
        route.extend(grid["cell_to_point"](cell) for cell in cells[1:-1])
        route.append(list(end[:2]))
        route = simplify_route_collinear(route)
        initial_ok, initial_reason = route_respects_initial_kinematics(
            route,
            start,
            vehicle_length=vehicle_length,
            max_steer=max_steer,
            heading_bins=heading_bins,
        )
        smoothed_route, smooth_reason = ackermann_smooth_route(
            route,
            start_heading=start[ActorStateEnum.THETA],
            vehicle_length=vehicle_length,
            max_steer=max_steer,
            sample_distance=max(search_resolution * 0.2, 0.05),
        )

        path = None
        collision_free = False
        if smoothed_route is not None:
            path = points_to_frenet_path(
                smoothed_route,
                start_heading=start[ActorStateEnum.THETA],
                speed=speed,
                dt=dt,
                max_points=max(int(horizon) + 1, 2),
            )
            collision_free = frenet_path_static_collision_free(
                path,
                collision_region,
                vehicle_length=vehicle_length,
                vehicle_width=vehicle_width,
            )

        kinematics_ok = (
            smoothed_route is not None
            and trajectory_respects_kinematic_model(
                path,
                vehicle_length=vehicle_length,
                max_steer=max_steer,
                curvature_tolerance=args.kpaths_curvature_tolerance,
            )
        )

        if (
            not duplicate
            and not loop
            and near_shortest
            and initial_ok
            and smoothed_route is not None
            and collision_free
            and kinematics_ok
        ):
            candidates.append(
                {
                    "path": path,
                    "route": smoothed_route,
                    "cells": cells,
                    "length": length,
                    "anchor": anchor_cell,
                    "generator": "kpaths",
                }
            )
            candidate_cells.append(cells)
        else:
            if duplicate:
                reject_counts["duplicate"] += 1
            if loop:
                reject_counts["loop"] += 1
            if not near_shortest:
                reject_counts["long"] += 1
            if not initial_ok:
                reject_counts["initial"] += 1
            if smoothed_route is None:
                reject_counts["turn_radius"] += 1
            if smoothed_route is not None and not collision_free:
                reject_counts["collision"] += 1
            if not kinematics_ok:
                reject_counts["kinematics"] += 1
            if debug:
                print(
                    "[kpaths] rejected grid route "
                    f"attempt={attempt} anchor={anchor_cell} "
                    f"len={length:.3f} overlap={overlap:.2f} "
                    f"reason={smooth_reason or initial_reason or 'constraints'}"
                )

        penalty_scale = float(diversity_penalty) * (1.0 + 0.25 * attempt)
        for cell in cells:
            cell_penalty[cell] = cell_penalty.get(cell, 0.0) + penalty_scale
        for a, b in zip(cells[:-1], cells[1:]):
            edge = tuple(sorted((a, b)))
            edge_penalty[edge] = edge_penalty.get(edge, 0.0) + 2.0 * penalty_scale

    accepted = select_best_kpath_candidates(
        candidates,
        k=k,
        max_overlap=max_overlap,
    )

    if debug:
        lengths = [round(waypoint_route_length(item["route"]), 3) for item in accepted]
        print(
            "[kpaths] grid accepted "
            f"{len(accepted)}/{int(k)} routes "
            f"candidates={len(candidates)}/{int(max_attempts)} "
            f"anchors={len(anchor_cells)} "
            f"lengths={lengths} "
            f"search_resolution={search_resolution:.3f} "
            f"rejects={reject_counts}"
        )

    return accepted


def generate_trajectories(
    start,
    end,
    args,
    *,
    static_polygons=None,
    display_offset=None,
    display_diff=None,
    vehicle_length=0.7,
    vehicle_width=0.7,
    vehicle_scale=1.0,
    max_steer=np.deg2rad(30.0),
    resolution=None,
):
    generator = str(getattr(args, "trajectory_generator", "kpaths")).lower()

    if generator == "kpaths":
        return generate_k_path_trajectories(
            start,
            end,
            args,
            static_polygons=static_polygons,
            display_offset=display_offset,
            display_diff=display_diff,
            vehicle_length=vehicle_length,
            vehicle_width=vehicle_width,
            vehicle_scale=vehicle_scale,
            max_steer=max_steer,
            resolution=resolution,
        )
    elif generator == "specialk":
        return generate_specialk_trajectories(
            start,
            end,
            args,
            static_polygons=static_polygons,
            display_offset=display_offset,
            display_diff=display_diff,
            vehicle_length=vehicle_length,
            vehicle_width=vehicle_width,
            vehicle_scale=vehicle_scale,
            max_steer=max_steer,
            resolution=resolution,
        )
    else:
        raise ValueError(f"Unsupported trajectory generator: {generator}")


def get_control(
    mppi,
    costmap,
    origin,
    resolution,
    robot_model,
    u_nom,
    initial_state,
    goal,
    path,
    agents,
    agent_predictions,
    args,
    static_polygons=None,
    u_prev=None,
    debug_tick=None,
):
    """
    Given the current state (x,y,v,theta) and the path, return the control
    """

    control_timing = {}
    tic = perf_counter()
    section_start = perf_counter()
    if True:  # u_nom is None:
        u_nom = nominal_controls_to_path(robot_model, initial_state, path, args)
    else:
        u_nom[:-1, :] = u_nom[1:, :]
        # u_nom[-1, :] = 0
    control_timing["nominal"] = perf_counter() - section_start

    # Apply clipping to u_nom (fix: assign back the clipped values)
    u_nom[:, 0] = np.clip(
        u_nom[:, 0], a_min=-CONTROL_LIMITS[0], a_max=CONTROL_LIMITS[0]
    )
    u_nom[:, 1] = np.clip(
        u_nom[:, 1], a_min=-CONTROL_LIMITS[1], a_max=CONTROL_LIMITS[1]
    )

    if getattr(args, "debug_steering", False):
        print(f"DEBUG: u_nom[0] = [{u_nom[0,0]:.4f}, {deg(u_nom[0,1]):.1f}deg]")
        print(f"DEBUG: u_nom[1] = [{u_nom[1,0]:.4f}, {deg(u_nom[1,1]):.1f}deg]")

    x_nom = np.zeros((args.horizon + 1, 4))
    pts_to_update = min(args.horizon + 1, len(path.x))
    x_nom[:pts_to_update, 0] = path.x[:pts_to_update]
    x_nom[:pts_to_update, 1] = path.y[:pts_to_update]
    x_nom[:pts_to_update, 2] = path.s_d[:pts_to_update]
    x_nom[:pts_to_update, 3] = path.yaw[:pts_to_update]

    if pts_to_update < x_nom.shape[0]:
        x_nom[pts_to_update:, 0] = path.x[pts_to_update - 1]
        x_nom[pts_to_update:, 1] = path.y[pts_to_update - 1]
        x_nom[pts_to_update:, 2] = 0
        x_nom[pts_to_update:, 3] = path.yaw[pts_to_update - 1]

    actors = []
    for polygon in static_polygons or []:
        if not getattr(polygon, "blocking", True):
            continue
        points = np.asarray(getattr(polygon, "points", []), dtype=np.float32)
        if points.ndim == 2 and points.shape[0] >= 3 and points.shape[1] >= 2:
            actors.append(
                {
                    "polygon": points[:, :2],
                    "static": True,
                    "polygon_class": getattr(polygon, "polygon_class", "polygon"),
                }
            )

    dynamic_actor_obstacles, dynamic_actor_debug = build_dynamic_obstacles_for_mppi(
        agents,
        agent_predictions,
        args.horizon,
        robot_model=robot_model,
        hard_clearance_margin=args.dynamic_hard_clearance_margin,
    )
    actors.extend(dynamic_actor_obstacles)
    if getattr(args, "debug_mppi", False) or getattr(args, "debug_steering", False):
        print(
            "[dynamic-obstacles] "
            f"tick={debug_tick} visible={len(agents)} horizon={args.horizon} "
            f"lookahead={args.horizon * args.tick_time:.2f}s "
            f"agents={dynamic_actor_debug}"
        )

    section_start = perf_counter()
    u, u_var, u_weights = mppi.find_control(
        costmap=costmap,
        origin=origin,
        resolution=resolution,
        x_init=initial_state,
        x_goal=goal,
        x_nom=x_nom,
        u_nom=u_nom,
        obstacles=actors,
        dt=args.tick_time,
    )
    control_timing["mppi"] = perf_counter() - section_start
    u_mppi = np.asarray(u, dtype=float).copy()
    pre_filter_weights = np.asarray(u_weights, dtype=float).copy()

    # # Debug: print MPPI result
    # print(f"DEBUG: MPPI result u[0] = [{u[0,0]:.4f}, {np.degrees(u[0,1]):.1f}°]")
    # print(
    #     f"DEBUG: Nominal vs MPPI: steering {np.degrees(u_nom[0,1]):.1f}° -> {np.degrees(u[0,1]):.1f}°"
    # )

    mppi_time = control_timing["mppi"]
    total_weight = np.sum(u_weights)
    static_union = None
    static_debug = None
    dynamic_debug = None
    control_timing["static_sample_filter"] = 0.0
    control_timing["static_final_check"] = 0.0
    control_timing["dynamic_sample_filter"] = 0.0
    control_timing["dynamic_final_check"] = 0.0
    static_hard_clearance = mppi_static_clearance_margin(
        robot_model,
        args.static_hard_clearance_margin,
    )
    if static_polygons and args.host_static_sample_filter:
        section_start = perf_counter()
        u_weights, static_union, static_debug = filter_static_collision_samples(
            robot_model=robot_model,
            initial_state=initial_state,
            u_nom=u_nom,
            u_variations=u_var,
            weights=u_weights,
            static_polygons=static_polygons,
            dt=args.tick_time,
            clearance=static_hard_clearance,
        )
        control_timing["static_sample_filter"] = perf_counter() - section_start
        static_safe_weight = float(np.sum(u_weights))
        if static_safe_weight > 0.0:
            sampled_controls = u_nom[np.newaxis, :, :] + u_var
            u = bounded_weighted_control_average(
                u_nom=u_nom,
                sampled_controls=sampled_controls,
                weights=u_weights,
                limits=CONTROL_LIMITS,
            )
            u[:, 0] = np.clip(
                u[:, 0],
                a_min=-CONTROL_LIMITS[0],
                a_max=CONTROL_LIMITS[0],
            )
            u[:, 1] = np.clip(
                u[:, 1],
                a_min=-CONTROL_LIMITS[1],
                a_max=CONTROL_LIMITS[1],
            )
        total_weight = np.sum(u_weights)
    elif static_polygons:
        static_union = blocking_static_polygon_union(static_polygons)

    if agents and args.host_dynamic_sample_filter:
        section_start = perf_counter()
        u_weights, dynamic_debug = filter_dynamic_collision_samples(
            robot_model=robot_model,
            initial_state=initial_state,
            u_nom=u_nom,
            u_variations=u_var,
            weights=u_weights,
            agents=agents,
            agent_predictions=agent_predictions,
            horizon=args.horizon,
            dt=args.tick_time,
            clearance=MIN_SEPARATION * scene_scale(robot_model),
        )
        control_timing["dynamic_sample_filter"] = perf_counter() - section_start
        dynamic_safe_weight = float(np.sum(u_weights))
        if dynamic_safe_weight > 0.0:
            sampled_controls = u_nom[np.newaxis, :, :] + u_var
            u = bounded_weighted_control_average(
                u_nom=u_nom,
                sampled_controls=sampled_controls,
                weights=u_weights,
                limits=CONTROL_LIMITS,
            )
            u[:, 0] = np.clip(
                u[:, 0],
                a_min=-CONTROL_LIMITS[0],
                a_max=CONTROL_LIMITS[0],
            )
            u[:, 1] = np.clip(
                u[:, 1],
                a_min=-CONTROL_LIMITS[1],
                a_max=CONTROL_LIMITS[1],
            )
        total_weight = np.sum(u_weights)

    # Enhanced emergency stop logic
    WEIGHT_THRESHOLD = 1e-10  # Threshold for extremely small weights
    MAX_EFFECTIVE_COST = 1e6  # Threshold for extremely high costs

    emergency_stop = False
    dynamic_hard_clearance = mppi_dynamic_collision_buffer(
        robot_model,
        args.dynamic_hard_clearance_margin,
    )

    if total_weight == 0:
        print("EMERGENCY STOP: No valid control found (zero weights)")
        emergency_stop = True
    elif total_weight < WEIGHT_THRESHOLD:
        print(
            f"EMERGENCY STOP: All paths have extremely high cost (total_weight={total_weight:.2e})"
        )
        emergency_stop = True
    else:
        # Check if the effective sample size is too low (indicates poor quality solutions)
        ess = (total_weight**2) / (np.sum(u_weights**2) + 1e-12)
        if ess < 0.1:  # Less than 0.1 effective samples
            print(f"EMERGENCY STOP: Poor quality control solution (ESS={ess:.3f})")
            emergency_stop = True

    if emergency_stop:
        recovery = find_safe_control_candidate(
            robot_model=robot_model,
            initial_state=initial_state,
            u_nom=u_nom,
            u_variations=u_var,
            weights=u_weights,
            dt=args.tick_time,
            static_union=static_union,
            static_clearance=static_hard_clearance,
            agents=agents,
            agent_predictions=agent_predictions,
            horizon=args.horizon,
            dynamic_clearance=dynamic_hard_clearance,
        )
        if recovery is not None and recovery.get("safe", False):
            u = np.asarray(recovery["controls"], dtype=float).copy()
            emergency_stop = False
            print(
                "RECOVERY: using safe control after invalid MPPI weights "
                f"source={recovery.get('source')} sample={recovery.get('sample_idx')}"
            )

    if not emergency_stop and static_union is not None:
        section_start = perf_counter()
        selected_trajectory = run_trajectory(
            vehicle=robot_model,
            initial_state=initial_state,
            controls=u,
            dt=args.tick_time,
        )
        if trajectory_collides_with_static(
            selected_trajectory,
            static_union,
            vehicle_length=robot_model.L,
            vehicle_width=robot_model.W,
            clearance=static_hard_clearance,
        ):
            recovery = find_safe_control_candidate(
                robot_model=robot_model,
                initial_state=initial_state,
                u_nom=u_nom,
                u_variations=u_var,
                weights=u_weights,
                dt=args.tick_time,
                static_union=static_union,
                static_clearance=static_hard_clearance,
                agents=agents,
                agent_predictions=agent_predictions,
                horizon=args.horizon,
                dynamic_clearance=dynamic_hard_clearance,
            )
            if recovery is not None and recovery.get("safe", False):
                u = np.asarray(recovery["controls"], dtype=float).copy()
                print(
                    "RECOVERY: selected MPPI average intersects a static obstacle; "
                    f"using {recovery.get('source')} control "
                    f"sample={recovery.get('sample_idx')}"
                )
            else:
                print(
                    "EMERGENCY STOP: Selected MPPI control intersects a static obstacle"
                )
                emergency_stop = True
        control_timing["static_final_check"] = perf_counter() - section_start

    if not emergency_stop and agents:
        section_start = perf_counter()
        selected_trajectory = run_trajectory(
            vehicle=robot_model,
            initial_state=initial_state,
            controls=u,
            dt=args.tick_time,
        )
        dynamic_summary = trajectory_dynamic_collision_summary(
            trajectory=selected_trajectory,
            agents=agents,
            agent_predictions=agent_predictions,
            horizon=args.horizon,
            robot_model=robot_model,
            clearance=dynamic_hard_clearance,
        )
        if dynamic_summary["collision"]:
            closest = dynamic_summary["closest"] or {}
            recovery = find_safe_control_candidate(
                robot_model=robot_model,
                initial_state=initial_state,
                u_nom=u_nom,
                u_variations=u_var,
                weights=u_weights,
                dt=args.tick_time,
                static_union=static_union,
                static_clearance=static_hard_clearance,
                agents=agents,
                agent_predictions=agent_predictions,
                horizon=args.horizon,
                dynamic_clearance=dynamic_hard_clearance,
            )
            if recovery is not None and recovery.get("safe", False):
                u = np.asarray(recovery["controls"], dtype=float).copy()
                print(
                    "RECOVERY: selected MPPI average intersects a dynamic agent; "
                    f"using {recovery.get('source')} control "
                    f"sample={recovery.get('sample_idx')}"
                )
            else:
                print(
                    "EMERGENCY STOP: Selected MPPI control intersects a dynamic agent "
                    f"agent={closest.get('agent_id')} step={closest.get('step')} "
                    f"dist={dynamic_summary['min_distance']:.3f} "
                    f"threshold={closest.get('threshold'):.3f}"
                )
                emergency_stop = True
        control_timing["dynamic_final_check"] = perf_counter() - section_start

    if emergency_stop:
        # Emergency stop: decelerate to zero velocity
        fallback_steer = float(u_nom[0, 1]) if u_nom is not None else 0.0
        fallback_steer = np.clip(
            fallback_steer,
            -CONTROL_LIMITS[1],
            CONTROL_LIMITS[1],
        )
        u[0] = [
            -initial_state[ActorStateEnum.VELOCITY] / args.tick_time,
            fallback_steer,
        ]
        print(
            f"  Applied emergency brake: accel={u[0][0]:.3f}, "
            f"steering={np.degrees(u[0][1]):.1f}°"
        )
        # Set weights to zero for consistency
        u_weights = np.zeros_like(u_weights)

    if getattr(args, "debug_steering", False):
        print_planned_steering_debug(
            tick=debug_tick,
            robot_model=robot_model,
            initial_state=initial_state,
            path=path,
            u_nom=u_nom,
            u_final=u,
            args=args,
            total_weight=total_weight,
            u_weights=u_weights,
            static_union=static_union,
            emergency_stop=emergency_stop,
        )
        print_mppi_filter_debug(
            tick=debug_tick,
            u_nom=u_nom,
            u_mppi=u_mppi,
            u_final=u,
            pre_filter_weights=pre_filter_weights,
            post_filter_weights=u_weights,
            static_debug=static_debug,
        )
        if dynamic_debug is not None:
            dynamic_free = np.asarray(dynamic_debug["collision_free"], dtype=bool)
            all_violate = bool(
                dynamic_debug.get("all_samples_violate_clearance", False)
            )
            print(
                "[dynamic-filter] "
                f"tick={debug_tick} safe={np.count_nonzero(dynamic_free)}/{dynamic_free.size} "
                f"min_dist={np.nanmin(dynamic_debug['min_distances']):.3f} "
                f"all_violate_clearance={all_violate}"
            )

    if getattr(args, "debug_timing", False) or getattr(args, "debug_steering", False):
        print(f"Time to find control: {mppi_time}")
        print(
            f"Total weight: {total_weight:.2e}, "
            f"ESS: {(total_weight ** 2) / (np.sum(u_weights ** 2) + 1e-12):.3f}"
        )

    # select a sample set of trajectories for review/visualization
    section_start = perf_counter()
    trajectories, trajectory_indices = rollout_trajectories(
        vehicle=robot_model,
        initial_state=initial_state,
        u_nom=u_nom,
        u_variations=u_var,
        weights=u_weights,
        dt=args.tick_time,
        return_indices=True,
    )
    trajectory_weights = np.asarray(u_weights, dtype=float)[trajectory_indices]
    # positive_display = trajectory_weights > 0.0
    # if np.any(positive_display):
    #     static_display_collisions = 0
    #     dynamic_display_collisions = 0
    #     for trajectory in trajectories[positive_display]:
    #         if static_union is not None and trajectory_collides_with_static(
    #             trajectory,
    #             static_union,
    #             vehicle_length=robot_model.L,
    #             vehicle_width=robot_model.W,
    #             clearance=static_hard_clearance,
    #         ):
    #             static_display_collisions += 1
    #         if agents:
    #             dynamic_summary = trajectory_dynamic_collision_summary(
    #                 trajectory=trajectory,
    #                 agents=agents,
    #                 agent_predictions=agent_predictions,
    #                 horizon=args.horizon,
    #                 robot_model=robot_model,
    #                 clearance=dynamic_hard_clearance,
    #             )
    #             if dynamic_summary["collision"]:
    #                 dynamic_display_collisions += 1
    #     if static_display_collisions or dynamic_display_collisions:
    #         print(
    #             "[rollout-display-audit] "
    #             f"tick={debug_tick} positive={int(np.count_nonzero(positive_display))} "
    #             f"static_colliding={static_display_collisions} "
    #             f"dynamic_colliding={dynamic_display_collisions}"
    #         )
    control_timing["rollout_display"] = perf_counter() - section_start

    control_timing["trajectory_agent_check"] = 0.0
    control_timing["total"] = perf_counter() - tic
    args._last_control_timing = dict(control_timing)
    if args.debug_timing:
        print(
            "[control-timing] "
            f"tick={debug_tick} "
            + " ".join(
                f"{key}={value * 1000.0:.2f}ms" for key, value in control_timing.items()
            )
        )

    return u, trajectories, trajectory_weights


def load_sdd_models(model_root, scene_id):
    scene_root = Path(model_root) / f"scene_{scene_id:03d}"

    state_space = np.load(scene_root / "state_space.npz")
    state_space = {key: state_space[key] for key in state_space.files}
    state_space_metadata = json.load((scene_root / "state_space.json").open("r"))
    model_metadata = json.load((scene_root / "model_metadata.json").open("r"))
    destination_classes = json.load((scene_root / "destination_classes.json").open("r"))

    destination_totals = [
        c.get("train_count", 0) for c in destination_classes["classes"]
    ]
    num_destinations = sum(destination_totals)
    classes = [
        {
            "id": c["class_id"],
            "prob": (
                c.get("train_count", 0) / num_destinations
                if num_destinations > 0
                else 0.0
            ),
        }
        for c in destination_classes["classes"]
    ]

    models = {}
    for c in classes:
        id = int(c["id"])
        count_path = scene_root / "transitions" / f"class_{id:03d}_counts.npz"
        transition_path = scene_root / "transitions" / f"class_{id:03d}_transition.npz"
        if count_path.exists():
            models[id] = {
                "counts": sparse.load_npz(count_path),
                "transitions": sparse.load_npz(transition_path),
            }

    count_path = scene_root / "transitions" / "global_counts.npz"
    transition_path = scene_root / "transitions" / "global_transition.npz"
    models["global"] = {
        "counts": sparse.load_npz(count_path),
        "transitions": sparse.load_npz(transition_path),
    }

    return {
        "state_space": state_space,
        "state_space_metadata": state_space_metadata,
        "model_metadata": model_metadata,
        "classes": classes,
        "models": models,
    }


def simulate(args, delivery_log=None):

    if args.show_sim:
        pygame.init()
        size = (args.width, args.height)
        screen = pygame.display.set_mode(size)
        surface = pygame.Surface(size, pygame.SRCALPHA)
        pygame.display.set_caption("Simulation")

        clock = pygame.time.Clock()
        pygame.font.init()
    else:
        screen = None

    # set the seed
    if args.seed is not None:
        seed(args.seed)
        print("Setting seed to: ", args.seed)
    else:
        seed(time())

    generator_args = GENERATOR_ARGS
    generator_args["seed"] = args.seed
    generator_args["max_time"] = args.max_time

    # load the models
    if args.data_source == "sdd":
        print("Loading SDD models...")
        sdd_models = load_sdd_models(
            model_root=args.sdd_model_root,
            scene_id=args.sdd_scene_id,
        )
    else:
        sdd_models = None

    sim = Simulation(
        generator_name=args.generator,
        generator_args=generator_args,
        num_actors=args.actors,
        tracks=args.tracks,
        data_source=args.data_source,
        sdd_processed_root=args.sdd_processed_root,
        sdd_scene_id=args.sdd_scene_id,
        limit_tracks=args.limit_tracks,
        pois_lambda=args.lambd,
        screen=surface if args.show_sim or args.record_data else None,
        tick_time=args.tick_time,
        screen_height=args.height,
        screen_width=args.width,
        margin=args.margin,
        #  ego_start=[0.05, 0.15],  # [0.25, 0.75]],
        ego_start=[[0.05, 0.95], [0.05, 0.15]],
        ego_heading=np.pi / 2.0,
        # ego_goal=[0.95, 0.85],  #
        ego_goal=[[0.05, 0.95], [0.7, 1]],
    )
    args.tick_time = sim.tick_time
    sim.debug_tick_timing = bool(args.debug_timing)

    if args.seed is not None:
        # Save the tasks (or reload them)
        pass

    tracker_update_interval = 1
    # tracker = Tracker(
    #     initial_timestep=0,
    #     scenario_map=None,
    #     args=args,
    #     device="cuda:0",
    #     dt=args.tick_time * tracker_update_interval,
    # )

    path = None

    # Simulation/Game Loop
    tracker_frame = 0

    action = None

    # Match the planner model to the scene-scaled ego actor. SDD scenes use a
    # per-scene scene_scale, so an unscaled wheelbase makes the model understeer.
    robot = Ackermann4(
        length=sim.ego.length,
        width=sim.ego.width,
        scale=sim.ego.size_scale,
        max_delta=CONTROL_LIMITS[1],
    )

    # initialize a controller
    Q = np.array([args.x_weight, args.y_weight, args.v_weight, args.theta_weight])
    Qf = np.array([FINAL_X_WEIGHT, FINAL_X_WEIGHT, FINAL_V_WEIGHT, FINAL_THETA_WEIGHT])
    R = np.array([args.a_weight, args.delta_weight])

    # Debug: print the weight matrix values
    print(f"DEBUG: Q matrix (state weights) = {Q}")
    print(f"DEBUG: R matrix (control weights) = {R}")
    print(f"DEBUG: Control/State ratio - A: {R[0]/Q[2]:.4f}, Delta: {R[1]/Q[3]:.4f}")
    if R[1] / Q[3] > 1.0:
        print("WARNING: Steering control weight is higher than heading state weight!")
        print("         This will make MPPI avoid steering even when needed.")
    dynamic_clearance_margin = mppi_dynamic_clearance_margin(
        robot, args.dynamic_clearance_margin
    )
    static_clearance_margin = mppi_static_clearance_margin(
        robot, args.static_clearance_margin
    )
    static_hard_clearance_margin = mppi_static_clearance_margin(
        robot, args.static_hard_clearance_margin
    )
    if args.debug_mppi:
        print(
            "[MPPI dynamic scale] "
            f"scene_scale={scene_scale(robot):.6f} "
            f"dynamic_collision_buffer={mppi_dynamic_collision_buffer(robot, args.dynamic_hard_clearance_margin):.6f} "
            f"clearance_margin={dynamic_clearance_margin:.6f} "
            f"raw_clearance_margin={args.dynamic_clearance_margin:.6f} "
            f"raw_hard_clearance_margin={args.dynamic_hard_clearance_margin:.6f} "
            f"static_hard_clearance_margin={static_hard_clearance_margin:.6f} "
            f"raw_static_hard_clearance_margin={args.static_hard_clearance_margin:.6f} "
            f"static_clearance_margin={static_clearance_margin:.6f} "
            f"raw_static_clearance_margin={args.static_clearance_margin:.6f}"
        )

    mppi = MPPI(
        vehicle_length=robot.L,
        vehicle_width=robot.W,
        samples=MPPI_SAMPLES,
        seed=args.seed,
        u_limits=CONTROL_LIMITS,
        u_dist_limits=CONTROL_VARIATION_LIMITS,
        M=args.mppi_m,
        Q=Q,
        Qf=Qf,
        R=R,
        method="Ignore",
        c_lambda=args.c_lambda,
        scan_range=SCAN_RANGE,
        debug=args.debug_mppi,
        dynamic_clearance_margin=dynamic_clearance_margin,
        dynamic_clearance_weight=args.dynamic_clearance_weight,
        dynamic_collision_cost=args.dynamic_collision_cost,
        static_clearance_margin=static_clearance_margin,
        static_clearance_weight=args.static_clearance_weight,
        static_collision_cost=args.static_collision_cost,
        static_hard_clearance_margin=static_hard_clearance_margin,
    )

    # self.controlNN = ControlPredictor("./models/tesla_car.model")

    control_variations = ControlVariations(
        vehicle=Unicycle(),
        samples=10,
        seed=args.seed,
        u_limits=[2, np.pi],
        u_dist_limits=[1, np.pi / 2],
    )

    # for now, assume an empty map -- we can add objects later
    grid_origin = sim.display_offset
    grid_resolution = sim.grid_resolution
    grid_rows = int(np.ceil(sim.grid_height / grid_resolution))
    grid_cols = int(np.ceil(sim.grid_width / grid_resolution))
    local_map = np.zeros((grid_rows, grid_cols))
    discrete_oce_tracker = None
    if args.oce_eval_method == "discrete":
        if sdd_models is None:
            print(
                "WARNING: discrete OCE requires --data-source sdd and loaded SDD models; "
                "falling back to trajectory OCE."
            )
            args.oce_eval_method = "trajectory"
        else:
            debug_dir = args.discrete_oce_debug_dir if args.debug_discrete_oce else None
            discrete_oce_tracker = DiscreteOCETracker(
                sdd_models,
                debug_dir=debug_dir,
                debug_top_k=args.discrete_oce_debug_top_k,
            )

    action = None
    previous_ego_state = np.asarray(sim.ego.x[: ActorStateEnum.DELTA], dtype=float)
    u = None
    stable_nominal_route = None
    stable_nominal_goal = None
    cached_paths = None
    cached_path_debug = None
    cached_nominal_route = None
    cached_best_trajectory = 0
    cached_control_trajectory = 0
    cached_oce_entropies = None
    next_route_replan_tick = None
    while True:
        loop_start = perf_counter()
        timing = {}
        if args.show_sim:
            section_start = perf_counter()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            timing["events"] = perf_counter() - section_start

        applied_action = action
        section_start = perf_counter()
        observation, reward, done, info = sim.tick(action=applied_action)
        timing["tick"] = perf_counter() - section_start
        current_ego_state = np.asarray(
            info["ego"]["pos"][: ActorStateEnum.DELTA],
            dtype=float,
        )
        if getattr(args, "debug_steering", False) and applied_action is not None:
            print_applied_steering_debug(
                tick=sim.ticks,
                previous_state=previous_ego_state,
                current_state=current_ego_state,
                action=applied_action,
                dt=args.tick_time,
                vehicle_length=sim.ego.length,
            )
        previous_ego_state = current_ego_state.copy()
        if done:
            break

        section_start = perf_counter()
        agent_predictions = {}
        if True:
            # if (sim.ticks - 1) % tracker_update_interval == 0:
            visible_agents = [
                actor for actor in info["actors"] if actor["visible"] == True
            ]
            if discrete_oce_tracker is not None:
                hmm_update_start = perf_counter()
                discrete_oce_tracker.update(info["actors"], sim.ticks)
                timing["hmm_update"] = perf_counter() - hmm_update_start

            u_nom = np.zeros((args.horizon, 2))
            for agent in visible_agents:
                u_nom[:, 0] = agent["pos"][2]  # constant velocity control
                # u_nom[:, 1] = 0               # zero steering angle
                agent_predictions[agent["id"]] = control_variations.predict(
                    x_init=agent["pos"][:4],
                    u_nom=u_nom,
                    dt=args.tick_time,
                )

            # # Update the tracker
            # agent_predictions = tracker.step(
            #     agents=visible_agents, horizon=args.horizon // tracker_update_interval, timesteps=1
            # )
        # else:
        #     agent_predictions = tracker.get_predictions()

        # if path is None:
        #     start = info["ego"]["pos"]
        #     end = info["goal"]
        #     path = generate_path(start, end, args)

        timing["predict"] = perf_counter() - section_start
        if args.debug_timing:
            print("Time to predict: ", timing["predict"])

        start = info["ego"]["pos"]
        end = info["goal"]
        section_start = perf_counter()

        replan_interval = max(1, int(args.replan_interval))
        should_replan_routes = (
            cached_paths is None
            or next_route_replan_tick is None
            or sim.ticks >= next_route_replan_tick
        )
        if should_replan_routes:
            section_start = perf_counter()
            paths = generate_trajectories(
                start,
                end,
                args,
                static_polygons=sim.static_polygons,
                display_offset=sim.display_offset,
                display_diff=sim.display_diff,
                vehicle_length=robot.L,
                vehicle_width=robot.W,
                vehicle_scale=robot.scale,
                max_steer=robot.max_delta,
                resolution=grid_resolution,
            )
            cached_paths = paths
            cached_path_debug = (
                getattr(args, "_last_specialk_debug", None)
                if str(args.trajectory_generator).lower() == "specialk"
                else None
            )
            timing["trajectories"] = perf_counter() - section_start

            next_route_replan_tick = sim.ticks + replan_interval
            if args.debug_paths or args.debug_steering:
                debug_routes = [path["route"] for path in paths]
                route_count = len(debug_routes)
                route_lengths = [
                    round(waypoint_route_length(route), 3) for route in debug_routes
                ]
                route_points = [len(route) for route in debug_routes]
                print(
                    "[paths] "
                    f"tick={sim.ticks} replanned paths={len(paths)} "
                    f"routes={route_count} "
                    f"points={route_points} "
                    f"lengths={route_lengths} "
                    f"next_replan_tick={next_route_replan_tick} "
                    f"generator={paths[0].get('generator')}"
                )
                if cached_path_debug:
                    print(
                        "[paths-roadmap] "
                        f"nodes={len(cached_path_debug.get('nodes', []))} "
                        f"edges={len(cached_path_debug.get('edges', []))} "
                        f"skeletons={len(cached_path_debug.get('skeletons', []))} "
                        f"local_anchors={len(cached_path_debug.get('local_anchors', []))} "
                        f"global_anchors={len(cached_path_debug.get('global_anchors', []))} "
                        f"connectivity={cached_path_debug.get('connectivity', {})} "
                        f"reasons={cached_path_debug.get('reasons', [])}"
                    )
        else:
            nominal_route = cached_nominal_route
            paths = cached_paths
            timing["trajectories"] = 0.0
            timing["filter_paths"] = 0.0

        best_trajectory = cached_best_trajectory
        control_trajectory = cached_control_trajectory
        oce_entropies = cached_oce_entropies
        timing["oce_eval"] = 0.0
        if should_replan_routes and args.use_oce_trajectory_eval:
            section_start = perf_counter()
            if args.oce_eval_method == "discrete":
                best_trajectory, oce_entropies, _oce_results = (
                    evaluate_candidate_paths_by_discrete_oce(
                        time_step=sim.ticks,
                        paths=[path["path"] for path in paths],
                        tracker=discrete_oce_tracker,
                        static_polygons=sim.static_polygons,
                        horizon=args.discrete_oce_horizon or args.horizon,
                        method=args.discrete_oce_method,
                        scan_range=SCAN_RANGE,
                        backend=args.discrete_oce_backend,
                        return_debug_tensors=args.debug_discrete_oce,
                        debug=args.debug_oce_eval,
                    )
                )
            else:
                best_trajectory, oce_entropies, _oce_results = (
                    evaluate_candidate_paths_by_oce(
                        time_step=sim.ticks,
                        grid=local_map,
                        origin=grid_origin,
                        resolution=grid_resolution,
                        paths=[path["path"] for path in paths],
                        agents=visible_agents,
                        predictions=agent_predictions,
                        prediction_interval=args.tick_time,
                        dt=args.tick_time,
                        debug=args.debug_oce_eval,
                    )
                )
            timing["oce_eval"] = perf_counter() - section_start
            if args.debug_oce_eval:
                print(
                    "[oce-eval] "
                    f"method={args.oce_eval_method} "
                    f"tick={sim.ticks} best={best_trajectory} "
                    f"entropies={np.asarray(oce_entropies).tolist() if oce_entropies is not None else None}"
                )
                if isinstance(_oce_results, dict) and "timing" in _oce_results:
                    print(f"[oce-eval-detail] {_oce_results['timing']}")

            # BUGBUG - temporarily disable OCE-based path selection to evaluate MPPI performance without OCE overhead
            best_trajectory = 0

            control_trajectory = choose_control_path_index(
                paths,
                best_trajectory,
                start_heading=start[ActorStateEnum.THETA],
                max_heading_error=np.deg2rad(args.max_control_path_heading_error_deg),
                debug=args.debug_steering or args.debug_oce_eval,
            )
            if args.debug_oce_eval and control_trajectory != best_trajectory:
                print(
                    "[oce-eval] "
                    f"best_trajectory={best_trajectory} control_trajectory={control_trajectory}"
                )
            cached_best_trajectory = best_trajectory
            cached_control_trajectory = control_trajectory
            cached_oce_entropies = oce_entropies
        elif should_replan_routes:
            best_trajectory = 0
            control_trajectory = choose_control_path_index(
                paths,
                best_trajectory,
                start_heading=start[ActorStateEnum.THETA],
                max_heading_error=np.deg2rad(args.max_control_path_heading_error_deg),
                debug=args.debug_steering or args.debug_oce_eval,
            )
            cached_best_trajectory = best_trajectory
            cached_control_trajectory = control_trajectory
            cached_oce_entropies = None
        else:
            best_trajectory = int(np.clip(best_trajectory, 0, len(paths) - 1))
            control_trajectory = int(np.clip(control_trajectory, 0, len(paths) - 1))
            if args.debug_oce_eval:
                print(
                    "[oce-eval] "
                    f"tick={sim.ticks} reused best_trajectory={best_trajectory} "
                    f"control_trajectory={control_trajectory} "
                    f"next_replan_tick={next_route_replan_tick}"
                )

        section_start = perf_counter()
        u, trajectories, trajectory_weights = get_control(
            mppi=mppi,
            costmap=local_map,
            origin=sim.display_offset,
            resolution=grid_resolution,
            robot_model=robot,
            u_nom=u,
            initial_state=info["ego"]["pos"][: ActorStateEnum.DELTA],
            goal=[*info["ego"]["goal"], 0, 0],
            path=paths[control_trajectory]["path"],
            agents=visible_agents,
            agent_predictions=agent_predictions,
            static_polygons=sim.static_polygons,
            args=args,
            debug_tick=sim.ticks,
        )
        timing["control"] = perf_counter() - section_start

        action = u[0]
        if np.isnan(action[0]):
            action = [0.0]

        timing["render"] = 0.0
        if args.show_sim:
            section_start = perf_counter()
            sim.render(
                actors=agent_predictions,
                trajectories=trajectories,
                trajectory_weights=trajectory_weights,
                path=[path["path"] for path in paths],
                debug_routes=[path["route"] for path in paths],
                debug_roadmap=cached_path_debug,
                selected_path_index=control_trajectory,
                prefix_str=args.prefix,
            )
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            # pygame.display.update()
            clock.tick(1 / args.tick_time * args.simulation_speed)
            timing["render"] = perf_counter() - section_start

        timing["total"] = perf_counter() - loop_start
        if args.debug_timing and sim.ticks % args.timing_interval == 0:
            print_loop_timing_debug(sim.ticks, timing)
            print_sim_tick_timing_debug(
                sim.ticks,
                getattr(sim, "last_tick_timing", {}),
            )
            print_sim_scan_timing_debug(
                sim.ticks,
                getattr(sim, "last_tick_timing", {}),
            )
            print_sim_observation_timing_debug(
                sim.ticks,
                getattr(sim, "last_tick_timing", {}),
            )
        if args.timing_output:
            append_timing_csv(
                args.timing_output,
                sim.ticks,
                timing,
                getattr(args, "_last_control_timing", {}),
                actors=len(info["actors"]),
                visible_actors=len(visible_agents),
            )

    return sim


def run_with_optional_profile(args):
    if not args.profile:
        simulate(args)
        return

    profile_output = Path(args.profile_output)
    profile_output.parent.mkdir(parents=True, exist_ok=True)
    profile_report_output = (
        Path(args.profile_report_output)
        if args.profile_report_output
        else profile_output.with_suffix(profile_output.suffix + ".txt")
    )
    profile_report_output.parent.mkdir(parents=True, exist_ok=True)

    profiler = cProfile.Profile()
    try:
        profiler.runcall(simulate, args)
    finally:
        profiler.dump_stats(str(profile_output))
        write_profile_report(
            profile_output,
            profile_report_output,
            sort_by=args.profile_sort,
            limit=args.profile_limit,
            project_filter=args.profile_project_filter,
            ignore_filter=args.profile_ignore_filter,
        )
        print(f"Wrote cProfile stats to {profile_output}")
        print(f"Wrote cProfile text report to {profile_report_output}")
        print(f"Open with: snakeviz {profile_output}")


def write_profile_report(
    profile_output,
    report_output,
    *,
    sort_by,
    limit,
    project_filter,
    ignore_filter,
):
    limit = max(1, int(limit))
    stream = io.StringIO()
    stream.write(f"Profile stats: {profile_output}\n")
    stream.write(f"Sort: {sort_by}, limit: {limit}\n\n")

    stream.write("== Full Profile ==\n")
    pstats.Stats(str(profile_output), stream=stream).strip_dirs().sort_stats(
        sort_by
    ).print_stats(limit)

    if project_filter:
        stream.write("\n\n== Project Functions ==\n")
        stream.write(f"Filter: {project_filter}\n")
        pstats.Stats(str(profile_output), stream=stream).sort_stats(
            sort_by
        ).print_stats(project_filter, limit)

    if ignore_filter:
        stream.write("\n\n== Full Profile, Infrastructure Rows Removed ==\n")
        stream.write(f"Ignoring: {ignore_filter}\n")
        filtered_stats = pstats.Stats(str(profile_output), stream=stream)
        filtered_stats.stats = {
            func: stat
            for func, stat in filtered_stats.stats.items()
            if not re.search(ignore_filter, f"{func[0]}:{func[1]}:{func[2]}")
        }
        filtered_stats.strip_dirs().sort_stats(sort_by).print_stats(limit)

    report_output.write_text(stream.getvalue())


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--height", default=SCREEN_HEIGHT, type=int, help="Screen vertical size"
    )
    argparser.add_argument(
        "--width", default=SCREEN_WIDTH, type=int, help="Screen horizontal size"
    )
    argparser.add_argument(
        "--margin", default=SCREEN_MARGIN, type=int, help="Screen horizontal size"
    )
    argparser.add_argument("-s", "--seed", default=None, type=int, help="Random Seed")
    argparser.add_argument(
        "-l",
        "--lambd",
        default=LAMBDA_TASKS,
        type=float,
        help="Exponential Spawn rate for Tasks",
    )
    argparser.add_argument(
        "-a",
        "--actors",
        default=NUM_ACTORS,
        type=int,
        help="Number of actors in the simulation",
    )
    argparser.add_argument(
        "-p", "--policy", default=DEFAULT_POLICY_NAME, help="Policy to use"
    )
    argparser.add_argument(
        "--simulation-speed",
        default=SIMULATION_SPEED,
        type=float,
        help="Simulator speed",
    )
    argparser.add_argument(
        "-t",
        "--tick-time",
        default=TICK_TIME,
        type=float,
        help="Length of Simulation Time Step",
    )
    argparser.add_argument(
        "-g",
        "--generator",
        default=DEFAULT_GENERATOR_NAME,
        help="Random Generator to use",
    )
    argparser.add_argument(
        "--tracks", default=None, type=str, help="Load pedestrian tracks from file"
    )
    argparser.add_argument(
        "--limit-tracks",
        default=None,
        type=int,
        help=(
            "Maximum number of loaded pedestrian tracks. Randomly selects tracks "
            "when the scenario has more tracks. Use 0 for no pedestrians."
        ),
    )
    argparser.add_argument(
        "--data-source",
        choices=["eth", "sdd", "random"],
        default=None,
        help="Scenario provider. Defaults to eth when --tracks is set, otherwise random.",
    )
    argparser.add_argument(
        "--sdd-processed-root",
        default="outputs/sdd_processed",
        help="Processed root for --data-source sdd.",
    )
    argparser.add_argument(
        "--sdd-model-root",
        default="outputs/models",
        help="Computed models/transitions for --data-source sdd.",
    )
    argparser.add_argument(
        "--sdd-scene-id",
        type=int,
        default=None,
        help="Processed scene id for --data-source sdd.",
    )
    argparser.add_argument(
        "--max-time", default=None, type=float, help="Maximum Length of Simulation"
    )
    argparser.add_argument(
        "--record-data", action="store_true", help="Record data to disk as frames"
    )
    argparser.add_argument(
        "--debug-steering",
        action="store_true",
        help=(
            "Print nominal/final steering diagnostics and measured ego heading "
            "response each tick."
        ),
    )
    argparser.add_argument(
        "--debug-timing",
        action="store_true",
        help="Print per-loop timing breakdowns for the main simulation loop.",
    )
    argparser.add_argument(
        "--timing-interval",
        default=1,
        type=int,
        help="Print --debug-timing output every N simulation ticks.",
    )
    argparser.add_argument(
        "--timing-output",
        default=None,
        help=(
            "Write per-tick timing data to a CSV file. This is more useful than "
            "cProfile for the monolithic simulation loop."
        ),
    )
    argparser.add_argument(
        "--debug-mppi",
        action="store_true",
        help="Enable verbose MPPI internal diagnostics.",
    )
    argparser.add_argument(
        "--debug-paths",
        action="store_true",
        help=(
            "Print path replanning details and draw generated route waypoints. "
            "For specialk, also draw the first-phase roadmap in the simulator."
        ),
    )
    argparser.add_argument(
        "--disable-oce-trajectory-eval",
        dest="use_oce_trajectory_eval",
        action="store_false",
        default=True,
        help="Disable PyCUDA OCE entropy evaluation for generated path selection.",
    )
    argparser.add_argument(
        "--debug-oce-eval",
        action="store_true",
        help="Print OCE trajectory entropy scores when OCE path evaluation is enabled.",
    )
    argparser.add_argument(
        "--oce-eval-method",
        choices=["trajectory", "discrete"],
        default="discrete",
        help="OCE path comparison backend.",
    )
    argparser.add_argument(
        "--discrete-oce-method",
        default="discrete_exact_entropy",
        help="Entropy method used by the discrete OCE backend.",
    )
    argparser.add_argument(
        "--discrete-oce-backend",
        choices=["auto", "gpu", "cpu"],
        default="auto",
        help="Execution backend for discrete OCE. auto uses GPU when available.",
    )
    argparser.add_argument(
        "--discrete-oce-horizon",
        type=int,
        default=None,
        help="Discrete OCE evaluation horizon. Defaults to --horizon.",
    )
    argparser.add_argument(
        "--debug-discrete-oce",
        action="store_true",
        help="Write per-tick discrete OCE state distribution debug files.",
    )
    argparser.add_argument(
        "--discrete-oce-debug-dir",
        default="results/discrete_oce_debug",
        help="Directory for discrete OCE JSON and heatmap debug files.",
    )
    argparser.add_argument(
        "--discrete-oce-debug-top-k",
        type=int,
        default=8,
        help="Number of highest-probability states to include in discrete OCE JSON.",
    )
    argparser.add_argument(
        "--host-static-sample-filter",
        action="store_true",
        help=(
            "Run the slow Python/Shapely static-obstacle filter over every MPPI "
            "sample. By default static polygons are handled by MPPI and only the "
            "selected final trajectory is checked on the host."
        ),
    )
    argparser.add_argument(
        "--host-dynamic-sample-filter",
        dest="host_dynamic_sample_filter",
        action="store_true",
        default=False,
        help=(
            "Run the slow Python dynamic-agent collision filter over every MPPI "
            "sample before recomputing the selected control. By default dynamic "
            "agents are handled by MPPI and only the selected final trajectory is "
            "checked on the host."
        ),
    )
    argparser.add_argument(
        "--profile",
        action="store_true",
        help="Run the simulation under cProfile and write stats for SnakeViz.",
    )
    argparser.add_argument(
        "--profile-output",
        default="results/simulation.prof",
        help="Path for cProfile stats output. SnakeViz can open this file directly.",
    )
    argparser.add_argument(
        "--profile-report-output",
        default=None,
        help=(
            "Optional text report path for cProfile stats. Defaults to "
            "<profile-output>.txt."
        ),
    )
    argparser.add_argument(
        "--profile-sort",
        default="cumulative",
        choices=["cumulative", "tottime", "time", "calls", "ncalls"],
        help="Sort key for the text cProfile report.",
    )
    argparser.add_argument(
        "--profile-limit",
        default=80,
        type=int,
        help="Number of rows to include in each text cProfile report section.",
    )
    argparser.add_argument(
        "--profile-project-filter",
        default=(
            "src/python/pedestrian|pedestrian/main.py|pedestrian/simulation.py|"
            "pedestrian/specialk.py|warp_mppi|discrete_oce"
        ),
        help=(
            "Regex filter for project functions in the text cProfile report. "
            "Use an empty string to disable the filtered section."
        ),
    )
    argparser.add_argument(
        "--profile-ignore-filter",
        default=(
            "debugpy|pydevd|queue.py|threading.py|socket.py|selectors.py|"
            "subprocess.py|prefork.py"
        ),
        help=(
            "Regex for debugger/wait infrastructure to remove from the extra "
            "filtered text profile section. Use an empty string to disable it."
        ),
    )
    argparser.add_argument(
        "--show-sim", action="store_true", help="Display the simulation window"
    )
    argparser.add_argument(
        "--config", type=str, help="Configuration file for the model"
    )
    argparser.add_argument("--model-iteration", type=int, help="Model version")
    argparser.add_argument("--model-dir", type=str, help="Location of the model files")
    argparser.add_argument(
        "--attention-radius", type=float, default=3.0, help="Model version"
    )
    argparser.add_argument("--device", type=str, default=None, help="Model version")
    argparser.add_argument("--samples", type=int, default=5, help="Model version")
    argparser.add_argument(
        "--history-len",
        type=int,
        default=10,
        help="Maximum number of samples to keep track of for agent routes",
    )
    argparser.add_argument(
        "--horizon", type=int, default=10, help="Model prediction horizon"
    )
    argparser.add_argument(
        "--incremental",
        help="Use Trajectron in online incremental mode",
        action="store_true",
    )
    argparser.add_argument(
        "--results-dir", type=str, help="Location of generated output"
    )
    argparser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Output prefix to identify saved results",
    )
    argparser.add_argument(
        "--robot-speed",
        default=ROBOT_SPEED,
        type=float,
        help="Speed of the robot (m/s)",
    )
    argparser.add_argument(
        "--steering-heading-gain",
        default=2.0,
        type=float,
        help="Nominal steering gain applied to trajectory heading error.",
    )
    argparser.add_argument(
        "--steering-cross-track-gain",
        default=0.75,
        type=float,
        help="Nominal steering gain applied to cross-track error.",
    )
    argparser.add_argument(
        "--steering-opposing-cross-track-scale",
        default=0.25,
        type=float,
        help=(
            "Scale cross-track steering when it opposes trajectory heading "
            "correction."
        ),
    )
    argparser.add_argument(
        "--route-heading-lookahead",
        default=None,
        type=float,
        help=(
            "Forward waypoint distance used when the route's first segment conflicts "
            "with the robot heading. Defaults to a vehicle-size-based distance."
        ),
    )
    argparser.add_argument(
        "--heading-mismatch-reference-speed",
        default=0.05,
        type=float,
        help=(
            "Initial reference speed used for cubic route generation when the "
            "route begins far from the robot's current heading."
        ),
    )
    argparser.add_argument(
        "--max-initial-route-heading-error-deg",
        default=35.0,
        type=float,
        help=(
            "Maximum allowed heading error between robot orientation and the first "
            "route segment before inserting a forward heading waypoint."
        ),
    )
    argparser.add_argument(
        "--nominal-rejoin-lookahead",
        default=1.5,
        type=float,
        help=(
            "Distance ahead on the stable nominal route that each replanned local "
            "route should rejoin."
        ),
    )
    argparser.add_argument(
        "--nominal-rejoin-search-distance",
        default=5.0,
        type=float,
        help=(
            "Maximum distance ahead on the stable nominal route to search for a "
            "heading-feasible local rejoin point."
        ),
    )
    argparser.add_argument(
        "--recovery-route-lookahead",
        default=1.0,
        type=float,
        help=(
            "Short forward planning distance used when no heading-feasible nominal "
            "rejoin route is available."
        ),
    )
    argparser.add_argument(
        "--trajectory-count",
        default=3,
        type=int,
        help=(
            "Number of candidate paths to generate. In frenet mode these are "
            "lateral offsets around the reference; in kpaths mode these are "
            "diverse near-shortest routes."
        ),
    )
    argparser.add_argument(
        "--replan-interval",
        default=1,
        type=int,
        help=(
            "Number of ticks to use generated route candidates before "
            "replanning. A value of 1 replans k-paths every step."
        ),
    )
    argparser.add_argument(
        "--trajectory-generator",
        choices=["kpaths", "specialk"],
        default="kpaths",
        help=(
            "Candidate trajectory generator. kpaths searches diverse "
            "kinematically feasible grid routes; specialk uses homotopy-aware "
            "roadmap skeletons refined with Ackermann state-lattice primitives."
        ),
    )
    argparser.add_argument(
        "--kpaths-heading-bins",
        default=16,
        type=int,
        help="Number of sampled goal orientations used for collision checking.",
    )
    argparser.add_argument(
        "--kpaths-search-resolution",
        default=None,
        type=float,
        help=(
            "Coarse A* grid cell size in meters. Defaults to at least the "
            "Ackermann minimum turn radius to keep path search fast and smooth."
        ),
    )
    argparser.add_argument(
        "--kpaths-near-shortest-factor",
        default=1.8,
        type=float,
        help="Maximum route length factor relative to the shortest kinematic route.",
    )
    argparser.add_argument(
        "--kpaths-max-overlap",
        default=0.65,
        type=float,
        help="Maximum allowed cell-overlap ratio with an accepted k-path.",
    )
    argparser.add_argument(
        "--kpaths-max-attempts",
        default=30,
        type=int,
        help="Maximum penalized A* search attempts used to find diverse k-paths.",
    )
    argparser.add_argument(
        "--kpaths-diversity-penalty",
        default=1.0,
        type=float,
        help="Penalty added to cells and edges used by accepted/rejected k-paths.",
    )
    argparser.add_argument(
        "--kpaths-turn-penalty",
        default=0.1,
        type=float,
        help="Additional A* path cost per radian of route heading change.",
    )
    argparser.add_argument(
        "--kpaths-motion-step",
        default=None,
        type=float,
        help="Legacy hybrid-A* option; ignored by the current k-path generator.",
    )
    argparser.add_argument(
        "--kpaths-goal-tolerance",
        default=None,
        type=float,
        help="Legacy hybrid-A* option; ignored by the current k-path generator.",
    )
    argparser.add_argument(
        "--kpaths-connect-distance",
        default=None,
        type=float,
        help="Legacy hybrid-A* option; ignored by the current k-path generator.",
    )
    argparser.add_argument(
        "--kpaths-curvature-tolerance",
        default=0.25,
        type=float,
        help=(
            "Allowed fractional curvature tolerance when validating converted "
            "k-path trajectories against the Ackermann steering limit."
        ),
    )
    argparser.add_argument(
        "--specialk-roadmap-samples-density",
        default=1.0,
        type=float,
        help=(
            "Target specialk free-space roadmap sample density in samples per "
            "square meter. Converted to scene units using scene_scale."
        ),
    )
    argparser.add_argument(
        "--specialk-roadmap-grid-step",
        default=None,
        type=float,
        help=(
            "Deterministic grid sample spacing for specialk roadmap coverage. "
            "Defaults from --specialk-roadmap-samples-density."
        ),
    )
    argparser.add_argument(
        "--specialk-obstacle-edge-step",
        default=None,
        type=float,
        help=(
            "Spacing for obstacle-edge offset samples used to cover narrow "
            "corridors. Defaults from --specialk-roadmap-samples-density."
        ),
    )
    argparser.add_argument(
        "--specialk-nearest",
        default=8,
        type=int,
        help="Nearest roadmap neighbors considered for specialk visibility edges.",
    )
    argparser.add_argument(
        "--specialk-connect-radius",
        default=None,
        type=float,
        help="Maximum specialk roadmap edge length. Defaults from turn radius.",
    )
    argparser.add_argument(
        "--specialk-raw-routes",
        default=0,
        type=int,
        help="Raw penalized roadmap route attempts. Zero derives from trajectory count.",
    )
    argparser.add_argument(
        "--specialk-route-candidates",
        default=0,
        type=int,
        help="Diverse route skeletons to lattice-refine. Zero derives from trajectory count.",
    )
    argparser.add_argument(
        "--specialk-max-overlap",
        default=0.6,
        type=float,
        help="Maximum overlap before same-signature specialk routes are redundant.",
    )
    argparser.add_argument(
        "--specialk-separation",
        default=None,
        type=float,
        help="Minimum mean separation for geometrically diverse same-signature routes.",
    )
    argparser.add_argument(
        "--specialk-clearance-weight",
        default=0.25,
        type=float,
        help="Roadmap edge clearance penalty weight for specialk.",
    )
    argparser.add_argument(
        "--specialk-obstacle-clearance",
        default=None,
        type=float,
        help=(
            "Extra obstacle inflation used by specialk in scene units. Defaults "
            "to STATIC_PLANNER_OBSTACLE_CLEARANCE scaled by the scene scale."
        ),
    )
    argparser.add_argument(
        "--specialk-diversity-penalty",
        default=1.2,
        type=float,
        help="Penalty applied to reused specialk roadmap edges and lattice cells.",
    )
    argparser.add_argument(
        "--specialk-corridor-radius",
        default=None,
        type=float,
        help="Route corridor radius for specialk lattice refinement.",
    )
    argparser.add_argument(
        "--specialk-lattice-resolution",
        default=None,
        type=float,
        help="SE(2) lattice xy resolution for specialk. Defaults from turn radius.",
    )
    argparser.add_argument(
        "--specialk-heading-bins",
        default=16,
        type=int,
        help="Number of heading bins in the specialk state lattice.",
    )
    argparser.add_argument(
        "--specialk-motion-step",
        default=None,
        type=float,
        help="Distance covered by each specialk Ackermann primitive.",
    )
    argparser.add_argument(
        "--specialk-goal-tolerance",
        default=None,
        type=float,
        help="Goal capture radius for specialk lattice search.",
    )
    argparser.add_argument(
        "--specialk-max-expansions",
        default=1200,
        type=int,
        help="Maximum lattice node expansions per specialk route.",
    )
    argparser.add_argument(
        "--specialk-time-budget-ms",
        default=0.0,
        type=float,
        help="Optional per-route specialk lattice time budget in milliseconds; zero disables.",
    )
    argparser.add_argument(
        "--max-control-path-heading-error-deg",
        default=90.0,
        type=float,
        help=(
            "When --follow-oce-selected-path is enabled, fall back to the nominal "
            "path if the selected candidate starts farther than this from the "
            "robot heading."
        ),
    )
    argparser.add_argument(
        "--max-path-nominal-deviation",
        default=1.25,
        type=float,
        help=(
            "Reject selected variation paths whose maximum distance from the stable "
            "nominal route exceeds this value. Use 0 to disable."
        ),
    )
    argparser.add_argument(
        "--k-eval", type=float, default=25.0, help="Number of samples to evaluate"
    )

    # MPPI params
    argparser.add_argument(
        "--c_lambda",
        type=float,
        default=DEFAULT_LAMBDA,
        help="Lambda value for weight normalization control",
    )
    argparser.add_argument(
        "--mppi_m",
        type=float,
        default=DEFAULT_METHOD_WEIGHT,
        help="M/Lambda value for method weights",
    )
    argparser.add_argument(
        "--dynamic-clearance-margin",
        type=float,
        default=MIN_SEPARATION,
        help="Extra soft dynamic-agent clearance band outside the hard collision radius.",
    )
    argparser.add_argument(
        "--dynamic-hard-clearance-margin",
        type=float,
        default=0.0,
        help=(
            "Extra hard dynamic-agent collision inflation. Keep at 0 so MPPI "
            "uses soft cost for social clearance without zeroing every close rollout."
        ),
    )
    argparser.add_argument(
        "--dynamic-clearance-weight",
        type=float,
        default=DEFAULT_METHOD_WEIGHT,
        help="Quadratic MPPI cost weight for dynamic-agent clearance violations.",
    )
    argparser.add_argument(
        "--dynamic-collision-cost",
        type=float,
        default=1.0e7,
        help="Hard MPPI cost added when a rollout intersects an inflated dynamic agent.",
    )
    argparser.add_argument(
        "--static-clearance-margin",
        type=float,
        default=STATIC_OBSTACLE_CLEARANCE,
        help="Extra soft static-obstacle clearance band outside the ego footprint.",
    )
    argparser.add_argument(
        "--static-hard-clearance-margin",
        type=float,
        default=STATIC_OBSTACLE_HARD_CLEARANCE,
        help=(
            "Hard static-obstacle clearance margin used by MPPI pruning and the "
            "host final safety check. Keep smaller than --static-clearance-margin."
        ),
    )
    argparser.add_argument(
        "--static-clearance-weight",
        type=float,
        default=DEFAULT_METHOD_WEIGHT,
        help="Quadratic MPPI cost weight for static-obstacle clearance violations.",
    )
    argparser.add_argument(
        "--static-collision-cost",
        type=float,
        default=1.0e7,
        help="Hard MPPI cost added when a rollout intersects a static obstacle.",
    )
    argparser.add_argument(
        "--x_weight", type=float, default=X_WEIGHT, help="Weight for x coordinate"
    )
    argparser.add_argument(
        "--y_weight", type=float, default=Y_WEIGHT, help="Weight for y coordinate"
    )
    argparser.add_argument(
        "--v_weight", type=float, default=V_WEIGHT, help="Weight for velocity"
    )
    argparser.add_argument(
        "--theta_weight", type=float, default=THETA_WEIGHT, help="Weight for theta"
    )
    argparser.add_argument(
        "--a_weight", type=float, default=A_WEIGHT, help="Weight for acceleration"
    )
    argparser.add_argument(
        "--delta_weight", type=float, default=DELTA_WEIGHT, help="Weight for delta"
    )

    # Model Parameters
    argparser.add_argument(
        "--offline_scene_graph",
        help="whether to precompute the scene graphs offline, options are 'no' and 'yes'",
        type=str,
        default="yes",
    )

    argparser.add_argument(
        "--dynamic_edges",
        help="whether to use dynamic edges or not, options are 'no' and 'yes'",
        type=str,
        default="yes",
    )

    argparser.add_argument(
        "--edge_state_combine_method",
        help="the method to use for combining edges of the same type",
        type=str,
        default="sum",
    )

    argparser.add_argument(
        "--edge_influence_combine_method",
        help="the method to use for combining edge influences",
        type=str,
        default="attention",
    )

    argparser.add_argument(
        "--edge_addition_filter",
        nargs="+",
        help="what scaling to use for edges as they're created",
        type=float,
        default=[0.25, 0.5, 0.75, 1.0],
    )  # We don't automatically pad left with 0.0, if you want a sharp
    # and short edge addition, then you need to have a 0.0 at the
    # beginning, e.g. [0.0, 1.0].

    argparser.add_argument(
        "--edge_removal_filter",
        nargs="+",
        help="what scaling to use for edges as they're removed",
        type=float,
        default=[1.0, 0.0],
    )  # We don't automatically pad right with 0.0, if you want a sharp drop off like
    # the default, then you need to have a 0.0 at the end.

    argparser.add_argument(
        "--override_attention_radius",
        action="append",
        help='Specify one attention radius to override. E.g. "PEDESTRIAN VEHICLE 10.0"',
        default=[],
    )

    argparser.add_argument(
        "--incl_robot_node",
        help="whether to include a robot node in the graph or simply model all agents",
        action="store_true",
    )

    argparser.add_argument(
        "--map_encoding", help="Whether to use map encoding or not", action="store_true"
    )

    argparser.add_argument(
        "--augment",
        help="Whether to augment the scene during training",
        action="store_true",
    )

    argparser.add_argument(
        "--node_freq_mult_train",
        help="Whether to use frequency multiplying of nodes during training",
        action="store_true",
    )

    argparser.add_argument(
        "--node_freq_mult_eval",
        help="Whether to use frequency multiplying of nodes during evaluation",
        action="store_true",
    )

    argparser.add_argument(
        "--scene_freq_mult_train",
        help="Whether to use frequency multiplying of nodes during training",
        action="store_true",
    )

    argparser.add_argument(
        "--scene_freq_mult_eval",
        help="Whether to use frequency multiplying of nodes during evaluation",
        action="store_true",
    )

    argparser.add_argument(
        "--scene_freq_mult_viz",
        help="Whether to use frequency multiplying of nodes during evaluation",
        action="store_true",
    )

    argparser.add_argument(
        "--no_edge_encoding",
        help="Whether to use neighbors edge encoding",
        action="store_true",
    )

    args = argparser.parse_args()

    run_with_optional_profile(args)
