import argparse
from importlib.metadata import distribution
from random import seed
from simulation import Simulation
from config import *
import pygame

from importlib import import_module
from os import path, mkdir
from time import time, sleep
from pickle import load, dump
from math import floor, sqrt
import numpy as np

from controller.ModelParameters.Ackermann import Ackermann4
from controller.mppi_gpu import MPPI
from controller.validate import visualize_variations, rollout_trajectories, validate_controls

LAMBDA = 0.1

from tracker.tracker import Tracker
from trajectory_planner.trajectory_planner import TrajectoryPlanner
from trajectory_planner.frenet_optimal_trajectory import (
    generate_target_course,
    frenet_optimal_planning,
    PlannerArgs,
    Frenet_path,
)
from trajectory_planner.trajectory_eval import evaluate

from Actor import STATE as ActorStateEnum


def generate_trajectories(start, end, args):
    from trajectory_planner.trajectory import Cubic

    initial_v = start[2]
    start_xy = list(start[:2])
    end_xy = list(end[:2])

    path = Cubic(waypoints=[start_xy, end_xy], dt=args.tick_time * 5.0)
    initial_heading = start[3]
    final_heading = 0
    path.np_trajectory(
        [start[2], args.robot_speed],
        initial_heading,
        final_heading,
        cubic_fn=Cubic.np_polynomial_time_scaling_3rd_order,
    )

    path = path.get_np_trajectory()

    from trajectory_planner.trajectory_planner import TrajectoryPlanner

    planner = TrajectoryPlanner(path, dt=args.tick_time)

    trajectories = planner.generate_trajectories(
        pos=start_xy,
        initial_v=initial_v,
        target_v=args.robot_speed,
        trajectories_requested=3,
        planning_horizon=args.horizon,
    )
    return trajectories


def get_control(mppi, costmap, origin, robot_model, u_nom, initial_state, goal, path, agents, args, u_prev=None):
    """
    Given the current state (x,y,v,theta) and the path, return the control
    """

    tic = time()
    if u_nom is None:
        u_nom = np.zeros((args.horizon, 2))

        # enumerate the path to find the first point that is in front of the vehicle
        v_est = initial_state[2]
        a_est = ROBOT_ACCELERATION
        for i in range(0, min(len(path.x) - 1, args.horizon)):
            dtheta = (path.yaw[i + 1] - path.yaw[i]) / args.tick_time

            if v_est < args.robot_speed:
                u_nom[i, 0] = a_est
            else:
                a_est = 0
                u_nom[i, 0] = 0

            u_nom[i, 1] = np.nan_to_num(np.arctan((dtheta) * robot_model.L / v_est))

            v_est = v_est + a_est * args.tick_time
    else:
        u_nom[:-1, :] = u_nom[1:, :]
        u_nom[-1, :] = 0

    np.clip(u_nom[:, 0], a_min=-CONTROL_LIMITS[0], a_max=CONTROL_LIMITS[0])
    np.clip(u_nom[:, 1], a_min=-CONTROL_LIMITS[1], a_max=CONTROL_LIMITS[1])

    x_nom = np.zeros((args.horizon + 1, 4))
    pts_to_update = min(args.horizon + 1, len(path.x))
    x_nom[:pts_to_update, 0] = path.x[:pts_to_update]
    x_nom[:pts_to_update, 1] = path.y[:pts_to_update]
    x_nom[:pts_to_update, 2] = path.s_d[:pts_to_update]
    x_nom[:pts_to_update, 3] = path.yaw[:pts_to_update]

    if pts_to_update < x_nom.shape[0]:
        x_nom[pts_to_update:, :2] = x_nom[pts_to_update - 1, :2][np.newaxis, :]

    actors = []
    distances = []
    for agent in agents:
        # for each agent, collect the center, bounding box, radius, closest point, and distance from the ego vehicle
        # BUGBUG -- this is a bit of a hack -- we're assuming the bounding box is a circle based on the width of the vehicle
        #           and not the worst case length.  We need to represent agents as a series circles, but for now this should
        #           be good enough.

        actors.append(
            [
                agent["pos"][0],
                agent["pos"][1],
                agent["extent"],
                0,
                0,
                0,
            ]
        )

    u, u_var, u_weights = mppi.find_control(
        costmap=costmap,
        origin=origin,
        resolution=GRID_RESOLUTION,
        x_init=initial_state,
        x_goal=goal,
        x_nom=x_nom,
        u_nom=u_nom,
        actors=actors,
        dt=args.tick_time,
    )
    mppi_time = time() - tic
    total_weight = np.sum(u_weights)
    if total_weight:
        u_weights = u_weights / total_weight
    else:
        # no valid control! Overwrite the control with emergency stop
        print("EMERGENCY STOP")
        u[0] = [-initial_state[ActorStateEnum.VELOCITY] / args.tick_time, 0]

    print(f"Time to find control: {mppi_time}")

    # select a sample set of trajectories for review/visualization
    trajectories = rollout_trajectories(
        vehicle=robot_model,
        initial_state=initial_state,
        u_nom=u_nom,
        u_variations=u_var,
        weights=u_weights,
        dt=args.tick_time,
    )

    return u, trajectories, u_weights


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

    sim = Simulation(
        # policy_name=args.policy,
        # policy_args={
        #     'max_v': 0.07,
        #     'min_v': 0,
        #     'max_accel': 0.05,
        #     'max_brake': 0.045,
        #     'screen': screen,
        # },
        generator_name=args.generator,
        generator_args=generator_args,
        num_actors=args.actors,
        tracks=args.tracks,
        pois_lambda=args.lambd,
        screen=surface if args.show_sim or args.record_data else None,
        tick_time=args.tick_time,
        #  ego_start=[0.05, 0.15],  # [0.25, 0.75]],
        ego_start=[0.05, [0.25, 0.75]],
        # ego_goal=[0.95, 0.85],  #
        ego_goal=[0.95, [0.25, 0.75]],
    )

    if args.seed is not None:
        # Save the tasks (or reload them)
        pass

    tracker_update_interval = 1
    tracker = Tracker(
        initial_timestep=0, scenario_map=None, args=args, device="cuda:0", dt=args.tick_time * tracker_update_interval
    )

    path = None

    # Simulation/Game Loop
    tracker_frame = 0

    action = None

    # initialize a acceleration/steering angle controlled robot
    robot = Ackermann4(length=0.7, width=0.7)

    # initialize a controller
    Q = np.array([args.x_weight, args.y_weight, args.v_weight, args.theta_weight])
    Qf = np.array([FINAL_X_WEIGHT, FINAL_X_WEIGHT, FINAL_V_WEIGHT, FINAL_THETA_WEIGHT])
    R = np.array([args.a_weight, args.delta_weight])
    mppi = MPPI(
        vehicle=robot,
        samples=5000,
        seed=args.seed,
        u_limits=CONTROL_LIMITS,
        u_dist_limits=CONTROL_VARIATION_LIMITS,
        c_lambda=args.c_lambda,
        Q=Q,
        Qf=Qf,
        R=R,
        M=args.mppi_m,
        method="None",
        scan_range=SCAN_RANGE,
        discount_factor=1.0,
        vehicle_length=robot.L,
    )

    # self.controlNN = ControlPredictor("./models/tesla_car.model")

    # for now, assume an empty map -- we can add objects later
    grid_origin = sim.display_offset
    grid_resolution = GRID_RESOLUTION
    grid_size = int(sim.display_diff / grid_resolution)
    local_map = np.zeros((grid_size, grid_size))

    action = None
    u = None
    while True:
        if args.show_sim:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

        observation, reward, done, info = sim.tick(action=action)
        if done == True:
            break

        agent_predictions = {}
        if (sim.ticks - 1) % tracker_update_interval == 0:
            visible_agents = [actor for actor in info["actors"] if actor["visible"] == True]

            # Update the tracker
            agent_predictions = tracker.step(agents=visible_agents, horizon=args.horizon, timesteps=1)

        # if path is None:
        #     start = info["ego"]["pos"]
        #     end = info["goal"]
        #     path = generate_path(start, end, args)

        start = info["ego"]["pos"]
        end = info["goal"]
        paths = generate_trajectories(start, end, args)
        # path_index, distance = project_position_to_path(info["ego"]["pos"], path=path)
        # path_index = min(len(path) - 1, sim.ticks)

        # if len(agent_predictions):
        #     best_trajectory = evaluate(
        #         grid=local_map,
        #         origin=grid_origin,
        #         resolution=grid_resolution,
        #         av_size=[sim.ego.LENGTH, sim.ego.WIDTH],
        #         trajectories=paths,
        #         agents=visible_agents,
        #         predictions=agent_predictions,
        #         stopping_threshold=0.5,
        #         prediction_interval=tracker_update_interval * args.tick_time,
        #         dt=args.tick_time,
        #     )
        # else:
        best_trajectory = 0

        u, trajectories, trajectory_weights = get_control(
            mppi=mppi,
            costmap=np.zeros((100, 100)),
            origin=sim.display_offset,
            robot_model=robot,
            u_nom=u,
            initial_state=info["ego"]["pos"][: ActorStateEnum.DELTA],
            goal=[*info["ego"]["goal"], 0, 0],
            path=paths[best_trajectory],
            agents=visible_agents,
            args=args,
        )

        action = u[0]
        if np.isnan(action[0]):
            action = [0.0]

        if args.show_sim:
            sim.render(
                actors=agent_predictions,
                trajectories=trajectories,
                trajectory_weights=trajectory_weights,
                path=paths,
                prefix_str=args.prefix,
            )
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            # pygame.display.update()
            clock.tick(1 / args.tick_time * args.simulation_speed)

    return sim


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--height", default=SCREEN_HEIGHT, type=int, help="Screen vertical size")
    argparser.add_argument("--width", default=SCREEN_WIDTH, type=int, help="Screen horizontal size")
    argparser.add_argument("--margin", default=SCREEN_MARGIN, type=int, help="Screen horizontal size")
    argparser.add_argument("-s", "--seed", default=None, type=int, help="Random Seed")
    argparser.add_argument("-l", "--lambd", default=LAMBDA, type=float, help="Exponential Spawn rate for Tasks")
    argparser.add_argument("-a", "--actors", default=NUM_ACTORS, type=int, help="Number of actors in the simulation")
    argparser.add_argument("-p", "--policy", default=DEFAULT_POLICY_NAME, help="Policy to use")
    argparser.add_argument("--simulation-speed", default=SIMULATION_SPEED, type=float, help="Simulator speed")
    argparser.add_argument("-t", "--tick-time", default=TICK_TIME, type=float, help="Length of Simulation Time Step")
    argparser.add_argument("-g", "--generator", default=DEFAULT_GENERATOR_NAME, help="Random Generator to use")
    argparser.add_argument("--tracks", default=None, type=str, help="Load pedestrian tracks from file")
    argparser.add_argument("--max-time", default=None, type=float, help="Maximum Length of Simulation")
    argparser.add_argument("--record-data", action="store_true", help="Record data to disk as frames")
    argparser.add_argument("--show-sim", action="store_true", help="Display the simulation window")
    argparser.add_argument("--config", type=str, help="Configuration file for the model")
    argparser.add_argument("--model-iteration", type=int, help="Model version")
    argparser.add_argument("--model-dir", type=str, help="Location of the model files")
    argparser.add_argument("--attention-radius", type=float, default=3.0, help="Model version")
    argparser.add_argument("--device", type=str, default=None, help="Model version")
    argparser.add_argument("--samples", type=int, default=5, help="Model version")
    argparser.add_argument(
        "--history-len", type=int, default=10, help="Maximum number of samples to keep track of for agent routes"
    )
    argparser.add_argument("--horizon", type=int, default=10, help="Model prediction horizon")
    argparser.add_argument("--incremental", help="Use Trajectron in online incremental mode", action="store_true")
    argparser.add_argument("--results-dir", type=str, help="Location of generated output")
    argparser.add_argument("--prefix", type=str, default=None, help="Output prefix to identify saved results")
    argparser.add_argument("--robot-speed", default=ROBOT_SPEED, type=float, help="Speed of the robot (m/s)")
    argparser.add_argument("--k-eval", type=float, default=25.0, help="Number of samples to evaluate")

    # MPPI params
    argparser.add_argument(
        "--c_lambda", type=float, default=DEFAULT_LAMBDA, help="Lambda value for weight normalization control"
    )
    argparser.add_argument(
        "--mppi_m", type=float, default=DEFAULT_METHOD_WEIGHT, help="M/Lambda value for method weights"
    )
    argparser.add_argument("--x_weight", type=float, default=X_WEIGHT, help="Weight for x coordinate")
    argparser.add_argument("--y_weight", type=float, default=Y_WEIGHT, help="Weight for y coordinate")
    argparser.add_argument("--v_weight", type=float, default=V_WEIGHT, help="Weight for velocity")
    argparser.add_argument("--theta_weight", type=float, default=THETA_WEIGHT, help="Weight for theta")
    argparser.add_argument("--a_weight", type=float, default=A_WEIGHT, help="Weight for acceleration")
    argparser.add_argument("--delta_weight", type=float, default=DELTA_WEIGHT, help="Weight for delta")

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

    argparser.add_argument("--map_encoding", help="Whether to use map encoding or not", action="store_true")

    argparser.add_argument("--augment", help="Whether to augment the scene during training", action="store_true")

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

    argparser.add_argument("--no_edge_encoding", help="Whether to use neighbors edge encoding", action="store_true")

    args = argparser.parse_args()

    simulate(args)
