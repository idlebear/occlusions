import gym
import json
import datetime as dt
import numpy as np
import argparse
import traceback
import time
from os import path
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy

from OcclusionGym import OcclusionEnv

from mpc_controller import MPC, Ackermann5, MPPI, rollout_trajectory
from Actor import STATE
from trajectory import generate_trajectory, validate_controls
from visibility_costmap import update_visibility_costmap
from Grid.VisibilityGrid import VisibilityGrid

from typing import Callable

from config import *


def main(args):
    x_fin = [10, -1.5, 0]
    vehicle = Ackermann5()

    if args.method == "rl":
        # Parallel environments
        if args.multipass:
            env = make_vec_env(
                OcclusionEnv,
                n_envs=args.instances,
                env_kwargs={"num_actors": args.actors},
                vec_env_cls=SubprocVecEnv,
            )
        else:
            env = make_vec_env(
                OcclusionEnv,
                n_envs=args.instances,
                env_kwargs={"num_actors": args.actors},
            )

        model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./occlusion_log")
        try:
            model.load("ppo_occlusions.model")
            print("Previous model loaded")
        except IOError:
            print("No model to load -- starting fresh")

        if not args.skip_training:
            for i in range(args.epochs):
                model.learn(total_timesteps=args.timesteps)
                model.save(f"ppo_occlusions_{i}.model")

    else:
        env = OcclusionEnv(num_actors=args.actors, seed=args.seed)

        M = args.visibility_weight

        if args.method == "mpc":
            Q = (
                np.array(
                    [
                        [5.0, 0, 0, 0, 0],
                        [0, 5.0, 0, 0, 0],
                        [0, 0, 1.0, 0, 0],
                        [0, 0, 0, 1.0, 0],
                        [0, 0, 0, 0, 0.01],
                    ]
                )
                / args.horizon
            )
            Qf = np.array(
                [
                    [8.0, 0, 0, 0, 0],
                    [0, 1.0, 0, 0, 0],
                    [0, 0, 1.0, 0, 0],
                    [0, 0, 0, 1.0, 0],
                    [0, 0, 0, 0, 0.0],
                ]
            )
            R = np.array([[1.0, 0], [0, 1.0 / np.pi]]) / args.horizon

            # M = np.array([2 / np.pi])  # Andersen
            mpc = MPC(
                mode=args.visibility_cost,
                vehicle=vehicle,
                planning_horizon=args.horizon,
                num_agents=args.actors,
                Q=Q,
                Qf=Qf,
                R=R,
                M=M,
                dt=env.sim.tick_time,
            )
        elif args.method == "mppi":
            mppi = MPPI(
                mode=args.visibility_cost,
                vehicle=vehicle,
                limits=(ACCEL_VARIATION, OMEGA_VARIATION),
                c_lambda=LAMBDA,
                Q=np.diag([X_WEIGHT, Y_WEIGHT, V_WEIGHT, THETA_WEIGHT, DELTA_WEIGHT]),
                M=M,
                seed=args.seed,
            )

    # start data logging
    if args.prefix != "":
        prefix = f"{args.prefix}-"
    else:
        prefix = ""
    results_str = (
        prefix
        + args.method
        + "-"
        + str(args.visibility_cost)
        + "-"
        + str(args.visibility_weight)
        + "-"
        + str(args.simulation_steps)
        + "-"
        + str(args.horizon)
        + "-"
        + str(args.samples)
        + "-"
        + str(args.seed)
        + ".csv"
    )
    results_file_name = path.join(RESULTS_DIR, results_str)
    f = open(results_file_name, "w")
    f.write(
        "policy,seed,run,samples,horizon,visibility-weight,t,x,y,v,theta,accel_requested,accel_allowed,u2,delta,costmap_time,planning_time\n"
    )
    f.flush

    for run in tqdm(range(args.runs)):
        # reset the environment and collect the first observation and current state
        obs, info = env.reset()
        count = 0

        controls_nom = np.array([[A_DES for _ in range(args.horizon)], [0 for _ in range(args.horizon)]])
        visibility_costmap = VisibilityGrid(dim=GRID_WIDTH, resolution=GRID_RESOLUTION)

        total_time = 0
        for step in tqdm(range(args.simulation_steps)):
            tic = time.time()

            if args.method == "random":
                # fixed forward motion (for testing)
                action = np.array([np.random.random() * 4 - 2, np.random.randint(-1, 2) * np.pi / 6])
            elif args.method == "rl":
                action, _states = model.predict(obs)
            elif args.method == "mpc":
                state = [
                    info["ego"]["x"][STATE.X],
                    info["ego"]["x"][STATE.Y],
                    info["ego"]["x"][STATE.VELOCITY],
                    info["ego"]["x"][STATE.THETA],
                    info["ego"]["x"][STATE.DELTA],
                ]
                actors = []
                # print(
                #     f"State = X:{state[0]:0.5}, Y:{state[1]:0.5}, V:{state[2]:0.5}, Th:{state[3]:0.5}, {state[4]:0.5}"
                # )
                for ac in info["actors"]:
                    radius = ac["extent"]  # + info["ego"]["extent"]
                    dist = np.sqrt((ac["x"][STATE.X] - state[0]) ** 2 + (ac["x"][STATE.Y] - state[1]) ** 2)
                    # print(
                    #     f"Dist:{dist}, Safe:{radius}, diff:{-dist+radius} {'AUUUGGGG' if -dist+radius > 0 else ''}"
                    # )
                    actors.append(
                        [
                            ac["x"][STATE.X],
                            ac["x"][STATE.Y],
                            radius,
                            *ac["min_pt"],
                            dist,
                        ]
                    )

                # only keep the closest, and drop the distance parameter
                if len(actors) > args.actors:
                    actors = sorted(actors, key=lambda actor: actor[-1])
                actors = [a[:-1] for a in actors[: args.actors]]

                while len(actors) < args.actors:
                    actors.append([1000, 1000, 0, 0, 0])  # placeholders are far, far away

                x = state[0]
                v = state[2]
                a = A_DES
                traj = []
                controls = []
                for _ in range(args.horizon):
                    x += v * env.sim.tick_time
                    v += a * env.sim.tick_time
                    if v >= V_DES:
                        v = V_DES
                        a = 0
                    # if x > x_fin[0]:
                    #     x = x_fin[0]
                    next_entry = [x, -LANE_WIDTH / 2.0, v, 0, 0]
                    traj.append(next_entry)
                    controls.append([a, 0])

                planning_start = time.time()
                try:
                    u, x = mpc.next(
                        obs=obs,
                        state=state,
                        goal=[
                            state[0] + 10,
                            -LANE_WIDTH / 2.0,
                            0,
                            0,
                            0,
                        ],  # moving carrot...
                        agents=actors,
                        trajectory=traj,
                        controls=controls,
                        warm_start=True,
                    )
                except SystemError as e:
                    print(traceback.format_exc())
                    print(e)
                    exit()

                print(f"Planning time: {time.time() - planning_start:0.5}")

                action = u[:, 0].full()
                u = np.array(u.full())
            elif args.method == "mppi":
                state = [
                    info["ego"]["x"][STATE.X],
                    info["ego"]["x"][STATE.Y],
                    info["ego"]["x"][STATE.VELOCITY],
                    info["ego"]["x"][STATE.THETA],
                    info["ego"]["x"][STATE.DELTA],
                ]

                actors = []
                if args.visibility_cost != "Nominal":
                    for ac in info["actors"]:
                        radius = 1.5  # ac["extent"]  # + info["ego"]["extent"]
                        actors.append([ac["x"][STATE.X], ac["x"][STATE.Y], radius, *ac["min_pt"]])

                ###
                # Build the visibility costmap
                waypoints = [
                    state[0:2],
                ]

                # add a waypoint for every 5 m for V_DES * args.horizon * dt * 2 -- double planning horizon
                for _ in range(int(V_DES * env.sim.tick_time * args.horizon * 2.0 / WAYPOINT_INTERVAL) + 1):
                    waypoints.append([waypoints[-1][0] + WAYPOINT_INTERVAL, -LANE_WIDTH / 2])

                # build the immediate planning trajectory
                obs_trajectory = generate_trajectory(
                    waypoints,
                    s=0,
                    d=(state[1] - (-LANE_WIDTH / 2)),
                    v=V_DES,
                    t=(args.horizon * env.sim.tick_time),
                    dt=env.sim.tick_time,
                )[0]

                # and a target trajectory to define the locations to observe
                target_trajectory = generate_trajectory(
                    waypoints,
                    s=0,
                    d=(state[1] - (LANE_WIDTH / 2)),
                    v=V_DES,
                    t=(args.horizon * env.sim.tick_time) * 2,
                    dt=env.sim.tick_time,
                )[0]

                origin = [
                    state[0] - (GRID_SIZE / 2) * GRID_RESOLUTION + GRID_RESOLUTION / 2.0,
                    state[1] - (GRID_SIZE / 2) * GRID_RESOLUTION + GRID_RESOLUTION / 2.0,
                ]

                costmap_tic = time.time()
                costmap = update_visibility_costmap(
                    costmap=visibility_costmap,
                    obs=obs,
                    map=info["map"],
                    origin=origin,
                    resolution=GRID_RESOLUTION,
                    obs_trajectory=obs_trajectory,
                    target_trajectory=target_trajectory,
                    v_des=V_DES,
                    dt=env.sim.tick_time,
                )
                costmap_toc = time.time()

                ###
                # Construct the nominal trajectory
                states_nom = [
                    state,
                ]
                x = state[0]
                v = state[2]
                a = A_DES
                for _ in range(args.horizon):
                    x += v * env.sim.tick_time
                    v += a * env.sim.tick_time
                    if v >= V_DES:
                        v = V_DES
                        a = 0
                    states_nom.append([x, -LANE_WIDTH / 2, v, 0, 0])

                states_nom = np.array(states_nom).T  # each state in a column

                control_tic = time.time()
                u, u_var = mppi.find_control(
                    costmap=visibility_costmap.visibility_costmap(),
                    origin=origin,
                    resolution=GRID_RESOLUTION,
                    x_nom=states_nom,
                    u_nom=controls_nom,
                    initial_state=np.array(state),
                    samples=args.samples,
                    actors=actors,
                    dt=env.sim.tick_time,
                )
                control_toc = time.time()

                if args.show_sim:
                    visibility_costmap.visualize()

                # warm start the controls for next time
                controls_nom[:, :-1] = u[:, 1:]
                # controls_nom[:, -1] = u[0, 0]

                if args.show_sim:
                    from mpc_controller import visualize_variations

                    visualize_variations(
                        vehicle,
                        initial_state=state,
                        u_nom=controls_nom,
                        u_variations=u_var,
                        u_weighted=u,
                        dt=env.sim.tick_time,
                    )

                if args.visibility_cost == "Nominal":
                    action = u[:, 0]
                else:
                    states = rollout_trajectory(vehicle=vehicle, state=state, controls=u.T, dt=env.sim.tick_time)
                    action = validate_controls(
                        vehicle=vehicle,
                        states=states,
                        controls=u.T,
                        obs=obs,
                        map=info["map"],
                        resolution=GRID_RESOLUTION,
                        dt=env.sim.tick_time,
                    )
            else:
                action = [8, 0]

            count += 1
            toc = time.time()

            step_tic = time.time()
            obs, rewards, done, info = env.step(action)
            step_toc = time.time()
            step_time = step_toc - step_tic

            if args.show_sim:
                env.render(u=u)

            # "policy,seed,run,samples,horizon,visibility-cost,t,x,y,v,theta,delta,accel_requested,accel_allowed,steering,costmap_time,planning_time\n"
            f.write(
                str(args.visibility_cost)
                + ","
                + str(args.seed)
                + ","
                + str(run)
                + ","
                + str(args.samples)
                + ","
                + str(args.horizon)
                + ","
                + str(args.visibility_weight)
                + ","
                + str(step * env.sim.tick_time)
                + ","
                + str(state[0])
                + ","
                + str(state[1])
                + ","
                + str(state[2])
                + ","
                + str(state[3])
                + ","
                + str(state[4])
                + ","
                + str(u[0, 0])
                + ","
                + str(action[0])
                + ","
                + str(u[1, 0])
                + ","
                + f"{costmap_toc-costmap_tic:0.5}"
                + ","
                + f"{control_toc-control_tic:0.5}"
                + "\n"
            )
            # print(
            #     f"Planner:{planner_time:0.5}(Costmap:{costmap_toc-costmap_tic:0.5}, MPPI:{control_toc-control_tic:0.5} ), Step:{step_time:0.5}, Render:{render_time:0.5}, Total:{time.time() - tic}"
            # )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--height", default=SCREEN_HEIGHT, type=int, help="Screen vertical size")
    argparser.add_argument("--width", default=SCREEN_WIDTH, type=int, help="Screen horizontal size")
    argparser.add_argument("--margin", default=SCREEN_MARGIN, type=int, help="Screen horizontal size")
    argparser.add_argument("-s", "--seed", default=None, type=int, help="Random Seed")
    argparser.add_argument(
        "-a",
        "--actors",
        default=NUM_ACTORS,
        type=int,
        help="Number of actors in the simulation",
    )
    argparser.add_argument(
        "--timesteps",
        default=MAX_TIMESTEPS,
        type=int,
        help="Number of timesteps/episodes to run the simulation",
    )
    argparser.add_argument(
        "-i",
        "--instances",
        default=NUM_INSTANCES,
        type=int,
        help="Number of instances to run at one time",
    )
    argparser.add_argument(
        "-e",
        "--epochs",
        default=NUM_EPOCHS,
        type=int,
        help="Number of instances to run at one time",
    )
    argparser.add_argument("--prefix", default="", help="Prefix on results file name")
    argparser.add_argument(
        "-g",
        "--generator",
        default=DEFAULT_GENERATOR_NAME,
        help="Random Generator to use",
    )
    argparser.add_argument(
        "--simulation_speed",
        default=SIMULATION_SPEED,
        type=float,
        help="Simulator speed",
    )
    argparser.add_argument(
        "-t",
        "--tick_time",
        default=TICK_TIME,
        type=float,
        help="Length of Simulation Time Step",
    )
    argparser.add_argument("--show-sim", action="store_true", help="Display the simulation window")
    argparser.add_argument(
        "--skip-training",
        action="store_true",
        help="skip the rl training and just run the inference engine",
    )
    argparser.add_argument("--debug", action="store_true", help="Dummy mode -- just display the env")
    argparser.add_argument(
        "--method",
        default="none",
        type=str,
        help="control method to be used: none, rl, mpc, mppi",
    )
    argparser.add_argument(
        "--samples",
        default=100,
        type=int,
        help="Number of samples to generate if sampling based method in use (e.g., MPPI)",
    )
    argparser.add_argument(
        "--horizon",
        default=10,
        type=int,
        help="Length of the planning horizon",
    )
    argparser.add_argument(
        "--runs",
        default=10,
        type=int,
        help="Number of times to repeat the experiment",
    )
    argparser.add_argument(
        "--simulation-steps",
        default=100,
        type=int,
        help="Number of steps to run the simulation",
    )
    argparser.add_argument(
        "--multipass",
        action="store_true",
        help="Run multiple environments simultaneously",
    )
    argparser.add_argument(
        "--visibility-cost",
        default="none",
        type=str,
        help="method to determine visibility cost: None, Andersen, Higgins, Ours",
    )

    argparser.add_argument(
        "--visibility-weight",
        default=1,
        type=float,
        help="Weight to apply to the visibility cost function -- a.k.a. M in the research paper",
    )

    args = argparser.parse_args()

    main(args)
