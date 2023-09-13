import gym
import json
import datetime as dt
import numpy as np
import argparse
import traceback
import time

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
    V_DES = 7.0
    A_DES = 3.0
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

            M = np.array([1]) / args.horizon  # Higgins
            # M = np.array([2 / np.pi])  # Anderson
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
            if args.visibility_cost == "Ours":
                M = 200
            elif args.visibility_cost == "Higgins":
                M = 0.9
            else:
                M = 0
            mppi = MPPI(
                mode=args.visibility_cost,
                vehicle=vehicle,
                limits=(3, np.pi / 2),
                c_lambda=250,
                Q=np.diag([2.0, 5.0, 5.0, 0, 0]),
                M=M,
                seed=args.seed,
            )
            controls_nom = np.array([[A_DES for _ in range(args.horizon)], [0 for _ in range(args.horizon)]])
            visibility_costmap = VisibilityGrid(dim=GRID_WIDTH, resolution=GRID_RESOLUTION)

    # reset the environment and collect the first observation and current state
    obs, info = env.reset()

    count = 0
    rev = 1

    total_time = 0
    while True:
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
                actors.append([ac["x"][STATE.X], ac["x"][STATE.Y], radius, *ac["min_pt"], dist])

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
            print(f"State = X:{state[0]:0.5}, Y:{state[1]:0.5}, V:{state[2]:0.5}, Th:{state[3]:0.5}, {state[4]:0.5}")
            for ac in info["actors"]:
                radius = 1.2  # ac["extent"]  # + info["ego"]["extent"]
                actors.append([ac["x"][STATE.X], ac["x"][STATE.Y], radius, *ac["min_pt"]])

            ###
            # Build the visibility costmap
            waypoints = [
                state[0:2],
            ]

            # add a waypoint for every 5 m for V_DES * args.horizon * dt * 2 -- double planning horizon
            for _ in range(int(V_DES * env.sim.tick_time * args.horizon * 2.0 / WAYPOINT_INTERVAL)):
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

            try:
                u, u_var = mppi.find_control(
                    costmap=visibility_costmap.normalized(),
                    origin=origin,
                    resolution=GRID_RESOLUTION,
                    x_nom=states_nom,
                    u_nom=controls_nom,
                    initial_state=np.array(state),
                    samples=args.samples,
                    actors=actors,
                    dt=env.sim.tick_time,
                )

                # warm start the controls for next time
                controls_nom[:, :-1] = u[:, 1:]
                # controls_nom[:, -1] = u[0, 0]

                from mpc_controller import visualize_variations

                visualize_variations(
                    vehicle,
                    initial_state=state,
                    u_nom=controls_nom,
                    u_variations=u_var,
                    u_weighted=u,
                    dt=env.sim.tick_time,
                )

            except Exception as e:
                print(traceback.format_exc())
                print(e)

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
            ## Pseudo Car
            # # fixed forward motion (for testing)
            # ego_state = info["ego"]
            # # best_dir = np.argmax(np.array(info['information_gain']))
            # # if best_dir == 0:
            # #     w = -np.pi/4
            # # elif best_dir == 1:
            # #     w = -ego_state['orientation']
            # # elif best_dir == 2:
            # #     w = np.pi/4
            # # else:
            # #     w = 0
            # w = 0  # fixed rotation (none)
            # if ego_state["speed"] < 5:
            #     action = (1, w)
            # else:
            #     action = (0, w)
            # action = (1, w)
            action = [8, 0]

        count += 1
        toc = time.time()

        planner_time = toc - tic
        total_time += planner_time
        print(f"Last time: {toc-tic:0.5}, Average calculation time: {total_time/count:0.5}")

        step_tic = time.time()
        obs, rewards, done, info = env.step(action)
        toc = time.time()
        step_time = toc - step_tic

        render_tic = time.time()
        env.render(u=u)
        toc = time.time()
        render_time = toc - render_tic

        print(
            f"Planner:{planner_time:0.5}({total_time/count:0.5}), Step:{step_time:0.5}, Render:{render_time:0.5}, Total:{time.time() - tic}"
        )


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
        "--multipass",
        action="store_true",
        help="Run multiple environments simultaneously",
    )
    argparser.add_argument(
        "--visibility-cost",
        default="none",
        type=str,
        help="method to determine visibility cost: none, Higgins, Ours",
    )

    args = argparser.parse_args()

    main(args)
