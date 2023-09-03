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

from mpc_controller import MPC, Ackermann5
from Actor import STATE

from typing import Callable

from config import *


def main(args):
    x_fin = [10, -1.5, 0]
    V_DES = 5.0
    A_DES = 5.0

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
                        [3.0, 0, 0, 0, 0],
                        [0, 5.0, 0, 0, 0],
                        [0, 0, 1.0, 0, 0],
                        [0, 0, 0, 1.0, 0],
                        [0, 0, 0, 0, 0.01],
                    ]
                )
                / PLANNING_HORIZON
            )
            Qf = np.array(
                [
                    [1.0, 0, 0, 0, 0],
                    [0, 1.0, 0, 0, 0],
                    [0, 0, 1.0, 0, 0],
                    [0, 0, 0, 1.0, 0],
                    [0, 0, 0, 0, 0.0],
                ]
            )
            R = np.array([[1.0, 0], [0, 1.0 / np.pi]]) / PLANNING_HORIZON

            M = np.array([1]) / PLANNING_HORIZON  # Higgins
            # M = np.array([2 / np.pi])  # Anderson
            mpc = MPC(
                mode="Higgins",  # 'Anderson', 'Higgins', 'None'
                vehicle=Ackermann5(),
                planning_horizon=PLANNING_HORIZON,
                num_agents=args.actors,
                Q=Q,
                Qf=Qf,
                R=R,
                M=M,
                dt=env.sim.tick_time,
            )

    # reset the environment and collect the first observation and current state
    obs, info = env.reset()

    count = 0
    rev = 1

    total_time = 0
    while True:
        print("Starting Action calc... ")
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
            print(f"State = X:{state[0]:0.5}, Y:{state[1]:0.5}, V:{state[2]:0.5}, Th:{state[3]:0.5}, {state[4]:0.5}")
            for ac in info["actors"]:
                radius = ac["extent"]  # + info["ego"]["extent"]
                dist = -np.sqrt((ac["x"][STATE.X] - state[0]) ** 2 + (ac["x"][STATE.Y] - state[1]) ** 2)
                print(f"Dist:{dist}, Safe:{radius}, diff:{dist+radius} {'AUUUGGGG' if dist+radius > 0 else ''}")
                actors.append([ac["x"][STATE.X], ac["x"][STATE.Y], radius, *ac["min_pt"]])
            while len(actors) < args.actors:
                actors.append([1000, 1000, 0, 0, 0])  # placeholders are far, far away

            x = state[0]
            v = state[2]
            a = A_DES
            traj = []
            controls = []
            for _ in range(mpc.planning_horizon):
                x += v * env.sim.tick_time
                v += a * env.sim.tick_time
                if v >= V_DES:
                    v = V_DES
                    a = 0
                # if x > x_fin[0]:
                #     x = x_fin[0]
                next_entry = [x, -1.5, v, 0, 0]
                traj.append(next_entry)
                controls.append([V_DES, 0])

            try:
                u, x = mpc.next(
                    obs=obs,
                    state=state,
                    goal=[state[0] + 10, -1.5, 0, 0, 0],  # moving carrot...
                    agents=actors,
                    trajectory=traj,
                    controls=controls,
                    warm_start=True,
                )
            except Exception as e:
                print(traceback.format_exc())
                print(e)

            # out = u[:, 0:4].full()
            # print(f"U0: {out[0,0]:6.5} | {out[0,1]:6.5} | {out[0,2]:6.5} | {out[0,3]:6.5}")
            # print(f"U1: {out[1,0]:6.5} | {out[1,1]:6.5} | {out[1,2]:6.5} | {out[1,3]:6.5}")

            # out = x[:, 0:5].full()
            # print(f"X0: {out[0,0]:6.5} | {out[0,1]:6.5} | {out[0,2]:6.5} | {out[0,3]:6.5} | {out[0,4]:6.5}")
            # print(f"X1: {out[1,0]:6.5} | {out[1,1]:6.5} | {out[1,2]:6.5} | {out[1,3]:6.5} | {out[1,4]:6.5}")
            # # print(f"X2: {out[2,0]:6.5} | {out[2,1]:6.5} | {out[2,2]:6.5} | {out[2,3]:6.5} | {out[2,4]:6.5}")
            # print(f"X3: {out[3,0]:6.5} | {out[3,1]:6.5} | {out[3,2]:6.5} | {out[3,3]:6.5} | {out[3,4]:6.5}")
            # # print(f"X4: {out[4,0]:6.5} | {out[4,1]:6.5} | {out[4,2]:6.5} | {out[4,3]:6.5} | {out[4,4]:6.5}")

            action = u[:, 0].full()
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

        total_time += toc - tic
        print(f"Last time: {toc-tic:0.5}, Average calculation time: {total_time/count:0.5}")
        print(count, count * env.sim.tick_time)

        obs, rewards, done, info = env.step(action)
        print(f"Step Completed: {time.time()-toc:0.5}")
        env.render(u=u.full().T)
        print(f"    ...and rendered.  Total time: { time.time() - tic:0.5}")


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
        help="control method to be used: none, rl, mpc",
    )
    argparser.add_argument(
        "--multipass",
        action="store_true",
        help="Run multiple environments simultaneously",
    )

    args = argparser.parse_args()

    main(args)
