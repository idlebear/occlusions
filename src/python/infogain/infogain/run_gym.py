import gym
import json
import datetime as dt
import numpy as np
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy

from OcclusionGym import OcclusionEnv

from mpc_controller import MPC, Vehicle, SkidSteer

from typing import Callable

from config import *

from mpc_controller import MPC


def main(args):
    x_fin = [50, -0.03, 0]
    V_DES = 1.0

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
            Q = np.array([[100.0, 0, 0], [0, 100.0, 0], [0, 0, 0.1]])
            Qf = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 0.001]])
            R = np.array([[0.01, 0], [0, 0.01]])

            planning_horizon = 15

            mpc = MPC(
                vehicle=Vehicle,
                planning_horizon=planning_horizon,
                num_agents=args.actors,
                Q=Q,
                Qf=Qf,
                R=R,
                dt=env.sim.tick_time,
            )

    # reset the environment and collect the first observation and current state
    obs, info = env.reset()

    count = 0
    rev = 1

    while True:
        if args.method == "random":
            # fixed forward motion (for testing)
            action = np.array([np.random.random() * 4 - 2, np.random.randint(-1, 2) * np.pi / 6])
        elif args.method == "rl":
            action, _states = model.predict(obs)
        elif args.method == "mpc":
            state = [info["ego"]["x"][0], info["ego"]["x"][1], info["ego"]["x"][4]]
            actors = []
            print(f"State: {state[0]}, {state[1]}, {state[2]}")
            for ac in info["actors"]:
                radius = max(ac["bbox"][2] - ac["bbox"][0], ac["bbox"][3] - ac["bbox"][1]) / 2.0
                actors.append([ac["x"][0], ac["x"][1], radius])
            while len(actors) < args.actors:
                actors.append([0, 0, 0])

            traj = [
                [info["ego"]["x"][0], -0.03, 0],
            ]
            for i in range(1, mpc.planning_horizon):
                next_entry = list(traj[-1])
                next_entry[0] += V_DES * env.sim.tick_time
                if next_entry[0] > x_fin[0]:
                    next_entry[0] = x_fin[0]
                traj.append(next_entry)

            u, x = mpc.next(obs=obs, goal=x_fin, agents=actors, trajectory=traj, warm_start=False)
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

            # SkidSteer
            if not count % 100:
                rev *= -1
            w1 = 15 + rev * np.sin(np.pi * 2 * count / 100) * 0.05
            w2 = 15
            action = [w1, w2]
            count += 1

        obs, rewards, done, info = env.step(action)
        env.render(u=u.full().T)


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
