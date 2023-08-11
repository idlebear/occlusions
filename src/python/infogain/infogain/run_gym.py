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

from mpc_controller import MPC, Vehicle

from typing import Callable

from config import *

from mpc_controller import MPC


def main(args):
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
            x_fin = [40, 0, 0]
            x_init = [0, 0, 0]
            v_des = 1

            Q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0.1]])

            Qf = np.array([[1 / 40.0, 0, 0], [0, 1 / 40.0, 0], [0, 0, 0.1 / 40.0]])

            R = np.array([[1, 0], [0, 1]])

            agents = [[10, 0.00, 2], [20, 4, 1]]
            # agents = [[20, 4, 2]]

            planning_horizon = 15
            control_len = Vehicle.control_len
            state_len = Vehicle.state_len

            # TODO: Using no active agents for the time being -- eventually we'll need to build a
            #       list of the near agents, and add factors so the remainder are ignored.
            mpc = MPC(
                state_len=state_len,
                control_len=control_len,
                planning_horizon=planning_horizon,
                num_agents=0,
                step_fn=Vehicle.runge_kutta_step,
                Q=Q,
                Qf=Qf,
                R=R,
                dt=env.sim.tick_time,
            )

    # reset the environment and collect the first observation and current state
    obs, info = env.reset()

    while True:
        if args.method == "random":
            # fixed forward motion (for testing)
            action = np.array(
                [np.random.random() * 4 - 2, np.random.randint(-1, 2) * np.pi / 6]
            )
        elif args.method == "rl":
            action, _states = model.predict(obs)
        elif args.method == "mpc":
            action = mpc.next(obs, info)
        else:
            # fixed forward motion (for testing)
            ego_state = info["ego"]
            # best_dir = np.argmax(np.array(info['information_gain']))
            # if best_dir == 0:
            #     w = -np.pi/4
            # elif best_dir == 1:
            #     w = -ego_state['orientation']
            # elif best_dir == 2:
            #     w = np.pi/4
            # else:
            #     w = 0
            w = 0  # fixed rotation (none)
            action = (1, w)

        obs, rewards, done, info = env.step(action)
        env.render()


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
    argparser.add_argument(
        "--show-sim", action="store_true", help="Display the simulation window"
    )
    argparser.add_argument(
        "--skip-training",
        action="store_true",
        help="skip the rl training and just run the inference engine",
    )
    argparser.add_argument(
        "--debug", action="store_true", help="Dummy mode -- just display the env"
    )
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
