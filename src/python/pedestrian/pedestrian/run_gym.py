import gym
import json
import datetime as dt
import argparse
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from PedestrianGym import PedestrianEnv
from config import *


def main(args):

    if args.debug:
        # set up debug version -- mo models
        env = PedestrianEnv(num_actors=args.actors, seed=args.seed, tracks=args.tracks)
    else:
        if args.multipass:
            env = make_vec_env(
                PedestrianEnv,
                n_envs=args.instances,
                env_kwargs={
                    "num_actors": args.actors,
                    "seed": args.seed,
                    "tracks": args.tracks,
                },
            )
        else:
            env = make_vec_env(
                PedestrianEnv,
                n_envs=args.instances,
                env_kwargs={
                    "num_actors": args.actors,
                    "seed": args.seed,
                    "tracks": args.tracks,
                },
            )

        if not args.demo:
            for i in range(10):
                model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./occlusion_log")
                try:
                    model.load("ppo_occlusions.model")
                    print("Previous model loaded")
                except IOError:
                    print("No model to load -- starting fresh")

                model.learn(total_timesteps=args.timesteps)

                model.save("ppo_occlusions.model")

        else:
            model = PPO("MlpPolicy", env, verbose=1)
            try:
                model.load("ppo_occlusions.model")
                print("Previous model loaded")
            except IOError:
                print("No model to load -- starting fresh")

        obs = env.reset()

    for i in range(2000):
        if args.debug:
            action = np.array([6, 0], dtype=np.float32)
        else:
            action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            break


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--height", default=SCREEN_HEIGHT, type=int, help="Screen vertical size")
    argparser.add_argument("--width", default=SCREEN_WIDTH, type=int, help="Screen horizontal size")
    argparser.add_argument("--margin", default=SCREEN_MARGIN, type=int, help="Screen horizontal size")
    argparser.add_argument("-s", "--seed", default=None, type=int, help="Random Seed")
    argparser.add_argument("-a", "--actors", default=NUM_ACTORS, type=int, help="Number of actors in the simulation")
    argparser.add_argument(
        "--timesteps", default=MAX_TIMESTEPS, type=int, help="Number of timesteps/episodes to run the simulation"
    )
    argparser.add_argument(
        "-i", "--instances", default=NUM_INSTANCES, type=int, help="Number of instances to run at one time"
    )
    argparser.add_argument("--prefix", default="", help="Prefix on results file name")
    argparser.add_argument("-g", "--generator", default=DEFAULT_GENERATOR_NAME, help="Random Generator to use")
    argparser.add_argument("--simulation_speed", default=SIMULATION_SPEED, type=float, help="Simulator speed")
    argparser.add_argument("-t", "--tick_time", default=TICK_TIME, type=float, help="Length of Simulation Time Step")
    argparser.add_argument("--show-sim", action="store_true", help="Display the simulation window")
    argparser.add_argument("--debug", action="store_true", help="Dummy mode -- just display the env")
    argparser.add_argument("--demo", action="store_true", help="Demo mode: show the model doing its thing")
    argparser.add_argument("--multipass", action="store_true", help="Run multiple environments simultaneously")
    argparser.add_argument("--tracks", default=None, type=str, help="Load pedestrian tracks from file")

    args = argparser.parse_args()

    main(args)
