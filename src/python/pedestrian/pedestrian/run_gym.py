import gymnasium as gym
import json
import datetime as dt
import argparse
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from PedestrianGym import PedestrianEnv
from config import *


def main(args):
    env_kwargs = {
        "num_actors": args.actors,
        "seed": args.seed,
        "tracks": args.tracks,
        "data_source": args.data_source,
        "sdd_processed_root": args.sdd_processed_root,
        "sdd_scene_id": args.sdd_scene_id,
        "limit_tracks": args.limit_tracks,
    }

    if args.debug:
        # set up debug version -- mo models
        env = PedestrianEnv(**env_kwargs)
    else:
        if args.multipass:
            env = make_vec_env(
                PedestrianEnv,
                n_envs=args.instances,
                env_kwargs=env_kwargs,
            )
        else:
            env = make_vec_env(
                PedestrianEnv,
                n_envs=args.instances,
                env_kwargs=env_kwargs,
            )

        if not args.demo:
            for i in range(10):
                model = PPO(
                    "MlpPolicy", env, verbose=1, tensorboard_log="./occlusion_log"
                )
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
        "--debug", action="store_true", help="Dummy mode -- just display the env"
    )
    argparser.add_argument(
        "--demo", action="store_true", help="Demo mode: show the model doing its thing"
    )
    argparser.add_argument(
        "--multipass",
        action="store_true",
        help="Run multiple environments simultaneously",
    )
    argparser.add_argument(
        "--tracks", default=None, type=str, help="Load pedestrian tracks from file"
    )
    argparser.add_argument(
        "--limit_tracks",
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
        help="Processed SDD root for --data-source sdd.",
    )
    argparser.add_argument(
        "--sdd-scene-id",
        type=int,
        default=None,
        help="Processed SDD scene id for --data-source sdd.",
    )

    args = argparser.parse_args()

    main(args)
