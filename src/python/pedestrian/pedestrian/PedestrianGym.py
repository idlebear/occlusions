import gym
from gym import spaces
import pygame
import cv2

from config import *
from simulation import Simulation

import numpy as np


class PedestrianEnv(gym.Env):
    """
    Occlusion environment, reshaped for OpenAI Gym environment -- basically a thin wrapper of the Simulation code
    """

    def __init__(self, **kwargs) -> None:

        super().__init__()

        # set defaults, then check for updated values
        num_actors = 5
        seed = 42
        tracks = None
        self.show_sim = False
        if kwargs is not None:
            try:
                num_actors = kwargs["num_actors"]
            except KeyError:
                pass
            try:
                seed = kwargs["seed"]
            except KeyError:
                pass
            try:
                tracks = kwargs["tracks"]
            except KeyError:
                pass

        pygame.init()
        size = (SCREEN_WIDTH, SCREEN_HEIGHT)
        self.screen = None
        self.surface = pygame.Surface(size, pygame.SRCALPHA)
        pygame.display.set_caption("Simulation")
        pygame.font.init()

        generator_args = GENERATOR_ARGS
        generator_args["seed"] = seed

        # Initialize the simulation
        self.sim = Simulation(
            screen=self.surface,
            num_actors=num_actors,
            tracks=tracks,
            generator_name="uniform",
            generator_args=generator_args,
            tick_time=TICK_TIME,
            ego_start=[0.05,[0.25,0.75]],
            ego_goal=[0.95,[0.25,0.75]],
        )

        # define the action space -- the car can go from full acceleration (100%/1) to full brake (-100%/-1) and can steer +/- 100%
        self.action_space = spaces.Box(
            low=np.array([-1, -1]).astype(np.float32), high=np.array([1, 1]).astype(np.float32), dtype=np.float32
        )

        # similarly the observation space is the current observation rendered as an array
        self.observation_space = spaces.Box(low=np.float32(0.0), high=np.float32(1.0), shape=self.sim.observation_shape)

    def step(self, action):
        """
        Step the environment, taking a single action and returning the resulting reward and new state.
        """
        return self.sim.tick(action)

    def reset(self):
        return self.sim.reset()

    def render(self, mode="human", close=False):
        self.sim.render()

        if mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None

            if self.screen is None:
                self.screen = pygame.display.set_mode((SCREEN_HEIGHT, SCREEN_WIDTH))
            self.screen.blit(self.surface, (0, 0))
            pygame.display.flip()

        image = pygame.surfarray.array3d(self.surface).swapaxes(0, 1)
        return cv2.resize(image, dsize=(500, 500), interpolation=cv2.INTER_CUBIC)
