import gym
from gym import spaces
import pygame
import cv2

from config import *
from simulation import Simulation

import numpy as np

NUM_ACCEL_ACTIONS = 3
NUM_TURN_ACTIONS = 3


class OcclusionEnv(gym.Env):
    '''
    Occlusion environment, reshaped for OpenAI Gym environment -- basically a thin wrapper of the Simulation code
    '''

    def __init__(self, **kwargs) -> None:

        super().__init__()

        # set defaults, then check for updated values
        try:
            num_actors = kwargs['num_actors']
        except KeyError:
            num_actors = 5

        try:
            seed = kwargs['seed']
        except KeyError:
            seed = 42

        pygame.init()
        size = (SCREEN_WIDTH, SCREEN_HEIGHT)
        self.screen = None
        self.surface = pygame.Surface(size, pygame.SRCALPHA)
        pygame.display.set_caption('Simulation')
        pygame.font.init()

        generator_args = GENERATOR_ARGS
        generator_args['seed'] = seed

        # Initialize the simulation
        self.sim = Simulation(
            screen=self.surface,
            num_actors=num_actors,
            generator_name='uniform',
            generator_args=generator_args,
            tick_time=TICK_TIME
        )

        # define the action space -- the car can go from full acceleration (100%/1) to full brake (-100%/-1) and can steer +/- 100%
        # self.action_space = spaces.Box(low=np.array([-1, -1]).astype(int), high=np.array([1, 1]).astype(np.int), dtype=int)
        # Ok, so a continuous action space doesn't seem to train very well -- let's try cutting down the options

        self.action_space = spaces.Discrete(NUM_ACCEL_ACTIONS * NUM_TURN_ACTIONS)

        # similarly the observation space is the current observation rendered as an image with the top layer being the occupancy
        self.observation_space = spaces.Box(low=0, high=255, shape=self.sim.observation_shape, dtype=np.uint8)

    def step(self, action):
        '''
        Step the environment, taking a single action and returning the resulting reward and new state.
        '''
        action = (action//NUM_ACCEL_ACTIONS-1, action % NUM_ACCEL_ACTIONS-1)
        return self.sim.tick(action)

    def reset(self):
        return self.sim.reset()

    def render(self, mode='human', close=False):
        self.sim.render(debug=True)

        if mode == 'human':
            if self.screen is None:
                self.screen = pygame.display.set_mode((SCREEN_HEIGHT, SCREEN_WIDTH))
            self.screen.blit(self.surface, (0, 0))
            pygame.display.flip()

        else:
            image = pygame.surfarray.array3d(self.surface).swapaxes(0, 1)
            return cv2.resize(image, dsize=(500, 500), interpolation=cv2.INTER_CUBIC)
