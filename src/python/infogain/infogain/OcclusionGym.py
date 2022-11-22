import gym 
from gym import spaces

from config import *
from simulation import Simulation

import numpy as np

class OcclusionEnv( gym.Env ):
    '''
    Occlusion environment, reshaped for OpenAI Gym environment -- basically a thin wrapper of the Simulation code 
    '''

    def __init__(self, policy_name, policy_args=None, generator_name='uniform', generator_args=None, num_actors=1, pois_lambda=0.01, screen=None, service_time=SERVICE_TIME,
                speed=ACTOR_SPEED, margin=SCREEN_MARGIN, screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT,
                max_time=MAX_SIMULATION_TIME, max_tasks=MAX_SERVICED_TASKS, record_data=False, sectors=1, delivery_log=None) -> None :

        super().__init__()

        # Initialize the simulation
        self.sim = Simulation(
            policy_name=policy_name,
            policy_args=policy_args,
            generator_name=generator_name,
            generator_args=generator_args,
            num_actors=num_actors,
            pois_lambda=pois_lambda,
            screen=screen,
            max_time=max_time,
            tick_time=TICK_TIME
        )

        # define the action space -- the car can go from full acceleration (100%/1) to full brake (-100%/-1) and can steer +/- 100%
        self.action_space = spaces.Box( low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float16)

        # similarly the observation space is the current observation rendered as an array
        self.observation_space = spaces.Box( low=0.0, high=1.0, shape=self.sim.observation_shape )


    def step( self, action ):
        '''
        Step the environment, taking a single action and returning the resulting reward and new state.  
        '''
        return self.sim.tick( action )

    def reset( self ):
        self.sim.reset()
        # take a single zero step and return the resulting observation
        return self.sim.step( (0,0) )

    def render( self, mode='human', close=False ):
        self.sim.render()
        pass