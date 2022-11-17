'''
Basic policy that does nothing
'''
from copy import deepcopy
from time import time
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image

from policies.flow.flow import flow
from policies.policy import BasePolicy
from policies.VelocityGrid import VelocityGrid


from config import EGO_X_OFFSET, EGO_Y_OFFSET

GRID_HEIGHT = 0.2
GRID_WIDTH = 1.0
GRID_RESOLUTION = 0.01
GRID_ORIGIN_Y_OFFSET = -GRID_HEIGHT / 2
GRID_ORIGIN_X_OFFSET = EGO_X_OFFSET

DEBUG = True
FORECAST_COUNT = 5
FORECAST_DT = 0.1

class AwesomePolicy(BasePolicy):
    def __init__(self, generator, policy_args=None) -> None:
        super().__init__(policy_args)

        self.generator = generator

        if policy_args is None:
            self.max_accel = 0
            self.max_brake = 0    # no brakes!
            self.max_v = 0
            self.max_v = 0    # no brakes!
            self.screen = None
        else:
            try:
                self.max_accel = policy_args['max_accel']
            except KeyError:
                self.max_accel = 0
            try:
                self.max_brake = policy_args['max_brake']
            except KeyError:
                self.max_brake = 0    # no brakes!

            try:
                self.max_v = policy_args['max_v']
            except KeyError:
                self.max_v = 0
            try:
                self.min_v = policy_args['min_v']
            except KeyError:
                self.min_v = 0    # no brakes!

            try:
                self.window = policy_args['window']
            except KeyError:
                self.window = None

        self.grid = VelocityGrid( height=GRID_HEIGHT, width=GRID_WIDTH, resolution=GRID_RESOLUTION, origin=(GRID_ORIGIN_Y_OFFSET, GRID_ORIGIN_Y_OFFSET) )

        if DEBUG:
            self.maps = []
            num_plots = FORECAST_COUNT + 1
            self.map_fig, self.map_ax = plt.subplots(num_plots, 1, figsize=(5, 15))
            H,W = self.grid.get_grid_size()
            for i in range(num_plots):
                self.maps.append( self.map_ax[i].imshow(np.zeros([H, W, 3], dtype=np.uint8)) )

            plt.show(block=False)


    def execute(self, ego, actors, visibility=None, current_time=0, max_solver_time=30):
        """basic policy

        Args:
            ego: the actor controlled by this policy
            actors (_type_): other known actors in the environment
        """

        # update the location of the grid
        self.grid.move_origin( (ego.pos[0]+GRID_ORIGIN_X_OFFSET, GRID_ORIGIN_Y_OFFSET) )
        self.grid.decay( 0.8 )

        if visibility is not None:
            self.grid.update( ego.pos, visibility=visibility, agents=actors )

        forecast = flow( self.grid.get_probability_map(), self.grid.get_velocity_map(), scale=GRID_RESOLUTION, timesteps=FORECAST_COUNT, dt=FORECAST_DT, mode='bilinear')
        
        if DEBUG:
            # draw the probability and velocity grid
            for i, (prob,v) in enumerate(forecast):
                map_img = Image.fromarray(np.flipud(((1-prob)*255.0).astype(np.uint8))).convert('RGB')
                self.maps[i].set_data(map_img)

            self.map_fig.canvas.draw()
            self.map_fig.canvas.flush_events()

        return False

    def draw(self):
        pass


def get_policy_fn(generator, policy_args=None):
    return AwesomePolicy(generator=generator, policy_args=policy_args)
