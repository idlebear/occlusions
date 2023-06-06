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
from simulation import get_location


from config import EGO_X_OFFSET, EGO_Y_OFFSET, GRID_HEIGHT, GRID_WIDTH, GRID_RESOLUTION
from config import GRID_ORIGIN_X_OFFSET, GRID_ORIGIN_Y_OFFSET


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


    def execute(self, ego, actors, visibility=None, tick_time=0.1, current_time=0, max_solver_time=30):
        """basic policy

        Args:
            ego: the actor controlled by this policy
            actors (_type_): other known actors in the environment
        """

        # update the location of the grid
        self.ego_pos = ego.pos
        self.ego_poly = ego.get_poly()
        self.grid.move_origin( (ego.pos[0]+GRID_ORIGIN_X_OFFSET, GRID_ORIGIN_Y_OFFSET) )
        self.grid.decay( 0.8 )

        if visibility is not None:
            self.grid.update( ego.pos, visibility=visibility, agents=actors )

        forecast = flow(self.grid.get_probability_map(), 
                        self.grid.get_velocity_map(), 
                        scale=GRID_RESOLUTION,
                        timesteps=FORECAST_COUNT, 
                        dt=FORECAST_DT, mode='bilinear')
        
        if DEBUG:
            # draw the probability and velocity grid
            for i, (prob,v) in enumerate(forecast):
                map_img = Image.fromarray(np.flipud(((1-prob)*255.0).astype(np.uint8))).convert('RGB')
                self.maps[i].set_data(map_img)

            self.map_fig.canvas.draw()
            self.map_fig.canvas.flush_events()

        # if ego.pos[1] > 0.1:
        #     ego.turn( -np.pi / 12.0, tick_time )
        # elif ego.pos[1] < -0.1:
        #     ego.turn( np.pi / 12.0, tick_time )
        # else:
        #     ego.turn( np.random.random() * np.pi/12.0 - np.pi/24.0, tick_time )

        steps = 20
        # u = [ (np.random.random() * 0.2 - 0.1, np.random.random() * np.pi/6.0 - np.pi/12.0) for s in range(steps) ]
        u = [ (0, np.pi/6.0) for s in range(steps//4) ] + [ (0, -np.pi/6.0) for s in range(steps//2)] + [ (0, np.pi/6.0) for s in range(steps//4) ] 
        self.rollout = ego.project( u, 0.1 )

        return False

    def draw(self):
        for (pos,orientation) in self.rollout:
            actor_pos = get_location(origin=self.ego_pos, location=pos)

            rot = np.array(
                [
                    [np.cos(orientation), np.sin(orientation)],
                    [-np.sin(orientation), np.cos(orientation)],
                ]
            )
            pts = (rot @ self.ego_poly.T).T + actor_pos
            self.window.draw_polygon(outline_colour='darkorange', fill_colour='peachpuff', points=pts)



def get_policy_fn(generator, policy_args=None):
    return AwesomePolicy(generator=generator, policy_args=policy_args)
