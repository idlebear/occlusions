'''
Basic policy that does nothing
'''
from copy import deepcopy
from time import time
from numpy import inf

from policies.policy import BasePolicy
from policies.VelocityGrid import VelocityGrid

from config import EGO_X_OFFSET, EGO_Y_OFFSET

GRID_HEIGHT = 0.5
GRID_WIDTH = 1.0
GRID_RESOLUTION = 0.02
GRID_ORIGIN_Y_OFFSET = -GRID_HEIGHT / 2
GRID_ORIGIN_X_OFFSET = EGO_X_OFFSET

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

        self.grid = VelocityGrid( height=GRID_HEIGHT, width=GRID_WIDTH, resolution=GRID_RESOLUTION, origin=(GRID_ORIGIN_Y_OFFSET, GRID_ORIGIN_Y_OFFSET), debug=True )

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

        return False

    def draw(self):
        pass


def get_policy_fn(generator, policy_args=None):
    return AwesomePolicy(generator=generator, policy_args=policy_args)
