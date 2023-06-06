'''
Collision avoidance policy
'''
from copy import deepcopy
from time import time
import numpy as np

from  policies.policy import BasePolicy
from simulation import get_location

import config


class RandomPolicy(BasePolicy):
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

    def execute(self, ego, actors, current_time=0, max_solver_time=30, dt=0.1):
        """basic policy

        Args:
            ego: the actor controlled by this policy
            actors (_type_): other known actors in the environment
        """

        # the stopping distance
        self.d_s = ego.speed / ego.max_brake

        if ego.speed:
            self.d_o_car = self.d_s * config.OPPONENT_CAR_SPEED / ego.speed
            self.d_o_ped = self.d_s * config.OPPONENT_PEDESTRIAN_SPEED / ego.speed
        else:
            self.d_o_car = 0
            self.d_o_ped = 0

        self.car_pts = [
            get_location(origin=ego.pos, location=ego.pos),
            get_location(origin=ego.pos, location=(ego.pos[0]+self.d_s, ego.pos[1]+self.d_o_car)),
            get_location(origin=ego.pos, location=(ego.pos[0]+self.d_s, ego.pos[1]-self.d_o_car)),
        ]

        self.ped_pts = [
            get_location(origin=ego.pos, location=ego.pos),
            get_location(origin=ego.pos, location=(ego.pos[0]+self.d_s, ego.pos[1]+self.d_o_ped)),
            get_location(origin=ego.pos, location=(ego.pos[0]+self.d_s, ego.pos[1]-self.d_o_ped)),
        ]

        a = self.generator.uniform(-self.max_brake, self.max_accel)
        ego.accelerate(a, dt)

        return False

    def draw(self):
        if self.window is not None:
            self.window.draw_polygon(outline_colour=(255, 0, 0, 200), fill_colour=(200, 0, 0, 200), points=self.car_pts, use_transparency=True)
            self.window.draw_polygon(outline_colour=(200, 200, 0, 200), fill_colour=(150, 200, 0, 200), points=self.ped_pts, use_transparency=True)


def get_policy_fn(generator, policy_args=None):
    return RandomPolicy(generator, policy_args=policy_args)
