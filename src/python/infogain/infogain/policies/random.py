'''
Collision avoidance policy
'''
from copy import deepcopy
from time import time
import numpy as np

from policies.policy import BasePolicy


class RandomPolicy(BasePolicy):
    def __init__(self, generator, policy_args=None) -> None:

        super().__init__(policy_args)

        self.generator = generator

        if policy_args is None:
            self.max_accel = 0
            self.max_brake = 0    # no brakes!
            self.max_v = 0
            self.max_v = 0    # no brakes!
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

    def execute(self, ego, actors, current_time=0, max_solver_time=30, dt=0.1):
        """basic policy

        Args:
            ego: the actor controlled by this policy
            actors (_type_): other known actors in the environment
        """

        a = self.generator.uniform(-self.max_brake, self.max_accel)
        ego.accelerate(a, dt)

        return False


def get_policy_fn(generator, policy_args=None):
    return RandomPolicy(generator, policy_args=policy_args)
