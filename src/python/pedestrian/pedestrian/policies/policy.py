'''
Basic policy that does nothing
'''
from copy import deepcopy
from time import time
from numpy import inf


class BasePolicy():
    def __init__(self, policy_args=None) -> None:
        pass

    def execute(self, ego, actors, current_time=0, max_solver_time=30):
        """basic policy

        Args:
            ego: the actor controlled by this policy
            actors (_type_): other known actors in the environment
        """

        # do nothing....

        return False


def get_policy_fn(generator, policy_args=None):
    return BasePolicy(policy_args=policy_args)
