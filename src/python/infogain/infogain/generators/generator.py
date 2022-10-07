import numpy as np
import random


class Generator:

    def __init__(self, **kwargs):
        try:
            self.min = kwargs['min']
        except KeyError:
            self.min = 0
        try:
            self.max = kwargs['max']
        except KeyError:
            self.max = 1
        try:
            self.seed = kwargs['seed']
        except KeyError:
            self.seed = None
        try:
            self.dim = kwargs['dim']
        except KeyError:
            self.dim = 2
        try:
            self.max_time = kwargs['max_time']
        except KeyError:
            self.max_time = None
        try:
            self.initial_tasks = kwargs['initial_tasks']
        except KeyError:
            self.initial_tasks = 0
        try:
            self.total_tasks = kwargs['total_tasks']
        except KeyError:
            self.total_tasks = 1000
        try:
            self.service_time = kwargs['service_time']
        except KeyError:
            self.service_time = 0

        self.reset()

    def reset(self):
        # TODO: Stopgap measure to set global seed here as well since some tasks are still using the random module.
        random.seed(self.seed)
        self.gen = np.random.default_rng(seed=self.seed)

    def uniform(self, low=0, high=1, size=None):
        return self.gen.uniform(low=low, high=high, size=size)

    def random(self, n=1):
        return self.gen.random(size=n)

    def poisson(self, lam):
        return self.gen.poisson(lam=lam)

    def normal(self, loc, scale):
        return self.gen.normal(loc=loc, scale=scale)
