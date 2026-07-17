# Ackermann specs -- roughly based on a typical sedan
from math import cos, sin, tan
import numpy as np

LENGTH = 2.2  # 2.875 is actual wheelbase, total length is 4.694
WIDTH = 1.849

MAX_V = 10
MIN_V = -10
MAX_A = 5.0
MIN_A = -4.3

MAX_DELTA = np.pi / 5.0
MIN_DELTA = -np.pi / 5.0

MAX_W = np.pi / 2.0
MIN_W = -np.pi / 2.0


# CARLA allows direct control of the steering angle so we have a 4 state model: x, y, v, theta
class Ackermann4:
    CONTROL_LEN = 2  # a, delta
    STATE_LEN = 4  # x, y, v, theta

    def __init__(self, length=None, width=None, max_delta=None, scale=1.0) -> None:
        self.scale = scale
        if length is None:
            self.L = LENGTH * self.scale
        else:
            self.L = length

        if width is None:
            self.W = WIDTH * self.scale
        else:
            self.W = width

        # set defaults limit on velocity and turning
        self.min_v = MIN_V
        self.max_v = MAX_V
        self.min_a = MIN_A
        self.max_a = MAX_A
        self.max_delta = MAX_DELTA if max_delta is None else float(max_delta)
        self.min_delta = -self.max_delta

    #   Step Function
    def ode(self, state, control):
        if control[1] > self.max_delta:
            control[1] = self.max_delta
        elif control[1] < self.min_delta:
            control[1] = self.min_delta

        dx = state[2] * cos(state[3])
        dy = state[2] * sin(state[3])
        dv = control[0]
        dtheta = state[2] * tan(control[1]) / self.L

        return np.array([dx, dy, dv, dtheta])


class Ackermann5:
    CONTROL_LEN = 2  # a, omega
    STATE_LEN = 5  # x, y, v, theta, delta

    def __init__(self, length=None, scale=1.0) -> None:
        self.scale = scale
        if length is None:
            self.L = LENGTH * self.scale
        else:
            self.L = length

        # set defaults limit on velocity and turning
        self.min_v = MIN_V
        self.max_v = MAX_V
        self.min_a = MIN_A
        self.max_a = MAX_A
        self.max_delta = MAX_DELTA
        self.min_delta = MIN_DELTA
        self.min_w = MIN_W
        self.max_w = MAX_W

    #   Step Function
    def ode(self, state, control):
        dx = state[2] * cos(state[3])
        dy = state[2] * sin(state[3])
        dv = control[0]
        dtheta = state[2] * tan(state[4]) / self.L
        ddelta = control[1]

        # return csi.vertcat(dx, dy, dv, dtheta, ddelta)
        return np.array([dx, dy, dv, dtheta, ddelta])
