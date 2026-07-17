"""
Implementation of a first order velocity model for the pedestrian controller
"""

MAX_V = 2.1
MIN_V = -2.1
MAX_A = 1.0
MIN_A = -1.0

import numpy as np


# CARLA allows direct control of the steering angle so we have a 4 state model: x, y, v, theta
class Unicycle:
    CONTROL_LEN = 2  # v, omega
    STATE_LEN = 4  # x, y, (v), theta

    def __init__(self, length=None, width=None) -> None:
        # set defaults limit on velocity and turning
        self.min_v = MIN_V
        self.max_v = MAX_V

    #   Step Function
    def ode(self, state, control):
        dx = control[0] * np.cos(state[3])
        dy = control[0] * np.sin(state[3])
        dtheta = control[1]

        return np.array([dx, dy, 0, dtheta])

    def predict(self, initial_state, control, steps, dt):
        state = initial_state
        for _ in range(steps):
            state = state + self.ode(state, control) * dt
            state[2] = np.clip(control[0], self.min_v, self.max_v)

        return state
