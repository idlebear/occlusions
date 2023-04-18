from config import DISTANCE_TOLERANCE, TICK_TIME, ACTOR_PATH_WIDTH, MAX_CAR_SPEED
from math import sqrt, atan2, cos, sin
import numpy as np
import pygame
from enum import Enum


class Actor:
    def __init__(self, id=0, pos=[0, 0], goal=None, speed=0, colour='grey', outline_colour='darkgrey', scale=1.0):
        self.id = id
        self.pos = pos

        self.reached_goal = False
        self.collided = False

        self.max_v = np.inf
        self.min_v = -np.inf
        self.max_brake = np.inf
        self.max_accel = np.inf
        self.max_omega = np.pi / 4.0

        self.colour = colour
        self.outline_colour = outline_colour
        self.scale = scale

        self.goal = goal
        if goal is not None:
            self.orientation = atan2((goal[1] - pos[1]), (goal[0] - pos[0]))
        else:
            # straight along the x-axis
            self.orientation = 0

        # unless specified, actor is initially at rest
        self.set_control([speed, 0])

        self.__update_bounding_box()

    def distance_to(self, pos):
        return np.linalg.norm(self.pos - pos)

    def __move(self, dt):
        """move towards the goal
        """
        self.orientation = self.orientation + self.omega * dt
        if self.orientation > np.pi:
            self.orientation -= 2 * np.pi
        elif self.orientation < -np.pi:
            self.orientation += 2 * np.pi

        self.pos = self.pos + self.v * dt
        self.__update_bounding_box()

        if self.goal is not None:
            if not self.reached_goal:
                dist = np.linalg.norm(self.goal - self.pos)
                if abs(dist - self.speed) < 0:
                    self.reached_goal = True

    def set_goal(self, goal):
        self.goal = goal
        self.reached_goal = False

    def __update_bounding_box(self):
        poly = self.get_poly()
        min_d = np.min(poly, axis=0)
        max_d = np.max(poly, axis=0)
        self.bounding_box = (*min_d, *max_d)

    def set_control(self, u):
        self.speed = np.clip(u[0], self.min_v, self.max_v)
        self.v = np.round(np.array([self.speed * np.cos(self.orientation), self.speed * np.sin(self.orientation)]), 5)
        self.omega = np.clip(u[1], -self.max_omega, self.max_omega)

    def tick(self, dt=TICK_TIME):
        """a time step
        """
        self.__move(dt)

    def at_goal(self):
        return self.reached_goal

    def _poly(self):
        return np.array([
            [0.01, 0.01],
            [-0.01, 0.01],
            [-0.01, -0.01],
            [0.01, -0.01],
            [0.01, 0.01],
        ]) * self.scale

    def get_size(self):
        return self.bounding_box[2] - self.bounding_box[0], self.bounding_box[3] - self.bounding_box[1]

    def contains(self, loc):
        return loc[0] >= self.bounding_box[0] and loc[1] >= self.bounding_box[1] and loc[0] <= self.bounding_box[2] and loc[1] <= self.bounding_box[3]

    def set_collided(self, colour='black'):
        self.colour = colour
        self.speed = 0
        self.collided = True

    def project(self, u, dt):
        '''
        Project a future position based on a supplied control and state
        '''

        virt_self = type(self)(id=self.id, pos=self.pos, goal=self.goal, speed=self.speed)
        states = []

        for (a, delta) in u:

            virt_self.accelerate(a, dt)
            virt_self.turn(delta, dt)
            virt_self.tick(dt)

            states.append((virt_self.pos, virt_self.orientation))

        return states

    def get_poly(self):
        self.rot = np.array(
            [
                [np.cos(self.orientation), -np.sin(self.orientation)],
                [np.sin(self.orientation), np.cos(self.orientation)],
            ]
        )
        poly = (self.rot @ self._poly().T).T + self.pos
        return poly

    def get_state(self):
        state = {
            'pos': self.pos,
            'v': self.v,
            'orientation': self.orientation,
            'speed': self.speed,
            'omega': self.omega,
            'collided': self.collided,
            'poly': self.get_poly(),
        }
        return state


class Car(Actor):
    def __init__(self, id=0, pos=[0, 0], goal=None, speed=1, colour='lightblue', outline_colour='dodgerblue', scale=1):
        super().__init__(id, pos=pos, goal=goal, speed=speed, colour=colour, outline_colour=outline_colour, scale=scale)
        self.max_v = MAX_CAR_SPEED
        self.min_v = 0
        self.max_brake = 1.5
        self.max_accel = 1.5

    def _poly(self):
        return np.array([
            [0.025, 0],
            [-0.025, 0.02],
            [-0.01, 0],
            [-0.025, -0.02],
            [0.025, 0],
        ]) * self.scale

    @staticmethod
    def check_width(scale):
        return 0.05 * scale


class Pedestrian(Actor):
    def __init__(self, id=0, pos=[0, 0], goal=None, speed=1, colour='blue', outline_colour='darkblue', scale=1):
        super().__init__(id, pos=pos, goal=goal, speed=speed, colour=colour, outline_colour=outline_colour, scale=scale)
        self.max_v = MAX_PEDESTRIAN_SPEED
        self.min_v = 0
        self.max_brake = 0.75
        self.max_accel = 0.75

    def _poly(self):
        return np.array([
            [0.01, 0],
            [0, 0.015],
            [-0.01, 0],
            [0, -0.015],
            [0.01, 0],
        ]) * self.scale

    @staticmethod
    def check_width(scale):
        return 0.035 * scale


class Obstacle(Actor):
    def __init__(self, id=0, pos=[0, 0], goal=None, speed=1, colour='grey', outline_colour='darkgrey', scale=1):
        super().__init__(id, pos, goal, speed, colour, outline_colour, scale)
        self.max_v = 0
        self.min_v = 0
        self.max_brake = 0
        self.max_accel = 0

    def _poly(self):
        return np.array([
            [0.01, 0.01],
            [-0.01, 0.01],
            [-0.01, -0.01],
            [0.01, -0.01],
            [0.01, 0.01],
        ]) * self.scale

    @staticmethod
    def check_width(scale):
        return 0.02 * scale


class Blank(Actor):
    def __init__(self, id=0, pos=[0, 0], goal=None, speed=1, colour='white', outline_colour='darkgrey', scale=1):
        super().__init__(id, pos, goal, speed, colour, outline_colour, scale)
        self.max_v = 0
        self.min_v = 0
        self.max_brake = 0
        self.max_accel = 0

    def _poly(self):
        return np.array([
            [0.01, 0.01],
            [-0.01, 0.01],
            [-0.01, -0.01],
            [0.01, -0.01],
            [0.01, 0.01],
        ]) * self.scale

    @staticmethod
    def check_width(scale):
        return 0.02 * scale
