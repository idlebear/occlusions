from config import (
    DISTANCE_TOLERANCE,
    TICK_TIME,
    ACTOR_PATH_WIDTH,
    MAX_CAR_SPEED,
    MAX_PEDESTRIAN_SPEED,
    CAR_IMAGE_LENGTH,
    CAR_IMAGE_WIDTH,
)
from math import sqrt, atan2, cos, sin
import numpy as np
import pygame
from enum import Enum, IntEnum


class STATE(IntEnum):
    X = 0
    Y = 1
    THETA = 2
    VX = 3
    VY = 4
    DELTA = 5


class Actor:
    def __init__(
        self,
        id=0,
        x=[0, 0, 0, 0, 0],
        goal=None,
        colour="grey",
        outline_colour="darkgrey",
        scale=1.0,
        ratio=1.0,
        params=None,
    ):
        self.id = id

        # Acceleration model -- state is [x,y,dx,dy,theta], with control [accel,omega] and
        # then state update is
        # x' = x + dx dt
        # y' = y + dy dt
        # theta' = theta + omega dt
        # dx' = dx + a cos(theta) dt
        # dy' = dy + a sin(theta) dt

        # unless specified, actor is initially at rest and no control is applied
        self.x = np.array(x)
        self.u = np.array([0.0, 0.0])
        self.speed = np.round(np.sqrt(self.x[STATE.VX] * self.x[STATE.VX] + self.x[STATE.VY] * self.x[STATE.VY]), 5)

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
        self.ratio = ratio

        self.goal = goal

        self.poly_def = np.array(
            [
                [0.01, 0.01, 1],
                [-0.01, 0.01, 1],
                [-0.01, -0.01, 1],
                [0.01, -0.01, 1],
                [0.01, 0.01, 1],
            ]
        ).T

        self.__update_bounding_box()

    def distance_to(self, pos):
        return np.linalg.norm(self.x[0:2] - pos[0:2])

    def __move(self, dt):
        """move towards the goal"""

        A = np.array(
            [
                [1, 0, 0, dt, 0],
                [0, 1, 0, 0, dt],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ]
        )
        B0 = np.array([0, 0, 0, np.cos(self.x[STATE.THETA]), np.sin(self.x[STATE.THETA])]) * dt
        B1 = np.array([0, 0, 1, 0, 0]) * dt

        self.x = A @ self.x + B0 * self.u[0] + B1 * self.u[1]
        # BUGBUG -- may need to add bounds to constrain the measured theta -- but-- that may
        #           lead to jumps in position...

    def set_goal(self, goal):
        self.goal = goal
        self.reached_goal = False
        # if goal is not None:
        #     dir = self.goal - self.pos
        #     self.orientation = atan2(dir[1], dir[0])
        # else:
        #     self.orientation = 0

    def __update_bounding_box(self):
        self.rot_bw = np.array(
            [
                [np.cos(self.x[STATE.THETA]), -np.sin(self.x[STATE.THETA]), 0],
                [np.sin(self.x[STATE.THETA]), np.cos(self.x[STATE.THETA]), 0],
                [0, 0, 1],
            ]
        )

        poly = self.get_poly()
        min_d = np.min(poly, axis=0)
        max_d = np.max(poly, axis=0)
        self.bounding_box = (*min_d, *max_d)

    def set_control(self, u):
        self.u[0] = np.clip(u[0], -self.max_brake, self.max_accel)
        self.u[1] = np.clip(u[1], -self.max_omega, self.max_omega)

    def tick(self, dt=TICK_TIME):
        """a time step"""
        self.__move(dt)

        self.__update_bounding_box()

        self.speed = np.round(np.sqrt(self.x[STATE.VX] * self.x[STATE.VX] + self.x[STATE.VY] * self.x[STATE.VY]), 5)

        if self.goal is not None:
            if not self.reached_goal:
                dist = np.linalg.norm(self.goal - [self.x[STATE.X], self.x[STATE.Y]])
                if abs(dist - self.speed) < 0:
                    self.reached_goal = True

    def at_goal(self):
        return self.reached_goal

    def get_size(self):
        return np.array(
            [
                self.bounding_box[2] - self.bounding_box[0],
                self.bounding_box[3] - self.bounding_box[1],
            ]
        )

    def contains(self, loc):
        return (
            loc[0] >= self.bounding_box[0]
            and loc[1] >= self.bounding_box[1]
            and loc[0] <= self.bounding_box[2]
            and loc[1] <= self.bounding_box[3]
        )

    def set_collided(self, colour="black"):
        self.colour = colour
        self.speed = 0
        self.x[STATE.VX] = 0
        self.x[STATE.VY] = 0
        self.collided = True

    def project(self, u, dt):
        """
        Project a future position based on a supplied control and state
        """

        virt_self = type(self)(id=self.id, x=self.x, goal=self.goal)
        states = []

        for control in u:
            virt_self.set_control(control)
            virt_self.tick(dt)
            states.append(list(virt_self.x))

        return states

    def get_poly(self):
        return (self.rot_bw @ self.poly_def)[0:2, ...].T * self.scale + self.x[0:2]

    def get_state(self):
        state = {
            "x": self.x,
            "speed": self.speed,
            "u": self.u,
            "collided": self.collided,
            "poly": self.get_poly(),
            "bbox": self.bounding_box,
        }
        return state

    def get_outline(self):
        return self.get_poly()

    def get_image(self):
        return None

    def is_real(self):
        return True


class Car(Actor):
    def __init__(
        self,
        id=0,
        x=[0, 0, 0, 0, 0],
        goal=None,
        colour="lightblue",
        outline_colour="dodgerblue",
        scale=1,
        image_prefix="",
    ):
        super().__init__(
            id,
            x=x,
            goal=goal,
            colour=colour,
            outline_colour=outline_colour,
            scale=scale,
        )
        self.max_v = MAX_CAR_SPEED
        self.min_v = 0
        self.max_brake = 1.5
        self.max_accel = 1.5

        self.poly_def = np.array(
            [
                [0.025, 0, 1],
                [-0.025, 0.02, 1],
                [-0.01, 0, 1],
                [-0.025, -0.02, 1],
                [0.025, 0, 1],
            ]
        ).T

        # art objects
        self.actor_image = pygame.image.load(f"assets/{image_prefix}car.svg")
        self.actor_image = pygame.transform.scale(self.actor_image, (CAR_IMAGE_LENGTH, CAR_IMAGE_WIDTH))

    @staticmethod
    def check_width(scale):
        return 0.05 * scale

    def get_image(self):
        return self.actor_image


# BUGBUG -- hacky car copy controlled by velocity only
class VelocityCar(Actor):
    def __init__(
        self,
        id=0,
        x=[0, 0, 0, 0, 0],
        goal=None,
        colour="orange",
        outline_colour="darkorange",
        scale=1,
        image_prefix="",
        params=None,
    ):
        super().__init__(
            id,
            x=x,
            goal=goal,
            colour=colour,
            outline_colour=outline_colour,
            scale=scale,
        )

        if params is None:
            self.max_omega = 5
        else:
            self.max_omega = params["max_omega"]

        self.poly_def = np.array(
            [
                [0.025, 0.02, 1],
                [-0.025, 0.02, 1],
                [-0.025, -0.02, 1],
                [0.025, -0.02, 1],
                [0.03, 0, 1],
                [0.025, 0.02, 1],
            ]
        ).T

        # art objects
        self.actor_image = pygame.image.load(f"assets/{image_prefix}car.svg")
        self.actor_image = pygame.transform.scale(self.actor_image, (CAR_IMAGE_LENGTH, CAR_IMAGE_WIDTH))

    def tick(self, dt=TICK_TIME):
        """a time step"""
        self.__move(dt)

        self.__update_bounding_box()
        self.speed = np.round(np.sqrt(self.x[STATE.VX] * self.x[STATE.VX] + self.x[STATE.VY] * self.x[STATE.VY]), 5)

        if self.goal is not None:
            if not self.reached_goal:
                dist = np.linalg.norm(self.goal - [self.x[STATE.VX], self.x[STATE.VY]])
                if abs(dist - self.speed) < 0:
                    self.reached_goal = True

    def __move(self, dt):
        self.speed = self.u[0]

        self.x[STATE.VX] = self.speed * np.cos(self.x[STATE.THETA])
        self.x[STATE.VY] = self.speed * np.sin(self.x[STATE.THETA])
        self.x[STATE.X] = self.x[STATE.X] + self.x[STATE.VX] * dt
        self.x[STATE.Y] = self.x[STATE.Y] + self.x[STATE.VY] * dt
        self.x[STATE.THETA] = self.x[STATE.THETA] + self.u[1] * dt

    def set_control(self, u):
        self.u[0] = np.clip(u[0], self.min_v, self.max_v)
        self.u[1] = np.clip(u[1], -self.max_omega, self.max_omega)

    def __update_bounding_box(self):
        self.rot_bw = np.array(
            [
                [np.cos(self.x[STATE.THETA]), -np.sin(self.x[STATE.THETA]), 0],
                [np.sin(self.x[STATE.THETA]), np.cos(self.x[STATE.THETA]), 0],
                [0, 0, 1],
            ]
        )

        poly = self.get_poly()
        min_d = np.min(poly, axis=0)
        max_d = np.max(poly, axis=0)
        self.bounding_box = (*min_d, *max_d)

    @staticmethod
    def check_width(scale):
        return 0.05 * scale

    def get_image(self):
        return self.actor_image


# Basic skid-steer model, as an ideallized differential drive
class SkidSteer(Actor):
    def __init__(
        self,
        id=0,
        x=[0, 0, 0, 0, 0],
        goal=None,
        colour="orange",
        outline_colour="darkorange",
        scale=1,
        image_prefix="",
        params=None,
    ):
        super().__init__(
            id,
            x=x,
            goal=goal,
            colour=colour,
            outline_colour=outline_colour,
            scale=scale,
        )

        if params is None:
            self.b = 1
            self.max_omega = 5
        else:
            self.b = params["b"]
            self.max_omega = params["max_omega"]

        self.poly_def = np.array(
            [
                [0.025, 0.02, 1],
                [-0.025, 0.02, 1],
                [-0.025, -0.02, 1],
                [0.025, -0.02, 1],
                [0.03, 0, 1],
                [0.025, 0.02, 1],
            ]
        ).T

        # art objects
        self.actor_image = pygame.image.load(f"assets/{image_prefix}car.svg")
        self.actor_image = pygame.transform.scale(self.actor_image, (CAR_IMAGE_LENGTH, CAR_IMAGE_WIDTH))

    def tick(self, dt=TICK_TIME):
        """a time step"""
        self.__move(dt)

        self.__update_bounding_box()
        self.speed = np.round(np.sqrt(self.x[STATE.VX] * self.x[STATE.VX] + self.x[STATE.VX] * self.x[STATE.VY]), 5)

        if self.goal is not None:
            if not self.reached_goal:
                dist = np.linalg.norm(self.goal - [self.x[STATE.X], self.x[STATE.Y]])
                if abs(dist - self.speed) < 0:
                    self.reached_goal = True

    def __move(self, dt):
        self.speed = (self.u[0] + self.u[1]) / 2
        w = (self.u[0] - self.u[1]) / self.b

        self.x[STATE.VX] = self.speed * np.cos(self.x[STATE.THETA])
        self.x[STATE.VY] = self.speed * np.sin(self.x[STATE.THETA])
        self.x[STATE.X] = self.x[STATE.X] + self.x[STATE.VX] * dt
        self.x[STATE.Y] = self.x[STATE.Y] + self.x[STATE.VY] * dt
        self.x[STATE.THETA] = self.x[STATE.THETA] + w * dt

    def set_control(self, u):
        self.u[0] = np.clip(u[0], -self.max_omega, self.max_omega)
        self.u[1] = np.clip(u[1], -self.max_omega, self.max_omega)

    def __update_bounding_box(self):
        self.rot_bw = np.array(
            [
                [np.cos(self.x[STATE.THETA]), -np.sin(self.x[STATE.THETA]), 0],
                [np.sin(self.x[STATE.THETA]), np.cos(self.x[STATE.THETA]), 0],
                [0, 0, 1],
            ]
        )

        poly = self.get_poly()
        min_d = np.min(poly, axis=0)
        max_d = np.max(poly, axis=0)
        self.bounding_box = (*min_d, *max_d)

    def _poly(self):
        return (
            np.array(
                [
                    [0.025, 0],
                    [-0.025, 0.02],
                    [-0.01, 0],
                    [-0.025, -0.02],
                    [0.025, 0],
                ]
            )
            * self.scale
        )

    def get_image(self):
        return None

    @staticmethod
    def check_width(scale):
        return 0.05 * scale


class Pedestrian(Actor):
    def __init__(
        self,
        id=0,
        x=[0, 0, 0, 0, 0],
        goal=None,
        colour="blue",
        outline_colour="darkblue",
        scale=1,
    ):
        super().__init__(
            id,
            x=x,
            goal=goal,
            colour=colour,
            outline_colour=outline_colour,
            scale=scale,
        )
        self.max_v = MAX_PEDESTRIAN_SPEED
        self.min_v = 0
        self.max_brake = 0.75
        self.max_accel = 0.75

        self.poly_def = np.array(
            [
                [0.01, 0, 1],
                [0, 0.015, 1],
                [-0.01, 0, 1],
                [0, -0.015, 1],
                [0.01, 0, 1],
            ]
        ).T

    @staticmethod
    def check_width(scale):
        return 0.035 * scale


class Obstacle(Actor):
    def __init__(
        self,
        id=0,
        x=[0, 0, 0, 0, 0],
        goal=None,
        colour="grey",
        outline_colour="darkgrey",
        scale=1,
        ratio=1,
    ):
        super().__init__(
            id,
            x=x,
            goal=goal,
            colour=colour,
            outline_colour=outline_colour,
            scale=scale,
        )

        self.max_v = 0
        self.min_v = 0
        self.max_brake = 0
        self.max_accel = 0

        self.poly_def = np.array(
            [
                [0.01, 0.01 * self.ratio, 1],
                [-0.01, 0.01 * self.ratio, 1],
                [-0.01, -0.01 * self.ratio, 1],
                [0.01, -0.01 * self.ratio, 1],
                [0.01, 0.01 * self.ratio, 1],
            ]
        ).T

    @staticmethod
    def check_width(scale):
        return 0.02 * scale


class Blank(Actor):
    def __init__(
        self,
        id=0,
        x=[0, 0, 0, 0, 0],
        goal=None,
        colour="white",
        outline_colour="darkgrey",
        scale=1,
    ):
        super().__init__(
            id,
            x=x,
            goal=goal,
            colour=colour,
            outline_colour=outline_colour,
            scale=scale,
        )
        self.max_v = 0
        self.min_v = 0
        self.max_brake = 0
        self.max_accel = 0

        self.poly_def = np.array(
            [
                [0.01, 0.01 * self.ratio, 1],
                [-0.01, 0.01 * self.ratio, 1],
                [-0.01, -0.01 * self.ratio, 1],
                [0.01, -0.01 * self.ratio, 1],
                [0.01, 0.01 * self.ratio, 1],
            ]
        ).T

    @staticmethod
    def check_width(scale):
        return 0.02 * scale

    def get_outline(self):
        return None

    def is_real(self):
        return False
