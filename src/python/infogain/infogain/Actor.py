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

import ModelParameters.GenericCar as GenericCar
import ModelParameters.Ackermann as Ackermann
import ModelParameters.SkidSteer as SkidSteer


class STATE(IntEnum):
    X = 0
    Y = 1
    THETA = 2
    VELOCITY = 3
    DELTA = 4


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

        self.reached_goal = False
        self.collided = False

        self.max_v = np.inf
        self.min_v = -np.inf
        self.min_a = -np.inf
        self.max_a = np.inf
        self.max_omega = 2 * np.pi / 3.0

        self.colour = colour
        self.outline_colour = outline_colour
        self.scale = scale
        self.ratio = ratio

        self.goal = goal

        self.poly_def = np.array(
            [
                [2.2, 0.8, 1],
                [-2.2, 0.8, 1],
                [-2.2, -0.8, 1],
                [2.2, -0.8, 1],
                [2.45, 0, 1],
                [2.2, 0.8, 1],
            ]
        ).T

        self.update_bounding_box()

    def distance_to(self, pos):
        return np.linalg.norm(self.x[0:2] - pos[0:2])

    def is_colliding(self, other):
        other_pts = [
            [other.bounding_box[0], other.bounding_box[1]],
            [other.bounding_box[0], other.bounding_box[3]],
            [other.bounding_box[2], other.bounding_box[1]],
            [other.bounding_box[2], other.bounding_box[3]],
        ]

        for pt in other_pts:
            if pt[0] >= self.bounding_box[0] and pt[0] <= self.bounding_box[2]:
                if pt[1] >= self.bounding_box[1] and pt[1] <= self.bounding_box[3]:
                    return True

        return False

    def __move(self, dt):
        """move towards the goal"""
        dx = self.u[0] * np.cos(self.x[STATE.THETA]) * dt
        dy = self.u[0] * np.sin(self.x[STATE.THETA]) * dt
        dtheta = self.u[1] * dt

        self.x[STATE.X] += dx
        self.x[STATE.Y] += dy
        self.x[STATE.THETA] += dtheta
        self.x[STATE.VELOCITY] = self.u[0]

    def set_goal(self, goal):
        self.goal = goal
        self.reached_goal = False

    def update_bounding_box(self):
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
        self.extent = max(np.linalg.norm(min_d - self.x[0:2]), np.linalg.norm(max_d - self.x[0:2]))

    def get_extent(self):
        return self.extent

    def set_control(self, u):
        self.u[0] = np.clip(u[0], self.max_v, self.max_v)
        self.u[1] = np.clip(u[1], -self.max_omega, self.max_omega)

    def tick(self, dt=TICK_TIME):
        """a time step"""
        self.__move(dt)
        self.update_bounding_box()

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
        self.x[Actor.STATE.V] = 0
        self.u[0] = 0
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
            "u": self.u,
            "collided": self.collided,
            "poly": self.get_poly(),
            "bbox": self.bounding_box,
            "extent": self.extent,
        }
        return state

    def get_outline(self):
        return self.get_poly()

    def get_extent(self):
        return self.extent

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
        image_scale=1,  # pixels / m
    ):
        super().__init__(
            id,
            x=x,
            goal=goal,
            colour=colour,
            outline_colour=outline_colour,
            scale=scale,
        )
        self.max_v = GenericCar.MAX_V
        self.min_v = GenericCar.MIN_V

        self.poly_def = np.array(
            [
                [2.5, 1.0, 1],
                [-2.5, 1.0, 1],
                [-2.5, -1.0, 1],
                [2.5, -1.0, 1],
                [2.75, 0, 1],
                [2.5, 1.0, 1],
            ]
        ).T

        self.actor_image = pygame.image.load(f"assets/{image_prefix}car.svg")
        self.actor_image = pygame.transform.scale(
            self.actor_image, (GenericCar.LENGTH * image_scale, GenericCar.WIDTH * image_scale)
        )

        super().update_bounding_box()

    def get_image(self):
        return self.actor_image


# Velocity based bicycle model
class AckermanCar(Actor):
    def __init__(
        self,
        id=0,
        x=[0, 0, 0, 0, 0, 0],
        goal=None,
        colour="orange",
        outline_colour="darkorange",
        scale=1,
        image_prefix="",
        image_scale=1.0,
    ):
        super().__init__(
            id,
            x=x,
            goal=goal,
            colour=colour,
            outline_colour=outline_colour,
            scale=scale,
        )

        self.max_v = Ackermann.MAX_V
        self.min_v = Ackermann.MIN_V
        self.max_a = Ackermann.MAX_A
        self.min_a = Ackermann.MIN_A
        self.max_omega = Ackermann.MAX_W
        self.max_delta = Ackermann.MAX_DELTA

        self.poly_def = np.array(
            [
                [2.5, 1.0, 1],
                [-2.5, 1.0, 1],
                [-2.5, -1.0, 1],
                [2.5, -1.0, 1],
                [2.75, 0, 1],
                [2.5, 1.0, 1],
            ]
        ).T

        self.actor_image = pygame.image.load(f"assets/{image_prefix}car.svg")
        self.actor_image = pygame.transform.scale(
            self.actor_image, (Ackermann.LENGTH * image_scale, Ackermann.WIDTH * image_scale)
        )

        super().update_bounding_box()

    def tick(self, dt=TICK_TIME):
        """a time step"""
        self.__move(dt)
        super().update_bounding_box()

        if self.goal is not None:
            if not self.reached_goal:
                dist = np.linalg.norm(self.goal - [self.x[STATE.VX], self.x[STATE.VY]])
                if abs(dist - self.speed) < 0:
                    self.reached_goal = True

    def __move(self, dt):
        self.x[STATE.X] = self.x[STATE.X] + self.x[STATE.VELOCITY] * np.cos(self.x[STATE.THETA]) * dt
        self.x[STATE.Y] = self.x[STATE.Y] + self.x[STATE.VELOCITY] * np.sin(self.x[STATE.THETA]) * dt
        self.x[STATE.THETA] = (
            self.x[STATE.THETA] + self.x[STATE.VELOCITY] * np.tan(self.x[STATE.DELTA]) / Ackermann.L * dt
        )
        # self.x[STATE.THETA] = self.x[STATE.THETA] + self.x[STATE.VELOCITY] * np.tan(self.u[1]) / Ackermann.L * dt

        self.x[STATE.VELOCITY] += self.u[0] * dt
        self.x[STATE.VELOCITY] = np.clip(self.x[STATE.VELOCITY], self.min_v, self.max_v)
        self.x[STATE.DELTA] += self.u[1] * dt
        self.x[STATE.DELTA] = np.clip(self.x[STATE.DELTA], -self.max_delta, self.max_delta)

    def set_control(self, u):
        self.u[0] = np.clip(u[0], self.min_a, self.max_a)
        self.u[1] = np.clip(u[1], -self.max_omega, self.max_omega)
        # self.u[1] = np.clip(u[1], -self.max_delta, self.max_delta)

    def get_image(self):
        return self.actor_image


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
                [0.40, 0, 1],
                [0, 0.2, 1],
                [-40, 0, 1],
                [0, -0.2, 1],
                [0.40, 0, 1],
            ]
        ).T

        super().update_bounding_box()


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
                [0.5, 0.5 * self.ratio, 1],
                [-0.5, 0.5 * self.ratio, 1],
                [-0.5, -0.5 * self.ratio, 1],
                [0.5, -0.5 * self.ratio, 1],
                [0.5, 0.5 * self.ratio, 1],
            ]
        ).T

        super().update_bounding_box()


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
                [0.5, 0.5 * self.ratio, 1],
                [-0.5, 0.5 * self.ratio, 1],
                [-0.5, -0.5 * self.ratio, 1],
                [0.5, -0.5 * self.ratio, 1],
                [0.5, 0.5 * self.ratio, 1],
            ]
        ).T

        super().update_bounding_box()

    def get_outline(self):
        return None

    def is_real(self):
        return False
