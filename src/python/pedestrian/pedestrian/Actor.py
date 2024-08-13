from config import EPSILON, TICK_TIME, OCC_PROB, TARGET_TOLERANCE, CONTROL_LIMITS
from math import sqrt, atan2, cos, sin
import numpy as np
import pygame
from enum import IntEnum

import controller.ModelParameters.Ackermann as Ackermann

from util.gaussian import create_gaussian
from util.uniform import create_uniform


class STATE(IntEnum):
    X = 0
    Y = 1
    VELOCITY = 2
    THETA = 3
    DELTA = 4


class Actor:
    serial = 0

    def __init__(
        self,
        id=0,
        x=[0, 0, 0, 0, 0],
        goal=None,
        colour="grey",
        outline_colour="darkgrey",
        track=None,
        resolution=0.1,
        radius=1.0,
        dt=0.1,
        actor_image=None,
        hidden_actor_image=None,
    ):
        self.id = id
        self.x = x
        self.u = None
        self.serial = Actor.serial
        Actor.serial += 1

        self.track = track
        self.goal = goal
        if self.track is not None:
            x, y, frame = self.track.pop(0)
            if len(self.track) > 0:
                goal_x, goal_y, _ = self.track[-1]
                orientation = np.arctan2(goal_y - y, goal_x - x)
                v = np.linalg.norm([self.track[0][0] - x, self.track[0][1] - y]) / dt
            else:
                orientation = 0
                v = 0
            self.x = np.array([x, y, orientation, v, 0])

        self.reached_goal = False
        self.collided = False
        self.visible = False

        self.max_v = np.inf
        self.min_v = -np.inf
        self.max_brake = np.inf
        self.max_accel = np.inf
        self.max_delta = np.pi / 6.0

        self.colour = colour
        self.outline_colour = outline_colour
        self.radius = radius

        self.poly_def = np.array(
            [
                [2.2, 0.8, 1],
                [2.45, 0, 1],
                [2.2, -0.8, 1],
                [-2.2, -0.8, 1],
                [-2.2, 0.8, 1],
                [2.2, 0.8, 1],
            ]
        ).T

        self.actor_image = actor_image
        self.hidden_actor_image = hidden_actor_image
        self.resolution = resolution

        self.update_position_properties()

    def distance_to(self, pos):
        return np.linalg.norm(self.x[:2] - pos[:2])

    def __move(self, dt):
        """move towards the goal"""
        if not self.reached_goal:
            if self.track is not None:
                # move along the track
                try:
                    next_pos = self.track.pop(0)
                    orientation = np.arctan2(next_pos[STATE.Y] - self.x[STATE.Y], next_pos[STATE.X] - self.x[STATE.X])
                    speed = (
                        np.sqrt((next_pos[STATE.X] - self.x[STATE.X]) ** 2 + (next_pos[STATE.Y] - self.x[STATE.Y]) ** 2)
                        / dt
                    )
                    self.x = [next_pos[STATE.X], next_pos[STATE.Y], speed, orientation, 0]
                except IndexError:
                    self.reached_goal = True
            else:
                if self.u is not None:
                    dx = self.x[STATE.VELOCITY] * np.cos(self.x[STATE.THETA]) * dt
                    dy = self.x[STATE.VELOCITY] * np.sin(self.x[STATE.THETA]) * dt
                    dtheta = self.x[STATE.VELOCITY] * np.tan(self.u[1]) / (self.LENGTH) * dt
                    dv = self.x[STATE.VELOCITY] + self.u[0] * dt
                else:
                    dx = self.x[STATE.VELOCITY] * np.cos(self.x[STATE.THETA]) * dt
                    dy = self.x[STATE.VELOCITY] * np.sin(self.x[STATE.THETA]) * dt
                    dtheta = 0
                    dv = 0

                if self.goal is not None:
                    dir = self.goal[:2] - self.x[:2]
                    dist = np.linalg.norm(dir)
                else:
                    dist = np.inf

                if dist > TARGET_TOLERANCE:  # np.linalg.norm([dx, dy]):
                    self.x[STATE.X] += dx
                    self.x[STATE.Y] += dy
                else:
                    # arrived at the goal
                    # self.x[:2] = self.goal[:2]
                    dv = -self.x[STATE.VELOCITY]
                    self.reached_goal = True

                self.x[STATE.THETA] += dtheta
                v = self.x[STATE.VELOCITY] + dv
                self.x[STATE.VELOCITY] = np.clip(v, a_min=self.min_v, a_max=self.max_v)

    def set_goal(self, goal):
        self.goal = goal
        if goal is not None:
            dir = self.goal[:2] - self.x[:2]
            self.x[STATE.THETA] = atan2(dir[STATE.Y], dir[STATE.X])
        else:
            self.x[STATE.THETA]

        self.reached_goal = False

    def set_control(self, u=None):
        if u is None:
            self.u = None
        else:
            self.u = np.array(
                [np.clip(u[0], -self.max_brake, self.max_accel), np.clip(u[1], -self.max_delta, self.max_delta)]
            )

    def get_v(self):
        return np.array([np.cos(self.x[STATE.THETA]), np.sin(self.x[STATE.THETA])]) * self.x[STATE.VELOCITY]

    def tick(self, dt=TICK_TIME):
        """a time step"""
        if self.x[STATE.VELOCITY] or self.u is not None:
            self.__move(dt)
        self.update_position_properties()

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
        # TODO: Update this to use the defined polygon and the GPU code
        #       that checks if a point is inside
        return (
            loc[0] >= self.bounding_box[0]
            and loc[1] >= self.bounding_box[1]
            and loc[0] <= self.bounding_box[2]
            and loc[1] <= self.bounding_box[3]
        )

    def set_collided(self, colour="black"):
        self.colour = colour
        self.collided = True

    def set_visible(self, visible=True):
        self.visible = visible

    def project(self, timesteps=1, u=None, dt=TICK_TIME):
        """
        Project a future position based on a supplied control and state
        """

        if self.track is None:
            virt_self = type(self)(id=self.id, x=self.x, goal=self.goal)
            states = []
            if u is None:
                for _ in range(timesteps):
                    virt_self.tick(dt)
                    states.append(list(virt_self.x))
            else:
                for control in u.T:
                    virt_self.set_control(control)
                    virt_self.tick(dt)
                    states.append(list(virt_self.x))
            return states
        else:
            if not len(self.track):
                return None
            return np.array(self.track[:timesteps])

    def get_poly(self):
        return (self.rot_bw @ self.poly_def)[0:2, ...].T + self.x[0:2]

    def get_pos(self):
        return np.array(self.x[:2])

    def update_position_properties(self):
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
        self.bounding_box = np.array([*min_d, *max_d])
        self.extent = max(np.linalg.norm(min_d - self.x[0:2]), np.linalg.norm(max_d - self.x[0:2]))

        # self.update_footprint()

    def get_bounding_box(self):
        return self.bounding_box

    def get_extent(self):
        return self.extent

    def get_image(self):
        if self.visible:
            return self.actor_image
        else:
            return self.hidden_actor_image

    def get_state(self):
        state = {
            "id": self.id,
            "pos": self.x,
            "size": [self.LENGTH, self.WIDTH],
            "heading": self.x[STATE.THETA],
            "goal": self.goal,
            "type": type(self).__name__.upper(),
            "u": self.u,
            "collided": self.collided,
            "poly": self.get_poly(),
            "bbox": self.get_bounding_box(),
            "extent": self.extent,
            "visible": self.visible,
            "future": self.project(timesteps=10),
        }
        return state


# Velocity based bicycle model
class Vehicle(Actor):
    LENGTH = 4.2
    WIDTH = 2.1

    def __init__(
        self,
        id=0,
        x=[0, 0, 0, 0, 0, 0],
        goal=None,
        colour="orange",
        outline_colour="darkorange",
        resolution=0.1,
        image_name=None,
        image_scale=1.0,
    ):
        if image_name is None:
            actor_image = None
            hidden_actor_image = None
        else:
            actor_image = pygame.image.load(f"assets/{image_name}.svg")
            actor_image = pygame.transform.scale(
                actor_image,
                (Vehicle.LENGTH * image_scale, Vehicle.WIDTH * image_scale),
            )
            try:
                hidden_actor_image = pygame.image.load(f"assets/hidden_{image_name}.svg")
                hidden_actor_image = pygame.transform.scale(
                    hidden_actor_image,
                    (Vehicle.LENGTH * image_scale, Vehicle.WIDTH * image_scale),
                )
            except FileNotFoundError:
                hidden_actor_image = actor_image

        super().__init__(
            id,
            x=x,
            goal=goal,
            colour=colour,
            outline_colour=outline_colour,
            resolution=resolution,
            actor_image=actor_image,
            hidden_actor_image=hidden_actor_image,
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
                [2.75, 0, 1],
                [2.5, -1.0, 1],
                [-2.5, -1.0, 1],
                [-2.5, 1.0, 1],
                [2.5, 1.0, 1],
            ]
        ).T

        self.update_position_properties()


class DeliveryBot(Actor):
    LENGTH = 0.7
    WIDTH = 0.5

    def __init__(
        self,
        id=0,
        x=[0, 0, 0, 0, 0, 0],
        goal=None,
        colour="orange",
        outline_colour="darkorange",
        resolution=0.1,
        image_name=None,
        image_scale=1.0,
    ):
        if image_name is None:
            actor_image = None
            hidden_actor_image = None
        else:
            actor_image = pygame.image.load(f"assets/{image_name}.svg")
            actor_image = pygame.transform.scale(
                actor_image,
                (DeliveryBot.LENGTH * image_scale, DeliveryBot.WIDTH * image_scale),
            )
            try:
                hidden_actor_image = pygame.image.load(f"assets/hidden_{image_name}.svg")
                hidden_actor_image = pygame.transform.scale(
                    hidden_actor_image,
                    (DeliveryBot.LENGTH * image_scale, DeliveryBot.WIDTH * image_scale),
                )
            except FileNotFoundError:
                hidden_actor_image = actor_image

        super().__init__(
            id,
            x=x,
            goal=goal,
            colour=colour,
            outline_colour=outline_colour,
            resolution=resolution,
            actor_image=actor_image,
            hidden_actor_image=hidden_actor_image,
        )

        self.max_v = 1.2
        self.min_v = -1.0
        self.max_brake = -CONTROL_LIMITS[0]
        self.max_accel = CONTROL_LIMITS[0]
        self.max_omega = np.pi / 4.0
        self.max_delta = CONTROL_LIMITS[1]

        self.poly_def = np.array(
            [
                [DeliveryBot.LENGTH / 2.0, DeliveryBot.WIDTH / 2.0, 1],
                [DeliveryBot.LENGTH / 2.0, -DeliveryBot.WIDTH / 2.0, 1],
                [-DeliveryBot.LENGTH / 2.0, -DeliveryBot.WIDTH / 2.0, 1],
                [-DeliveryBot.LENGTH / 2.0, DeliveryBot.WIDTH / 2.0, 1],
                [DeliveryBot.LENGTH / 2.0, DeliveryBot.WIDTH / 2.0, 1],
            ]
        ).T

        self.update_position_properties()


class Pedestrian(Actor):
    WIDTH = 0.45
    LENGTH = 0.3  # in X direction

    def __init__(
        self,
        id=0,
        x=[0, 0, 0, 0, 0],
        goal=None,
        track=None,
        resolution=0.1,
        colour="lightblue",
        outline_colour="dodgerblue",
        image_name=None,
        image_scale=1.0,
    ):

        if image_name is None:
            actor_image = None
            hidden_actor_image = None
        else:
            actor_image = pygame.image.load(f"assets/{image_name}.svg")
            actor_image = pygame.transform.scale(
                actor_image,
                (Pedestrian.LENGTH * image_scale, Pedestrian.WIDTH * image_scale),
            )
            try:
                hidden_actor_image = pygame.image.load(f"assets/hidden_{image_name}.svg")
                hidden_actor_image = pygame.transform.scale(
                    hidden_actor_image,
                    (Pedestrian.LENGTH * image_scale, Pedestrian.WIDTH * image_scale),
                )
            except FileNotFoundError:
                hidden_actor_image = actor_image

        super().__init__(
            id,
            x=x,
            goal=goal,
            track=track,
            resolution=resolution,
            colour=colour,
            outline_colour=outline_colour,
            actor_image=actor_image,
            hidden_actor_image=hidden_actor_image,
        )

        self.max_v = 2.0
        self.min_v = -2.0
        self.max_brake = 0.75
        self.max_accel = 0.75

        self.poly_def = np.array(
            [
                [Pedestrian.LENGTH / 2.0, Pedestrian.WIDTH / 2.0, 1],
                [Pedestrian.LENGTH / 2.0, -Pedestrian.WIDTH / 2.0, 1],
                [-Pedestrian.LENGTH / 2.0, -Pedestrian.WIDTH / 2.0, 1],
                [-Pedestrian.LENGTH / 2.0, Pedestrian.WIDTH / 2.0, 1],
                [Pedestrian.LENGTH / 2.0, Pedestrian.WIDTH / 2.0, 1],
                # [Pedestrian.LENGTH / 2.0, 0.0, 1],
                # [0.0, Pedestrian.WIDTH / 2.0, 1],
                # [-Pedestrian.LENGTH / 2.0, 0.0, 1],
                # [0.0, -Pedestrian.WIDTH / 2.0, 1],
                # [Pedestrian.LENGTH / 2.0, 0.0, 1],
            ]
        ).T

        self.update_position_properties()

    # def get_probability(self, height, width, origin, scale):
    #     if not self.reached_goal:
    #         if self.probability == "Uniform":
    #             prob = create_uniform(
    #                 height=height,
    #                 width=width,
    #                 origin=origin,
    #                 centre=self.get_pos(scaled=True),
    #                 sigma=self.get_radius(scaled=True),
    #                 scale=scale,
    #             )
    #         else:
    #             prob = create_gaussian(
    #                 height=height,
    #                 width=width,
    #                 origin=origin,
    #                 centre=self.get_pos(scaled=True),
    #                 sigma=self.get_radius(scaled=True),
    #                 scale=scale,
    #             )
    #         prob = OCC_PROB * prob / np.max(prob)
    #     else:
    #         prob = np.zeros((height, width))
    #     return prob
