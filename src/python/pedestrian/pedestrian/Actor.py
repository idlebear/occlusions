from config import EPSILON, TICK_TIME, OCC_PROB, TARGET_TOLERANCE, CONTROL_LIMITS
from math import sqrt, atan2, cos, sin
import numpy as np
import pygame
from enum import IntEnum
from pathlib import Path

import controller.ModelParameters.Ackermann as Ackermann

from util.gaussian import create_gaussian
from util.uniform import create_uniform

ASSET_ROOT = Path(__file__).resolve().parent / "assets"
TRACK_HEADING_LOOKAHEAD_STEPS = 15
TRACK_HEADING_MIN_DISPLACEMENT = 0.05


def scaled_poly_points(points, size_scale):
    points = np.asarray(points, dtype=float) * size_scale
    return np.column_stack([points, np.ones(points.shape[0])]).T


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
        size_scale=1.0,
    ):
        self.id = id
        self.x = x
        self.u = None
        self.serial = Actor.serial
        Actor.serial += 1
        self.size_scale = size_scale
        self.length = getattr(self, "LENGTH", 0.0) * self.size_scale
        self.width = getattr(self, "WIDTH", 0.0) * self.size_scale

        self.track = track
        self.goal = goal
        if self.track is not None:
            x, y, frame = self.track.pop(0)
            self.x = np.array([x, y, 0, 0, 0], dtype=float)
            v, orientation = self._estimate_track_motion(
                current_xy=np.asarray([x, y], dtype=float),
                dt=dt,
                previous_theta=0.0,
            )
            self.x = np.array([x, y, v, orientation, 0], dtype=float)

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

        self.poly_def = scaled_poly_points(
            [
                [2.2, 0.8],
                [2.45, 0],
                [2.2, -0.8],
                [-2.2, -0.8],
                [-2.2, 0.8],
                [2.2, 0.8],
            ],
            self.size_scale,
        )

        self.actor_image = actor_image
        self.hidden_actor_image = hidden_actor_image
        self.resolution = resolution

        self.update_position_properties()

    def distance_to(self, pos):
        return np.linalg.norm(self.x[:2] - pos[:2])

    def _estimate_track_motion(self, current_xy, dt, previous_theta, fallback_xy=None):
        if len(self.track) == 0:
            if fallback_xy is None:
                return 0.0, previous_theta
            delta = current_xy - np.asarray(fallback_xy, dtype=float)
            distance = float(np.linalg.norm(delta))
            if distance <= EPSILON:
                return 0.0, previous_theta
            return distance / dt, atan2(delta[1], delta[0])

        max_steps = min(TRACK_HEADING_LOOKAHEAD_STEPS, len(self.track))
        for index in range(max_steps):
            future_xy = np.asarray(self.track[index][:2], dtype=float)
            delta = future_xy - current_xy
            distance = float(np.linalg.norm(delta))
            if distance >= TRACK_HEADING_MIN_DISPLACEMENT:
                steps = index + 1
                return distance / (steps * dt), atan2(delta[1], delta[0])

        next_xy = np.asarray(self.track[0][:2], dtype=float)
        delta = next_xy - current_xy
        distance = float(np.linalg.norm(delta))
        if distance <= EPSILON:
            if fallback_xy is None:
                return 0.0, previous_theta
            fallback_delta = current_xy - np.asarray(fallback_xy, dtype=float)
            fallback_distance = float(np.linalg.norm(fallback_delta))
            if fallback_distance <= EPSILON:
                return 0.0, previous_theta
            return fallback_distance / dt, previous_theta
        return distance / dt, previous_theta

    def __move(self, dt):
        """move towards the goal"""
        if not self.reached_goal:
            if self.track is not None:
                # move along the track
                try:
                    next_pos = self.track.pop(0)
                    current_xy = np.asarray(
                        [next_pos[STATE.X], next_pos[STATE.Y]],
                        dtype=float,
                    )
                    speed, orientation = self._estimate_track_motion(
                        current_xy=current_xy,
                        dt=dt,
                        previous_theta=self.x[STATE.THETA],
                        fallback_xy=self.x[:2],
                    )
                    self.x = [
                        next_pos[STATE.X],
                        next_pos[STATE.Y],
                        speed,
                        orientation,
                        0,
                    ]
                except IndexError:
                    self.reached_goal = True
            else:
                if self.u is not None:
                    velocity_midpoint = np.clip(
                        self.x[STATE.VELOCITY] + 0.5 * self.u[0] * dt,
                        a_min=self.min_v,
                        a_max=self.max_v,
                    )
                    effective_length = max(self.length, EPSILON)
                    dtheta = (
                        velocity_midpoint * np.tan(self.u[1]) / effective_length * dt
                    )
                    theta_midpoint = self.x[STATE.THETA] + 0.5 * dtheta
                    dx = velocity_midpoint * np.cos(theta_midpoint) * dt
                    dy = velocity_midpoint * np.sin(theta_midpoint) * dt
                    dv = self.u[0] * dt
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
            # Debug: print control values before and after clipping
            u_raw = np.array(u)
            u_clipped = np.array(
                [
                    np.clip(u[0], -self.max_brake, self.max_accel),
                    np.clip(u[1], -self.max_delta, self.max_delta),
                ]
            )

            # Only print if there's significant clipping
            if abs(u_raw[1] - u_clipped[1]) > 0.01:  # Only for steering
                print(
                    f"STEERING CLIPPED: raw={np.degrees(u_raw[1]):.1f}°, applied={np.degrees(u_clipped[1]):.1f}°, limit={np.degrees(self.max_delta):.1f}°"
                )

            self.u = u_clipped

    def get_v(self):
        return (
            np.array([np.cos(self.x[STATE.THETA]), np.sin(self.x[STATE.THETA])])
            * self.x[STATE.VELOCITY]
        )

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
        self.extent = max(
            np.linalg.norm(min_d - self.x[0:2]), np.linalg.norm(max_d - self.x[0:2])
        )

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
            "size": [self.length, self.width],
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
        size_scale=1.0,
    ):
        length = Vehicle.LENGTH * size_scale
        width = Vehicle.WIDTH * size_scale
        if image_name is None:
            actor_image = None
            hidden_actor_image = None
        else:
            actor_image = pygame.image.load(str(ASSET_ROOT / f"{image_name}.svg"))
            actor_image = pygame.transform.scale(
                actor_image,
                (length * image_scale, width * image_scale),
            )
            try:
                hidden_actor_image = pygame.image.load(
                    str(ASSET_ROOT / f"hidden_{image_name}.svg")
                )
                hidden_actor_image = pygame.transform.scale(
                    hidden_actor_image,
                    (length * image_scale, width * image_scale),
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
            size_scale=size_scale,
        )

        self.max_v = Ackermann.MAX_V
        self.min_v = Ackermann.MIN_V
        self.max_a = Ackermann.MAX_A
        self.min_a = Ackermann.MIN_A
        self.max_omega = Ackermann.MAX_W
        self.max_delta = Ackermann.MAX_DELTA

        self.poly_def = scaled_poly_points(
            [
                [2.5, 1.0],
                [2.75, 0],
                [2.5, -1.0],
                [-2.5, -1.0],
                [-2.5, 1.0],
                [2.5, 1.0],
            ],
            self.size_scale,
        )

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
        size_scale=1.0,
    ):
        length = DeliveryBot.LENGTH * size_scale
        width = DeliveryBot.WIDTH * size_scale
        if image_name is None:
            actor_image = None
            hidden_actor_image = None
        else:
            actor_image = pygame.image.load(str(ASSET_ROOT / f"{image_name}.svg"))
            actor_image = pygame.transform.scale(
                actor_image,
                (length * image_scale, width * image_scale),
            )
            try:
                hidden_actor_image = pygame.image.load(
                    str(ASSET_ROOT / f"hidden_{image_name}.svg")
                )
                hidden_actor_image = pygame.transform.scale(
                    hidden_actor_image,
                    (length * image_scale, width * image_scale),
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
            size_scale=size_scale,
        )

        self.max_v = 1.2
        self.min_v = -1.0
        self.max_brake = CONTROL_LIMITS[0]
        self.max_accel = CONTROL_LIMITS[0]
        self.max_omega = np.pi / 4.0
        self.max_delta = CONTROL_LIMITS[1]

        self.poly_def = scaled_poly_points(
            [
                [DeliveryBot.LENGTH / 2.0, DeliveryBot.WIDTH / 2.0],
                [DeliveryBot.LENGTH / 2.0, -DeliveryBot.WIDTH / 2.0],
                [-DeliveryBot.LENGTH / 2.0, -DeliveryBot.WIDTH / 2.0],
                [-DeliveryBot.LENGTH / 2.0, DeliveryBot.WIDTH / 2.0],
                [DeliveryBot.LENGTH / 2.0, DeliveryBot.WIDTH / 2.0],
            ],
            self.size_scale,
        )

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
        size_scale=1.0,
        dt=TICK_TIME,
    ):
        length = Pedestrian.LENGTH * size_scale
        width = Pedestrian.WIDTH * size_scale

        if image_name is None:
            actor_image = None
            hidden_actor_image = None
        else:
            actor_image = pygame.image.load(str(ASSET_ROOT / f"{image_name}.svg"))
            actor_image = pygame.transform.scale(
                actor_image,
                (length * image_scale, width * image_scale),
            )
            try:
                hidden_actor_image = pygame.image.load(
                    str(ASSET_ROOT / f"hidden_{image_name}.svg")
                )
                hidden_actor_image = pygame.transform.scale(
                    hidden_actor_image,
                    (length * image_scale, width * image_scale),
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
            size_scale=size_scale,
            dt=dt,
        )

        self.max_v = 2.0
        self.min_v = -2.0
        self.max_brake = 0.75
        self.max_accel = 0.75

        self.poly_def = scaled_poly_points(
            [
                [Pedestrian.LENGTH / 2.0, Pedestrian.WIDTH / 2.0],
                [Pedestrian.LENGTH / 2.0, -Pedestrian.WIDTH / 2.0],
                [-Pedestrian.LENGTH / 2.0, -Pedestrian.WIDTH / 2.0],
                [-Pedestrian.LENGTH / 2.0, Pedestrian.WIDTH / 2.0],
                [Pedestrian.LENGTH / 2.0, Pedestrian.WIDTH / 2.0],
            ],
            self.size_scale,
        )

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
