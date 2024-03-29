from copy import deepcopy
from importlib import import_module
from math import sqrt, exp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pygame
from random import random, expovariate, seed
import visilibity as vis

from policies.VelocityGrid import VelocityGrid
from policies.flow.flow import flow

# from polycheck import Vertex, VertexList, PolygonList
# from polycheck import visibility_from_region, faux_scan
from polygpu import faux_scan, visibility_from_region

# from dogm_py import LaserMeasurementGridParams
# from dogm_py import LaserMeasurementGrid
# from dogm_py import DOGMParams
# from dogm_py import DOGM
# from dogm_py import VectorFloat
# from dogm_py import renderOccupancyGrid, renderDynamicOccupancyGrid

# from dogm_py import renderMeasurement

from Grid.OccupancyGrid import OccupancyGrid

import ModelParameters.Ackermann as Ackermann


# local functions/imports
from Actor import Actor, Pedestrian, Car, Obstacle, Blank, SkidSteer, AckermanCar, STATE
from config import *

DEBUG_INFORMATION_GAIN = True
DEBUG = 0
FORECAST_COUNT = 5
FORECAST_INTERVAL = 0.1

# rewards - a high penalty for colliding with anything, a small penalty for
# deviating from the desired velocity, a slightly smaller one for deviating from
# the desired Y position, and a positive reward for moving forward
REWARD_COLLISION = -100000  # note that this includes leaving the road surface!
REWARD_DEVIATION_V = 100.0
REWARD_DEVIATION_Y = -10
REWARD_DEVIATION_X = 10
REWARD_PASSING = 1000
REWARD_GOAL = 100


INFORMATION_GAIN_TRAJECTORIES = 3


# dynamic occupancy grid paramaters
MAP_DYANMIC_PARTICLES = 20000
MAP_NEW_BORN_PARTICLES = 10000
MAP_PARTICLE_PERSISTANCE = 0.5
MAP_PROCESS_NOISE_SIGMA = 0.001
MAP_PROCESS_NOISE_VELOCITY = 0.001
MAP_BIRTH_PROBABILITY = 0.2
MAP_STDDEV_VELOCITY = 1.0
MAP_BASE_VELOCITY = ACTOR_SPEED

# Occupancy Grid Properties
GRID_OCCUPANCY_THRESHOLD = 0.6
GRID_VELOCITY_THRESHOLD = 1.0
GRID_VELOCITY_MAX = ACTOR_SPEED * 1.5

DESIRED_LANE_POSITION = -LANE_WIDTH / 2

GOAL = [1000, DESIRED_LANE_POSITION]


def get_location(origin, location):
    return [
        location[0] - origin[0] - EGO_X_OFFSET,
        location[1] - origin[1] - EGO_Y_OFFSET,
    ]


class Window:
    def __init__(self, screen, screen_width, screen_height, margin, scale=1.0):
        self.screen = screen

        self.sim_time_text = pygame.font.SysFont("dejavuserif", 15)
        self.elapsed_time_text = pygame.font.SysFont("dejavuserif", 10)
        self.status_font = pygame.font.SysFont("roboto", STATUS_FONT_SIZE)

        self._xmargin = margin * 0.5
        self._ymargin = margin * 0.5
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._env_size = screen_width - margin
        self._draw_scale = self._env_size // scale  # pixels/m (unit, whatever...)
        self._scale = scale
        self._border_offset = 10

        self.tmp_screen = pygame.Surface((self.screen.get_width(), self.screen.get_height()), flags=pygame.SRCALPHA)

    # def _get_location_on_screen(self, origin, location):
    #     return [
    #         int(self._xmargin + (location[0]-origin[0] - EGO_X_OFFSET)*self._env_size),
    #         int(self._ymargin + (location[1]-origin[1] - EGO_Y_OFFSET)*self._env_size)
    #     ]

    def clear(self):
        self.screen.fill(SCREEN_BACKGROUND_COLOUR)

    def get_drawing_scale(self):
        return self._draw_scale

    def draw_line(self, start, end, colour, width=2):
        sx = self._xmargin + start[0] * self._env_size / self._scale
        sy = self._ymargin + start[1] * self._env_size / self._scale
        ex = self._xmargin + end[0] * self._env_size / self._scale
        ey = self._ymargin + end[1] * self._env_size / self._scale

        pygame.draw.line(self.screen, color=colour, start_pos=(sx, sy), end_pos=(ex, ey), width=width)

    def draw_circle(self, centre, colour, radius=2):
        cx = self._xmargin + centre[0] * self._env_size / self._scale
        cy = self._ymargin + centre[1] * self._env_size / self._scale
        radius = int(radius * self._env_size / self._scale)

        pygame.draw.circle(self.screen, color=colour, center=(cx, cy), radius=radius)

    def draw_rect(self, colour, location, height, width=None):
        if width is None:
            width = height

        pygame.draw.rect(
            self.screen,
            colour,
            (
                self._xmargin + (location[0] - width / 2.0) * self._env_size / self._scale,
                self._ymargin + (location[1] - height / 2.0) * self._env_size / self._scale,
                width * self._env_size / self._scale,
                height * self._env_size / self._scale,
            ),
        )

    def draw_polygon(
        self,
        outline_colour,
        fill_colour,
        points,
        width=ACTOR_PATH_WIDTH,
        use_transparency=False,
    ):
        points = [
            [
                self._xmargin + x * self._env_size / self._scale,
                self._ymargin + y * self._env_size / self._scale,
            ]
            for x, y in points
        ]

        if use_transparency:
            self.tmp_screen.fill((0, 0, 0, 0))
            if fill_colour is not None:
                pygame.draw.polygon(self.tmp_screen, fill_colour, points, 0)
            pygame.draw.polygon(self.tmp_screen, outline_colour, points, width=width)
            self.screen.blit(self.tmp_screen, (0, 0))
        else:
            if fill_colour is not None:
                pygame.draw.polygon(self.screen, fill_colour, points, 0)
            pygame.draw.polygon(self.screen, outline_colour, points, width=width)

    # Quick image rotation
    #   https://stackoverflow.com/questions/4183208/how-do-i-rotate-an-image-around-its-center-using-pygame
    def draw_image(self, image, center, orientation):
        center = (
            self._xmargin + center[0] * self._env_size / self._scale,
            self._ymargin + center[1] * self._env_size / self._scale,
        )
        rotated_image = pygame.transform.rotate(image, np.rad2deg(orientation))
        new_rect = rotated_image.get_rect(center=image.get_rect(center=center).center)
        self.screen.blit(rotated_image, new_rect)

    def draw_status(self, collisions, sim_time):
        #  draw the limits of the environment
        pygame.draw.rect(
            self.screen,
            SCREEN_OUTLINE_COLOUR,
            (
                self._xmargin - self._border_offset,
                self._ymargin - self._border_offset,
                self._env_size + self._border_offset * 2,
                self._env_size + self._border_offset * 2,
            ),
            2,
        )

        collisions_str = f"Collisions: {collisions}"
        time_str = f"Sim Time: {sim_time:.4f}"

        text_width, text_height = self.status_font.size(collisions_str)
        time_text_width, time_text_height = self.status_font.size(time_str)

        x_avg_offset = self._env_size + self._xmargin - text_width - STATUS_XMARGIN * 2
        y_avg_offset = self._env_size + self._ymargin - text_height - STATUS_YMARGIN

        pygame.draw.rect(
            self.screen,
            SCREEN_BACKGROUND_COLOUR,
            (
                x_avg_offset - STATUS_XMARGIN,
                y_avg_offset - STATUS_YMARGIN,
                text_width + STATUS_XMARGIN * 2,
                text_height + STATUS_YMARGIN,
            ),
            0,
        )
        pygame.draw.rect(
            self.screen,
            SCREEN_OUTLINE_COLOUR,
            (
                x_avg_offset - STATUS_XMARGIN,
                y_avg_offset - STATUS_YMARGIN,
                text_width + STATUS_XMARGIN * 2,
                text_height + STATUS_YMARGIN,
            ),
            2,
        )
        text = self.status_font.render(collisions_str, False, STATUS_FONT_COLOUR)
        self.screen.blit(text, (x_avg_offset + STATUS_XMARGIN / 3, y_avg_offset))

        pygame.draw.rect(
            self.screen,
            SCREEN_BACKGROUND_COLOUR,
            (
                self._xmargin + STATUS_XMARGIN / 2,
                self._xmargin + STATUS_YMARGIN,
                time_text_width + STATUS_XMARGIN,
                time_text_height + STATUS_YMARGIN,
            ),
            0,
        )

        text = self.status_font.render(time_str, False, STATUS_FONT_COLOUR)
        self.screen.blit(text, (self._xmargin + STATUS_XMARGIN, self._ymargin + STATUS_YMARGIN))

    def save_screen(self, path):
        pygame.image.save(self.screen, path)


class Simulation:
    def __init__(
        self,
        generator_name="uniform",
        generator_args=None,
        num_actors=1,
        limit_actors=False,
        pois_lambda=0.01,
        screen=None,
        service_time=SERVICE_TIME,
        speed=ACTOR_SPEED,
        margin=SCREEN_MARGIN,
        screen_width=SCREEN_WIDTH,
        screen_height=SCREEN_HEIGHT,
        tick_time=TICK_TIME,
        record_data=False,
    ):
        self.num_actors = num_actors
        self.limit_actors = limit_actors
        self.actor_target_speed = speed
        self.pois_lambda = pois_lambda

        self.record_data = record_data
        self.tick_time = tick_time

        self.service_time = service_time

        if screen is not None or record_data:
            self.window = Window(
                screen=screen,
                screen_width=screen_width,
                screen_height=screen_height,
                margin=margin,
                scale=WINDOW_SIZE,
            )
            self.image_scale = self.window.get_drawing_scale()
        else:
            self.window = None
            self.image_scale = None

        # load the draw method
        self.load_generator(generator_name=generator_name, generator_args=generator_args)

        # self.grid = VelocityGrid(
        #     height=GRID_HEIGHT,
        #     width=GRID_WIDTH,
        #     resolution=GRID_RESOLUTION,
        #     origin=(GRID_ORIGIN_Y_OFFSET, GRID_ORIGIN_Y_OFFSET),
        # )

        self.observation_shape = [
            int(GRID_HEIGHT / GRID_RESOLUTION),
            int(GRID_WIDTH / GRID_RESOLUTION),
        ]

        self.maps = None
        self.ig_images = None
        self.ig_val_images = None

        # Construct the occupancy grid
        self.obs = OccupancyGrid(
            dim=GRID_WIDTH,
            resolution=GRID_RESOLUTION,
            origin=(0, 0),
        )
        # # Construct the dynamic occupancy grid
        # # DOGM params
        # dogm_params = DOGMParams(
        #     size=GRID_WIDTH,
        #     resolution=GRID_RESOLUTION,
        #     particle_count=MAP_DYANMIC_PARTICLES,
        #     new_born_particle_count=MAP_NEW_BORN_PARTICLES,
        #     persistance_prob=MAP_PARTICLE_PERSISTANCE,
        #     stddev_process_noise_position=MAP_PROCESS_NOISE_SIGMA,
        #     stddev_process_noise_velocity=MAP_PROCESS_NOISE_VELOCITY,
        #     birth_prob=MAP_BIRTH_PROBABILITY,
        #     stddev_velocity=MAP_STDDEV_VELOCITY,
        #     init_max_velocity=MAP_BASE_VELOCITY,
        # )
        # self.dogm = DOGM(params=dogm_params)

        # # Create a LaserMeasurementGrid object that converts the range based laserscan update into a cartesian
        # # grid.  The cartesian grid is then used as an update to the occupancy grid.
        # lmg_params = LaserMeasurementGridParams(
        #     fov=SCAN_FOV * 180.0 / np.pi,
        #     angle_increment=SCAN_ANGLE_INCREMENT * 180.0 / np.pi,
        #     max_range=SCAN_RANGE,
        #     resolution=SCAN_RESOLUTION,
        #     stddev_range=SCAN_STDDEV_RANGE,
        # )
        # self.lmg = LaserMeasurementGrid(
        #     params=lmg_params, size=GRID_WIDTH, resolution=GRID_RESOLUTION
        # )

        # # TODO: ROI is fixed for the time being -- should move this to the ego/agent and make it relative to the vehicle speed.  Note that we should also
        # #       position the ROI relative to the AV, but for now, we'll centre it on the road.
        # #
        # # ROI is determined by the current speed, expected agent speeds
        # y_ratio = OPPONENT_CAR_SPEED / ACTOR_SPEED

        # # self.roi = []
        # # for x in range(2, GRID_SIZE // 2, 2):
        # #     max_y = int(round(x * y_ratio))
        # #     for y in range(-max_y, max_y + 1, 2):
        # #         if abs(y) < GRID_SIZE // 2:
        # #             self.roi.append([x + GRID_SIZE // 2, y + GRID_SIZE // 2])
        # # self.roi = np.array(self.roi).astype(int)

        self.reset()

    def reset(self):
        # reset the random number generator
        self.generator.reset()

        self.ego = AckermanCar(
            id=0,
            x=np.array([0.0, DESIRED_LANE_POSITION, 0, 0, 0, 0]),
            goal=None,
            colour="red",
            outline_colour="darkred",
            scale=1.0,
            resolution=GRID_RESOLUTION,
            image_scale=self.image_scale,
        )

        self.actor_list = []
        self.generated_actors = 0

        self.sim_time = 0.0
        self.next_time = 0.0
        self.sim_start_time = 0.0
        self.last_distance = GOAL[0]

        self.next_agent_x = -np.inf

        self.ticks = 0
        self.collisions = 0

        self.information_gain = None

        self.obs.reset()
        self.probability_map = None

        return (
            self._get_next_observation(self._calculate_future_visibility(), self.tick_time),
            self._get_info(),
        )

    def load_generator(self, generator_name, generator_args):
        # load the generator
        self.generator_name = generator_name
        self.generator_args = generator_args
        gen_mod = import_module("." + self.generator_name, package="generators")
        generator_fn = gen_mod.get_generator_fn()
        self.generator = generator_fn(**self.generator_args)

    ############################################################################
    # Plotting and drawing functions
    ############################################################################

    def _draw_line(self, start, end, colour, width=2):
        start = get_location(origin=self.ego.x[0:2], location=start)
        end = get_location(origin=self.ego.x[0:2], location=end)
        self.window.draw_line(start, end, colour, width)

    def _draw_circle(self, centre, colour, radius=1):
        centre = get_location(origin=self.ego.x[0:2], location=centre)
        self.window.draw_circle(centre, colour, radius)

    def _get_actor_outline(self, actor):
        actor_pos = get_location(origin=self.ego.x[0:2], location=(0, 0))
        poly = actor.get_poly()
        return actor_pos + poly

    def _draw_road(self):
        x = self.ego.x[0]
        y = 0

        loc = get_location(origin=self.ego.x[0:2], location=(x, y))
        self.window.draw_rect(ROAD_COLOUR, (loc[0], loc[1]), 2 * LANE_WIDTH, WINDOW_SIZE * 2)

        x = (x // 4) * 4.0
        for _ in range(int(WINDOW_SIZE * 2 // 4)):
            loc = get_location(origin=self.ego.x[0:2], location=(x - WINDOW_SIZE / 2, y - 0.2))
            self.window.draw_rect(ROAD_MARKING_COLOUR, (loc[0], loc[1]), 0.4, 2.5)
            x += 4

    def _draw_actor(self, actor):
        if not actor.is_real():
            return

        actor_image = actor.get_image()
        if actor_image is not None:
            actor_pos = get_location(origin=self.ego.x[0:2], location=actor.x[0:2])
            # drawing with y inverted reverse the rotation to correct the display
            self.window.draw_image(image=actor_image, center=actor_pos, orientation=-actor.x[STATE.THETA])
        else:
            actor_poly = actor.get_poly()
            if actor_poly is not None:
                # shift the poly relative to the ego instead of the origin
                actor_poly += get_location(origin=self.ego.x[0:2], location=(0, 0))

                self.window.draw_polygon(
                    outline_colour=actor.outline_colour,
                    fill_colour=actor.colour,
                    points=actor_poly,
                )

        # else: actor has no image or outline (blank)

    def _draw_ego(self):
        self._draw_actor(self.ego)

    def _draw_visibility(self):
        # if self.visibility is not None:
        #     pts = []
        #     for i in range(self.visibility.n()):
        #         pts.append(get_location(origin=self.ego.x, location=[self.visibility[i].x(), self.visibility[i].y()]))

        #     TGREEN = (150, 220, 150, 100)
        #     TBLACK = (0, 0, 0, 200)
        #     # TODO: Drawing this filled polygon is hella slow, taking the sim from faster than real-time to somewhere around
        #     #       half speed.  Removed for now, but it's just not as pretty...  there may be an update to resolve the issue
        #     #       in pygame, but for the time being, our poly is going to be clear....
        #     self.window.draw_polygon(outline_colour=TBLACK, fill_colour=None, points=pts, use_transparency=True)
        pass

    def _draw_status(self):
        self.window.draw_status(self.collisions, self.sim_time)

    def _draw_scan(self):
        pass
        # for i in range(0, SCAN_RAYS, 10):
        #     angle = SCAN_START_ANGLE + i * SCAN_ANGLE_INCREMENT
        #     ex = self.ego.x[0] + np.cos(angle) * self.scan_data[i]
        #     ey = self.ego.x[1] + np.sin(angle) * self.scan_data[i]
        #     self._draw_line(self.ego.x, (ex, ey), (200, 200, 0, 255), 2)

    ##################################################################################
    # Simulator step functions
    ##################################################################################

    def calculate_visibility(self):
        # calculate the visibility polygon
        shapes = []

        # environment poly is counter clockwise and large enough to be off screen
        ox = self.ego.x[0] - 2
        oy = -2
        shapes.append(
            vis.Polygon(
                [
                    vis.Point(ox, oy),
                    vis.Point(ox + 4.0, oy),
                    vis.Point(ox + 4.0, oy + 4.0),
                    vis.Point(ox, oy + 4.0),
                ]
            )
        )

        for actor in self.actor_list:
            if type(actor) is Blank:
                continue

            if actor.x[0] > self.ego.x[0] + EGO_X_OFFSET and actor.x[0] < self.ego.x[0] + EGO_X_OFFSET + 1.5:
                pts = actor.get_poly()
                poly_pts = [vis.Point(pt[0], pt[1]) for pt in pts[-1:0:-1]]
                shapes.append(vis.Polygon(poly_pts))

        vis_poly = None
        env = vis.Environment(shapes)
        if env.is_valid(EPSILON):
            observer = vis.Point(self.ego.x[0], self.ego.x[1])
            vis_poly = vis.Visibility_Polygon(observer, env, EPSILON)

        return vis_poly

    def _generate_parkade(self):
        x = max(self.next_agent_x, self.ego.x[0] + WINDOW_SIZE)

        while len(self.actor_list) < self.num_actors and x - self.ego.x[0] < WINDOW_SIZE * 2.0:
            # left side
            rnd = self.generator.uniform()
            if rnd < 0.6:
                # create a parked car
                self.generated_actors += 1
                rnd = self.generator.uniform()
                if rnd < 0:
                    # some small percentage of cars are really badly parked
                    y = -LANE_WIDTH + 1
                else:
                    y = -LANE_WIDTH - CAR_OFFSET

                x_wiggle = self.generator.uniform() * CAR_WIGGLE
                orientation = self.generator.uniform() * CAR_ROTATION - CAR_ROTATION / 2

                actor = Obstacle(
                    id=self.ticks,
                    x=np.array([x + x_wiggle, y, orientation, 0, 0]),
                    scale=Ackermann.LENGTH,
                    ratio=Ackermann.WIDTH / Ackermann.LENGTH,
                    resolution=GRID_RESOLUTION,
                    image="red_car",
                    image_scale=self.image_scale,
                )
                self.actor_list.append(actor)
            else:
                # blank space
                pass

            # right side
            # rnd = self.generator.uniform()
            # if rnd < 0.6:
            #     # create a parked car
            #     y = LANE_WIDTH + CAR_OFFSET
            #     x_wiggle = self.generator.uniform() * CAR_WIGGLE
            #     orientation = self.generator.uniform() * CAR_ROTATION - CAR_ROTATION / 2
            #     self.generated_actors += 1

            #     actor = Obstacle(
            #         id=self.ticks,
            #         x=np.array([x + x_wiggle, y, orientation + np.pi, 0, 0]),
            #         scale=Ackermann.LENGTH,
            #         ratio=Ackermann.WIDTH / Ackermann.LENGTH,
            #         resolution=GRID_RESOLUTION,
            #         image="red_car",
            #         image_scale=self.image_scale,
            #     )
            #     self.actor_list.append(actor)
            # else:
            #     # blank space
            #     pass

            x = x + CAR_SPACING
            self.next_agent_x = x

    def _generate_new_agents(self):
        x = max(self.next_agent_x, self.ego.x[0] + WINDOW_SIZE / 2)

        while len(self.actor_list) < self.num_actors:
            rnd = self.generator.uniform()
            if rnd < 0.4:
                self.generated_actors += 1
                scale = 1 + 9 * self.generator.uniform()
                if rnd < 0.2:
                    y = -LANE_WIDTH - np.ceil(scale / 2) - self.generator.uniform() * 5
                else:
                    y = LANE_WIDTH + np.ceil(scale / 2) + self.generator.uniform() * 5
                actor = Obstacle(
                    id=self.ticks,
                    x=np.array([x + scale // 2, y, 0, 0, 0]),
                    scale=scale,
                    resolution=GRID_RESOLUTION,
                )
            # elif rnd < 0.4:
            #     # oncoming traffic
            #     scale = 1

            #     v = OPPONENT_CAR_SPEED
            #     y = LANE_WIDTH / 2

            #     actor = Car(
            #         id=self.ticks,
            #         x=[x + GenericCar.LENGTH / 2, y, -v, -np.pi, 0],
            #         goal=np.array([0, y]),
            #         scale=scale,
            #         image_prefix="red_",
            #         image_scale=self.image_scale,
            # resolution=GRID_RESOLUTION,
            #     )
            # elif rnd < 1:
            #     # same side traffic
            #     scale = 1

            #     v = OPPONENT_CAR_SPEED * 0.5
            #     y = -LANE_WIDTH / 2

            #     actor = Car(
            #         id=self.ticks,
            #         x=np.array([x + GenericCar.LENGTH / 2, y, v, 0, 0]),
            #         goal=np.array([self.ego.x[0] + 100000, y]),
            #         scale=scale,
            # resolution=GRID_RESOLUTION,
            #         image_prefix="red_",
            #         image_scale=self.image_scale,
            #     )
            # elif rnd < 0.95:
            #     scale = 1

            #     v = OPPONENT_PEDESTRIAN_SPEED
            #     y = 5.0
            #     theta = np.pi / 2
            #     if rnd < 0.7875:
            #         v = -v
            #         y = -y
            #         theta = -theta

            #     actor = Pedestrian(
            #         id=self.ticks,
            #         x=np.array([x + width / 2, y, 0, -v, theta]),
            #         goal=np.array([x + width / 2, -y]),
            #         scale=scale,
            # resolution=GRID_RESOLUTION,
            #     )
            else:
                # do nothing (space)
                scale = 1 + 9 * self.generator.uniform()
                y = 3 + scale // 2 + self.generator.uniform() * 10
                actor = Blank(
                    id=self.ticks,
                    x=np.array([x + scale / 2, y, 0, 0, 0]),
                    scale=scale,
                )

            width = actor.get_extent() * 2
            self.actor_list.append(actor)
            x += width + 0.005 * self.generator.uniform()
            self.next_agent_x = x

    def _get_next_observation(self, scan_data, dt):
        self.obs.move_origin(self.ego.x[0:2])
        self.obs.update(
            X=[*self.ego.x[0:2], self.ego.x[STATE.THETA]],
            angle_min=SCAN_START_ANGLE,
            angle_inc=SCAN_ANGLE_INCREMENT,
            ranges=scan_data,
            min_range=0,
            max_range=SCAN_RANGE + 1,
        )
        self.probability_map = self.obs.probabilityMap()
        return self.probability_map

        # grid_data = self.lmg.generateGrid(
        #     VectorFloat(scan_data), self.ego.x[STATE.THETA] * 180.0 / np.pi
        # )
        # self.dogm.updateGrid(grid_data, self.ego.x[0], self.ego.x[1], dt)
        # return renderOccupancyGrid(
        #     self.dogm
        # )  # , GRID_OCCUPANCY_THRESHOLD, GRID_VELOCITY_THRESHOLD, GRID_VELOCITY_MAX)

        # # # BUGBUG: Add some probable occupany to the area outside the roadway to allow the RL engine to learn about the
        # # #         undesireable area(s), where we don't want the car to travel.  Did not have a positive affect in training,
        # # #         but left here as a comment to inform the next attempt.
        # # rs, re = int(self.observation_shape[0]/2-ROAD_WIDTH/GRID_RESOLUTION/2), int(self.observation_shape[0]/2+ROAD_WIDTH/GRID_RESOLUTION/2)
        # # probability_map[0:rs, :] = np.maximum(probability_map[0:rs, :], 0.1)
        # # probability_map[re+1:, :] = np.maximum(probability_map[re+1:, :], 0.1)
        # observation = np.append(np.expand_dims(probability_map, axis=2), velocity_map, axis=2)

        # observation = (observation * 255.0).astype(np.uint8)
        # return observation

    #
    # @brief: construct a 2x map centred on the ego car of the obstacles in the environment as a
    #         ground truth
    #
    def _get_map(self):
        map = np.zeros([GRID_SIZE * 2, GRID_SIZE * 2])
        for actor in self.actor_list:
            if actor.is_real() and actor.distance_to(self.ego.x) <= SCAN_RANGE:
                if type(actor) == Obstacle:
                    footprint = actor.get_footprint()
                    dy, dx = footprint.shape

                    mx = GRID_SIZE + int((actor.bounding_box[0] - self.ego.x[0]) / GRID_RESOLUTION)
                    my = GRID_SIZE + int((actor.bounding_box[1] - self.ego.x[1]) / GRID_RESOLUTION)
                    ix = 0
                    iy = 0

                    if mx > GRID_SIZE * 2 or my > GRID_SIZE * 2 or mx + dx < 0 or my + dy < 0:
                        continue

                    if mx < 0:
                        dx = dx + mx
                        ix = -mx
                        mx = 0
                    if mx + dx >= GRID_SIZE * 2:
                        dx = GRID_SIZE * 2 - mx

                    if my < 0:
                        dy = dy + my
                        iy = -my
                        my = 0
                    if my + dy >= GRID_SIZE * 2:
                        dy = GRID_SIZE * 2 - my

                    map[my : my + dy, mx : mx + dx] = footprint[iy : iy + dy, ix : ix + dx]

        return map

    def _get_info(self):
        info = {}
        info["ego"] = self.ego.get_state()
        actor_states = []
        for actor in self.actor_list:
            if actor.is_real() and actor.distance_to(self.ego.x) <= SCAN_RANGE:
                actor_state = actor.get_state()
                poly = actor.get_poly()
                min_angle = np.pi / 2
                min_pt = None
                for pt in poly:
                    angle = abs(np.arctan((pt[1] - self.ego.x[1]) / (pt[0] - self.ego.x[0])))
                    if angle < min_angle:
                        min_angle = angle
                        min_pt = pt
                actor_state["min_pt"] = min_pt
                actor_states.append(actor_state)

        info["map"] = self._get_map()
        info["actors"] = actor_states
        info["information_gain"] = self.information_gain
        return info

    def _calculate_future_visibility(self):
        # create the scan of the environment
        # calculate the visibility polygon
        shapes = []

        # build a list of polygons in the environment
        polygons = []
        for actor in self.actor_list:
            if type(actor) is Blank or actor.collided:
                continue
            polygons.append(actor.get_poly())
        polygons = np.array(polygons)

        scan_data = faux_scan(
            polygons,
            origin=self.ego.x[0:2],
            angle_start=SCAN_START_ANGLE + self.ego.x[STATE.THETA],
            angle_inc=SCAN_ANGLE_INCREMENT,
            num_rays=SCAN_RAYS,
            max_range=SCAN_RANGE,
            resolution=SCAN_RESOLUTION,
        )

        # clear any rays that didn't hit anything
        scan_data[scan_data == -1] = SCAN_RANGE + 1
        return scan_data.astype(np.float32)

    # def calculate_information_gain(self, occupancy_grid):
    #     obs_pts = [
    #         [
    #             [self.ego.x[0] + 0.04, self.ego.x[1] - 0.04],
    #             [self.ego.x[0] + 0.08, self.ego.x[1] - 0.04],
    #             [self.ego.x[0] + 0.12, self.ego.x[1] - 0.04],
    #         ],
    #         [
    #             [self.ego.x[0] + 0.04, self.ego.x[1]],
    #             [self.ego.x[0] + 0.08, self.ego.x[1]],
    #             [self.ego.x[0] + 0.12, self.ego.x[1]],
    #         ],
    #         [
    #             [self.ego.x[0] + 0.04, self.ego.x[1] + 0.04],
    #             [self.ego.x[0] + 0.08, self.ego.x[1] + 0.04],
    #             [self.ego.x[0] + 0.12, self.ego.x[1] + 0.04],
    #         ],
    #     ]

    #     if DEBUG_INFORMATION_GAIN:
    #         self.obs_pts = []
    #         for row in obs_pts:
    #             grid_pts = []
    #             for pt in row:
    #                 x = int((pt[0] - self.ego.x[0]) / GRID_RESOLUTION + GRID_SIZE // 2)
    #                 y = int((pt[1] - self.ego.x[1]) / GRID_RESOLUTION + GRID_SIZE // 2)
    #                 grid_pts.append([x, y])
    #             self.obs_pts.append(grid_pts)

    #     ig_results = []
    #     for row in obs_pts:
    #         total_ig = 0
    #         for pt in row:
    #             total_ig += self._calculate_information_gain_from(
    #                 pt, occupancy_grid=occupancy_grid
    #             )
    #         ig_results.append(total_ig)

    #     return ig_results

    # def _calculate_information_gain_from(self, location, occupancy_grid):
    #     # only one position to sample from, map it to the grid, relative to the AV
    #     obs_x = ((location[0] - self.ego.x[0]) / GRID_RESOLUTION) + GRID_SIZE // 2
    #     obs_y = ((location[1] - self.ego.x[1]) / GRID_RESOLUTION) + GRID_SIZE // 2
    #     obs_pts = np.array([obs_x, obs_y]).reshape(1, 2)

    #     values = occupancy_grid[self.roi[:, 1], self.roi[:, 0]]

    #     # results are num observation points rows by num region of interest points columns
    #     result = np.zeros((obs_pts.shape[0], self.roi.shape[0]))

    #     visibility_from_region(occupancy_grid, obs_pts, self.roi, result)

    #     # multiply the results by the probability of the cell being occupied based on current observation
    #     result = result * values
    #     log_result = -np.log(result + 0.00000001) * result
    #     self.log_result = log_result

    #     # BUGBUG: Notice that we are only calculating for one point at a time.
    #     return np.sum(log_result)

    def draw_control_output(self, u):
        start = self.ego.x[0:2]
        states = self.ego.project(u, self.tick_time)

        for state in states:
            self._draw_line(start, state[0:2], colour="blue", width=1)
            start = state[0:2]
            self._draw_circle(start, radius=0.23, colour="blue")

    def draw_obs(self):
        if DEBUG_INFORMATION_GAIN:
            obs = (self.obs.probabilityMap() * 255.0).astype(np.uint8)

            if self.ig_images is None:
                num_maps = 1
                num_rows = 1
                self.ig_fig, self.ig_ax = plt.subplots(num_rows, num_maps, num=FIG_IG_MAPS, figsize=(15, 15))

                self.ig_images = []
                self.ig_images.append(self.ig_ax.imshow(np.zeros([GRID_SIZE, GRID_SIZE, 3], dtype=np.uint8)))

            map_img = Image.fromarray(obs).convert("RGB")
            self.ig_images[0].set_data(map_img)

            self.ig_fig.canvas.draw()
            self.ig_fig.canvas.flush_events()

    # def draw_information_gain(self, ig_results):
    #     if DEBUG_INFORMATION_GAIN:
    #         min_ig = np.min(ig_results)
    #         max_ig = np.max(ig_results) - min_ig
    #         normalized_ig = ((ig_results - min_ig) / max_ig * 255.0).astype(np.uint8)

    #         if self.ig_images is None:
    #             num_maps = 2
    #             num_rows = 1
    #             self.ig_fig, self.ig_ax = plt.subplots(
    #                 num_rows, num_maps, num=FIG_IG_MAPS, figsize=(15, 15)
    #             )

    #             self.ig_images = []
    #             self.ig_images.append(
    #                 self.ig_ax[0].imshow(
    #                     np.zeros([GRID_SIZE, GRID_SIZE, 3], dtype=np.uint8)
    #                 )
    #             )
    #             self.ig_images.append(
    #                 self.ig_ax[1].imshow(np.zeros([3, 1, 3], dtype=np.uint8))
    #             )

    #         grid = (renderOccupancyGrid(self.dogm) * 255.0).astype(np.uint8)
    #         for pt in self.roi:
    #             grid[pt[1], pt[0]] = 128
    #         for row in self.obs_pts:
    #             for pt in row:
    #                 grid[pt[1], pt[0]] = 250

    #         map_img = Image.fromarray(grid).convert("RGB")
    #         self.ig_images[0].set_data(map_img)

    #         ig_results = np.array(ig_results).reshape(3, 1)
    #         min_ig_sum = np.min(ig_results)
    #         max_ig_sum = np.max(ig_results) - min_ig_sum
    #         ig_img = ((ig_results - min_ig_sum) / max_ig_sum * 255.0).astype(np.uint8)
    #         ig_img = Image.fromarray(ig_img).convert("RGB")
    #         self.ig_images[1].set_data(ig_img)

    #         # BUGBUG: Temporary code to draw the info gain values at the pixel level if required
    #         # num_maps = 3
    #         # num_rows = 3
    #         # if self.ig_val_images is None:
    #         #     self.ig_val_fig, self.ig_val_ax = plt.subplots(num_rows, num_maps, num=FIG_IG_MAPS+1, figsize=(15, 15))

    #         #     self.ig_val_images = []
    #         #     for i in range(num_rows):
    #         #         ig_img = []
    #         #         for j in range(num_maps):
    #         #             ig_img.append(self.ig_val_ax[i, j].imshow(np.zeros([GRID_SIZE//2, GRID_SIZE//2, 3], dtype=np.uint8)))
    #         #         self.ig_val_images.append(ig_img)

    #         # min_ig = np.min(self.log_result)
    #         # max_ig = np.max(self.log_result) - min_ig

    #         # self.log_result = (self.log_result - min_ig) / max_ig
    #         # for i in range(num_rows):
    #         #     for j in range(num_maps):
    #         #         map = np.zeros((GRID_SIZE//2, GRID_SIZE//2))
    #         #         vals = self.log_result[i*num_maps + j]
    #         #         for k, pt in enumerate(self.roi):
    #         #             map[pt[1]//2, pt[0]//2] = vals[k]

    #         #         map = Image.fromarray((map * 255.0).astype(np.uint8)).convert('RGB')
    #         #         self.ig_val_images[i][j].set_data(map)

    #         # self.ig_val_fig.canvas.draw()
    #         # self.ig_val_fig.canvas.flush_events()

    #         self.ig_fig.canvas.draw()
    #         self.ig_fig.canvas.flush_events()

    ##################################################################################
    # Simulator step functions
    ##################################################################################

    def _tick_actor(self, actor, tick_time):
        """step of simulation for each actor

        Args:
            actor_index (_type_): the index of the actor
        """
        actor.tick()

    def tick(self, action):
        """[summary]"""

        # one clock tick for the simulation time
        self.sim_time += self.tick_time
        self.ticks += 1

        # apply the requested action to the ego vehicle
        self.ego.set_control(action)

        if not self.limit_actors or self.generated_actors < self.num_actors:
            # self._generate_new_agents()
            self._generate_parkade()

        # move everyone
        finished_actors = []
        collisions = 0
        self.ego.tick(self.tick_time)
        for i, actor in enumerate(self.actor_list[::-1]):
            if actor.is_real():
                actor.tick(self.tick_time)

                # if not actor.collided and self.ego.is_colliding(actor):
                #     collisions += 1
                #     actor.set_collided()

            if actor.at_goal() or (
                actor.x[0] < self.ego.x[0] and actor.distance_to(self.ego.x[0:2]) > WINDOW_SIZE * 2 / 3
            ):
                finished_actors.append(actor)

        if abs(self.ego.x[STATE.Y]) > LANE_WIDTH:
            collisions += 1  # off the road

        # clean up
        passed = len(finished_actors)
        for actor in finished_actors:
            self.actor_list.remove(actor)

        # update the observation
        self.obs.decay(0.95)
        self.scan_data = self._calculate_future_visibility()
        observation = self._get_next_observation(self.scan_data, self.tick_time)
        # self.information_gain = self.calculate_information_gain(observation)

        # calculate the reward
        # y_reward = ((self.ego.x[1] - DESIRED_LANE_POSITION)**2)*REWARD_DEVIATION_Y
        y_reward = -exp(-((self.ego.x[1] - DESIRED_LANE_POSITION) ** 2))
        # v_reward = (self.actor_target_speed**2 - (self.actor_target_speed - self.ego.speed)**2)*REWARD_DEVIATION_V
        v_reward = exp(-((self.actor_target_speed - self.ego.x[STATE.VELOCITY]) ** 2))
        p_reward = passed * REWARD_PASSING
        # c_reward = collisions * REWARD_COLLISION
        g_distance = np.linalg.norm([self.ego.x[0] - GOAL[0], self.ego.x[1] - GOAL[1]])
        if g_distance < GOAL[0]:
            d_reward = 1 - (g_distance / GOAL[0]) ** (0.4)
            if g_distance < 0.05:
                print("Reached goal!")
                done = True
        else:
            d_reward = 0
        # self.last_distance = g_distance

        # check if this episode is finished
        done = collisions != 0

        g_reward = 0
        # if done:
        #     g_reward = (GOAL[0] - g_distance) / GOAL[0] * REWARD_GOAL

        reward = g_reward + v_reward + y_reward + p_reward + d_reward

        # print(f'Reward: v:{v_reward}, y:{y_reward}, passing: {p_reward}, goal:{g_reward}, total: {reward}')

        return observation, reward, done, self._get_info()

    def render(self, debug=False, u=None):
        if self.window is not None:
            self.window.clear()

            self._draw_road()
            for actor in self.actor_list:
                self._draw_actor(actor)

            self._draw_ego()
            self._draw_visibility()
            # self._draw_status()

            self._draw_scan()

            if u is not None:
                self.draw_control_output(u)

        # self.draw_information_gain(self.information_gain)

        if debug:
            # if self.maps is None:
            #     self.maps = []
            #     num_plots = FORECAST_COUNT + 1
            #     self.map_fig, self.map_ax = plt.subplots(num_plots, 1, figsize=(5, 15))
            #     H, W, D = self.grid.get_grid_size()
            #     for i in range(num_plots):
            #         self.maps.append(self.map_ax[i].imshow(np.zeros([H, W, 3], dtype=np.uint8)))

            #     plt.show(block=False)

            # # draw the probability and velocity grids
            # forecast = flow(self.grid.get_probability_map(), self.grid.get_velocity_map(), scale=self.grid.resolution,
            #                 timesteps=FORECAST_COUNT, dt=FORECAST_INTERVAL, mode='bilinear')

            # for i, (prob, v) in enumerate(forecast):
            #     map_img = Image.fromarray(np.flipud(((1-prob)*255.0).astype(np.uint8))).convert('RGB')
            #     self.maps[i].set_data(map_img)

            if self.maps is None:
                self.map_fig, self.map_ax = plt.subplots(num=FIG_MAPS)
                self.maps = self.map_ax.imshow(np.ones((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8))
                plt.show(block=False)

            map_img = Image.fromarray((self.probability_map * 255.0).astype(np.uint8)).convert("RGB")
            self.maps.set_data(map_img)

            self.map_fig.canvas.draw()
            self.map_fig.canvas.flush_events()
