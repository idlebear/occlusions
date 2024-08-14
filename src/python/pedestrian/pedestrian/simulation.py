from copy import deepcopy
from importlib import import_module
from math import sqrt, exp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pygame
from random import random, expovariate, seed

import visilibity as vis

from Grid.OccupancyGrid import OccupancyGrid
from util.xkcdColour import XKCD_ColourPicker

# local functions/imports
from Actor import DeliveryBot, Pedestrian, Vehicle, STATE
from config import *
from polygpu import faux_scan, visibility_from_region


DEBUG = 0
FORECAST_COUNT = 5
FORECAST_INTERVAL = 0.1

# rewards - a high penalty for colliding with anything, a small penalty for
# deviating from the desired velocity, a slightly smaller one for deviating from
# the desired Y position, and a positive reward for moving forward
REWARD_COLLISION = -100000  # note that this includes leaving the road surface!
REWARD_DEVIATION_V = -100.0
REWARD_DEVIATION_Y = -100.0
REWARD_FORWARD_MOTION = 0.01  # a small positive reward for not dying


MAX_V = 10.0  # define a max v for scaling the observation output to keep it in the
# range [0,1]


def get_location(origin, location):
    return [location[0] - origin[0], 1 - (location[1] - origin[1])]


class Window:
    def __init__(self, screen, screen_width, screen_height, margin, display_origin=(0, 0), display_size=10.0):
        self.screen = screen

        self.sim_time_text = pygame.font.SysFont("dejavuserif", 15)
        self.elapsed_time_text = pygame.font.SysFont("dejavuserif", 10)
        self.status_font = pygame.font.SysFont("roboto", STATUS_FONT_SIZE)

        self._xmargin = margin * 0.5
        self._ymargin = margin * 0.5
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._env_size = min(screen_height, screen_width) - margin
        self._scale = self._env_size / display_size  # pixels per meter
        self._display_size = display_size
        self._border_offset = 10
        self._origin = display_origin

        self.tmp_screen = pygame.Surface((self.screen.get_width(), self.screen.get_height()), flags=pygame.SRCALPHA)

    # def _get_location_on_screen(self, origin, location):
    #     return [
    #         int(self._xmargin + (location[0]-origin[0] - EGO_X_OFFSET)*self._env_size),
    #         int(self._ymargin + (location[1]-origin[1] - EGO_Y_OFFSET)*self._env_size)
    #     ]

    def clear(self):
        self.screen.fill(SCREEN_BACKGROUND_COLOUR)

    def get_drawing_scale(self):
        return self._scale

    def draw_polyline(self, points, colour, width=2):
        start = points[0]
        for end in points[1:]:
            self.draw_line(start, end, colour, width)
            start = end

    def draw_line(self, start, end, colour, width=2):
        sx = self._xmargin + int((start[0] - self._origin[0]) * self._scale)
        sy = self._ymargin + int((start[1] - self._origin[1]) * self._scale)
        ex = self._xmargin + int((end[0] - self._origin[0]) * self._scale)
        ey = self._ymargin + int((end[1] - self._origin[1]) * self._scale)

        pygame.draw.line(self.screen, color=colour, start_pos=(sx, sy), end_pos=(ex, ey), width=width)

    def draw_circle(self, center, colour, radius=2):
        cx = self._xmargin + int((center[0] - self._origin[0]) * self._scale)
        cy = self._ymargin + int((center[1] - self._origin[1]) * self._scale)
        radius = int(radius * self._scale)

        pygame.draw.circle(self.screen, color=colour, center=(cx, cy), radius=radius)

    def draw_rect(self, colour, center, height, width=None):
        if width is None:
            width = height

        pygame.draw.rect(
            self.screen,
            colour,
            (
                self._xmargin + int((center[0] - width / 2.0 - self._origin[0]) * self._scale),
                self._ymargin + int((center[1] - height / 2.0 - self._origin[1]) * self._scale),
                int(width * self._scale),
                int(height * self._scale),
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
                self._xmargin + int((x - self._origin[0]) * self._scale),
                self._ymargin + int((y - self._origin[1]) * self._scale),
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
            self._xmargin + int((center[0] - self._origin[0]) * self._scale),
            self._ymargin + int((center[1] - self._origin[1]) * self._scale),
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
        tracks=None,
        ego_start=None,
        ego_goal=None,
        pois_lambda=0.01,
        screen=None,
        speed=ACTOR_SPEED,
        margin=SCREEN_MARGIN,
        screen_width=SCREEN_WIDTH,
        screen_height=SCREEN_HEIGHT,
        tick_time=TICK_TIME,
        record_data=False,
    ):
        self.record_data = record_data
        self.tick_time = tick_time

        # load the random generator method
        self.load_generator(generator_name=generator_name, generator_args=generator_args)

        self.num_actors = num_actors
        self.actor_target_speed = speed
        self.pois_lambda = pois_lambda
        if tracks is not None:
            self.track_data, self.display_offset, self.display_diff = self.load_tracks(tracks)
        else:
            self.track_data = None
            self.tracks = None
            self.display_diff = DEFAULT_DISPLAY_SIZE  # meters
            self.display_offset = [0, 0]

        if screen is not None or record_data:
            self.window = Window(
                screen=screen,
                screen_width=screen_width,
                screen_height=screen_height,
                margin=margin,
                display_size=self.display_diff,
                display_origin=self.display_offset,
            )
            self.image_scale = self.window.get_drawing_scale()
        else:
            self.window = None
            self.image_scale = 1.0

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

        self.ego_start = ego_start
        self.ego_goal = ego_goal

        colours = XKCD_ColourPicker()
        self.colours = colours.values(30, "red")

        # if DEBUG:
        #     self.maps = []
        #     num_plots = FORECAST_COUNT + 1
        #     self.map_fig, self.map_ax = plt.subplots(num_plots, 1, figsize=(5, 15))
        #     H,W,D = self.grid.get_grid_size()
        #     for i in range(num_plots):
        #         self.maps.append( self.map_ax[i].imshow(np.zeros([H, W, 3], dtype=np.uint8)) )

        #     plt.show(block=False)

        self.reset()

    def load_tracks(self, tracks):
        objects = {}
        parsed_lines = []

        min_x = np.inf
        max_x = -np.inf
        min_y = np.inf
        max_y = -np.inf

        with open(tracks, "rb") as f:
            # parse the file, separating each space separated line into a list of floats
            lines = f.readlines()

        for line in lines:
            frame, id, x, y = [float(x) for x in line.split()]
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
            parsed_lines.append([frame, id, x, y])

        diff_y = max_y - min_y
        diff_x = max_x - min_x

        if diff_x > diff_y:
            max_diff = diff_x
            min_y = min_y - (diff_x - diff_y) / 2
        else:
            max_diff = diff_y
            min_x = min_x - (diff_y - diff_x) / 2

        display_scale = 1.0 / max_diff
        display_offset = [min_x, min_y]

        # Frames are recorded at 25 frames/second, but only every 10th frame is
        # annotated (0.4s/sample).  Insert correct number of frames to match the desired time
        # step by interpolating between the frames.  For simulation step time of 0.1 seconds,
        # we have to insert four extra frames.

        # And, somewhat an artifact of the previous processing, frames increment in 10's
        # so the calculation becomes ((frame - last_frame) / 10) * (0.4 / dt)
        for line in parsed_lines:
            frame, id, x, y = line

            if id not in objects:
                objects[id] = [[x, y, frame]]
            else:
                # interpolate between the last frame and the current frame
                last_frame = objects[id][-1][2]
                additional_frames = int((frame - last_frame) / 10 * (0.4 / self.tick_time))
                step_x = (x - objects[id][-1][0]) / additional_frames
                step_y = (y - objects[id][-1][1]) / additional_frames
                for i in range(1, additional_frames + 1):
                    objects[id].append(
                        [
                            objects[id][-1][0] + step_x,
                            objects[id][-1][1] + step_y,
                            last_frame + 10.0 * (i / additional_frames),
                        ]
                    )

        return objects, display_offset, max_diff

    def reset(self):
        # reset the random number generator
        self.generator.reset()

        if self.ego_start is None:
            sx, sy = self.generator.random(n=2) * self.display_diff + self.display_offset
        else:
            sx = self.ego_start[0]
            if type(sx) is float:
                sx = self.display_diff * sx + self.display_offset[0]
            else:
                r = sx[1] - sx[0]
                sx = self.display_offset[0] + (r * float(self.generator.random(n=1)) + sx[0]) * self.display_diff
            sy = self.ego_start[1]
            if type(sy) is float:
                sy = self.display_diff * sy + self.display_offset[1]
            else:
                r = sy[1] - sy[0]
                sy = self.display_offset[1] + (r * float(self.generator.random(n=1)) + sy[0]) * self.display_diff

        if self.ego_goal is None:
            gx, gy = self.generator.random(n=2) * self.display_diff + self.display_offset
        else:
            gx = self.ego_goal[0]
            if type(gx) is float:
                gx = self.display_diff * gx + self.display_offset[0]
            else:
                r = gx[1] - gx[0]
                gx = self.display_offset[0] + (r * float(self.generator.random(n=1)) + gx[0]) * self.display_diff
            gy = self.ego_goal[1]
            if type(gy) is float:
                gy = self.display_diff * gy + self.display_offset[1]
            else:
                r = gy[1] - gy[0]
                gy = self.display_offset[1] + (r * float(self.generator.random(n=1)) + gy[0]) * self.display_diff

        self.ego = DeliveryBot(
            id=0,
            x=np.array([sx, sy, 0, 0, 0]),
            goal=[gx, gy],
            colour="red",
            outline_colour="darkred",
            resolution=GRID_RESOLUTION,
            image_name="robot",
            image_scale=self.image_scale,
        )
        self.ego.set_visible(True)

        self.actor_list = []
        if self.track_data is not None:
            self.tracks = deepcopy(self.track_data)
        else:
            self.tracks = None

        self.sim_time = 0.0
        self.next_time = 0.0
        self.sim_start_time = 0.0

        self.next_agent_x = -np.inf

        self.ticks = 0
        self.collisions = 0

        self.obs.reset()

        return self._get_next_observation(self._calculate_scan())

    def translate_coordinates(self, x, y):
        """
        Translate the x, y coordinates from the percentage of the sim ( range 0.0-1.0) to world coordinates
        """
        return [x * self.display_diff + self.display_offset[0], y * self.display_diff + self.display_offset[1]]

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

    def _draw_actor(self, actor):
        actor_image = actor.get_image()
        if actor_image is not None:
            actor_pos = actor.get_pos()
            # drawing with y inverted reverse the rotation to correct the display
            self.window.draw_image(image=actor_image, center=actor_pos, orientation=-actor.x[STATE.THETA])
        else:
            actor_poly = actor.get_poly()
            self.window.draw_polygon(
                outline_colour=actor.outline_colour,
                fill_colour=actor.colour,
                points=actor_poly,
            )

    def _draw_ego(self):
        self.window.draw_circle(self.ego.goal[:2], colour="red", radius=0.2)
        self._draw_actor(self.ego)

    def _draw_path(self, path, colours=["red"]):
        if type(path) == list:
            for i, p in enumerate(path):
                for pos in zip(p.x, p.y):
                    self.window.draw_circle(pos[:2], colour=colours[i % len(colours)], radius=0.05)
        else:
            for pos in zip(path.x, path.y):
                self.window.draw_circle(pos[:2], colour=colours[0], radius=0.05)

    def _draw_status(self):
        self.window.draw_status(self.collisions, self.sim_time)

    def _draw_visibility(self):
        vis_poly = self.calculate_visibility()
        if vis_poly is not None:
            pts = []
            for i in range(vis_poly.n()):
                pt = vis_poly[i]
                pts.append([pt.x(), pt.y()])

            self.window.draw_polygon(
                outline_colour="black",
                fill_colour=None,
                points=pts,
            )

    def draw_polyline(self, points, colour, width=2):
        self.window.draw_polyline(points, colour, width)

    ##################################################################################
    # Simulator step functions
    ##################################################################################

    def calculate_visibility(self):
        # calculate the visibility polygon
        shapes = []

        # environment poly is counter clockwise and large enough to be off screen
        ox = self.ego.x[0] + self.display_diff
        oy = self.ego.x[1] + self.display_diff
        shapes.append(
            vis.Polygon(
                [
                    vis.Point(ox, oy),
                    vis.Point(ox - self.display_diff * 2.0, oy),
                    vis.Point(ox - self.display_diff * 2.0, oy - self.display_diff * 2.0),
                    vis.Point(ox, oy - self.display_diff * 2.0),
                ]
            )
        )

        for actor in self.actor_list:
            pts = actor.get_poly()
            poly_pts = [vis.Point(pt[0], pt[1]) for pt in pts[0:-1]]
            shapes.append(vis.Polygon(poly_pts))

        vis_poly = None
        env = vis.Environment(shapes)
        if env.is_valid(EPSILON):
            observer = vis.Point(self.ego.x[0], self.ego.x[1])
            vis_poly = vis.Visibility_Polygon(observer, env, EPSILON)

        return vis_poly

    def _generate_new_agents(self):
        if self.tracks is not None:
            activated = []
            for id, track in self.tracks.items():
                if self.sim_time >= track[0][2] * self.tick_time:
                    self.actor_list.append(
                        Pedestrian(
                            id=str(int(id)) if type(id) is int or type(id) is float else id,
                            track=track.copy(),
                            image_name="pedestrian",
                            image_scale=self.image_scale,
                        )
                    )
                    activated.append(id)
            for id in activated:
                del self.tracks[id]

        else:
            while len(self.actor_list) < self.num_actors:
                rnd = self.generator.uniform()

                x = self.generator.random(n=2) * self.display_diff + self.display_offset
                goal = self.generator.random(n=2) * self.display_diff + self.display_offset
                heading = np.arctan2(goal[1] - x[1], goal[0] - x[0])
                v = float(0.2 + self.generator.random() * 1.0)

                actor = Pedestrian(
                    id=self.ticks,
                    x=np.array([x[0], x[1], heading, v, 0]),
                    goal=goal,
                    image_name="pedestrian",
                    image_scale=self.image_scale,
                )

                self.actor_list.append(actor)

    def _get_next_observation(self, scan_data):
        # update the observation
        self.obs.move_origin(self.ego.x[0:2])
        self.obs.update(
            X=[*self.ego.x[0:2], self.ego.x[STATE.THETA]],
            angle_min=SCAN_START_ANGLE,
            angle_inc=SCAN_ANGLE_INCREMENT,
            ranges=scan_data,
            min_range=0,
            max_range=SCAN_RANGE + 1,
        )
        return self.obs.probabilityMap()

    def _get_info(self):
        info = {}
        info["ego"] = self.ego.get_state()
        info["goal"] = self.ego.goal
        info["time"] = self.sim_time

        actor_states = []
        for actor in self.actor_list:
            if actor.distance_to(self.ego.x) <= SCAN_RANGE:
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

        info["map"] = None  # self._get_map()
        info["actors"] = actor_states
        info["information_gain"] = None  # self.information_gain
        return info

    def _calculate_scan(self):
        # create the scan of the environment

        # build a list of polygons in the environment
        polygons = []
        for actor in self.actor_list:
            polygons.append(actor.get_poly())
        polygons = np.array(polygons)

        scan_data, indices = faux_scan(
            polygons,
            origin=self.ego.x[0:2],
            angle_start=SCAN_START_ANGLE + self.ego.x[STATE.THETA],
            angle_inc=SCAN_ANGLE_INCREMENT,
            num_rays=SCAN_RAYS,
            max_range=SCAN_RANGE,
            resolution=SCAN_RESOLUTION,
        )

        # TODO: change this from binary visiblity to a count of the number of rays
        #       that hit each actor
        visible_actors = [self.actor_list[i] for i in list(set(indices)) if i < len(self.actor_list)]
        for actor in self.actor_list:
            if actor in visible_actors:
                actor.set_visible(True)
            else:
                actor.set_visible(False)

        # clear any rays that didn't hit anything
        scan_data[scan_data == -1] = SCAN_RANGE + 1
        return scan_data.astype(np.float32)

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
        self.ego.tick(self.tick_time)

        self._generate_new_agents()

        # move everyone
        finished_actors = []
        collisions = 0
        for i, actor in enumerate(self.actor_list[::-1]):
            actor.tick(self.tick_time)

            if actor.at_goal():
                finished_actors.append(actor)

        # clean up
        for actor in finished_actors:
            self.actor_list.remove(actor)

        # degrade previous sensor information
        self.obs.decay(0.95)

        # update the observation
        self.scan_data = self._calculate_scan()
        observation = self._get_next_observation(self.scan_data)
        info = self._get_info()

        # calculate the reward
        reward = 0

        # check if this episode is finished
        done = collisions != 0 or self.ego.at_goal()

        return observation, reward, done, info

    def render(self, actors=None, trajectories=None, trajectory_weights=None, horizon=1, path=None, prefix_str=None):
        if self.window is not None:
            self.window.clear()

            for actor in self.actor_list:
                try:
                    prediction = actors[actor.id][0]
                    for traj in prediction:
                        colour = self.colours[actor.serial % len(self.colours)]
                        self.draw_polyline(traj, colour=colour)
                    future = actor.project(horizon)
                    if future is not None:
                        self.draw_polyline(future, colour="black")
                except KeyError:
                    pass
                self._draw_actor(actor)

            self._draw_path(path)

            self._draw_ego()
            if trajectories is not None:
                min_weight = np.min(trajectory_weights)
                range_weight = np.max(trajectory_weights) - min_weight
                if range_weight:
                    trajectory_weights = (trajectory_weights - min_weight) / range_weight

                for weight, trajectory in zip(trajectory_weights, trajectories):
                    self.draw_polyline(trajectory, colour=[*EGO_TRAJECTORY_COLOUR, int(50 + weight * 205.0)])
            self._draw_visibility()
            self._draw_status()

            if prefix_str is None:
                prefix_str = "pedestrian"
            self.window.save_screen(f"results/{prefix_str}_{self.ticks:05}.png")

        if DEBUG:
            pass
            # # draw the probability and velocity grid
            # for i, (prob, v) in enumerate(forecast):
            #     map_img = Image.fromarray(np.flipud(((1 - prob) * 255.0).astype(np.uint8))).convert("RGB")
            #     self.maps[i].set_data(map_img)

            # self.map_fig.canvas.draw()
            # self.map_fig.canvas.flush_events()

    def futures(self, timesteps=1):
        # get the future states of the actors
        futures = []
        for actor in self.actor_list:
            future = actor.project(timesteps)
            if future is not None:
                futures.append(future)
        return futures
