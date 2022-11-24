
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

# local functions/imports
from Actor import Actor, Pedestrian, Car, Obstacle, Blank
from config import *

DEBUG = 0
FORECAST_COUNT = 5
FORECAST_INTERVAL = 0.1

# rewards - a high penalty for colliding with anything, a small penalty for
# deviating from the desired velocity, a slightly smaller one for deviating from
# the desired Y position, and a positive reward for moving forward
REWARD_COLLISION = -100000     # note that this includes leaving the road surface!
REWARD_DEVIATION_V = -10.0
REWARD_DEVIATION_Y = -100.0
REWARD_FORWARD_MOTION = 0.01   # a small positive reward for not dying

DESIRED_LANE_POSITION = -LANE_WIDTH / 2

MAX_V = 10.0  # define a max v for scaling the observation output to keep it in the
# range [0,1]


def get_location(origin, location):
    return [location[0]-origin[0] - EGO_X_OFFSET, 1 - location[1] + EGO_Y_OFFSET]


class Window:
    def __init__(self, screen, screen_width, screen_height, margin):
        self.screen = screen

        self.sim_time_text = pygame.font.SysFont('dejavuserif', 15)
        self.elapsed_time_text = pygame.font.SysFont('dejavuserif', 10)
        self.status_font = pygame.font.SysFont('roboto', STATUS_FONT_SIZE)

        self._xmargin = margin * 0.5
        self._ymargin = margin * 0.5
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._env_size = screen_width - margin
        self._border_offset = 10

        self.tmp_screen = pygame.Surface((self.screen.get_width(), self.screen.get_height()), flags=pygame.SRCALPHA)

    def _get_location_on_screen(self, origin, location):
        return [
            int(self._xmargin + (location[0]-origin[0] - EGO_X_OFFSET)*self._env_size),
            int(self._ymargin + self._env_size - (location[1] - EGO_Y_OFFSET)*self._env_size)
        ]

    def clear(self):
        self.screen.fill(SCREEN_BACKGROUND_COLOUR)

    def draw_rect(self, colour, location, height, width=None):
        if width is None:
            width = height

        pygame.draw.rect(
            self.screen,
            colour,
            (self._xmargin+(location[0] - width / 2.0)*self._env_size,
             self._ymargin+(location[1] - height / 2.0)*self._env_size,
             width * self._env_size,
             height * self._env_size
             )
        )

    def draw_polygon(self, outline_colour, fill_colour, points, use_transparency=False):
        points = [[self._xmargin+x*self._env_size, self._ymargin+y*self._env_size] for x, y in points]

        if use_transparency:
            self.tmp_screen.fill((0, 0, 0, 0))
            if fill_colour is not None:
                pygame.draw.polygon(self.tmp_screen, fill_colour, points, 0)
            pygame.draw.polygon(self.tmp_screen, outline_colour, points, ACTOR_PATH_WIDTH)
            self.screen.blit(self.tmp_screen, (0, 0))
        else:
            if fill_colour is not None:
                pygame.draw.polygon(self.screen, fill_colour, points, 0)
            pygame.draw.polygon(self.screen, outline_colour, points, ACTOR_PATH_WIDTH)

    def draw_status(self, collisions, sim_time):
        #  draw the limits of the environment
        pygame.draw.rect(self.screen,
                         SCREEN_OUTLINE_COLOUR,
                         (self._xmargin-self._border_offset, self._ymargin-self._border_offset, self._env_size+self._border_offset*2, self._env_size+self._border_offset*2), 2)

        collisions_str = f'Collisions: {collisions}'
        time_str = f'Sim Time: {sim_time:.4f}'

        text_width, text_height = self.status_font.size(collisions_str)
        time_text_width, time_text_height = self.status_font.size(time_str)

        x_avg_offset = self._env_size + self._xmargin - text_width - STATUS_XMARGIN*2
        y_avg_offset = self._env_size + self._ymargin - text_height - STATUS_YMARGIN

        pygame.draw.rect(self.screen,
                         SCREEN_BACKGROUND_COLOUR,
                         (x_avg_offset-STATUS_XMARGIN, y_avg_offset-STATUS_YMARGIN, text_width + STATUS_XMARGIN*2, text_height + STATUS_YMARGIN), 0)
        pygame.draw.rect(self.screen,
                         SCREEN_OUTLINE_COLOUR,
                         (x_avg_offset-STATUS_XMARGIN, y_avg_offset-STATUS_YMARGIN, text_width + STATUS_XMARGIN*2, text_height + STATUS_YMARGIN), 2)
        text = self.status_font.render(collisions_str, False, STATUS_FONT_COLOUR)
        self.screen.blit(text, (x_avg_offset+STATUS_XMARGIN/3, y_avg_offset))

        pygame.draw.rect(self.screen,
                         SCREEN_BACKGROUND_COLOUR,
                         (self._xmargin + STATUS_XMARGIN/2, self._xmargin + STATUS_YMARGIN, time_text_width + STATUS_XMARGIN, time_text_height + STATUS_YMARGIN), 0)

        text = self.status_font.render(time_str, False, STATUS_FONT_COLOUR)
        self.screen.blit(text, (self._xmargin+STATUS_XMARGIN, self._ymargin+STATUS_YMARGIN))


class Simulation:
    def __init__(self, generator_name='uniform', generator_args=None, num_actors=1, pois_lambda=0.01, screen=None, service_time=SERVICE_TIME,
                 speed=ACTOR_SPEED, margin=SCREEN_MARGIN, screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT,
                 max_time=MAX_SIMULATION_TIME, tick_time=TICK_TIME, record_data=False, ):
        self.num_actors = num_actors
        self.actor_target_speed = speed
        self.pois_lambda = pois_lambda

        self.record_data = record_data
        self.max_time = max_time
        self.tick_time = tick_time

        self.service_time = service_time

        if screen is not None or record_data:
            self.window = Window(screen=screen, screen_width=screen_width, screen_height=screen_height, margin=margin)
        else:
            self.window = None

        # load the draw method
        self.load_generator(generator_name=generator_name, generator_args=generator_args)

        self.grid = VelocityGrid(height=GRID_HEIGHT, width=GRID_WIDTH, resolution=GRID_RESOLUTION, origin=(GRID_ORIGIN_Y_OFFSET, GRID_ORIGIN_Y_OFFSET))
        self.observation_shape = self.grid.get_grid_size()
        # if DEBUG:
        #     self.maps = []
        #     num_plots = FORECAST_COUNT + 1
        #     self.map_fig, self.map_ax = plt.subplots(num_plots, 1, figsize=(5, 15))
        #     H,W,D = self.grid.get_grid_size()
        #     for i in range(num_plots):
        #         self.maps.append( self.map_ax[i].imshow(np.zeros([H, W, 3], dtype=np.uint8)) )

        #     plt.show(block=False)

        self.reset()

    def reset(self):
        # reset the random number generator
        self.generator.reset()

        self.ego = Car(
            id=0,
            pos=np.array([0.0, DESIRED_LANE_POSITION]),
            goal=None,
            speed=0,
            colour='red',
            outline_colour='darkred',
            scale=1.1
        )

        self.actor_list = []

        self.sim_time = 0.
        self.next_time = 0.
        self.sim_start_time = 0.

        self.next_agent_x = -np.inf

        self.ticks = 0
        self.collisions = 0

        self.grid.reset()

        return self._get_next_observation()

    def load_generator(self, generator_name, generator_args):
        # load the generator
        self.generator_name = generator_name
        self.generator_args = generator_args
        gen_mod = import_module('.'+self.generator_name, package='generators')
        generator_fn = gen_mod.get_generator_fn()
        self.generator = generator_fn(**self.generator_args)

    ############################################################################
    # Plotting and drawing functions
    ############################################################################

    def _rotate_actor_outline(self, actor, flip=False):
        flip = -1 if flip else 1
        rot = np.array(
            [
                [np.cos(actor.orientation), -flip * np.sin(actor.orientation)],
                [flip * np.sin(actor.orientation), np.cos(actor.orientation)],
            ]
        )
        return (rot @ actor.get_poly().T).T

    def _get_actor_outline(self, actor):
        actor_pos = get_location(origin=self.ego.pos, location=actor.pos)
        actor_outline = self._rotate_actor_outline(actor, flip=True)

        return (actor_outline + np.array(actor_pos))

    def _draw_road(self):
        x = int(self.ego.pos[0]-0.5)
        y = 0

        loc = get_location(origin=self.ego.pos, location=(x, y))
        self.window.draw_rect(ROAD_COLOUR, (loc[0], loc[1]), 2 * LANE_WIDTH, 10)

        for _ in range(15):
            loc = get_location(origin=self.ego.pos, location=(x,  y - 0.0001))
            self.window.draw_rect(ROAD_MARKING_COLOUR, (loc[0], loc[1]), 0.005, 0.1)
            x += 0.2

    def _draw_actor(self, actor):
        if type(actor) is not Blank:
            pts = self._get_actor_outline(actor)
            self.window.draw_polygon(outline_colour=actor.outline_colour, fill_colour=actor.colour, points=pts)

    def _draw_ego(self):
        self._draw_actor(self.ego)

    def _draw_visibility(self):
        if self.visibility is not None:
            pts = []
            for i in range(self.visibility.n()):
                pts.append(get_location(origin=self.ego.pos, location=[self.visibility[i].x(), self.visibility[i].y()]))

            TGREEN = (150, 220, 150, 100)
            TBLACK = (0, 0, 0, 200)
            # TODO: Drawing this filled polygon is hella slow, taking the sim from faster than real-time to somewhere around
            #       half speed.  Removed for now, but it's just not as pretty...  there may be an update to resolve the issue
            #       in pygame, but for the time being, our poly is going to be clear....
            self.window.draw_polygon(outline_colour=TBLACK, fill_colour=None, points=pts, use_transparency=True)

    def _draw_status(self):
        self.window.draw_status(self.collisions, self.sim_time)

    ##################################################################################
    # Simulator step functions
    ##################################################################################

    def calculate_visibility(self):

        # calculate the visibility polygon
        shapes = []

        # environment poly is counter clockwise and large enough to be off screen
        ox = self.ego.pos[0]-2
        oy = -2
        shapes.append(
            vis.Polygon([
                vis.Point(ox, oy),
                vis.Point(ox+4.0, oy),
                vis.Point(ox+4.0, oy+4.0),
                vis.Point(ox, oy+4.0),
            ])
        )

        for actor in self.actor_list:
            if type(actor) is Blank:
                continue

            if actor.pos[0] > self.ego.pos[0]+EGO_X_OFFSET and actor.pos[0] < self.ego.pos[0]+EGO_X_OFFSET+1.5:
                pts = self._rotate_actor_outline(actor) + actor.pos
                poly_pts = [vis.Point(pt[0], pt[1]) for pt in pts[-1:0:-1]]
                shapes.append(vis.Polygon(poly_pts))

        vis_poly = None
        env = vis.Environment(shapes)
        if env.is_valid(EPSILON):
            observer = vis.Point(self.ego.pos[0], self.ego.pos[1])
            vis_poly = vis.Visibility_Polygon(observer, env, EPSILON)

        return vis_poly

    def _generate_new_agents(self):
        x = max(self.next_agent_x, self.ego.pos[0] + (1.0))

        while (len(self.actor_list) < self.num_actors):
            rnd = self.generator.uniform()
            if rnd < 0.3:
                scale = 1 + 9 * self.generator.uniform()
                width = Obstacle.check_width(scale) + 0.005

                if rnd < 0.25:
                    y = - LANE_WIDTH * 1.5 - self.generator.uniform()*0.1 - width/2
                else:
                    y = LANE_WIDTH * 1.5 + self.generator.uniform()*0.1 + width/2

                actor = Obstacle(
                    id=self.ticks,
                    pos=np.array([x+width/2, y]),
                    speed=0.0,
                    scale=scale
                )
            elif rnd < 0.6:

                # oncoming traffic
                scale = 1
                width = Car.check_width(scale) * 2

                v = OPPONENT_CAR_SPEED
                y = LANE_WIDTH / 2

                actor = Car(
                    id=self.ticks,
                    pos=np.array([x+width/2, y]),
                    goal=np.array([self.ego.pos[0]-EGO_X_OFFSET, y]),
                    speed=v,
                    scale=scale
                )
            elif rnd < 0.9:

                # same side traffic
                scale = 1
                width = Car.check_width(scale) * 5

                v = OPPONENT_CAR_SPEED*0.5
                y = -LANE_WIDTH / 2

                actor = Car(
                    id=self.ticks,
                    pos=np.array([x+width/2, y]),
                    goal=np.array([self.ego.pos[0]+100000, y]),
                    speed=v,
                    scale=scale
                )
            # elif rnd < 0.55:

            #     scale = 1
            #     width = Car.check_width(scale)

            #     v = OPPONENT_CAR_SPEED

            #     y = 2
            #     if rnd < 0.625:
            #         y = -y

            #     actor = Car(
            #         id=self.ticks,
            #         pos=np.array([x+width/2, y]),
            #         goal=np.array([x+width/2, -y]),
            #         speed=v,
            #         scale=scale
            #     )
            # elif rnd < 0.95:

            #     scale = 1
            #     width = Pedestrian.check_width(scale)

            #     v = OPPONENT_PEDESTRIAN_SPEED
            #     y = 0.4
            #     if rnd < 0.7875:
            #         y = -y

            #     actor = Pedestrian(
            #         id=self.ticks,
            #         pos=np.array([x+width/2, y]),
            #         goal=np.array([x+width/2, -y]),
            #         speed=v,
            #         scale=scale
            #     )
            else:
                # do nothing (space)
                scale = 1 + 9 * self.generator.uniform()
                width = Blank.check_width(scale)

                y = 0.1 + self.generator.uniform()*0.3
                actor = Blank(
                    id=self.ticks,
                    pos=np.array([x+width/2, y]),
                    speed=0.0,
                    scale=scale
                )

            self.actor_list.append(actor)
            x += width + 0.005 * self.generator.uniform()
            self.next_agent_x = x

    def _get_next_observation(self):
        # calculate the visibility polygon and use it to determine which agents are visible to the
        # AV.  Note that this is an alternative to implementing some sort LIDAR simulation
        self.visibility = self.calculate_visibility()

        if self.visibility is not None:
            actors = [self.ego]
            for actor in self.actor_list:
                if type(actor) is not Blank:
                    for i in range(self.visibility.n()):
                        if actor.contains((self.visibility[i].x(), self.visibility[i].y())):
                            actors.append(actor)
                            break

            # update the observation
            self.grid.update(self.ego.pos, visibility=self.visibility, agents=actors)

        observation = np.append(np.expand_dims(self.grid.get_probability_map(), axis=2), self.grid.get_velocity_map()/MAX_V, axis=2)
        return observation

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
        """[summary]
        """

        # one clock tick for the simulation time
        self.sim_time += self.tick_time
        self.ticks += 1

        # apply the requested action to the ego vehicle
        self.ego.accelerate(action[0], dt=self.tick_time)
        self.ego.turn(action[0], dt=self.tick_time)

        self._generate_new_agents()

        # move everyone
        finished_actors = []
        collisions = 0
        self.ego.tick(self.tick_time)
        for i, actor in enumerate(self.actor_list[::-1]):
            actor.tick(self.tick_time)
            if not actor.collided and actor.distance_to(self.ego.pos) <= COLLISION_DISTANCE:
                collisions += 1
                actor.set_collided()

            if actor.at_goal() or (actor.pos[0] < self.ego.pos[0] and actor.distance_to(self.ego.pos) > 0.5):
                finished_actors.append(actor)

        if abs(self.ego.pos[1]) > LANE_WIDTH:
            collisions += 1  # off the road

        # clean up
        for actor in finished_actors:
            self.actor_list.remove(actor)

        # update the observation
        observation = self._get_next_observation()

        # calculate the reward
        y_error = abs(self.ego.pos[1] - DESIRED_LANE_POSITION)
        v_error = abs(self.ego.speed - self.actor_target_speed)
        reward = collisions * REWARD_COLLISION + y_error * REWARD_DEVIATION_Y + v_error * REWARD_DEVIATION_V

        # check if this episode is finished
        done = collisions != 0

        return observation, reward, done, {}

    def render(self):
        if self.window is not None:
            self.window.clear()

            self._draw_road()
            for actor in self.actor_list:
                self._draw_actor(actor)

            self._draw_ego()
            self._draw_visibility()
            self._draw_status()

        # if DEBUG:
        #     # draw the probability and velocity grid
        #     for i, (prob,v) in enumerate(forecast):
        #         map_img = Image.fromarray(np.flipud(((1-prob)*255.0).astype(np.uint8))).convert('RGB')
        #         self.maps[i].set_data(map_img)

        #     self.map_fig.canvas.draw()
        #     self.map_fig.canvas.flush_events()
