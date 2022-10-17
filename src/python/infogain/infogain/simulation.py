
from copy import deepcopy
from importlib import import_module
from math import sqrt, exp
from matplotlib.pyplot import show
import numpy as np
import pygame
from random import random, expovariate, seed
import visilibity as vis


# local functions/imports
from Actor import Actor, Pedestrian, Car, Obstacle, Blank
from config import *


class Simulation:
    def __init__(self, policy_name, policy_args=None, generator_name='uniform', generator_args=None, num_actors=1, pois_lambda=0.01, screen=None, service_time=SERVICE_TIME,
                 speed=ACTOR_SPEED, margin=SCREEN_MARGIN, screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT,
                 max_time=MAX_SIMULATION_TIME, max_tasks=MAX_SERVICED_TASKS, record_data=False, sectors=1, delivery_log=None):
        self.num_actors = num_actors
        self.actor_speed = speed
        self.pois_lambda = pois_lambda
        self.screen = screen
        self.record_data = record_data

        if screen is not None or record_data:
            self.sim_time_text = pygame.font.SysFont('dejavuserif', 15)
            self.elapsed_time_text = pygame.font.SysFont('dejavuserif', 10)
            self.status_font = pygame.font.SysFont('roboto', STATUS_FONT_SIZE)

        self.service_time = service_time
        self._xmargin = margin * 0.5
        self._ymargin = margin * 0.5
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._env_size = screen_width - margin
        self._border_offset = 10
        self.max_time = max_time
        self.max_tasks = max_tasks

        self.delivery_log = delivery_log

        # load the draw method
        self.load_generator(generator_name=generator_name, generator_args=generator_args)

        # load the policy
        self.load_policy(policy_name=policy_name, policy_args=policy_args)

        # preload all the the tasks
        self.reset()

    def reset(self, task_list=None):
        # reset the random number generator
        self.generator.reset()

        self.ego = Car(
            id=0,
            pos=np.array([0.0, -LANE_WIDTH / 2]),
            goal=None,
            speed=0.7,
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

    def load_policy(self, policy_name, policy_args):
        # load the policy
        self.policy_name = policy_name
        self.policy_args = policy_args
        policy_mod = import_module('.'+self.policy_name, package='policies')
        self._policy = policy_mod.get_policy_fn(self.generator, policy_args)

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

    def _get_location_on_screen(self, location):
        return [
            int(self._xmargin + (location[0]-self.ego.pos[0] - EGO_X_OFFSET)*self._env_size),
            int(self._ymargin + self._env_size - (location[1] - EGO_Y_OFFSET)*self._env_size)
        ]

    def _get_actor_outline(self, actor, flip=False):
        flip = -1 if flip else 1
        rot = np.array(
            [
                [np.cos(actor.orientation), -flip * np.sin(actor.orientation)],
                [flip * np.sin(actor.orientation), np.cos(actor.orientation)],
            ]
        )
        return (rot @ actor.get_poly().T).T

    def _get_actor_outline_on_screen(self, actor):
        actor_screen_pos = self._get_location_on_screen(actor.pos)
        actor_screen_outline = self._get_actor_outline(actor, flip=True) * self._env_size

        return (actor_screen_outline + np.array(actor_screen_pos))

    def _draw_rect(self, location, color, size):
        pygame.draw.rect(self.screen,
                         color,
                         (location[0] - size/2.0, location[1] - size/2.0, size, size))

    def _draw_road(self):
        x = int(self.ego.pos[0]-0.5)
        y = 0

        loc = self._get_location_on_screen((x, LANE_WIDTH))
        pygame.draw.rect(self.screen, ROAD_COLOUR, (loc[0], loc[1], self._env_size * 2.5, 2*LANE_WIDTH*self._env_size))

        for _ in range(15):
            loc = self._get_location_on_screen((x,  y - int(self._env_size*0.0001)))
            pygame.draw.rect(self.screen, ROAD_MARKING_COLOUR, (loc[0], loc[1], int(self._env_size * 0.1), int(self._env_size*0.005)))
            x += 0.2

    def _draw_task(self, location, color, size, outlines=False):
        if not outlines:
            pygame.draw.circle(self.screen,
                               color,
                               (location[0], location[1]), size, 0)
        else:
            pygame.draw.circle(self.screen,
                               SCREEN_OUTLINE_COLOUR,
                               (location[0], location[1]), size, 2)

    def _draw_actor(self, actor):
        if type(actor) is not Blank:
            pts = self._get_actor_outline_on_screen(actor)
            pygame.draw.polygon(self.screen, actor.colour, pts, 0)
            pygame.draw.polygon(self.screen, actor.outline_colour, pts, ACTOR_PATH_WIDTH)

    def _draw_ego(self):
        self._draw_actor(self.ego)

        loc = self._get_location_on_screen(self.ego.pos)
        pygame.draw.circle(self.screen, color='tomato', center=(loc[0], loc[1]), radius=self.d_s*self._env_size, width=2)

    def _draw_visibility(self):
        if self.visibility is not None:
            pts = []
            for i in range(self.visibility.n()):
                pts.append(self._get_location_on_screen([self.visibility[i].x(), self.visibility[i].y()]))

            TGREEN = (150, 220, 150, 100)
            TBLACK = (0, 0, 0, 200)
            vis_screen = pygame.Surface((self.screen.get_width(), self.screen.get_height()), flags=pygame.SRCALPHA)
            pygame.draw.polygon(vis_screen, TGREEN, pts, 0)
            pygame.draw.polygon(vis_screen, TBLACK, pts, ACTOR_PATH_WIDTH)
            self.screen.blit(vis_screen, (0, 0))

    def _draw_status(self):

        #  draw the limits of the environment
        pygame.draw.rect(self.screen,
                         SCREEN_OUTLINE_COLOUR,
                         (self._xmargin-self._border_offset, self._ymargin-self._border_offset, self._env_size+self._border_offset*2, self._env_size+self._border_offset*2), 2)

        collisions_str = f'Collisions: {self.collisions}'
        time_str = f'Sim Time: {self.sim_time:.4f}'

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
        # pygame.draw.rect(self.screen,
        #                  SCREEN_OUTLINE_COLOUR,
        #                  (self._xmargin + STATUS_XMARGIN/2, self._xmargin + STATUS_YMARGIN, time_text_width + STATUS_XMARGIN, time_text_height + STATUS_YMARGIN), 2)

        text = self.status_font.render(time_str, False, STATUS_FONT_COLOUR)
        self.screen.blit(text, (self._xmargin+STATUS_XMARGIN, self._ymargin+STATUS_YMARGIN))

    ##################################################################################
    # Simulator step functions
    ##################################################################################
    def calculate_visibility(self):

        # the stopping distance
        self.d_s = self.ego.speed / self.ego.max_brake

        if self.ego.speed:
            self.d_o = self.d_s * OPPONENT_RISK_SPEED / self.ego.speed
        else:
            self.d_o = 0

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
                pts = self._get_actor_outline(actor) + actor.pos
                poly_pts = [vis.Point(pt[0], pt[1]) for pt in pts[-1:0:-1]]
                shapes.append(vis.Polygon(poly_pts))

        vis_poly = None
        env = vis.Environment(shapes)
        if env.is_valid(EPSILON):
            observer = vis.Point(self.ego.pos[0], self.ego.pos[1])
            vis_poly = vis.Visibility_Polygon(observer, env, EPSILON)

        return vis_poly

    ##################################################################################
    # Simulator step functions
    ##################################################################################

    def _tick_actor(self, actor, tick_time):
        """step of simulation for each actor

        Args:
            actor_index (_type_): the index of the actor
        """
        actor.tick()

        # TODO: should check for collisions....

    def tick(self, tick_time, max_simulation_time):
        """[summary]
        """

        # one clock tick for the simulation time
        self.sim_time += tick_time
        self.ticks += 1

        x = max(self.next_agent_x, self.ego.pos[0] + (1.5 + EGO_X_OFFSET))

        dx = 0.005 * self.generator.uniform()

        while (len(self.actor_list) < self.num_actors):
            rnd = self.generator.uniform()
            if rnd < 0.4:
                scale = 1 + 9 * self.generator.uniform()
                width = Obstacle.check_width(scale)

                if rnd < 0.25:
                    y = - LANE_WIDTH * 1.5 - self.generator.uniform()*0.3 - width/2
                else:
                    y = LANE_WIDTH * 1.5 + self.generator.uniform()*0.3 + width/2

                actor = Obstacle(
                    id=self.ticks,
                    pos=np.array([x+dx+width/2, y]),
                    speed=0.0,
                    scale=scale
                )
            elif rnd < 0.5:

                # oncoming traffic
                scale = 1
                width = Car.check_width(scale) * 2

                v = OPPONENT_CAR_SPEED
                y = LANE_WIDTH / 2

                actor = Car(
                    id=self.ticks,
                    pos=np.array([x+dx+width/2, y]),
                    goal=np.array([self.ego.pos[0]-EGO_X_OFFSET, y]),
                    speed=v,
                    scale=scale
                )
            elif rnd < 0.75:

                scale = 1
                width = Car.check_width(scale)

                v = OPPONENT_CAR_SPEED

                y = 0.3
                if rnd < 0.625:
                    y = -y

                actor = Car(
                    id=self.ticks,
                    pos=np.array([x+dx+width/2, y]),
                    goal=np.array([x+dx+width/2, -y]),
                    speed=v,
                    scale=scale
                )
            elif rnd < 0.95:

                scale = 1
                width = Pedestrian.check_width(scale)

                v = OPPONENT_PEDESTRIAN_SPEED
                y = 0.2
                if rnd < 0.7875:
                    y = -y

                actor = Pedestrian(
                    id=self.ticks,
                    pos=np.array([x+dx+width/2, y]),
                    goal=np.array([x+dx+width/2, -y]),
                    speed=v,
                    scale=scale
                )
            else:
                # do nothing (space)
                scale = 1 + 9 * self.generator.uniform()
                width = Blank.check_width(scale)

                y = 0.1 + self.generator.uniform()*0.3
                actor = Blank(
                    id=self.ticks,
                    pos=np.array([x+dx+width/2, y]),
                    speed=0.0,
                    scale=scale
                )

            self.actor_list.append(actor)
            dx += actor.get_width() + 0.005 * self.generator.uniform()
            self.next_agent_x = x + dx

        if max_simulation_time is not None:
            if self.sim_time > max_simulation_time:
                return -1

        # calculate the cone of danger for an assumed velocity -- this should be in the policy but we'll
        # do it here for now for visualization
        self.visibility = self.calculate_visibility()

        self._policy.execute(self.ego, self.actor_list, self.sim_time, tick_time)

        finished_actors = []
        for i, actor in enumerate(self.actor_list[::-1]):
            actor.tick(tick_time)
            if not actor.collided and actor.distance_to(self.ego.pos) <= COLLISION_DISTANCE:
                self.collisions += 1
                actor.colour = 'black'
                actor.speed = 0
                actor.collided = True

            if actor.at_goal() or (actor.pos[0] < self.ego.pos[0] and actor.distance_to(self.ego.pos) > 0.5):
                finished_actors.append(actor)

        self.ego.tick(tick_time)

        # clean up
        for actor in finished_actors:
            self.actor_list.remove(actor)

        for actor in self.actor_list:
            if actor.at_goal():
                actor.set_goal(self.generator.random(2))

        if self.screen is not None:

            #  draw the limits of the environment
            self.screen.fill(SCREEN_BACKGROUND_COLOUR)

            self._draw_road()

            for actor in self.actor_list:
                self._draw_actor(actor)

            self._draw_ego()

            # visibility first as it will (currently) nuke everything else
            self._draw_visibility()

            self._draw_status()
