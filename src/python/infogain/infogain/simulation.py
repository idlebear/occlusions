
from copy import deepcopy
from importlib import import_module
from math import sqrt, exp
from matplotlib.pyplot import show
import numpy as np
import pygame
from random import random, expovariate, seed


# local functions/imports
from Actor import Actor, Pedestrian, Car
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
            pos=np.array([0.0, 0.0]),
            goal=None,
            speed=0,
            colour='red',
            outline_colour='darkred',
            scale=5
        )

        self.actor_list = []

        self.sim_time = 0.
        self.next_time = 0.
        self.sim_start_time = 0.

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
            int(self._xmargin + (location[0]-self.ego.pos[0] + 0.25)*self._env_size),
            int(self._ymargin + self._env_size - (location[1]-self.ego.pos[1]+0.5)*self._env_size)
        ]

    def _draw_rect(self, location, color, size):
        pygame.draw.rect(self.screen,
                         color,
                         (location[0] - size/2.0, location[1] - size/2.0, size, size))

    def _draw_road(self):
        x = int(self.ego.pos[0]-0.5)
        for _ in range(15):
            loc = self._get_location_on_screen((x, -int(self._env_size*0.0001)))
            pygame.draw.rect(self.screen, 'yellow', (loc[0], loc[1], int(self._env_size * 0.1), int(self._env_size*0.005)))
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

        actor_screen_pos = self._get_location_on_screen(actor.pos)

        rot = np.array(
            [
                [np.cos(actor.orientation), np.sin(actor.orientation)],
                [-np.sin(actor.orientation), np.cos(actor.orientation)],
            ]
        )

        pts = np.array([
            [5, 0],
            [-5, 4],
            [-2, 0],
            [-5, -4],
            [5, 0],
        ]) * actor.scale

        pts = ((rot @ pts.T) + np.array(actor_screen_pos).reshape([2, 1])).T

        pygame.draw.polygon(self.screen, actor.colour, pts, 0)
        pygame.draw.polygon(self.screen, actor.outline_colour, pts, ACTOR_PATH_WIDTH)

    def _draw_status(self):

        collisions_str = f'Collisions: {self.collisions}'

        text_width, text_height = self.status_font.size(collisions_str)

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

        while (len(self.actor_list) < self.num_actors):
            x = self.ego.pos[0]
            dx = self.generator.uniform(low=0.25, high=0.75)

            if self.generator.uniform() < 0.5:
                actor = Car(
                    id=self.ticks,
                    pos=np.array([x+dx, 0.25]),
                    goal=np.array([x+dx, -0.25]),
                    speed=0.75,
                    scale=5
                )
            else:
                actor = Pedestrian(
                    id=self.ticks,
                    pos=np.array([x+dx, 0.25]),
                    goal=np.array([x+dx, -0.25]),
                    speed=0.25,
                    scale=3.5
                )
            self.actor_list.append(actor)

        if max_simulation_time is not None:
            if self.sim_time > max_simulation_time:
                return -1

        self._policy.execute(self.ego, self.actor_list, self.sim_time, tick_time)

        finished_actors = []
        for i, actor in enumerate(self.actor_list[::-1]):
            actor.tick(tick_time)
            if not actor.collided and actor.distance_to(self.ego.pos) <= COLLISION_DISTANCE:
                self.collisions += 1
                actor.colour = 'black'
                actor.speed = 0
                actor.collided = True

            if actor.at_goal() or (actor.collided and actor.distance_to(self.ego.pos) > 0.5):
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

            #  draw the limits of the environment
            pygame.draw.rect(self.screen,
                             SCREEN_OUTLINE_COLOUR,
                             (self._xmargin-self._border_offset, self._ymargin-self._border_offset, self._env_size+self._border_offset*2, self._env_size+self._border_offset*2), 2)

            for actor in self.actor_list:
                self._draw_actor(actor)

            self._draw_actor(self.ego)

            self._draw_status()
