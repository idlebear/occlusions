from config import DISTANCE_TOLERANCE, TICK_TIME, ACTOR_PATH_WIDTH
from math import sqrt, atan2, cos, sin
import numpy as np
import pygame


class Actor:
    def __init__(self, id=0, pos=[0, 0], goal=None, speed=1.0, colour='grey', outline_colour='darkgrey', scale=1.0):
        self.id = id
        self.pos = pos

        self.speed = speed
        self.reached_goal = False
        self.collided = False

        self.max_v = np.inf
        self.min_v = -np.inf
        self.max_brake = np.inf
        self.max_accel = np.inf
        self.max_delta = np.pi / 6.0

        self.colour = colour
        self.outline_colour = outline_colour
        self.scale = scale

        self.goal = goal
        if goal is not None:
            self.orientation = atan2((goal[1] - pos[1]), (goal[0] - pos[0]))
        else:
            # straight along the x-axis
            self.orientation = 0

        self.__update_v_and_rot()
        
    def distance_to(self, pos):
        return np.linalg.norm(self.pos - pos)

    def __move(self, dt):
        """move towards the goal
        """
        if not self.reached_goal:
            if self.goal is not None:
                dir = self.goal - self.pos
                dist = np.linalg.norm(dir)
            else:
                dist = np.inf

            if (dist > self.speed*dt):
                self.pos = np.round(self.pos + self.v * dt, 5)
            else:
                # arrived at the goal
                self.pos = self.goal
                self.reached_goal = True

            self.__update_v_and_rot()

    def set_goal(self, goal):
        self.goal = goal
        if goal is not None:
            dir = self.goal - self.pos
            self.orientation = atan2(dir[1], dir[0])
        else:
            self.orientation = 0
        self.__update_v_and_rot()

        self.reached_goal = False

    def __update_v_and_rot(self):
        self.v = np.round( np.array( [ self.speed * np.cos( self.orientation ), self.speed * np.sin(self.orientation)]), 5)
        self.rot = np.array(
            [
                [np.cos(self.orientation), -np.sin(self.orientation)],
                [np.sin(self.orientation), np.cos(self.orientation)],
            ]
        )
        poly = (self.rot @ self.get_poly().T).T + self.pos
        min_d = np.min( poly, axis=0 )
        max_d = np.max( poly, axis=0 )
        self.bounding_box = ( *min_d, *max_d )

    def accelerate(self, a, dt):
        a = np.clip(a, -self.max_brake, self.max_accel)
        self.speed = np.clip(self.speed + a * dt, self.min_v, self.max_v)
        self.__update_v_and_rot()
        
    def turn(self, delta, dt):
        delta = np.clip(delta, -self.max_delta, self.max_delta)
        self.orientation = self.orientation + delta * dt
        if self.orientation > np.pi:
            self.orientation -= 2 * np.pi
        elif self.orientation < -np.pi:
            self.orientation -= 2 * np.pi
        self.__update_v_and_rot()
        
    def get_v(self):
        return self.v

    def tick(self, dt=TICK_TIME):
        """a time step
        """
        if self.speed:
            self.__move(dt)

    def at_goal(self):
        return self.reached_goal

    def get_poly(self):
        return np.array([
            [0.01, 0.01],
            [-0.01, 0.01],
            [-0.01, -0.01],
            [0.01, -0.01],
            [0.01, 0.01],
        ]) * self.scale

    def get_width(self):
        return 0.02 * self.scale

    def contains( self, loc ):
        return loc[0] >= self.bounding_box[0] and loc[1] >= self.bounding_box[1] and loc[0] <= self.bounding_box[2] and loc[1] <= self.bounding_box[3]
            
    def set_collided( self, colour='black' ):
        self.colour = colour
        self.speed = 0
        self.__update_v_and_rot()
        self.collided = True

    def project( self, u, dt ):
        '''
        Project a future position based on a supplied control and state
        '''
        
        virt_self = type(self)( id=self.id, pos=self.pos, goal=self.goal, speed=self.speed )
        states = []

        for (a,delta) in u:
            
            virt_self.accelerate( a, dt )
            virt_self.turn( delta, dt )
            virt_self.tick(dt)

            states.append( (virt_self.pos, virt_self.orientation))

        return states
        

class Car(Actor):
    def __init__(self, id=0, pos=[0, 0], goal=None, speed=1, colour='lightblue', outline_colour='dodgerblue', scale=1):
        super().__init__(id, pos, goal, speed, colour, outline_colour, scale)
        self.max_v = 0.4
        self.min_v = 0
        self.max_brake = 0.75
        self.max_accel = 0.75

    def get_poly(self):
        return np.array([
            [0.025, 0],
            [-0.025, 0.02],
            [-0.01, 0],
            [-0.025, -0.02],
            [0.025, 0],
        ]) * self.scale

    def get_width(self):
        return 0.05 * self.scale

    @staticmethod
    def check_width(scale):
        return 0.05 * scale


class Pedestrian(Actor):
    def __init__(self, id=0, pos=[0, 0], goal=None, speed=1, colour='blue', outline_colour='darkblue', scale=1):
        super().__init__(id, pos, goal, speed, colour, outline_colour, scale)
        self.max_v = 0.3
        self.min_v = 0
        self.max_brake = 0.75
        self.max_accel = 0.75

    def get_poly(self):
        return np.array([
            [0.01, 0],
            [0, 0.015],
            [-0.01, 0],
            [0, -0.015],
            [0.01, 0],
        ]) * self.scale

    def get_width(self):
        return 0.035 * self.scale

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

    def get_poly(self):
        return np.array([
            [0.01, 0.01],
            [-0.01, 0.01],
            [-0.01, -0.01],
            [0.01, -0.01],
            [0.01, 0.01],
        ]) * self.scale

    def get_width(self):
        return 0.02 * self.scale

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

    def get_poly(self):
        return np.array([
            [0.01, 0.01],
            [-0.01, 0.01],
            [-0.01, -0.01],
            [0.01, -0.01],
            [0.01, 0.01],
        ]) * self.scale

    def get_width(self):
        return 0.02 * self.scale

    @staticmethod
    def check_width(scale):
        return 0.02 * scale
