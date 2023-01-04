
#
# Occupancy Grid
#
from math import ceil, floor, log, exp, sqrt, sin, cos, isnan, pi
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from random import shuffle
from scipy import signal
from shapely.geometry import Polygon, Point, MultiPoint
from threading import Lock
import time

from policies.VelocityGrid.util import clamp

from config import MAX_CAR_SPEED

import polycheck

# Clamp probability to maximum and minimum values to prevent overflow and subsequent
# irrational behaviour
MIN_PROBABILITY = -60.0
MAX_PROBABILITY = 60.0


"""
For each sensed element in the environment, we store both the probability of occupancy and the
apparent velocity (assuming an oracle or some other subsystem provides it)

"""


class VelocityGrid:

    def __init__(self, height, width, resolution=1, origin=(0, 0), pUnk=0.5, pOcc=0.8, pFree=0.3, debug=False):
        self.resolution = resolution
        self.pOcc = pOcc
        self.logOcc = log(pOcc / (1-pOcc))
        self.pFree = pFree
        self.logFree = log(pFree / (1-pFree))
        self.l0 = log(pUnk/(1-pUnk))

        self.origin = origin
        self.grid_rows = int(height / resolution)
        self.grid_cols = int(width / resolution)

        self.debug = debug
        if self.debug:
            self.maps = []
            num_plots = 3
            self.map_fig, self.map_ax = plt.subplots(num_plots, 1, figsize=(5, 15))
            for i in range(num_plots):
                self.maps.append(self.map_ax[i].imshow(np.zeros([self.grid_rows, self.grid_cols, 3], dtype=np.uint8)))

            plt.show(block=False)

        # allocate a mutex/lock
        self.mutex = Lock()

        # complete the initialization
        self.reset()

    def reset(self):
        self.mutex.acquire()
        try:
            self.probabilityMap = np.ones([self.grid_rows, self.grid_cols]).astype(np.float32) * self.l0
            self.velocityMap = np.zeros([self.grid_rows, self.grid_cols, 2]).astype(np.float32)

            # set up a grid of points in a convenient form for checking visibility
            xx, yy = np.meshgrid(np.arange(self.grid_cols), np.arange(self.grid_rows), indexing='xy')
            xx = xx * self.resolution + 0.5 * self.resolution
            yy = yy * self.resolution + 0.5 * self.resolution
            self.grid_pts = np.array([[x, y] for x, y in zip(xx.flatten(), yy.flatten())])

        finally:
            self.mutex.release()

    def get_grid_size(self):
        return [*self.probabilityMap.shape, 3]

    def decay(self, rate):
        self.mutex.acquire()
        try:
            self.probabilityMap *= rate
        finally:
            self.mutex.release()

    def move_origin(self, new_origin):
        # calculate the offset that places the desired center in a block and
        # apply that shift to the stored value for the center.  The result is
        # a new center that is *close* to the desired value, but not exact
        # allowing the grid to keep the correct relative position of existing
        # occupancy
        dx = floor((new_origin[0] - self.origin[0]) / self.resolution)
        dy = floor((new_origin[1] - self.origin[1]) / self.resolution)

        if not dx and not dy:
            return

        self.origin = (self.origin[0] + dx * self.resolution, self.origin[1] + dy * self.resolution)

        if dx > 0:
            old_x_min = min(self.grid_cols, dx)
            old_x_max = self.grid_cols
            new_x_min = 0
            new_x_max = max(0, self.grid_cols - dx)
        else:
            old_x_min = 0
            old_x_max = max(0, self.grid_cols + dx)
            new_x_min = min(self.grid_cols, -dx)
            new_x_max = self.grid_cols

        if dy > 0:
            old_y_min = min(self.grid_rows, dy)
            old_y_max = self.grid_rows
            new_y_min = 0
            new_y_max = max(0, self.grid_rows - dy)
        else:
            old_y_min = 0
            old_y_max = max(0, self.grid_rows + dy)
            new_y_min = min(self.grid_rows, -dy)
            new_y_max = self.grid_rows

        if old_x_max - old_x_min > 0 and old_y_max - old_y_min > 0:
            tmp_prob_grid = np.ones_like(self.probabilityMap) * self.l0
            tmp_vel_grid = np.zeros_like(self.velocityMap)

            tmp_prob_grid[new_y_min:new_y_max, new_x_min:new_x_max] = self.probabilityMap[old_y_min:old_y_max, old_x_min:old_x_max]
            self.probabilityMap = tmp_prob_grid

            tmp_vel_grid[new_y_min:new_y_max, new_x_min:new_x_max, :] = self.velocityMap[old_y_min:old_y_max, old_x_min:old_x_max, :]
            self.velocityMap = tmp_vel_grid

    def update(self, X, visibility, agents):
        '''
        @desc Update the grid with the latest observation data -- note that this currently
        requires the use of rays to indicate line of sight.  With no actual lidar/scanner
        that might be an issue.
        '''
        self.mutex.acquire()
        try:

            view_polygon = None
            if visibility is not None:
                pts = []
                for i in range(visibility.n()):
                    pts.append([visibility[i].x(), visibility[i].y()])

                # view_polygon = Polygon(pts)
                # construct a copy of the view polygon, relative to the vehicle/origin
                view_polygon = np.array(pts) - self.origin

                res = np.zeros([len(self.grid_pts), 1]).reshape(-1, 1).astype(np.uint32)
                polycheck.contains(view_polygon, self.grid_pts, res)
                res = res.reshape(self.probabilityMap.shape)
                self.probabilityMap += res * (self.logFree - self.l0)

                # zero any existing velocities
                self.velocityMap *= 0

                # add the visible agents to the velocity map
                for agent in agents:
                    dx, dy = agent.get_size()
                    corrected_pos = agent.pos - self.origin

                    for x in np.arange(corrected_pos[0] - dx/2, corrected_pos[0] + dx/2, self.resolution):
                        for y in np.arange(corrected_pos[1] - dy/2, corrected_pos[1] + dy/2, self.resolution):
                            cx = int(x/self.resolution)
                            cy = int(y/self.resolution)

                            if cx >= 0 and cy >= 0:
                                try:
                                    self.probabilityMap[cy, cx] = self.probabilityMap[cy, cx] + self.logOcc - self.l0
                                    self.velocityMap[cy, cx, :] = agent.v
                                except IndexError:
                                    pass  # off the grid

                # for x in range(self.grid_cols):
                #     for y in range(self.grid_rows):
                #         cell_loc = np.array([self.origin[0] + (x+0.5) * self.resolution, self.origin[1] + (y+0.5) * self.resolution])

                #         assigned = False
                #         for agent in agents:
                #             if agent.contains(cell_loc):
                #                 self.probabilityMap[y, x] = self.probabilityMap[y, x] + self.logOcc - self.l0
                #                 self.velocityMap[y, x, :] = agent.v
                #                 assigned = True
                #                 break

                # if not assigned:
                #     # pretend we can see it...
                #     self.probabilityMap[y, x] = self.probabilityMap[y, x] + self.logFree - self.l0
                #     # TODO: This is hella-slow -- going to assume visibility for the time being since
                #     #       we aren't looking at occlusions from buildings, just dodging.  Simplify until
                #     #       we know we have a working system...
                # #     if view_polygon is not None and view_polygon.contains(Point(*cell_loc)):
                # #         self.probabilityMap[y, x] = self.probabilityMap[y, x] + self.logFree - self.l0

                self.probabilityMap = np.clip(self.probabilityMap, MIN_PROBABILITY, MAX_PROBABILITY)

            if self.debug:
                # draw the probability and velocity grid
                map_img = Image.fromarray(np.flipud(((1-self.__probabilityMap())*255.0).astype(np.uint8))).convert('RGB')
                self.maps[0].set_data(map_img)

                map_img = Image.fromarray(np.flipud((self.velocityMap[:, :, 0]/MAX_CAR_SPEED)*255.0)).astype(np.uint8).convert('RGB')
                self.maps[1].set_data(map_img)
                map_img = Image.fromarray(np.flipud((self.velocityMap[:, :, 1]/MAX_CAR_SPEED)*255.0)).astype(np.uint8).convert('RGB')
                self.maps[2].set_data(map_img)

                self.map_fig.canvas.draw()
                self.map_fig.canvas.flush_events()

        finally:
            self.mutex.release()

    def get_probability_map(self):
        # return the current state as a probabilistic representation
        self.mutex.acquire()
        try:
            return self.__probabilityMap()
        finally:
            self.mutex.release()

    def __probabilityMap(self):
        # return the current state as a probabilistic representation
        num = np.zeros_like(self.probabilityMap)
        denom = np.ones_like(self.probabilityMap)
        np.exp(self.probabilityMap, out=num)
        denom = denom + num
        return np.divide(num, denom)

    def get_velocity_map(self):
        # return the current state as a free/occupied space representation
        self.mutex.acquire()
        try:
            return self.__velocityMap()
        finally:
            self.mutex.release()

    def __velocityMap(self):
        return np.array(self.velocityMap)

    def plot(self, fig, robotPos=None, title=None):
        '''
        Plots an occupancy grid with belief probabilities shown in grayscale
        Input:
            fig = handle of figure to generate
        Output:
            Plot window showing occupancy grid
        '''
        import matplotlib.pyplot as plt
        import matplotlib.image as img

        self.mutex.acquire()
        try:
            plt.figure(fig)
            plt.clf()
            ax = plt.subplot(111)
            plt.gca().set_aspect('equal', adjustable='box')
            if title:
                plt.title('Occupancy Grid Map ('+title+')')
            else:
                plt.title('Occupancy Grid Map')

            # plt.axis( [self.xMin, self.xMax, self.yMin, self.yMax] )
            plt.axis('off')

            map = 1.-self.__probabilityMap()

            plt.imshow(map, cmap='Greys', vmin=0, vmax=1)

            if robotPos is not None:
                plt.plot(robotPos[0], robotPos[1], 'b*')

            plt.show()

            if title:
                plt.savefig(title+'.png', bbox_inches='tight')
        finally:
            self.mutex.release()
