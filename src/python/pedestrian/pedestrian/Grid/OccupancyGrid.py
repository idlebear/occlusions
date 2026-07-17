#
# Occupancy Grid
#
from math import ceil, floor, log, exp, sqrt, sin, cos, isnan, pi
import numpy as np
from random import shuffle

from scipy import signal
from threading import Lock
try:
    from shapely.geometry import Point, Polygon
except Exception:  # pragma: no cover - shapely is optional outside SDD runs
    Point = None
    Polygon = None

try:
    from Grid.OccupancyGrid import bresenham
except:
    from Grid import bresenham


# Clamp probability to maximum and minimum values to prevent overflow and subsequent
# irrational behaviour -- limiting the values closer to zero makes the occupancy grid more
# responsive to dynamic objects in the environment
MIN_PROBABILITY = -50.0
MAX_PROBABILITY = 50.0


"""
For each sensed element in the environment, we capture two categories of information:

    - the type of object
    - the the footprint of that object

Another thought is to have an occupancy grid for each class of object.  Objects that don't move, have
zero velocity and never move.  Objects that have a velocity are stored in their own gaussian grid.  Then, when
the time ticks over, the probabilistic occupancy grid is a the sum of all the occupied cells, but each
shifted according to its class.

"""


class OccupancyGrid:
    def __init__(
        self,
        dim,
        resolution=1,
        origin=(0, 0),
        invertY=False,
        pUnk=0.5,
        pOcc=0.7,
        pFree=0.3,
        static_polygons=None,
        origin_mode="center",
        static_p_occ=1.0 - 1.0e-6,
    ):
        self.dim = dim
        self.resolution = resolution
        self.pOcc = pOcc
        self.pFree = pFree
        self.pUnk = pUnk
        self.l0 = log(pUnk / (1 - pUnk))
        self.origin = origin
        if origin_mode not in {"center", "lower_left"}:
            raise ValueError("origin_mode must be 'center' or 'lower_left'")
        self.origin_mode = origin_mode
        self.grid_size = int(dim / resolution)
        self.base_grid = np.ones([self.grid_size, self.grid_size]) * self.l0
        self.static_p_occ = float(static_p_occ)
        self.static_polygons = list(static_polygons or [])
        self._rasterize_static_polygons()
        self.grid = np.array(self.base_grid)

        # included because either ROS or Carla is wierd and inverts the expected y axis
        self.invertY = -1 if invertY else 1

        # allocate a mutex/lock
        self.mutex = Lock()

    def reset(self, origin=(0.0, 0.0)):
        self.grid = np.array(self.base_grid)
        self.origin = origin

    def copy(self):
        dup = OccupancyGrid(
            self.dim,
            self.resolution,
            (self.origin[0], self.origin[1]),
            self.invertY == -1,
            self.pUnk,
            self.pOcc,
            self.pFree,
            static_polygons=[],
            origin_mode=self.origin_mode,
            static_p_occ=self.static_p_occ,
        )
        dup.base_grid = np.array(self.base_grid)
        dup.grid = np.array(self.grid)
        dup.static_polygons = list(self.static_polygons)
        return dup

    def world_to_cell(self, point):
        x, y = float(point[0]), float(point[1])
        if self.origin_mode == "lower_left":
            ix = int(floor((x - self.origin[0]) / self.resolution))
            iy = int(floor((y - self.origin[1]) / self.resolution))
        else:
            ix = int(floor((x - self.origin[0]) / self.resolution + self.grid_size // 2))
            iy = int(floor((y - self.origin[1]) / self.resolution + self.grid_size // 2))
        return ix, iy

    def cell_center_to_world(self, ix, iy):
        if self.origin_mode == "lower_left":
            return (
                self.origin[0] + (float(ix) + 0.5) * self.resolution,
                self.origin[1] + (float(iy) + 0.5) * self.resolution,
            )
        return (
            self.origin[0] + (float(ix) + 0.5 - self.grid_size // 2) * self.resolution,
            self.origin[1] + (float(iy) + 0.5 - self.grid_size // 2) * self.resolution,
        )

    def _polygon_points(self, static_polygon):
        if hasattr(static_polygon, "blocking") and not getattr(static_polygon, "blocking", True):
            return None
        if isinstance(static_polygon, dict):
            if not bool(static_polygon.get("blocking", True)):
                return None
            for key in ("points", "polygon", "vertices"):
                if key in static_polygon:
                    return static_polygon[key]
            return None
        return getattr(static_polygon, "points", static_polygon)

    def _rasterize_static_polygons(self):
        if not self.static_polygons:
            return
        if Polygon is None or Point is None:
            raise ImportError("shapely is required to rasterize static polygons")

        occ = np.clip(self.static_p_occ, 1.0e-9, 1.0 - 1.0e-9)
        occ_log_odds = log(occ / (1.0 - occ))
        for static_polygon in self.static_polygons:
            points = self._polygon_points(static_polygon)
            if points is None:
                continue
            points = np.asarray(points, dtype=float)
            if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
                continue
            polygon = Polygon(points[:, :2])
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if polygon.is_empty:
                continue

            minx, miny, maxx, maxy = polygon.bounds
            min_cell = self.world_to_cell((minx, miny))
            max_cell = self.world_to_cell((maxx, maxy))
            min_ix = max(0, min(min_cell[0], max_cell[0]) - 1)
            max_ix = min(self.grid_size - 1, max(min_cell[0], max_cell[0]) + 1)
            min_iy = max(0, min(min_cell[1], max_cell[1]) - 1)
            max_iy = min(self.grid_size - 1, max(min_cell[1], max_cell[1]) + 1)
            for iy in range(min_iy, max_iy + 1):
                for ix in range(min_ix, max_ix + 1):
                    if polygon.covers(Point(*self.cell_center_to_world(ix, iy))):
                        self.base_grid[iy, ix] = occ_log_odds

    def decay(self, rate):
        self.mutex.acquire()
        try:
            self.grid *= rate
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
            old_x_min = min(self.grid_size, dx)
            old_x_max = self.grid_size
            new_x_min = 0
            new_x_max = max(0, self.grid_size - dx)
        else:
            old_x_min = 0
            old_x_max = max(0, self.grid_size + dx)
            new_x_min = min(self.grid_size, -dx)
            new_x_max = self.grid_size

        if dy > 0:
            old_y_min = min(self.grid_size, dy)
            old_y_max = self.grid_size
            new_y_min = 0
            new_y_max = max(0, self.grid_size - dy)
        else:
            old_y_min = 0
            old_y_max = max(0, self.grid_size + dy)
            new_y_min = min(self.grid_size, -dy)
            new_y_max = self.grid_size

        tmp_grid = np.ones_like(self.grid) * self.l0
        if old_x_max - old_x_min > 0 and old_y_max - old_y_min > 0:
            tmp_grid[new_y_min:new_y_max, new_x_min:new_x_max] = self.grid[old_y_min:old_y_max, old_x_min:old_x_max]
        self.grid = tmp_grid

    def update(self, X, angle_min, angle_inc, ranges, min_range, max_range):
        self.mutex.acquire()
        try:
            # TODO: Decoupled update from centering to allow updates while maintaining the current position
            #
            # # keep the origin at the centre of the grid
            # self.move_origin(X[0:2])

            n = len(ranges)

            # indices = [i for i in range(n)]
            # # process the rays in a random order to allow for an even distribution in cases where time is
            # # limited and not every scan can be processed
            # shuffle(indices)

            for i in range(n):  # len(indices)):
                bearing = angle_min + angle_inc * i  # indices[i]
                r = ranges[i]  # [indices[i]]
                if r == -1:
                    continue

                if r > max_range:
                    r = max_range

                cellList = self.__inverseScanner(X, rng=r, bearing=bearing, alpha=self.resolution * 2)
                self.__updateCells(cellList, clip=False)

            self.grid = np.clip(self.grid, MIN_PROBABILITY, MAX_PROBABILITY)

        finally:
            self.mutex.release()

    def updateCells(self, cellList, free_only=False):
        self.mutex.acquire()
        try:
            self.__updateCells(cellList, free_only, clip=True)
        finally:
            self.mutex.release()

    def __updateCells(self, cellList, free_only=False, clip=True):
        if free_only:
            _min_prob = 0
        else:
            _min_prob = MIN_PROBABILITY

        cellList = np.asarray(cellList)
        if not len(cellList):
            return

        ix = cellList[:, 0].astype(np.intp, copy=False)
        iy = cellList[:, 1].astype(np.intp, copy=False)
        il = cellList[:, 2]

        mask = (ix > 0) & (ix < self.grid_size) & (iy > 0) & (iy < self.grid_size)
        if free_only:
            mask &= il < 0.5
        if not np.any(mask):
            return

        updates = np.log(il[mask] / (1 - il[mask])) - self.l0
        np.add.at(self.grid, (iy[mask], ix[mask]), updates)

        if clip:
            self.grid = np.clip(self.grid, _min_prob, MAX_PROBABILITY)

    def mungeCells(self, cellList):
        # Loop through each cell from measurement model
        for j in range(len(cellList)):
            ix = int(cellList[j][0])
            iy = int(cellList[j][1])
            il = cellList[j][2]

            # Calculate updated log odds
            self.grid[iy, ix] = clamp(
                self.grid[iy, ix] + log(il / (1 - il)) - self.l0, MIN_PROBABILITY, MAX_PROBABILITY
            )

    def __inverseScanner(self, X, rng, bearing, alpha=0.2):
        """
        Calculates the inverse measurement model for a laser scanner through
        raytracing with Bresenham's algorithm, assigns low probability of object
        on ray, high probability at end. Returns matrix of cell probabilities.
        (Based on code from the ME640 reference)

        Input:
            X = Robot pose
            rng = Range to measured object
            bearing = Angle to measured object
            alpha = uncertain depth of sensor

        Output:
            m = Matrix representing the inverse measurement model

        BUGBUG: Ignoring minimum range for now (an exercise left for the reader)
        """

        # shoot rays from the current position relative to the current centre
        sx = int((X[0] - self.origin[0]) / self.resolution + self.grid_size // 2)
        sy = int((X[1] - self.origin[1]) / self.resolution + self.grid_size // 2)

        rng = rng / self.resolution

        # Calculate position of measured object (endpoint of ray) and
        # verify that it is also within the current map
        ex = sx + floor(rng * cos(bearing + X[2]))
        ey = sy + floor(rng * sin(bearing + X[2]))

        # Get coordinates of all cells traversed by laser ray
        cx, cy = bresenham(sx, sy, ex, ey)

        n = len(cx)
        cx = cx.reshape([n, 1])
        cy = cy.reshape([n, 1])
        oc = self.pFree * np.ones([n, 1])

        # TODO: Implement a minimum range check here and don't update any cells that
        #       are too close.

        half_alpha = alpha / (2 * self.resolution)
        trim = 0
        dx = cx - sx
        dy = cy - sy

        for i in range(len(cx) - 1, -1, -1):
            if cx[i, 0] < 0 or cx[i, 0] > self.grid_size - 1 or cy[i, 0] < 0 or cy[i, 0] > self.grid_size - 1:
                # out of range
                trim += 1
                continue

            diff = rng - sqrt(dx[i, 0] * dx[i, 0] + dy[i, 0] * dy[i, 0])
            if abs(diff) < half_alpha:
                oc[i] = self.pOcc
            elif diff > half_alpha:
                break
        cellList = np.hstack([cx[0 : (n - trim), :], cy[0 : (n - trim), :], oc[0 : (n - trim), :]])

        return cellList

    def probabilityMap(self):
        # return the current state as a probabilistic representation
        self.mutex.acquire()
        try:
            return self.__probabilityMap()
        finally:
            self.mutex.release()

    def __probabilityMap(self):
        # return the current state as a probabilistic representation
        return 1.0 / (1.0 + np.exp(-self.grid))

    def obstacleMap(self):
        # return the current state as a free/occupied space representation
        self.mutex.acquire()
        try:
            return self.__obstacleMap()
        finally:
            self.mutex.release()

    def __obstacleMap(self):
        # return the current state as a free/occupied space representation
        return self.grid > -0.5

    def plot(self, fig, robotPos=None, title=None):
        """
        Plots an occupancy grid with belief probabilities shown in grayscale
        Input:
            fig = handle of figure to generate
        Output:
            Plot window showing occupancy grid
        """
        import matplotlib.pyplot as plt
        import matplotlib.image as img

        self.mutex.acquire()
        try:
            plt.figure(fig)
            plt.clf()
            ax = plt.subplot(111)
            plt.gca().set_aspect("equal", adjustable="box")
            if title:
                plt.title("Occupancy Grid Map (" + title + ")")
            else:
                plt.title("Occupancy Grid Map")

            # plt.axis( [self.xMin, self.xMax, self.yMin, self.yMax] )
            plt.axis("off")

            map = 1.0 - self.__probabilityMap()

            plt.imshow(map, cmap="Greys", vmin=0, vmax=1)

            if robotPos is not None:
                plt.plot(robotPos[0], robotPos[1], "b*")

            plt.show()

            if title:
                plt.savefig(title + ".png", bbox_inches="tight")
        finally:
            self.mutex.release()
