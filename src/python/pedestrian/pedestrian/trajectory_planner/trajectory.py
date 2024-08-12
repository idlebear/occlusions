# Copyright (C) 2021 - idlebear

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from math import pi, sqrt, atan2, cos, sin, pow, ceil
import numpy as np

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from enum import IntEnum


class Cubic:

    class TrajVar(IntEnum):
        t = 0
        x = 1
        y = 2
        vx = 3
        vy = 4
        v = 5
        yaw = 6
        num_vals = 7

    def __init__(self, waypoints, dt):

        self.waypoints = waypoints
        self.dt = dt
        self.traj = None

        # calculate the angle/orientation for each segment
        self.angles = []
        last_wp = None
        for wp in self.waypoints:
            if last_wp is not None:
                if wp == last_wp:
                    self.angles.append(self.angles[-1])
                else:
                    dx = wp[0] - last_wp[0]
                    dy = wp[1] - last_wp[1]
                    self.angles.append(atan2(dy, dx))
            last_wp = wp

        # copy the last angle for the last waypoint
        self.angles.append(self.angles[-1])

    def trajectory(self, waypoint_speeds, yaw_i, yaw_f, cubic_fn=None):
        self.X = []
        self.V = []
        self.t = []
        self.yaw = []
        self.v = []

        v_iter = iter(waypoint_speeds)

        self.angles[-1] = yaw_f

        for i in range(1, len(self.waypoints)):

            if self.waypoints[i] == self.waypoints[i - 1]:
                continue

            if i < len(self.waypoints) - 1:
                angle_out = self.angles[i] - self.angles[i - 1]
                if angle_out > np.pi:
                    angle_out -= np.pi * 2
                elif angle_out < -np.pi:
                    angle_out -= np.pi * 2
                angle_out = angle_out / 2 + self.angles[i - 1]
            else:
                angle_out = self.angles[i]

            # initial speed is either the last one from the previous interval or
            # we're just starting
            try:
                v0 = self.V[-1]
                vi = sqrt(v0[0] * v0[0] + v0[1] * v0[1])
            except IndexError:
                vi = next(v_iter)
                v0 = (vi * cos(yaw_i), vi * sin(yaw_i))

            # with multiple speeds available, we can limit the acceleration over
            # each interval and keep it within reasonable bounds
            try:
                vf = next(v_iter)
            except StopIteration:
                # no more speeds -- reuse the final velocity
                vf = waypoint_speeds[-1]
            v1 = (vf * cos(angle_out), vf * sin(angle_out))

            # traversal time is the approximate (euclidean) distance divided by the
            # average desired velocity
            dx = self.waypoints[i][0] - self.waypoints[i - 1][0]
            dy = self.waypoints[i][1] - self.waypoints[i - 1][1]
            dist = sqrt(dx * dx + dy * dy)
            T = dist / ((vi + vf) / 2)
            try:
                t0 = self.t[-1]
            except IndexError:
                t0 = 0

            if cubic_fn is None:
                cubic_fn = Cubic.polynomial_time_scaling_3rd_order
            (ts, x, v) = cubic_fn(self.waypoints[i - 1], self.waypoints[i], v0, v1, t0, t0 + T, self.dt)

            self.t.extend(ts)
            self.V.extend(v)
            self.X.extend(x)

        # given all of the velocity vectors, calculate the orientation(s) of the robot and the
        # magnitude of the velocity
        self.yaw = [atan2(vy, vx) for vx, vy in self.V]
        self.v = [sqrt(vx * vx + vy * vy) for vx, vy in self.V]

    def np_trajectory(self, waypoint_speeds, yaw_i, yaw_f, cubic_fn=None):

        v_iter = iter(waypoint_speeds)
        self.angles[-1] = yaw_f

        for i in range(1, len(self.waypoints)):

            if self.waypoints[i] == self.waypoints[i - 1]:
                continue

            if i < len(self.waypoints) - 1:
                angle_out = self.angles[i] - self.angles[i - 1]
                if angle_out > np.pi:
                    angle_out -= np.pi * 2
                elif angle_out < -np.pi:
                    angle_out -= np.pi * 2
                angle_out = angle_out / 2 + self.angles[i - 1]
            else:
                angle_out = self.angles[i]

            # initial speed is either the last one from the previous interval or
            # we're just starting
            try:
                v0 = (self.traj[Cubic.TrajVar.vx, -1], self.traj[Cubic.TrajVar.vy, -1])
                vi = sqrt(v0[0] * v0[0] + v0[1] * v0[1])
            except TypeError:
                vi = next(v_iter)
                v0 = (vi * cos(yaw_i), vi * sin(yaw_i))

            # with multiple speeds available, we can limit the acceleration over
            # each interval and keep it within reasonable bounds
            try:
                vf = next(v_iter)
            except StopIteration:
                # no more speeds -- reuse the final velocity
                vf = waypoint_speeds[-1]
            v1 = (vf * cos(angle_out), vf * sin(angle_out))

            # traversal time is the approximate (euclidean) distance divided by the
            # average desired velocity
            dx = self.waypoints[i][0] - self.waypoints[i - 1][0]
            dy = self.waypoints[i][1] - self.waypoints[i - 1][1]
            dist = sqrt(dx * dx + dy * dy)
            T = dist / ((vi + vf) / 2)
            try:
                t0 = self.traj[Cubic.TrajVar.t, -1]
            except TypeError:
                t0 = 0

            if cubic_fn is None:
                cubic_fn = Cubic.np_polynomial_time_scaling_3rd_order
            results = cubic_fn(self.waypoints[i - 1], self.waypoints[i], v0, v1, t0, t0 + T, self.dt)

            try:
                self.traj = np.hstack([self.traj, results])
            except ValueError:
                self.traj = results

    def get_trajectory(self):
        return [[x, y, v, yaw] for [x, y], v, yaw in zip(self.X, self.v, self.yaw)]

    def get_np_trajectory(self):
        if self.traj is None:
            raise ValueError("Numpy Trajectory not initialized")
        return np.vstack(
            [
                self.traj[Cubic.TrajVar.x],
                self.traj[Cubic.TrajVar.y],
                self.traj[Cubic.TrajVar.v],
                self.traj[Cubic.TrajVar.yaw],
            ]
        ).T

    @staticmethod
    def test():
        dt = 0.05
        # waypoints = [[0, 0], [0.5, 0], [0.5, -0.5], [1, -0.5], [1, 0], [1, 0.5],
        #              [1.5, 0.5], [1.5, 0], [1.5, -0.5], [1, -0.5], [1, 0],
        #              [1, 0.5], [0.5, 0.5], [0.5, 0], [0, 0]]
        waypoints = [
            [0, 0],
            [1.5, 0],
        ]

        cubic_trajectory = Cubic(waypoints=waypoints, dt=dt)

        # set the target parameters
        initial_yaw = np.pi / 2.0
        final_yaw = np.pi / 2.0
        cubic_trajectory.trajectory(
            [0.25, 0.5], initial_yaw, final_yaw, cubic_fn=Cubic.polynomial_time_scaling_3rd_order
        )

        return cubic_trajectory

    @staticmethod
    def np_test():
        dt = 0.05
        waypoints = [
            [0, 0],
            [1.5, 0],
        ]

        cubic_trajectory = Cubic(waypoints=waypoints, dt=dt)

        # set the target parameters
        initial_yaw = np.pi / 2.0
        final_yaw = np.pi / 2.0
        cubic_trajectory.np_trajectory(
            [0.25, 0.5], initial_yaw, final_yaw, cubic_fn=Cubic.np_polynomial_time_scaling_3rd_order
        )

        return cubic_trajectory

    def plot(self, scale=0.1):
        fig, ax = plt.subplots(2, 1)

        ax[0].plot(self.t, [x for x, _ in self.X])
        ax[0].plot(self.t, [y for _, y in self.X])

        ax[1].plot(self.t, [x for x, _ in self.V])
        ax[1].plot(self.t, [y for _, y in self.V])

        ax[1].plot(self.t, self.v)

        fig2, ax2 = plt.subplots()
        for wp in self.waypoints:
            c = patches.Circle([wp[0], wp[1]], scale, linewidth=1, edgecolor="k", facecolor="gray")
            ax2.add_patch(c)

        for x, yaw in zip(self.X, self.yaw):
            ax2.plot((x[0], x[0] + 0.2 * cos(yaw)), (x[1], x[1] + 0.2 * sin(yaw)), "b-")

        ax2.plot([x for x, _ in self.X], [y for _, y in self.X])
        plt.gca().set_aspect("equal", adjustable="box")
        plt.show()

        plt.close(fig)
        plt.close(fig2)

    def np_plot(self, scale=0.1):
        fig, ax = plt.subplots(2, 1)

        ax[0].plot(self.traj[Cubic.TrajVar.t, :], self.traj[Cubic.TrajVar.x, :])
        ax[0].plot(self.traj[Cubic.TrajVar.t, :], self.traj[Cubic.TrajVar.y, :])

        ax[1].plot(self.traj[Cubic.TrajVar.t, :], self.traj[Cubic.TrajVar.vx, :])
        ax[1].plot(self.traj[Cubic.TrajVar.t, :], self.traj[Cubic.TrajVar.vy, :])

        ax[1].plot(self.traj[Cubic.TrajVar.t, :], self.traj[Cubic.TrajVar.v, :])

        fig2, ax2 = plt.subplots()
        for wp in self.waypoints:
            c = patches.Circle([wp[0], wp[1]], scale, linewidth=1, edgecolor="k", facecolor="gray")
            ax2.add_patch(c)

        ax2.plot(
            (
                self.traj[Cubic.TrajVar.x, :],
                self.traj[Cubic.TrajVar.x, :] + 0.2 * np.cos(self.traj[Cubic.TrajVar.yaw, :]),
            ),
            (
                self.traj[Cubic.TrajVar.y, :],
                self.traj[Cubic.TrajVar.y, :] + 0.2 * np.sin(self.traj[Cubic.TrajVar.yaw, :]),
            ),
            "b-",
        )

        ax2.plot(self.traj[Cubic.TrajVar.x, :], self.traj[Cubic.TrajVar.y, :])
        plt.gca().set_aspect("equal", adjustable="box")
        plt.show()

        plt.close(fig)
        plt.close(fig2)

    @staticmethod
    def eval_cubic(a0, a1, a2, a3, t):
        return (a0 * t * t * t + a1 * t * t + a2 * t + a3, 3.0 * a0 * t * t + 2 * a1 * t + a2)

    @staticmethod
    def _calculate_coefficients(p0, p1, v0, v1, T):
        # mat = np.array([
        #     0, 0, 0, 1,
        #     T*T*T, T*T, T, 1,
        #     0, 0, 1, 0,
        #     3*T*T, 2*T, 1, 0
        # ]).reshape((4, 4))
        # inv_T_mat = np.linalg.inv(mat)

        inv_T_mat = np.array(
            [
                2.0 / (T * T * T),
                -2.0 / (T * T * T),
                1.0 / (T * T),
                1.0 / (T * T),
                -3.0 / (T * T),
                3.0 / (T * T),
                -2.0 / T,
                -1.0 / T,
                0.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
            ]
        ).reshape((4, 4))

        x = np.array([p0[0], p1[0], v0[0], v1[0]]).reshape((-1, 1))
        y = np.array([p0[1], p1[1], v0[1], v1[1]]).reshape((-1, 1))

        ax = np.squeeze(inv_T_mat @ x)
        ay = np.squeeze(inv_T_mat @ y)

        return (ax, ay)

    @staticmethod
    def eval_cubic(a0, a1, a2, a3, t):
        return ((1 / 6.0) * a0 * t * t * t + (1 / 2.0) * a1 * t * t + a2 * t + a3, (1 / 2.0) * a0 * t * t + a1 * t + a2)

    #
    # Cubic spline based on an informal paper by G.W.Lucas, published at the Rossum project:
    #
    #   http://rossum.sourceforge.net/papers/CalculationsForRobotics/CubicPath.htm
    #
    # This one is approximately twice as fast as the lab version, with similar results.  There
    # is a small variation between the two that expands as the trajectory gets longer.  Don't
    # know yet if this is significant.
    #
    @staticmethod
    def polynomial_time_scaling_3rd_order(p0, p1, v0, v1, t0, t1, dt):
        """
        Given a position, velocity and orientation for both the starting and ending position, as well
        as the time interval, derive the cubic equation that describes the trajectory.
        """

        # total duration
        T = t1 - t0

        ax = 6.0 * ((v1[0] + v0[0]) * T - 2.0 * (p1[0] - p0[0])) / pow(T, 3)
        bx = -2.0 * ((v1[0] + 2.0 * v0[0]) * T - 3.0 * (p1[0] - p0[0])) / pow(T, 2)

        ay = 6.0 * ((v1[1] + v0[1]) * T - 2.0 * (p1[1] - p0[1])) / pow(T, 3)
        by = -2.0 * ((v1[1] + 2.0 * v0[1]) * T - 3.0 * (p1[1] - p0[1])) / pow(T, 2)

        X = []
        V = []
        ts = []
        for t in np.arange(0, T, dt):
            ts.append(t + t0)
            (x, vx) = Cubic.eval_cubic(ax, bx, v0[0], p0[0], t)
            (y, vy) = Cubic.eval_cubic(ay, by, v0[1], p0[1], t)
            X.append((x, y))
            V.append((vx, vy))

        # add the final endpoint
        ts.append(t1)
        (x, vx) = Cubic.eval_cubic(ax, bx, v0[0], p0[0], T)
        (y, vy) = Cubic.eval_cubic(ay, by, v0[1], p0[1], T)
        X.append((x, y))
        V.append((vx, vy))

        return (ts, X, V)

    #
    # Polynomial version, reimplmented to use a single numpy array for output instead of
    # growing lists.
    #
    @staticmethod
    def np_polynomial_time_scaling_3rd_order(p0, p1, v0, v1, t0, t1, dt):
        """
        Given a position, velocity and orientation for both the starting and ending position, as well
        as the time interval, derive the cubic equation that describes the trajectory.
        """

        # total duration
        T = t1 - t0

        ax = 6.0 * ((v1[0] + v0[0]) * T - 2.0 * (p1[0] - p0[0])) / pow(T, 3)
        bx = -2.0 * ((v1[0] + 2 * v0[0]) * T - 3.0 * (p1[0] - p0[0])) / pow(T, 2)

        ay = 6.0 * ((v1[1] + v0[1]) * T - 2.0 * (p1[1] - p0[1])) / pow(T, 3)
        by = -2.0 * ((v1[1] + 2.0 * v0[1]) * T - 3.0 * (p1[1] - p0[1])) / pow(T, 2)

        n = ceil(T / dt)
        results = np.zeros((Cubic.TrajVar.num_vals, n + 1))

        for i in range(n):
            t = dt * i
            (x, vx) = Cubic.eval_cubic(ax, bx, v0[0], p0[0], t)
            (y, vy) = Cubic.eval_cubic(ay, by, v0[1], p0[1], t)
            results[:, i] = [t + t0, x, y, vx, vy, sqrt(vx * vx + vy * vy), atan2(vy, vx)]

        # need to insert the final point, which may or may not be an integer
        # multiple of dt...
        (x, vx) = Cubic.eval_cubic(ax, bx, v0[0], p0[0], T)
        (y, vy) = Cubic.eval_cubic(ay, by, v0[1], p0[1], T)
        results[:, n] = [t1, x, y, vx, vy, sqrt(vx * vx + vy * vy), atan2(vy, vx)]

        return results


if __name__ == "__main__":
    import timeit

    num = 1000
    t = timeit.timeit(stmt="Cubic.np_test()", setup="from __main__ import Cubic", number=num)
    print("time: {}, per iteration: {}".format(t, t / num))

    t = timeit.timeit(stmt="Cubic.test()", setup="from __main__ import Cubic", number=num)
    print("time: {}, per iteration: {}".format(t, t / num))

    whatever = Cubic.np_test()
    whatever.np_plot()
