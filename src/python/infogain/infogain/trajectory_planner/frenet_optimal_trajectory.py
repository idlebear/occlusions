"""
The MIT License (MIT)

Copyright (c) 2016 - 2022 Atsushi Sakai and other contributors:
https://github.com/AtsushiSakai/PythonRobotics/contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Frenet optimal trajectory generator

author: Atsushi Sakai (@Atsushi_twi)

Ref:

- [Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame](https://www.researchgate.net/profile/Moritz_Werling/publication/224156269_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame/links/54f749df0cf210398e9277af.pdf)

- [Optimal trajectory generation for dynamic street scenarios in a Frenet Frame](https://www.youtube.com/watch?v=Cj6tAQe7UCY)

"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import math

try:
    from trajectory_planner import cubic_spline_planner
except:
    import cubic_spline_planner


class quintic_polynomial:
    def __init__(self, xs, vxs, axs, xe, vxe, axe, T):
        # calc coefficient of quintic polynomial
        self.xs = xs
        self.vxs = vxs
        self.axs = axs
        self.xe = xe
        self.vxe = vxe
        self.axe = axe

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array(
            [[T**3, T**4, T**5], [3 * T**2, 4 * T**3, 5 * T**4], [6 * T, 12 * T**2, 20 * T**3]]
        )
        b = np.array(
            [xe - self.a0 - self.a1 * T - self.a2 * T**2, vxe - self.a1 - 2 * self.a2 * T, axe - 2 * self.a2]
        )
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + self.a3 * t**3 + self.a4 * t**4 + self.a5 * t**5

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + 3 * self.a3 * t**2 + 4 * self.a4 * t**3 + 5 * self.a5 * t**4

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2 + 20 * self.a5 * t**3

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t**2

        return xt


class quartic_polynomial:
    def __init__(self, xs, vxs, axs, vxe, axe, T):
        # calc coefficient of quintic polynomial
        self.xs = xs
        self.vxs = vxs
        self.axs = axs
        self.vxe = vxe
        self.axe = axe

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * T**2, 4 * T**3], [6 * T, 12 * T**2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * T, axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + self.a3 * t**3 + self.a4 * t**4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + 3 * self.a3 * t**2 + 4 * self.a4 * t**3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt


class Frenet_path:
    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []
        self.stopping = False


def calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0, args):
    MAX_ROAD_WIDTH = args["max_road_width"]
    # D_ROAD_W = args["D_ROAD_W"]
    DT = args["time_tick"]
    # D_T_S = args["D_T_S"]

    target_speed = args["target_speed"]

    KJ = args["KJ"]
    KT = args["KT"]
    KD = args["KD"]
    KLON = args["KLON"]
    KLAT = args["KLAT"]

    if args["mode"] == "center_only":
        offset_list = [0]
    else:
        offset = -args["max_road_width"]
        if args["trajectory_fanout"]:
            fanout_step = offset / args["trajectory_fanout"]
        for i in range(args["trajectory_fanout"]):
            offset_list.append(offset)
            offset += fanout_step
        offset_list.append(0)
        for i in range(args["trajectory_fanout"]):
            offset += fanout_step
            offset_list.append(offset)

    # tv_list -- vehicle final velocity
    # tv_list = [TARGET_SPEED * 0.25, TARGET_SPEED * 0.5, TARGET_SPEED * 0.75, TARGET_SPEED]
    tv_list = [
        args["target_speed"],
    ]

    # time_list -- trajectory length (time)
    time_list = [t for t in np.arange(args["min_t"], args["max_t"] + 0.0001, args["step_t"])]

    frenet_paths = []

    # ----------
    # Regular timed paths, grouped by path length
    for Ti in time_list:
        for di in offset_list:
            fp = Frenet_path()
            lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

            for tv in tv_list:
                tfp = copy.deepcopy(fp)
                lon_qp = quartic_polynomial(s0, c_speed, 0, tv, 0.0, Ti)

                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                # square of diff from target speed
                ds = (target_speed - tfp.s_d[-1]) ** 2

                tfp.cd = KJ * Jp + KT * Ti + KD * tfp.d[-1] ** 2
                tfp.cv = KJ * Js + KT * Ti + KD * ds
                tfp.cf = KLON * tfp.cv + KLAT * tfp.cd

                frenet_paths.append(tfp)

    return frenet_paths


def calc_global_paths(fplist, csp):
    for fp in fplist:
        # calc global positions
        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            iyaw = csp.calc_yaw(fp.s[i])
            di = fp.d[i]
            fx = ix + di * math.cos(iyaw + math.pi / 2.0)
            fy = iy + di * math.sin(iyaw + math.pi / 2.0)
            fp.x.append(fx)
            fp.y.append(fy)

        # calc yaw and ds
        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(math.atan2(dy, dx))
            fp.ds.append(math.sqrt(dx**2 + dy**2))

        try:
            fp.yaw.append(fp.yaw[-1])
            fp.ds.append(fp.ds[-1])
        except IndexError:
            pass  # empty trajectory

        # calc curvature
        for i in range(len(fp.yaw) - 1):
            try:
                fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])
            except ZeroDivisionError:
                fp.c.append(0)

    return fplist


# def check_collision(fp, ob):
#     for i in range(len(ob[:, 0])):
#         d = [((ix - ob[i, 0]) ** 2 + (iy - ob[i, 1]) ** 2) for (ix, iy) in zip(fp.x, fp.y)]

#         collision = any([di <= ROBOT_RADIUS**2 for di in d])

#         if collision:
#             return False

#     return True


def check_paths(fplist, args):
    MAX_SPEED = args["max_v"]
    MAX_ACCEL = args["max_a"]
    MAX_CURVATURE = args["max_curvature"]

    okind = []
    for i in range(len(fplist)):
        # if fplist[i].stopping:
        #     if any([v > 2*MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
        #         continue
        #     okind.append(i)
        #     continue
        # if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
        #     continue
        # elif any([abs(a) > MAX_ACCEL for a in fplist[i].s_dd]):  # Max accel check
        #     continue
        # elif any([abs(c) > MAX_CURVATURE for c in fplist[i].c]):  # Max curvature check
        #     continue
        # elif not check_collision(fplist[i], ob):
        #     continue

        okind.append(i)

    return [fplist[i] for i in okind]


def frenet_optimal_planning(csp, s0, c_speed, c_d, c_d_d, c_d_dd, args):
    fplist = calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0, args)
    fplist = calc_global_paths(fplist, csp)
    # fplist = check_paths(fplist, ob, args)

    # # find minimum cost path
    # mincost = float("inf")
    # bestpath = None
    # for fp in fplist:
    #     if mincost >= fp.cf:
    #         mincost = fp.cf
    #         bestpath = fp

    return fplist


def generate_target_course(x, y):
    return cubic_spline_planner.Spline2D(x, y)
    # s = np.arange(0, csp.s[-1], 0.1)

    # rx, ry, ryaw, rk = [], [], [], []
    # for i_s in s:
    #     ix, iy = csp.calc_position(i_s)
    #     rx.append(ix)
    #     ry.append(iy)
    #     ryaw.append(csp.calc_yaw(i_s))
    #     rk.append(csp.calc_curvature(i_s))
    # return rx, ry, ryaw, rk, csp
