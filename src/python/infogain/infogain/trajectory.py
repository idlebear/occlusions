#
# Simple trajectory planning interface to PythonRobotics code
#
from trajectory_planner.frenet_optimal_trajectory import (
    frenet_optimal_planning,
    generate_target_course,
)
from config import *
from numpy import arctan

import ModelParameters.Ackermann as Ackermann


def generate_trajectory(waypoints, s, d, v, t, dt):
    """
    initialize the frenet trajectory planner
    """

    s0 = s
    c_d = d
    c_d_d = 0
    c_d_dd = 0

    wx = [pt[0] for pt in waypoints]
    wy = [pt[1] for pt in waypoints]

    target_course = generate_target_course(wx, wy)

    args = {
        "mode": "center_only",
        "min_t": t,
        "max_t": t,
        "step_t": 1,
        "time_tick": dt,
        "target_speed": v,
        "trajectory_fanout": 0,
        # road parameters
        "max_curvature": 1,
        "max_road_width": LANE_WIDTH,
        "D_T_S": 3.0 / 3.6,  # target speed sampling length [m/s]
        # vehicle parameters
        "max_v": Ackermann.MAX_V,  # maximum speed [m/s]
        "min_v": Ackermann.MIN_V,  # minimum speed [m/s]
        "max_a": Ackermann.MAX_A,  # maximum acceleration [m/ss]
        "min_a": Ackermann.MIN_A,  # maximum acceleration [m/ss]
        # cost weights
        "KJ": 0.1,
        "KT": 0.1,
        "KD": 1.0,
        "KLON": 1.0,
        "KLAT": 1.0,
    }

    return frenet_optimal_planning(
        csp=target_course,
        s0=s0,
        c_speed=v,
        c_d=c_d,
        c_d_d=c_d_d,
        c_d_dd=c_d_dd,
        args=args,
    )


def extract_acceleration_and_steering_control(vehicle, state, a_des, trajectory):
    controls = []

    v = state[2]
    for step in range(1, len(trajectory.x)):
        if v < trajectory.s_d[step - 1]:
            a = a_des
            v = v + a_des * (trajectory.t[step] - trajectory.t[step - 1])
        else:
            a = 0

        dyaw = trajectory.yaw[step] - trajectory.yaw[step - 1]
        delta = arctan(dyaw * vehicle.L / trajectory.s_d[step - 1]) / (trajectory.t[step] - trajectory.t[step - 1])

        controls.append([a, delta])

    return controls
