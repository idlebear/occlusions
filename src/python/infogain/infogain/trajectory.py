#
# Simple trajectory planning interface to PythonRobotics code
#
from trajectory_planner.frenet_optimal_trajectory import (
    frenet_optimal_planning,
    generate_target_course,
)
from config import *
import numpy as np
from scipy.signal import convolve2d


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
        delta = np.arctan(dyaw * vehicle.L / trajectory.s_d[step - 1]) / (
            trajectory.t[step] - trajectory.t[step - 1]
        )

        controls.append([a, delta])

    return controls


def create_2d_gaussian(width, centre_x=0, centre_y=0, sigma=1, scale=1):
    """create_2d_gaussian -- creates a 2D gaussian kernal at the
           specified offset, and scale

    @param width     the number of steps in the gaussian kernel
    @param offset_x  the distance from center of the peak value in X.
    @param offset_y  the distance from center of the peak value in Y.
    @param std       sigma -- the standard deviation of the curve
    @param scale     numerical width of each step
    """
    X = range(width)
    x, y = np.meshgrid(X, X)

    xc = int(width / 2.0 - scale * centre_x)
    yc = int(width / 2.0 - scale * centre_y)

    sigma = sigma * scale

    # since the gaussian may be off-center (and is definitely truncated), normalize
    # using the sum of elements
    gus = np.exp(-(np.power(x - xc, 2) + np.power(y - yc, 2)) / (2 * sigma * sigma))
    return gus / np.sum(gus)


def validate_controls(vehicle, states, controls, obs, map, resolution, dt) -> np.array:
    N_controls = len(controls)

    # the map is centered on the vehicle (state [0:2]) but is double size -- hack off
    # the margins
    map = map[GRID_SIZE // 2 : 3 * GRID_SIZE // 2, GRID_SIZE // 2 : 3 * GRID_SIZE // 2]

    # remove all the static structures/objects  - blot any inaccuracies as well
    pedestrian_grid = np.where(map != 0, 0, obs)
    pedestrian_grid = np.where(pedestrian_grid > 0.8, 0, pedestrian_grid)

    # construct a gaussian to blur movement based on the possible -- size is measured in
    # grid cells.  If the size is too small, then we make it one, and only inflate every
    # n steps.
    max_pedestrian_move = MAX_PEDESTRIAN_SPEED * dt / resolution
    steps_between_blur = max(1, np.round(1.0 / max_pedestrian_move))
    max_pedestrian_move = int(np.ceil(max_pedestrian_move))

    # blur = create_2d_gaussian(width=2 * max_pedestrian_move + 1, centre_x=0, centre_y=0, sigma=2.0, scale=0.5)
    blur = np.ones([2 * max_pedestrian_move + 1, 2 * max_pedestrian_move + 1]) / (
        2 * max_pedestrian_move + 1
    )
    # / (
    #     (2 * max_pedestrian_move + 1) * (2 * max_pedestrian_move + 1)
    # )

    # construct a stack of maps showing possible future occupancy (inflated)
    future_maps = []
    for i in range(N_controls):
        if i % steps_between_blur == 0:
            blurred_grid = np.clip(
                convolve2d(in1=pedestrian_grid, in2=blur, mode="same"), 0, 1
            )
            next_ped_grid = blurred_grid + map
            pedestrian_grid = np.where(map != 0, 0, blurred_grid)
        future_maps.append(next_ped_grid)

    # check the projected states (skipping the first -- that has to be ok)
    initial_state = states[0]
    future_states = states[1:]
    current_state = 0
    for state in future_states:
        x = int((state[0] - initial_state[0]) / resolution + GRID_SIZE // 2)
        y = int((state[1] - initial_state[1]) / resolution + GRID_SIZE // 2)
        if future_maps[current_state][y, x] > OCCUPANCY_THRESHOLD:
            # potential collision
            break
        current_state += 1
    if current_state >= N_controls:
        # no collisions detected
        return controls[0, :]

    # current state has a possible collision -- assume that we must be able to stop in the previous
    # step at state current_state -1
    current_state -= 1
    max_v = abs(vehicle.min_a) * dt * current_state

    # while current_state > 0:
    #     current_state -= 1
    #     max_v = max_v + abs(vehicle.min_a) * dt
    #     if future_states[current_state][2] < max_v:
    #         # we're within stopping acceleration, should be safe to proceed.
    #         return controls[0, :]

    # fell all the way back to the initial state -- calculate the maximum allowable
    # acceleration.
    dv = max_v - initial_state[2]
    required_a = np.round(dv / dt, 3)

    # now check if it's valid
    requested_a = controls[0, 0]
    if required_a < vehicle.min_a:
        # emergency braking required
        print("EMERGENCY brake deployed!")
        return np.array([EMERGENCY_BRAKE, controls[0, 1]])
    elif required_a < requested_a:
        print("Slowing...!")
        return np.array([required_a, controls[0, 1]])

    # all good
    return controls[0, :]
