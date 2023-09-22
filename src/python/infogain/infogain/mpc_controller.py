import numpy as np
import casadi as csi
from enum import Enum

import ModelParameters.Ackermann as Ackermann
import ModelParameters.GenericCar as GenericCar


from config import LANE_WIDTH, DISCOUNT_FACTOR, EXP_OVERFLOW_LIMIT, SCAN_RANGE


# Basic step function -- apply the control to advance one step
def euler(vehicle, state, control):
    return vehicle.ode(state, control)


#
# Also define the Runge-Kutta variant as it is (apparently) a much
# better approximation of the first order derivative
#
# https://en.wikipedia.org/wiki/Runge-Kutta_methods
def runge_kutta_step(vehicle, state, control, dt):
    k1 = vehicle.ode(state, control)
    k2 = vehicle.ode(state + k1 * (dt / 2), control)
    k3 = vehicle.ode(state + k2 * (dt / 2), control)
    k4 = vehicle.ode(state + k3 * dt, control)

    return (k1 + 2 * (k2 + k3) + k4) / 6.0


# wrapper to allow easy switch between methods. (simplifies validation)
def step_fn(vehicle, state, control, dt=None):
    # return euler(vehicle=vehicle, state=state, control=control)
    return runge_kutta_step(vehicle=vehicle, state=state, control=control, dt=dt)


def rollout_trajectory(vehicle, state, controls, dt):
    states = [state]
    state = np.array(state)
    for control in controls:
        step = step_fn(vehicle=vehicle, state=state, control=control, dt=dt)
        state = state + step * dt
        states.append(state)
    return states


class Vehicle:
    control_len = 2
    state_len = 3

    # set limits on velocity and turning
    min_v = GenericCar.MIN_V
    max_v = GenericCar.MAX_V
    min_w = GenericCar.MIN_W
    max_w = GenericCar.MAX_W
    max_delta = GenericCar.MAX_DELTA
    min_delta = GenericCar.MIN_DELTA

    def __init__(self) -> None:
        pass

    #   Step Functions
    #   --------------
    #        x(k+1) = x(k) + v cos(theta(k)),
    #        y(k+1) = y(k) + v sin(theta(k)),
    #        theta(k+1) = theta(k) + w,
    #  next_state = [v*cos(theta)*dt, v*sin(theta)*dt
    @staticmethod
    def ode(state, control):
        ds0 = control[0] * csi.cos(state[2])
        ds1 = control[0] * csi.sin(state[2])
        ds2 = control[1]

        return csi.vertcat(ds0, ds1, ds2)


class Ackermann3:
    CONTROL_LEN = 2  # v, omega
    STATE_LEN = 5  # x, y, theta

    def __init__(self, length=None) -> None:
        if length is None:
            self.L = Ackermann.L
        else:
            self.L = length

        # set defaults limit on velocity and turning
        self.min_v = Ackermann.MIN_V
        self.max_v = Ackermann.MAX_V
        self.min_a = Ackermann.MIN_A
        self.max_a = Ackermann.MAX_A
        self.max_delta = Ackermann.MAX_DELTA
        self.min_delta = Ackermann.MIN_DELTA
        self.min_w = Ackermann.MIN_W
        self.max_w = Ackermann.MAX_W

    #   Step Function
    def ode(self, state, control):
        ds0 = control[0] * csi.cos(state[2])
        ds1 = control[0] * csi.sin(state[2])
        ds2 = control[0] * csi.tan(control[1]) / self.L

        return csi.vertcat(ds0, ds1, ds2)


class Ackermann4:
    CONTROL_LEN = 2  # a, delta
    STATE_LEN = 4  # x, y, v, theta

    def __init__(self, length=None) -> None:
        if length is None:
            self.L = Ackermann.L
        else:
            self.L = length

        # set defaults limit on velocity and turning
        self.min_v = Ackermann.MIN_V
        self.max_v = Ackermann.MAX_V
        self.min_a = Ackermann.MIN_A
        self.max_a = Ackermann.MAX_A
        self.max_delta = Ackermann.MAX_DELTA
        self.min_delta = Ackermann.MIN_DELTA
        self.min_w = Ackermann.MIN_W
        self.max_w = Ackermann.MAX_W

    #   Step Function
    def ode(self, state, control):
        dx = state[2] * csi.cos(state[3])
        dy = state[2] * csi.sin(state[3])
        dv = control[0]
        dtheta = state[2] * csi.tan(control[1]) / self.L

        return csi.vertcat(dx, dy, dv, dtheta)


class Ackermann5:
    CONTROL_LEN = 2  # a, omega
    STATE_LEN = 5  # x, y, v, theta, delta

    def __init__(self, length=None) -> None:
        if length is None:
            self.L = Ackermann.L
        else:
            self.L = length

        # set defaults limit on velocity and turning
        self.min_v = Ackermann.MIN_V
        self.max_v = Ackermann.MAX_V
        self.min_a = Ackermann.MIN_A
        self.max_a = Ackermann.MAX_A
        self.max_delta = Ackermann.MAX_DELTA
        self.min_delta = Ackermann.MIN_DELTA
        self.min_w = Ackermann.MIN_W
        self.max_w = Ackermann.MAX_W

    #   Step Function
    def ode(self, state, control):
        dx = state[2] * csi.cos(state[3])
        dy = state[2] * csi.sin(state[3])
        dv = control[0]
        dtheta = state[2] * csi.tan(state[4]) / self.L
        ddelta = control[1]

        # return csi.vertcat(dx, dy, dv, dtheta, ddelta)
        return np.array([dx, dy, dv, dtheta, ddelta])


class MPC:
    def __init__(
        self,
        mode,
        vehicle,
        planning_horizon,
        num_agents,
        Q,
        Qf,
        R,
        M,
        dt,
    ) -> None:
        assert len(Q) == vehicle.STATE_LEN
        assert len(Qf) == vehicle.STATE_LEN
        assert len(R) == vehicle.CONTROL_LEN

        self.state_len = vehicle.STATE_LEN
        self.control_len = vehicle.CONTROL_LEN
        self.step_fn = step_fn

        # set limits for the sandbox (states)
        self.min_x = -1.0
        self.max_x = np.inf
        self.min_y = -LANE_WIDTH
        self.max_y = LANE_WIDTH

        self.dt = dt
        self.planning_horizon = planning_horizon

        # define the ego-state of the robot
        self.state = csi.SX.sym("x", self.state_len, 1)
        self.X = csi.SX.sym("X", self.state_len, self.planning_horizon + 1)

        # define an array to hold the predicted controls and initialize it
        self.control = csi.SX.sym("u", self.control_len, 1)
        self.U = csi.SX.sym("U", self.control_len, self.planning_horizon)

        # define initial and final states
        self.X_i = csi.SX.sym("init-state", self.state.shape)
        self.X_f = csi.SX.sym("final-state", self.state.shape)

        # define obstacles in the env which must be avoided based on the agents/objects
        # in the environment
        #
        # [x,y,radius,min_pt_x,min_pt_y]
        self.num_agents = num_agents
        if self.num_agents:
            self.agents = csi.SX.sym("agents", 5, num_agents)
        else:
            self.agents = None

        # # define trajectory states and controls
        self.X_ideal = csi.SX.sym("traj-states", self.state_len, self.planning_horizon)
        self.U_ideal = csi.SX.sym("traj-controls", self.control_len, self.planning_horizon)

        # define a forward step
        self.state_inc = self.step_fn(vehicle, self.state, self.control, dt)
        self.step_fn = csi.Function(
            "StepFn",
            [self.state, self.control],
            [self.state_inc],
            ["x", "u"],
            ["dot_x"],
        )

        #  Formulate the problem assuming we want some sort of trade-off
        #  between the cost of using a control and deviation from our target
        #
        #  J = SUM[ (x-x_f)' Q (x-x_f) + u R u + v M v ] over horizon + xf Qf xf
        self.Q = Q
        self.Qf = Qf
        self.R = R
        self.M = M

        # TODO: make the controls responsive to the vehicle limits instead of hard
        #       coding
        #
        # reshape` the optimization variable as a single vector
        optimization_vars = self.U.reshape((-1, 1))
        controls_len = optimization_vars.shape[0]
        optimization_vars = csi.vertcat(optimization_vars, self.X.reshape((-1, 1)))
        self.u_i = csi.DM.zeros(optimization_vars.shape)
        self.lbu = csi.DM.zeros(optimization_vars.shape)
        self.ubu = csi.DM.zeros(optimization_vars.shape)

        # bounds on control inputs
        self.lbu[0:controls_len:2] = vehicle.min_a
        self.lbu[1:controls_len:2] = vehicle.min_w
        self.ubu[0:controls_len:2] = vehicle.max_a
        self.ubu[1:controls_len:2] = vehicle.max_w

        # bounds on states
        self.lbu[controls_len::5] = self.min_x
        self.lbu[controls_len + 1 :: 5] = self.min_y * 2
        self.lbu[controls_len + 2 :: 5] = vehicle.min_v
        self.lbu[controls_len + 3 :: 5] = -csi.inf
        self.lbu[controls_len + 4 :: 5] = vehicle.min_delta

        self.ubu[controls_len::5] = self.max_x
        self.ubu[controls_len + 1 :: 5] = self.max_y * 2
        self.ubu[controls_len + 2 :: 5] = vehicle.max_v
        self.ubu[controls_len + 3 :: 5] = csi.inf
        self.ubu[controls_len + 4 :: 5] = vehicle.max_delta

        # Calculate J by summing the costs over the entire trajectory
        J = 0

        # set the initial state constraint
        g = []
        g.append(self.X[:, 0] - self.X_i[:, 0])
        # and add constraints for all the interval points (multiple shooting)
        for i in range(self.planning_horizon):
            # take a step by applying the control
            step = self.step_fn(x=self.X[:, i], u=self.U[:, i])
            X_next = self.X[:, i] + step["dot_x"] * self.dt

            # add the cost of the controls to the index function
            err_U = self.U[:, i] - self.U_ideal[:, i]
            # err_U = self.U[:, i]  # minimize control effort

            err_X = X_next - self.X_ideal[:, i]

            J = J + err_X.T @ self.Q @ err_X + err_U.T @ self.R @ err_U

            if mode == "Anderson":
                J += self.anderson_visibility_cost(i + 1)
            elif mode == "Higgins":
                J += self.higgins_visibility_cost(i + 1)

            # add the difference between this segment end and the start of the
            # next to the constraint list (for multiple shooting)
            g.append(X_next - self.X[:, i + 1])

        # and a terminal error
        err_X = self.X[:, -1] - self.X_f
        J = J + err_X.T @ self.Qf @ err_X

        # set the state constraints to zero (upper and lower) for an equality
        # constraint - this ensures the segments line up
        self.lbg = np.zeros(
            [
                (self.planning_horizon + 1) * self.state_len,
            ]
        )
        self.ubg = np.zeros(
            [
                (self.planning_horizon + 1) * self.state_len,
            ]
        )

        if self.num_agents:
            # add constraints to make sure we stay away from all the agents
            for i in range(1, self.planning_horizon + 1):
                for ag in range(self.num_agents):
                    dx = self.X[0, i] - self.agents[0, ag]
                    dy = self.X[1, i] - self.agents[1, ag]

                    # BUGBUG -- should really be considering the robot radius here too, but
                    #           there isn't one included right now
                    g.append(-(dx * dx + dy * dy) + self.agents[2, ag] ** 2)
        # else:
        #     # insert bogus obstacles for testing
        #     for i in range(self.planning_horizon + 1):
        #         dx = self.X[0, i] - 1
        #         dy = self.X[1, i] - 0.2
        #         g = csi.vertcat(g, -(dx * dx + dy * dy) + 0.08)

        #     self.lbg = np.hstack(
        #         [
        #             self.lbg,
        #             np.ones((self.planning_horizon + 1,)) * (-np.inf),
        #         ]
        #     )
        #     self.ubg = np.hstack([self.ubg, np.zeros((self.planning_horizon + 1,))])

        #  Create an NLP solver
        #
        # Set the solver arguments
        args = {
            "print_time": 0,
            # "common_options": {"final_options": {"dump_in": True, "dump_out": True}},
            "ipopt": {
                "max_iter": 5000,
                "print_level": 0,
                "acceptable_tol": 1e-8,
                "acceptable_obj_change_tol": 1e-6,
                "check_derivatives_for_naninf": "Yes",
            },
        }

        # Assign the problem elements
        if self.agents is not None:
            p = csi.vertcat(
                self.X_i,
                self.X_ideal.reshape((-1, 1)),
                self.X_f,
                self.U_ideal.reshape((-1, 1)),
                self.agents.reshape((-1, 1)),
            )
        else:
            p = csi.vertcat(
                self.X_i,
                self.X_ideal.reshape((-1, 1)),
                self.X_f,
                self.U_ideal.reshape((-1, 1)),
            )

        prob = {
            "f": J,
            "x": optimization_vars,
            "g": csi.vertcat(*g),
            "p": p,
        }
        self.solver = csi.nlpsol("solver", "ipopt", prob, args)

    def higgins_visibility_cost(self, index):
        J_vis = 0
        r_fov = SCAN_RANGE
        r_fov_2 = r_fov**2

        for ag in range(self.num_agents):
            dx = self.X[0, index] - self.agents[0, ag]
            dy = self.X[1, index] - self.agents[1, ag]
            d_agent_2 = dx * dx + dy * dy
            d_agent = csi.sqrt(d_agent_2)

            inner = self.agents[2, ag] / d_agent * (r_fov_2 - d_agent_2)
            if inner > EXP_OVERFLOW_LIMIT:
                score = self.M * inner
            else:
                score = self.M * csi.log(1 + csi.exp(inner))
            J_vis += score

        return J_vis

    def anderson_visibility_cost(self, index):
        J_vis = 0

        for ag in range(self.num_agents):
            angle = csi.arctan((self.agents[4, ag] - self.X[1, index]) / (self.agents[3, ag] - self.X[0, index]))
            j_vis += -self.M * angle

        return J_vis

    # def next(self, obs, state, goal, trajectory, controls, agents, warm_start=False):
    def next(self, obs, state, goal, agents, trajectory, controls=None, warm_start=False):
        # p = csi.vertcat(state['ego'], goal, *controls, *trajectory)

        if len(agents):
            p = csi.vertcat(state, *trajectory, goal, *controls, *agents)
            # p = csi.vertcat(state, goal, *agents)
            lbg = np.hstack(
                [
                    self.lbg,
                    np.ones((self.num_agents * (self.planning_horizon),)) * (-np.inf),
                ]
            )
            ubg = np.hstack([self.ubg, np.zeros((self.num_agents * (self.planning_horizon),))])
        else:
            p = csi.vertcat(state, *trajectory, goal, *controls)
            # p = csi.vertcat(state, goal)
            lbg = self.lbg
            ubg = self.ubg

        sol = self.solver(x0=self.u_i, lbx=self.lbu, ubx=self.ubu, lbg=lbg, ubg=ubg, p=p)
        res = self.solver.stats()
        if not res["success"]:
            raise SystemError("ERROR: Solver failed!")

        print(f"Cost: {sol['f']}")
        opt = sol["x"]
        u_opt = opt[: self.planning_horizon * self.control_len].reshape((self.control_len, self.planning_horizon))
        x_opt = opt[self.planning_horizon * self.control_len :].reshape((self.state_len, self.planning_horizon + 1))

        if warm_start:
            # keep the future states for the next lop
            self.u_i[0 : (self.planning_horizon - 1) * self.control_len] = opt[
                self.control_len : self.planning_horizon * self.control_len
            ]
            self.u_i[(self.planning_horizon - 1) * self.control_len : self.planning_horizon * self.control_len] = opt[
                (self.planning_horizon - 1) * self.control_len : self.planning_horizon * self.control_len
            ]

            self.u_i[self.planning_horizon * self.control_len : -self.state_len] = opt[
                self.planning_horizon * self.control_len + self.state_len :
            ]
            self.u_i[-self.state_len :] = opt[-self.state_len :]

        return u_opt, x_opt


# @brief: MPPI (Class) -- sampling based implementation of Model Predictive Control
#
class MPPI:
    class visibility_method(Enum):
        OURS = 0
        HIGGINS = 1
        ANDERSON = 2
        NONE = 3

    def __init__(self, mode, vehicle, limits, c_lambda=1, Q=None, M=1, seed=None) -> None:
        if mode == "Ours":
            self.mode = MPPI.visibility_method.OURS
        elif mode == "Higgins":
            self.mode = MPPI.visibility_method.HIGGINS
        elif mode == "Anderson":
            self.mode = MPPI.visibility_method.ANDERSON
        else:
            self.mode = MPPI.visibility_method.NONE

        self.vehicle = vehicle
        self.limits = limits
        self.c_lambda = c_lambda
        self.seed = seed

        if Q is None:
            self.Q = np.eye(vehicle.state_size)
        else:
            self.Q = Q
        self.M = M

        self.gen = np.random.default_rng(seed)

    # @brief: find_control -- sample the vehicle motion model to find the value
    #         of variations in the control signal.
    #
    def find_control(
        self,
        costmap,
        origin,
        resolution,
        x_nom,
        u_nom,
        initial_state,
        actors,
        samples,
        dt,
    ):
        u_N, u_M = u_nom.shape
        u_dist = np.zeros([samples, u_N, u_M])  # one extra for weights
        u_weight = np.zeros([samples, u_M])

        for i in range(samples):
            self.__rollout(
                costmap,
                origin,
                resolution,
                x_nom,
                u_nom,
                initial_state,
                actors,
                dt,
                u_dist[i, ...],
                u_weight[i, ...],
            )

        u_weighted = np.zeros_like(u_nom)
        weights = np.exp(-1.0 / (self.c_lambda) * np.sum(u_weight, axis=1)).reshape((-1, 1))
        total_weight = np.sum(weights)

        for step in range(u_M):
            u_dist[:, :, step] = weights * u_dist[:, :, step]
            u_weighted[:, step] = u_nom[:, step] + np.sum(u_dist[:, :, step], axis=0) / total_weight

        return u_weighted, u_dist

    #
    # @brief: __rollout (internal function) -- given an initial state, costmap, etc., perform
    #         a trajectory rollout to see where we end up
    #
    # @returns: reward, control disturbances
    #
    def __rollout(
        self,
        costmap,
        origin,
        resolution,
        x_nom,
        u_nom,
        initial_state,
        actors,
        dt,
        u_dist,
        u_weight,
    ):
        control_len, steps = u_nom.shape

        state = np.array(initial_state)
        for step in range(steps):
            u_i = np.array(u_nom[:, step])
            for control_num in range(control_len):
                noise = self.gen.random() * self.limits[control_num] * 2 - self.limits[control_num]
                u_i[control_num] += noise
                u_dist[control_num, step] = noise

            dstep = step_fn(self.vehicle, state, control=u_i, dt=dt)
            state = state + dstep * dt

            # penalize error in trajectory
            state_err = x_nom[:, step + 1] - state
            step_score = state_err.T @ self.Q @ state_err

            # check for obstacles
            step_score += self.obstacle_cost(state=state, actors=actors)

            if self.mode == MPPI.visibility_method.OURS:
                step_score += self.our_visibility_cost(
                    costmap=costmap,
                    origin=origin,
                    resolution=resolution,
                    state=state,
                    step=step,
                )
            elif self.mode == MPPI.visibility_method.HIGGINS:
                step_score += self.higgins_visibility_cost(state, actors=actors)
            elif self.mode == MPPI.visibility_method.ANDERSON:
                step_score += self.anderson_visibility_cost(state, actors=actors)

            u_weight[step] = step_score

    def anderson_visibility_cost(self, state, actors):
        J_vis = 0

        for act in actors:
            # BUGBUG: hacky test to see if the obstacle is still ahead of the AV -- works in this
            #         linear environment but will need to be modified and expanded to a general
            #         case for more involved envs.
            if act[3] > state[0]:
                angle = np.arctan((act[4] - state[0]) / (act[3] - state[0]))
                # in anticipation of obstacles on both sides, use abs() instead of reversing the sign
                # J_vis += -self.M * angle
                J_vis += abs(self.M * angle)

        return J_vis

    def higgins_visibility_cost(self, state, actors):
        J_vis = 0
        r_fov = SCAN_RANGE
        r_fov_2 = r_fov**2

        for act in actors:
            dx = state[0] - act[0]
            dy = state[1] - act[1]
            d_agent_2 = dx * dx + dy * dy
            d_agent = np.sqrt(d_agent_2)

            # if the exp is going to overflow, just use the max value
            inner = act[2] / d_agent * (r_fov_2 - d_agent_2)
            if inner > EXP_OVERFLOW_LIMIT:
                score = self.M * inner
            else:
                score = self.M * np.log(1 + np.exp(inner))

            J_vis += score

        return J_vis

    def obstacle_cost(self, state, actors):
        for act in actors:
            dx = state[0] - act[0]
            dy = state[1] - act[1]
            d_agent_2 = dx * dx + dy * dy

            # TODO: Use multiple circle of dm2 to check collisions -- use this approximation for now
            #       so we're consistant
            if d_agent_2 < act[2]:
                return 10000

        return 0

    def our_visibility_cost(self, costmap, origin, resolution, state, step):
        map_x = int((state[0] - origin[0]) / resolution)
        map_y = int((state[1] - origin[1]) / resolution)

        max_y, max_x = costmap.shape
        if map_x < 0 or map_x > max_x - 1 or map_y < 0 or map_y > max_y - 1:
            raise ValueError("Planning outside of available cost map!")

        # NOTE: Reversing the sense of the cost map to make it a penalty instead of a reward
        return self.M * (1 - costmap[map_y, map_x]) * (DISCOUNT_FACTOR**step)


import matplotlib.pyplot as plt

global fig, ax, plot_lines, weighed_line, plot_backgrounds
fig, ax = None, None


def run_trajectory(vehicle, initial_state, controls, dt):
    N, M = controls.shape

    traj = np.zeros((len(initial_state), M + 1))
    traj[:, 0] = initial_state

    state = np.array(initial_state)
    for m in range(M):
        u = controls[:, m]
        step = step_fn(vehicle=vehicle, state=state, control=u, dt=dt)
        state += step * dt
        traj[:, m + 1] = state

    return traj


def visualize_variations(vehicle, initial_state, u_nom, u_variations, u_weighted, dt):
    # visualizing!

    global fig, ax, plot_lines, weighted_line, plot_backgrounds
    if fig is None:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        plot_lines = None
        weighted_line = None
        plot_backgrounds = []

    n_samples, n_controls, n_steps = u_variations.shape
    new_traj_pts = []
    for i in range(n_samples):
        u_var = np.array(u_nom)
        u_var = u_var + u_variations[i, ...]

        traj = run_trajectory(vehicle=vehicle, initial_state=initial_state, controls=u_var, dt=dt)
        new_traj_pts.append(np.expand_dims(traj, axis=0))

    new_traj_pts = np.vstack(new_traj_pts)

    traj = run_trajectory(vehicle=vehicle, initial_state=initial_state, controls=u_weighted, dt=dt)

    if plot_lines is None:
        plot_lines = ax[0].plot(new_traj_pts[:, 0, :].T, new_traj_pts[:, 1, :].T)
        plot_initialized = True
        weighted_line = ax[1].plot(traj[0, :], traj[1, :])

        ax[0].axis("equal")
        ax[1].axis("equal")
        plt.show(block=False)
    else:
        for line, data in zip(plot_lines, new_traj_pts):
            line.set_data(data[0, :], data[1, :])

        weighted_line[0].set_data(traj[0, :], traj[1, :])

        ax[0].relim()
        ax[0].autoscale_view()
        ax[1].relim()
        ax[1].autoscale_view()

        plt.pause(0.01)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    vehicle = Ackermann5()
    roller = MPPI(vehicle, (2, np.pi / 2.0), 1, 42)

    samples = 10
    steps = 20
    dt = 0.2
    L = 3.0

    u_nom = np.array([[1 for i in range(steps)], [0 for i in range(steps)]])

    origin = [-1.0, -6]
    resolution = 0.1

    mapsize = 120
    peak_x = 8.0
    peak_y = 0

    costmap = np.zeros([mapsize, mapsize])

    for i in range(mapsize):
        for j in range(mapsize):
            x_offset = origin[0] + i * resolution
            y_offset = origin[1] + j * resolution

            value = (5 - abs(y_offset - peak_y)) / 10
            if y_offset > 0.1:
                value = 100

            # if peak_x - x_offset < 0:
            #     value = 100

            costmap[j, i] = value

    initial = np.array([0, 0, 0, 0, 0]).astype(np.float64)

    u, u_variations = roller.find_control(costmap, origin, resolution, u_nom, initial, samples, dt)

    state = np.array(initial)
    last_distance = 10000.0

    for i in range(steps):
        control = np.array(u[:, i])
        step = step_fn(vehicle, state, control, dt)
        state = state + step * dt

        dx = peak_x - state[0]
        dy = peak_y - state[1]
        dist = np.sqrt(dx * dx + dy * dy)

        flag = ""
        if last_distance < dist:
            flag = " (**) "
        last_distance = dist

        print(
            f"State: {state[0]:0.4},{state[1]:0.4},{state[2]:0.4} -- Control: {control[0]:0.4},{control[0]:0.4} -- Distance: {dist:0.4} {flag}"
        )

    visualize_variations(
        Ackermann5(),
        initial_state=initial,
        u_nom=u_nom,
        u_variations=u_variations,
        u_weighted=u,
        dt=dt,
    )
