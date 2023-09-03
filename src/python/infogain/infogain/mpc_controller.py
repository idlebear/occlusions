import numpy as np
import casadi as csi

import ModelParameters.Ackermann as Ackermann
import ModelParameters.GenericCar as GenericCar


from config import LANE_WIDTH


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


class Ackermann5:
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

        return csi.vertcat(dx, dy, dv, dtheta, ddelta)


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
        r_fov = 15
        r_fov_2 = r_fov**2

        for ag in range(self.num_agents):
            dx = self.X[0, index] - self.agents[0, ag]
            dy = self.X[1, index] - self.agents[1, ag]
            d_agent_2 = dx * dx + dy * dy
            d_agent = csi.sqrt(d_agent_2)

            # TODO: Reformulate for overflow!
            J_vis += self.M * csi.log(1 + csi.exp(self.agents[2, ag] / d_agent * (r_fov_2 - d_agent_2)))

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


if __name__ == "__main__":
    # Problem parameters
    x_fin = [40, 1, 0, 0, 0]
    x_init = [0, 0, 0, 0, 0]
    v_des = 1

    Q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0.1]])

    Qf = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 0.2]])

    R = np.array([[1, 0], [0, 1]])

    x_cur = x_init

    step = 1

    T = 20
    dt = 0.1

    agents = [[10, 0.00, 2], [25, 2, 2]]
    # agents = [[20, 4, 2]]

    planning_horizon = 10
    control_len = Vehicle.control_len
    state_len = Vehicle.state_len

    mpc = MPC(
        state_len=state_len,
        control_len=control_len,
        planning_horizon=planning_horizon,
        num_agents=len(agents),
        step_fn=Vehicle.runge_kutta_step,
        Q=Q,
        Qf=Qf,
        R=R,
        dt=dt,
    )

    x_hist = None
    traj_hist = None

    print(x_cur)
    x_ideal = list(x_init)

    for t in np.arange(0, T, dt):
        traj = []
        dx = v_des * dt

        # traj = [[float(x_cur[0]) + dx * i, x_fin[1], 0] for i in range(1, planning_horizon+1)]
        # controls = [[v_des, 0] for _ in range(planning_horizon)]

        # Solve the NLP
        state = {"ego": x_cur}
        # u_opt, x_opt = mpc.next(None, state, goal=x_fin, trajectory=traj, controls=controls, agents=agents, warm_start=True)
        u_opt, x_opt = mpc.next(None, state, goal=x_fin, agents=agents, warm_start=False)

        # just use the optimizer's belief for the mo
        x_cur = x_opt[:, 1].toarray()

        if len(agents):
            print(
                f"X: {float(x_cur[0]):5.3}, Y: {float(x_cur[1]):5.3}, Dist: {np.linalg.norm( x_cur[0:2] - np.array(agents[0][0:2]).reshape(-1,1))}"
            )
        else:
            print(f"X: {float(x_cur[0]):5.3}, Y: {float(x_cur[1]):5.3}, Theta: {float(x_cur[2]):5.3}")

        if x_hist is None:
            x_hist = x_cur
            # traj_hist = np.array(traj[0]).reshape(-1, 1)
        else:
            x_hist = np.hstack([x_hist, x_cur])
            # traj_hist = np.hstack([traj_hist, np.array(traj[0]).reshape(-1, 1)])

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plt.plot(np.arange(0, T, dt), x_hist[0, :])

    fig, ax = plt.subplots()
    plt.plot(x_hist[0, :], x_hist[1, :])
    # plt.plot(traj_hist[0, :], traj_hist[1, :])
    ax.axis("equal")
    plt.show()

    print("done")
