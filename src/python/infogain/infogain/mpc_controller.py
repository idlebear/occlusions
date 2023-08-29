import numpy as np
import casadi as csi


class Vehicle:
    control_len = 2
    state_len = 3

    def __init__(self) -> None:
        pass

    #
    #   Step Functions
    #   --------------
    #
    #   Define a simple velocity model that takes a state and control
    #   and generates the next (predicted) state
    #
    #   We're using a simple velocity/angle model meaning the controls are
    #   v and w, and the state is (x, y, theta), with the relation
    #
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

    @staticmethod
    def euler(state, control, dt=None):
        return Vehicle.ode(state, control)

    #
    # Also define the Runge-Kutta variant as it is (apparently) a much
    # better approximation of the first order derivative
    #
    # https://en.wikipedia.org/wiki/Runge-Kutta_methods
    @staticmethod
    def runge_kutta_step(state, control, dt):
        k1 = Vehicle.ode(state, control)
        k2 = Vehicle.ode(state + k1 * (dt / 2), control)
        k3 = Vehicle.ode(state + k2 * (dt / 2), control)
        k4 = Vehicle.ode(state + k3 * dt, control)

        return (k1 + 2 * (k2 + k3) + k4) / 6.0


class SkidSteer(Vehicle):
    control_len = 2
    state_len = 3
    b = 0.05

    def __init__(self) -> None:
        pass

    #
    #   Step Functions
    #   --------------
    #
    #   Define a simple velocity model that takes a state and control
    #   and generates the next (predicted) state
    #
    #   We're using a simple velocity/angle model meaning the controls are
    #   v and w, and the state is (x, y, theta), with the relation
    #
    #        x(k+1) = x(k) + v cos(theta(k)),
    #        y(k+1) = y(k) + v sin(theta(k)),
    #        theta(k+1) = theta(k) + w,
    #  next_state = [v*cos(theta)*dt, v*sin(theta)*dt
    @staticmethod
    def ode(state, control):
        v = (control[0] + control[1]) / 2
        w = (control[0] - control[1]) / SkidSteer.b

        ds0 = v * csi.cos(state[2])
        ds1 = v * csi.sin(state[2])
        ds2 = w

        return csi.vertcat(ds0, ds1, ds2)

    @staticmethod
    def euler(state, control, dt=None):
        return SkidSteer.ode(state, control)

    #
    # Also define the Runge-Kutta variant as it is (apparently) a much
    # better approximation of the first order derivative
    #
    # https://en.wikipedia.org/wiki/Runge-Kutta_methods
    @staticmethod
    def runge_kutta_step(state, control, dt=0.01):
        k1 = SkidSteer.ode(state, control)
        k2 = SkidSteer.ode(state + k1 * (dt / 2.0), control)
        k3 = SkidSteer.ode(state + k2 * (dt / 2.0), control)
        k4 = SkidSteer.ode(state + k3 * dt, control)

        return (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


class Ackermann(Vehicle):
    control_len = 2  # accel, omega(dot delta)
    state_len = 5  # x, y, v, theta, delta
    L = 0.06

    def __init__(self) -> None:
        pass

    #
    #   Step Functions
    #   --------------
    #
    #   Define a simple velocity model that takes a state and control
    #   and generates the next (predicted) state
    #
    @staticmethod
    def ode(state, control):
        ds0 = state[2] * csi.cos(state[3])
        ds1 = state[2] * csi.sin(state[3])
        ds2 = control[0]
        ds3 = 1.0 / Ackermann.L * csi.tan(state[4])
        ds4 = control[1]

        return csi.vertcat(ds0, ds1, ds2, ds3, ds4)

    @staticmethod
    def euler(state, control, dt=None):
        return SkidSteer.ode(state, control)

    #
    # Also define the Runge-Kutta variant as it is (apparently) a much
    # better approximation of the first order derivative
    #
    # https://en.wikipedia.org/wiki/Runge-Kutta_methods
    @staticmethod
    def runge_kutta_step(state, control, dt=0.01):
        k1 = SkidSteer.ode(state, control)
        k2 = SkidSteer.ode(state + k1 * (dt / 2.0), control)
        k3 = SkidSteer.ode(state + k2 * (dt / 2.0), control)
        k4 = SkidSteer.ode(state + k3 * dt, control)

        return (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


class MPC:
    class MPC_Callback(csi.Callback):
        def __init__(self, name, sim, opts={}):
            super().__init__()
            self.sim = sim

        # Number of inputs and outputs

        def get_n_in(self, *args) -> "casadi_int":
            return 2

        def get_n_out(self, *args) -> "casadi_int":
            return 1

        # initialize
        def init(self):
            # initializing
            pass

        def eval(self, arg):
            x = arg[0]
            y = arg[1]

    # TODO:  Make the trajectory tracking an optional arg

    def __init__(
        self,
        vehicle,
        planning_horizon,
        num_agents,
        Q,
        Qf,
        R,
        dt,
    ) -> None:
        self.state_len = vehicle.state_len
        self.control_len = vehicle.control_len
        self.step_fn = vehicle.runge_kutta_step

        # set limits on velocity and turning
        self.min_v = -2.0
        self.max_v = 2.0
        self.max_delta = 2.0 * np.pi / 3.0
        self.min_w = -np.pi
        self.max_w = np.pi

        # set limits for the sandbox (states)
        self.min_x = 0.0
        self.max_x = 100.0
        self.min_y = -0.15
        self.max_y = 0.15

        self.dt = dt
        self.planning_horizon = planning_horizon

        # define the ego-state of the robot
        self.state = csi.SX.sym("x", self.state_len, 1)
        self.X = csi.SX.sym("X", self.state_len, self.planning_horizon + 1)

        # define an array to hold the predicted controls and initialize it
        self.control = csi.SX.sym("u", self.control_len, 1)
        self.U = csi.SX.sym("U", self.control_len, self.planning_horizon)

        # define initial and final states
        # self.X_i = csi.SX.sym("init-state", self.state.shape)
        self.X_f = csi.SX.sym("final-state", self.state.shape)

        # define obstacles in the env which must be avoided
        self.num_agents = num_agents
        if self.num_agents:
            self.agents = csi.SX.sym("agents", 3, num_agents)
        else:
            self.agents = None

        # # define trajectory states and controls
        self.X_ideal = csi.SX.sym("traj-states", self.state_len, self.planning_horizon)
        # self.U_ideal = csi.SX.sym('traj-controls', self.control_len, self.planning_horizon)

        # define a forward step
        self.state_inc = self.step_fn(self.state, self.control, dt)
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
        #  J = SUM[ (x-x_f)' Q (x-x_f) + u R u ] over horizon + xf Qf xf
        self.Q = Q
        self.Qf = Qf
        self.R = R

        # reshape` the optimization variable as a single vector
        optimization_vars = self.U.reshape((-1, 1))
        controls_len = optimization_vars.shape[0]
        optimization_vars = csi.vertcat(optimization_vars, self.X.reshape((-1, 1)))
        self.u_i = np.zeros(optimization_vars.shape)
        self.lbu = np.zeros(optimization_vars.shape)
        self.ubu = np.zeros(optimization_vars.shape)

        # bounds on control inputs
        self.lbu[0:controls_len:2] = self.min_v
        self.lbu[1:controls_len:2] = self.min_w  # max_w
        self.ubu[0:controls_len:2] = self.max_v
        self.ubu[1:controls_len:2] = self.max_w  # max_w

        # bounds on states
        self.lbu[controls_len::3] = self.min_x
        self.lbu[controls_len + 1 :: 3] = self.min_y
        self.lbu[controls_len + 2 :: 3] = -np.inf

        self.ubu[controls_len::3] = self.max_x
        self.ubu[controls_len + 1 :: 3] = self.max_y
        self.ubu[controls_len + 2 :: 3] = np.inf

        # Calculate J by summing the costs over the entire trajectory
        J = 0

        for i in range(1, self.planning_horizon):
            # calculate the error and add it to the index function
            # err_X = self.X[:, i+1] - self.X_ideal[:, i]
            err_X = self.X[:, i] - self.X_ideal[:, i]
            J = J + err_X.T @ self.Q @ err_X

        # and a terminal error
        err_X = self.X[:, -1] - self.X_f
        J = J + err_X.T @ self.Qf @ err_X

        # set the initial state constraint
        g = []
        g = csi.vertcat(g, self.X[:, 0] - self.X_ideal[:, 0])
        # and add constraints for all the interval points (multiple shooting)
        for i in range(self.planning_horizon):
            # take a step by applying the control
            step = self.step_fn(x=self.X[:, i], u=self.U[:, i])
            X_next = self.X[:, i] + step["dot_x"] * self.dt

            # add the cost of the controls to the index function
            # err_U = self.U[:, i] - self.U_ideal[:, i]
            err_U = self.U[:, i]  # minimize control effort
            J = J + err_U.T @ self.R @ err_U

            # add the difference between this segment end and the start of the
            # next to the constraint list (for multiple shooting)
            g = csi.vertcat(g, X_next - self.X[:, i + 1])

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
            for i in range(self.planning_horizon + 1):
                for ag in range(num_agents):
                    dx = self.X[0, i] - self.agents[0, ag]
                    dy = self.X[1, i] - self.agents[1, ag]

                    # BUGBUG -- should really be considering the robot radius here too, but
                    #           there isn't one included right now
                    g = csi.vertcat(g, -(dx * dx + dy * dy) + self.agents[2, ag] * self.agents[2, ag])
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
            "ipopt": {
                "max_iter": 100,
                "print_level": 0,
            },
        }

        # Assign the problem elements
        if self.agents is not None:
            p = csi.vertcat(self.X_ideal.reshape((-1, 1)), self.X_f, self.agents.reshape((-1, 1)))
        else:
            p = csi.vertcat(self.X_ideal.reshape((-1, 1)), self.X_f)

        prob = {
            "f": J,
            "x": optimization_vars,
            "g": g,
            "p": p,
        }
        self.solver = csi.nlpsol("solver", "ipopt", prob, args)

    # def next(self, obs, state, goal, trajectory, controls, agents, warm_start=False):
    def next(self, obs, goal, agents, trajectory, controls=None, warm_start=False):
        # p = csi.vertcat(state['ego'], goal, *controls, *trajectory)

        if len(agents):
            p = csi.vertcat(*trajectory, goal, *agents)
            lbg = np.hstack(
                [
                    self.lbg,
                    np.ones((self.num_agents * (self.planning_horizon + 1),)) * (-np.inf),
                ]
            )
            ubg = np.hstack([self.ubg, np.zeros((self.num_agents * (self.planning_horizon + 1),))])
        else:
            p = csi.vertcat(*trajectory, goal)
            lbg = self.lbg
            ubg = self.ubg

        sol = self.solver(x0=self.u_i, lbx=self.lbu, ubx=self.ubu, lbg=lbg, ubg=ubg, p=p)
        res = self.solver.stats()
        if not res["success"]:
            raise SystemError("ERROR: Solver failed!")

        print(f"Cost: {sol['f']}")
        opt = sol["x"]
        u_opt = opt[: self.planning_horizon * self.control_len].reshape((self.control_len, -1))
        x_opt = opt[self.planning_horizon * self.control_len :].reshape((self.state_len, -1))

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
    x_fin = [40, 1, 0]
    x_init = [0, 0, 0]
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
    control_len = SkidSteer.control_len
    state_len = SkidSteer.state_len

    mpc = MPC(
        state_len=state_len,
        control_len=control_len,
        planning_horizon=planning_horizon,
        num_agents=len(agents),
        step_fn=SkidSteer.runge_kutta_step,
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
