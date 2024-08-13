import numpy as np
from enum import Enum
from controller.ModelParameters.Ackermann import Ackermann4

from config import LANE_WIDTH, DISCOUNT_FACTOR, SCAN_RANGE
from controller.validate import visualize_variations, run_trajectory, step_fn, runge_kutta_step, euler

import matplotlib.pyplot as plt

global fig, ax, plot_lines, nom_line, weighted_line, plot_backgrounds
fig, ax = None, None


def rollout_trajectory(vehicle, state, controls, dt):
    states = [state]
    state = np.array(state)
    for control in controls:
        step = step_fn(vehicle=vehicle, state=state, control=control, dt=dt)
        state = state + step * dt
        states.append(state)
    return states


# @brief: MPPI (Class) -- sampling based implementation of Model Predictive Control
#
class MPPI:
    class visibility_method(Enum):
        OURS = 0
        HIGGINS = 1
        ANDERSEN = 2
        NONE = 3
        IGNORE = 4

    def __init__(
        self, mode=None, vehicle=Ackermann4(), limits=None, c_lambda=20000, Q=None, R=None, M=1, seed=None
    ) -> None:

        if mode is not None:
            if mode[0:4] == "Ours":
                self.mode = MPPI.visibility_method.OURS
            elif mode == "Higgins":
                self.mode = MPPI.visibility_method.HIGGINS
            elif mode == "Andersen":
                self.mode = MPPI.visibility_method.ANDERSEN
            else:
                raise ValueError("Unknown visibility method")
        else:
            self.mode = MPPI.visibility_method.NONE

        self.vehicle = vehicle
        self.limits = limits
        self.c_lambda = c_lambda
        self.seed = seed

        if Q is None:
            self.Q = np.eye(vehicle.STATE_LEN)
        else:
            self.Q = Q
        self.M = M

        if R is None:
            self.R = np.eye(vehicle.CONTROL_LEN)
        else:
            self.R = R

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
        u_dist = np.zeros([samples, u_N, u_M])
        u_weight = np.zeros([samples])

        for i in range(samples):
            u_weight[i] = self.__rollout(
                costmap,
                origin,
                resolution,
                x_nom,
                u_nom,
                initial_state,
                actors,
                dt,
                u_dist[i, ...],
            )

        u_weighted = np.zeros_like(u_nom)
        weights = np.exp(-1.0 / (self.c_lambda) * u_weight).reshape((-1, 1))
        assert np.sum(weights) > 0
        total_weight = np.sum(weights)
        weights = weights / total_weight

        for step in range(u_M):
            u_dist[:, :, step] = weights * u_dist[:, :, step]
            u_weighted[:, step] = u_nom[:, step] + np.sum(u_dist[:, :, step], axis=0) 

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
    ):
        control_len, steps = u_nom.shape

        state = np.array(initial_state)
        u_weight = 0
        for step in range(steps):
            u_step = np.array(u_nom[:, step])
            for control_num in range(control_len):
                noise = self.gen.random() * self.limits[control_num] * 2 - self.limits[control_num]
                u_step[control_num] += noise
                u_dist[control_num, step] = noise

            dstep = step_fn(self.vehicle, state, control=u_step, dt=dt)
            state = state + dstep * dt

            # penalize error in trajectory
            state_err = x_nom[:, step + 1] - state
            u_weight += state_err.T @ self.Q @ state_err

            # penalize any control action
            control_err = u_step - u_nom[:, step]
            u_weight += control_err.T @ self.R @ control_err

            # check for obstacles
            u_weight += self.obstacle_cost(state=state, actors=actors)

            if self.mode == MPPI.visibility_method.OURS:
                u_weight += self.our_visibility_cost(
                    costmap=costmap,
                    origin=origin,
                    resolution=resolution,
                    state=state,
                    step=step,
                )
            elif self.mode == MPPI.visibility_method.HIGGINS:
                u_weight += self.higgins_visibility_cost(state, [dstep[0], dstep[1]], actors=actors)
            elif self.mode == MPPI.visibility_method.ANDERSEN:
                u_weight += self.andersen_visibility_cost(state, [dstep[0], dstep[1]], actors=actors)

        return u_weight

    def andersen_visibility_cost(self, state, vx, vy, actors):
        J_vis = 0

        for act in actors:
            # distance to min point on the obstacle
            dx = act[3] - state[0]
            dy = act[4] - state[1]
            dot = dx * vx + dy * vy
            if dot > 0:
                d = np.sqrt(dx * dx + dy * dy)
                v = np.sqrt(vx * vx + vy * vy)
                angle = np.arccos(dot / (d * v))
                J_vis += self.M * angle

                # BUGBUG: A hack to take only the next obstacle into consideration -- otherwise
                #         the error scales with the number of occlusions, pushing the vehicle
                #         further and further away
                break

        return J_vis

    def higgins_visibility_cost(self, state, vx, vy, actors):
        J_vis = 0
        r_fov = SCAN_RANGE
        r_fov_2 = r_fov**2

        for act in actors:
            # BUGBUG -- may need to check if the obstacle is ahead of the vehicle
            dx = act[0] - state[0]
            dy = act[1] - state[1]
            d_agent_2 = dx * dx + dy * dy
            d_agent = np.sqrt(d_agent_2)

            # if the exp is going to overflow, just use the max value
            inner = act[2] / d_agent * (r_fov_2 - d_agent_2)
            with np.errstate(over="raise"):
                try:
                    score = np.log(1 + np.exp(inner))
                except FloatingPointError:
                    score = inner

            J_vis += self.M * score * score

        return J_vis

    def obstacle_cost(self, state, actors):
        for act in actors:
            dx = state[0] - act[0]
            dy = state[1] - act[1]
            d_agent_2 = dx * dx + dy * dy

            # TODO: Use multiple circle of dm2 to check collisions -- use this approximation for now
            #       so we're consistant
            if d_agent_2 < act[2]:
                return 10000000.0

        return 0

    def our_visibility_cost(self, costmap, origin, resolution, state, step):
        map_x = int((state[0] - origin[0]) / resolution)
        map_y = int((state[1] - origin[1]) / resolution)

        max_y, max_x = costmap.shape
        if map_x < 0 or map_x > max_x - 1 or map_y < 0 or map_y > max_y - 1:
            raise ValueError("Planning outside of available cost map!")

        reward = -self.M * (costmap[map_y, map_x]) * (DISCOUNT_FACTOR**step)

        return reward


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from controller.ModelParameters.Ackermann import Ackermann5

    vehicle = Ackermann5()
    roller = MPPI(vehicle, (2, np.pi / 2.0), 1, 42)

    samples = 25
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

    u, u_variations, weights = roller.find_control(costmap, origin, resolution, u_nom, initial, samples, dt)

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
        weights=weights,
        dt=dt,
    )
