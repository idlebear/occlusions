from trajectory_planner.frenet_optimal_trajectory import (
    generate_target_course,
    frenet_optimal_planning,
    PlannerArgs,
    Frenet_path,
)
import numpy as np

DEFAULT_PLANNING_HORIZON = 10
TICK = 0.1


class TrajectoryPlanner:
    def __init__(self, waypoints: np.ndarray, dt=None) -> None:
        # self._t = data[:, 0]

        self.waypoints = waypoints
        self.dt = dt if dt is not None else 0.1
        self.trajectories = None

        self.tx, self.ty, self.yaw, self.curvature, self._csp = generate_target_course(self.waypoints)

    def project_position_to_path(self, position):
        # Find the closest point on the path

        if self.trajectories is None:
            return None

        source_traj = self.trajectories[1][0]  # first trajectory is always the centre line

        distances = [
            np.linalg.norm(np.array(position[:2]) - np.array(point[:2]))
            for point in [[x, y] for x, y in zip(source_traj.x, source_traj.y)]
        ]

        return np.argmin(distances)

    def generate_trajectories(
        self,
        pos,
        trajectories_requested: int = 1,
        initial_v=None,
        target_v=None,
        planning_horizon: int = DEFAULT_PLANNING_HORIZON,
    ) -> list:

        min_index = self.project_position_to_path(pos)

        planner_args = PlannerArgs(
            min_predict_time=planning_horizon * self.dt,
            max_predict_time=planning_horizon * self.dt,
            predict_step=self.dt,
            time_tick=self.dt,
            target_speed=target_v,
            stopping_time=None,  # EMERGENCY_STOPPING_TIME,
            trajectories_requested=trajectories_requested,
            generate_planning_path=True,
        )

        if min_index is None:
            s0 = 0
        else:
            s0 = self.trajectories[1][0].s[min_index]

        self.trajectories = frenet_optimal_planning(
            self._csp,
            s0,
            initial_v,
            0,  # self._c_d,
            0,  # self._c_d_d,
            0,  # self._c_d_dd,
            0,  # acceleration
            planner_args,
        )

        return self.get_working_trajectories()

    def get_working_trajectories(self) -> list:
        return self.trajectories[1]

    def get_planning_trajectory(self) -> Frenet_path:
        return self.trajectories[-1][0]
