import numpy as np


def dedupe_waypoints(waypoints, tolerance=1.0e-6):
    deduped = []
    for waypoint in waypoints:
        if (
            not deduped
            or np.linalg.norm(np.asarray(waypoint) - np.asarray(deduped[-1]))
            > tolerance
        ):
            deduped.append(list(waypoint))
    return deduped


def frenet_lateral_offsets(count, max_d):
    count = max(1, int(count))
    max_d = max(0.0, float(max_d))
    if count == 1 or max_d <= 1.0e-9:
        return [0.0]

    remaining = count - 1
    negative_count = int(np.ceil(remaining / 2.0))
    positive_count = remaining - negative_count
    negative = (
        np.linspace(-max_d, -max_d / negative_count, negative_count).tolist()
        if negative_count > 0
        else []
    )
    positive = (
        np.linspace(max_d, max_d / positive_count, positive_count).tolist()
        if positive_count > 0
        else []
    )

    offsets = [0.0]
    for index in range(max(len(negative), len(positive))):
        if index < len(negative):
            offsets.append(float(negative[index]))
        if index < len(positive):
            offsets.append(float(positive[index]))
    return offsets[:count]


def sample_spline_route(csp, *, spacing=0.1, start_s=0.0, end_s=None):
    max_s = float(csp.s[-1])
    start_s = float(np.clip(start_s, 0.0, max_s))
    end_s = max_s if end_s is None else float(np.clip(end_s, start_s, max_s))
    spacing = max(float(spacing), 1.0e-3)
    samples = np.arange(start_s, end_s + 0.5 * spacing, spacing, dtype=float)
    if samples.size == 0 or samples[-1] < end_s:
        samples = np.concatenate([samples, np.asarray([end_s], dtype=float)])

    route = []
    for s_value in samples:
        x, y = csp.calc_position(float(np.clip(s_value, 0.0, max_s)))
        if x is not None and y is not None:
            route.append([float(x), float(y)])
    return dedupe_waypoints(route)


def project_pose_to_spline_frenet(csp, pose, *, min_s=0.0, search_step=0.05):
    pose = np.asarray(pose, dtype=float)
    point = pose[:2]
    max_s = float(csp.s[-1])
    search_step = max(float(search_step), 1.0e-3)
    samples = np.arange(0.0, max_s + 0.5 * search_step, search_step, dtype=float)
    if samples.size == 0 or samples[-1] < max_s:
        samples = np.concatenate([samples, np.asarray([max_s], dtype=float)])

    positions = []
    valid_s = []
    for s_value in samples:
        x, y = csp.calc_position(float(np.clip(s_value, 0.0, max_s)))
        if x is not None and y is not None:
            valid_s.append(float(s_value))
            positions.append([float(x), float(y)])
    if not positions:
        fallback_yaw = float(pose[3]) if pose.size > 3 else 0.0
        return {
            "s": 0.0,
            "d": 0.0,
            "yaw": fallback_yaw,
            "center": point.astype(float),
            "distance": 0.0,
        }

    positions = np.asarray(positions, dtype=float)
    valid_s = np.asarray(valid_s, dtype=float)
    nearest = int(np.argmin(np.linalg.norm(positions - point, axis=1)))
    s_value = float(np.clip(max(valid_s[nearest], float(min_s)), 0.0, max_s))
    center = np.asarray(csp.calc_position(s_value), dtype=float)
    yaw = float(csp.calc_yaw(s_value))
    normal = np.asarray([-np.sin(yaw), np.cos(yaw)], dtype=float)
    delta = point - center
    return {
        "s": s_value,
        "d": float(delta @ normal),
        "yaw": yaw,
        "center": center,
        "distance": float(np.linalg.norm(delta)),
    }
