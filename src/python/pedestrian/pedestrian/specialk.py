from dataclasses import dataclass
from heapq import heappop, heappush
from time import perf_counter

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from Actor import STATE as ActorStateEnum
from config import GRID_RESOLUTION, STATIC_PLANNER_OBSTACLE_CLEARANCE
from trajectory_planner.frenet_optimal_trajectory import Frenet_path

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - scipy is part of the project deps.
    cKDTree = None


@dataclass(frozen=True)
class SpecialKParams:
    count: int
    speed: float
    dt: float
    horizon: int
    vehicle_length: float
    vehicle_width: float
    max_steer: float
    display_offset: np.ndarray
    display_diff: float
    scene_scale: float
    grid_resolution: float
    heading_bins: int
    roadmap_samples_density: float
    roadmap_samples: int
    roadmap_grid_step: float
    obstacle_edge_step: float
    nearest: int
    connect_radius: float
    raw_routes: int
    route_candidates: int
    max_overlap: float
    separation: float
    clearance_weight: float
    diversity_penalty: float
    corridor_radius: float
    lattice_resolution: float
    motion_step: float
    goal_tolerance: float
    max_expansions: int
    time_budget_ms: float
    curvature_tolerance: float
    obstacle_clearance: float
    debug: bool
    show_roadmap: bool


def wrap_angle(angle):
    return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi


def heading_bin(angle, heading_bins):
    return int(
        np.round((float(angle) % (2.0 * np.pi)) / (2.0 * np.pi / heading_bins))
    ) % int(heading_bins)


def heading_from_bin(index, heading_bins):
    return float(index) * 2.0 * np.pi / float(heading_bins)


def vehicle_footprint_polygon(state, length, width):
    x, y, _v, theta = np.asarray(state, dtype=float)[:4]
    half_length = float(length) / 2.0
    half_width = float(width) / 2.0
    corners = np.asarray(
        [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width],
        ],
        dtype=float,
    )
    c = np.cos(theta)
    s = np.sin(theta)
    rotation = np.asarray([[c, -s], [s, c]], dtype=float)
    return Polygon(corners @ rotation.T + np.asarray([x, y], dtype=float))


def blocking_static_polygon_union(static_polygons):
    polygons = []
    for static_polygon in static_polygons or []:
        if not getattr(static_polygon, "blocking", True):
            continue
        points = np.asarray(getattr(static_polygon, "points", []), dtype=float)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
            continue
        polygon = Polygon(points[:, :2])
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_empty:
            polygons.append(polygon)
    return unary_union(polygons) if polygons else None


def buffered_obstacle_union(static_polygons, clearance=0.0):
    obstacle_union = blocking_static_polygon_union(static_polygons)
    if obstacle_union is None or obstacle_union.is_empty:
        return None
    clearance = max(0.0, float(clearance))
    return obstacle_union.buffer(clearance) if clearance > 0.0 else obstacle_union


def obstacle_centroids(static_polygons):
    centroids = []
    for idx, static_polygon in enumerate(static_polygons or []):
        if not getattr(static_polygon, "blocking", True):
            continue
        points = np.asarray(getattr(static_polygon, "points", []), dtype=float)
        if points.ndim != 2 or points.shape[0] < 3:
            continue
        polygon = Polygon(points[:, :2])
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty:
            continue
        centroid = polygon.centroid
        centroids.append((idx, np.asarray([centroid.x, centroid.y], dtype=float)))
    return centroids


def point_in_bounds(point, params):
    x, y = np.asarray(point, dtype=float)[:2]
    min_x = float(params.display_offset[0])
    min_y = float(params.display_offset[1])
    return (
        min_x <= x <= min_x + params.display_diff
        and min_y <= y <= min_y + params.display_diff
    )


def point_is_free(point, obstacle_union, params):
    if not point_in_bounds(point, params):
        return False
    return (
        obstacle_union is None
        or obstacle_union.is_empty
        or not obstacle_union.covers(Point(float(point[0]), float(point[1])))
    )


def segment_is_free(a, b, obstacle_union):
    if obstacle_union is None or obstacle_union.is_empty:
        return True
    return not LineString([a, b]).intersects(obstacle_union)


def dedupe_points(points, tolerance=1.0e-6):
    deduped = []
    for point in points:
        p = np.asarray(point, dtype=float)[:2]
        if not deduped or np.linalg.norm(p - np.asarray(deduped[-1])) > tolerance:
            deduped.append(p.tolist())
    return deduped


def polyline_length(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)))


def polyline_sample(points, spacing):
    points = np.asarray(dedupe_points(points), dtype=float)
    if points.ndim != 2 or points.shape[0] < 2:
        return points
    lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    cumulative = np.zeros(points.shape[0], dtype=float)
    cumulative[1:] = np.cumsum(lengths)
    total = float(cumulative[-1])
    if total <= 1.0e-9:
        return points[:1]
    distances = np.arange(0.0, total + 0.5 * spacing, max(float(spacing), 1.0e-3))
    if distances[-1] < total:
        distances = np.concatenate([distances, np.asarray([total])])
    sampled = []
    segment = 0
    for distance in distances:
        while segment < len(lengths) - 1 and cumulative[segment + 1] < distance:
            segment += 1
        length = lengths[segment]
        if length <= 1.0e-9:
            sampled.append(points[segment, :2])
            continue
        u = np.clip((distance - cumulative[segment]) / length, 0.0, 1.0)
        sampled.append(
            points[segment, :2] + u * (points[segment + 1, :2] - points[segment, :2])
        )
    return np.asarray(sampled, dtype=float)


def point_to_polyline_distance(point, route_line):
    if route_line is None or route_line.is_empty:
        return 0.0
    return float(route_line.distance(Point(float(point[0]), float(point[1]))))


def halton(index, base):
    value = 0.0
    inv_base = 1.0 / float(base)
    fraction = inv_base
    while index > 0:
        value += fraction * (index % base)
        index //= base
        fraction *= inv_base
    return value


def add_unique_node(nodes, point, obstacle_union, params, tolerance=1.0e-4):
    point = np.asarray(point, dtype=float)[:2]
    if not point_is_free(point, obstacle_union, params):
        return None
    for idx, existing in enumerate(nodes):
        if np.linalg.norm(point - existing) <= tolerance:
            return idx
    nodes.append(point)
    return len(nodes) - 1


def obstacle_boundary_polygons(obstacle_union):
    if obstacle_union is None or obstacle_union.is_empty:
        return []
    if isinstance(obstacle_union, Polygon):
        return [obstacle_union]
    if isinstance(obstacle_union, MultiPolygon):
        return [polygon for polygon in obstacle_union.geoms if not polygon.is_empty]
    polygons = []
    for geometry in getattr(obstacle_union, "geoms", []):
        if isinstance(geometry, Polygon) and not geometry.is_empty:
            polygons.append(geometry)
    return polygons


def add_obstacle_boundary_nodes(nodes, obstacle_union, params):
    epsilon = max(0.05 * float(params.grid_resolution), 1.0e-4)
    edge_step = max(float(params.obstacle_edge_step), params.grid_resolution)
    for polygon in obstacle_boundary_polygons(obstacle_union):
        centroid = np.asarray([polygon.centroid.x, polygon.centroid.y], dtype=float)
        exterior = np.asarray(polygon.exterior.coords[:-1], dtype=float)
        if exterior.ndim != 2 or exterior.shape[0] < 3:
            continue
        for a, b in zip(exterior, np.roll(exterior, -1, axis=0)):
            edge = b - a
            length = float(np.linalg.norm(edge))
            if length <= 1.0e-9:
                continue
            samples = max(1, int(np.ceil(length / edge_step)))
            for sample_idx in range(samples + 1):
                t = sample_idx / float(samples)
                point = a + t * edge
                direction = point - centroid
                norm = float(np.linalg.norm(direction))
                if norm <= 1.0e-9:
                    continue
                add_unique_node(
                    nodes,
                    point + direction / norm * epsilon,
                    obstacle_union,
                    params,
                    tolerance=0.35 * params.grid_resolution,
                )


def build_roadmap_nodes(start_xy, goal_xy, static_polygons, obstacle_union, params):
    nodes = []
    start_idx = add_unique_node(nodes, start_xy, obstacle_union, params)
    goal_idx = add_unique_node(nodes, goal_xy, obstacle_union, params)
    if start_idx is None:
        nodes.append(np.asarray(start_xy, dtype=float)[:2])
        start_idx = 0
    if goal_idx is None:
        nodes.append(np.asarray(goal_xy, dtype=float)[:2])
        goal_idx = len(nodes) - 1

    min_x, min_y = params.display_offset[:2]
    max_x = min_x + params.display_diff
    max_y = min_y + params.display_diff

    add_obstacle_boundary_nodes(nodes, obstacle_union, params)

    offset = max(
        1.1 * params.obstacle_clearance,
        params.grid_resolution,
        0.5 * params.vehicle_width,
    )
    edge_step = max(float(params.obstacle_edge_step), params.grid_resolution)
    for static_polygon in static_polygons or []:
        if not getattr(static_polygon, "blocking", True):
            continue
        raw_points = np.asarray(getattr(static_polygon, "points", []), dtype=float)
        if raw_points.ndim != 2 or raw_points.shape[0] < 3:
            continue
        polygon = Polygon(raw_points[:, :2])
        if polygon.is_empty:
            continue
        centroid = np.asarray([polygon.centroid.x, polygon.centroid.y], dtype=float)
        for vertex in np.asarray(polygon.exterior.coords[:-1], dtype=float):
            direction = vertex - centroid
            norm = float(np.linalg.norm(direction))
            if norm <= 1.0e-9:
                continue
            for scale in (1.0, 1.75):
                add_unique_node(
                    nodes,
                    vertex + direction / norm * offset * scale,
                    obstacle_union,
                    params,
                )
        exterior = np.asarray(polygon.exterior.coords[:-1], dtype=float)
        for a, b in zip(exterior, np.roll(exterior, -1, axis=0)):
            edge = b - a
            length = float(np.linalg.norm(edge))
            if length <= 1.0e-9:
                continue
            samples = max(1, int(np.floor(length / edge_step)))
            for sample_idx in range(1, samples + 1):
                t = sample_idx / float(samples + 1)
                point = a + t * edge
                direction = point - centroid
                norm = float(np.linalg.norm(direction))
                if norm <= 1.0e-9:
                    continue
                add_unique_node(
                    nodes,
                    point + direction / norm * offset,
                    obstacle_union,
                    params,
                    tolerance=0.35 * params.grid_resolution,
                )

    grid_step = float(params.roadmap_grid_step)
    if grid_step > 0.0:
        xs = np.arange(min_x + 0.5 * grid_step, max_x, grid_step)
        ys = np.arange(min_y + 0.5 * grid_step, max_y, grid_step)
        for y in ys:
            for x in xs:
                add_unique_node(
                    nodes,
                    [x, y],
                    obstacle_union,
                    params,
                    tolerance=0.35 * params.grid_resolution,
                )

    # Low discrepancy free-space samples give reproducible PRM-like coverage.
    sample_index = 1
    attempts = max(params.roadmap_samples * 6, params.roadmap_samples + 1)
    while len(nodes) < params.roadmap_samples + 2 and sample_index <= attempts:
        x = min_x + halton(sample_index, 2) * (max_x - min_x)
        y = min_y + halton(sample_index, 3) * (max_y - min_y)
        add_unique_node(
            nodes,
            [x, y],
            obstacle_union,
            params,
            tolerance=0.35 * params.grid_resolution,
        )
        sample_index += 1

    return np.asarray(nodes, dtype=float), start_idx, goal_idx


def build_roadmap_edges(nodes, obstacle_union, params):
    graph = {idx: [] for idx in range(nodes.shape[0])}
    if nodes.shape[0] <= 1:
        return graph

    if cKDTree is not None:
        tree = cKDTree(nodes)
        query_count = min(nodes.shape[0], max(params.nearest + 1, 2))
        distances, indices = tree.query(nodes, k=query_count)
    else:
        distances = []
        indices = []
        for point in nodes:
            d = np.linalg.norm(nodes - point, axis=1)
            order = np.argsort(d)[: max(params.nearest + 1, 2)]
            distances.append(d[order])
            indices.append(order)

    edge_seen = set()

    def add_edge(source, target):
        source = int(source)
        target = int(target)
        dist = float(np.linalg.norm(nodes[source] - nodes[target]))
        if source == target or dist <= 1.0e-9:
            return False
        key = tuple(sorted((source, target)))
        if key in edge_seen:
            return False
        if not segment_is_free(nodes[source], nodes[target], obstacle_union):
            return False
        clearance = (
            params.connect_radius
            if obstacle_union is None or obstacle_union.is_empty
            else float(
                LineString([nodes[source], nodes[target]]).distance(obstacle_union)
            )
        )
        clearance_cost = params.clearance_weight / max(clearance, 0.05)
        cost = dist * (1.0 + clearance_cost)
        graph[source].append((target, cost, dist))
        graph[target].append((source, cost, dist))
        edge_seen.add(key)
        return True

    for source in range(nodes.shape[0]):
        for dist, target in zip(
            np.atleast_1d(distances[source]), np.atleast_1d(indices[source])
        ):
            target = int(target)
            dist = float(dist)
            if source == target or dist <= 1.0e-9 or dist > params.connect_radius:
                continue
            add_edge(source, target)
    return graph


def graph_components(graph, node_count):
    components = []
    component_index = np.full(node_count, -1, dtype=int)
    for node in range(node_count):
        if component_index[node] >= 0:
            continue
        stack = [node]
        component_index[node] = len(components)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour, _cost, _length in graph.get(current, []):
                neighbour = int(neighbour)
                if component_index[neighbour] >= 0:
                    continue
                component_index[neighbour] = component_index[node]
                stack.append(neighbour)
        components.append(component)
    return components, component_index


def graph_add_edge(graph, nodes, source, target, obstacle_union, params):
    source = int(source)
    target = int(target)
    if source == target:
        return False
    if any(int(neighbour) == target for neighbour, _cost, _length in graph[source]):
        return False
    if not segment_is_free(nodes[source], nodes[target], obstacle_union):
        return False
    distance = float(np.linalg.norm(nodes[source] - nodes[target]))
    if distance <= 1.0e-9:
        return False
    clearance = (
        params.connect_radius
        if obstacle_union is None or obstacle_union.is_empty
        else float(LineString([nodes[source], nodes[target]]).distance(obstacle_union))
    )
    cost = distance * (1.0 + params.clearance_weight / max(clearance, 0.05))
    graph[source].append((target, cost, distance))
    graph[target].append((source, cost, distance))
    return True


def attach_endpoint_to_roadmap(graph, nodes, endpoint_idx, obstacle_union, params):
    distances = np.linalg.norm(nodes - nodes[int(endpoint_idx)], axis=1)
    order = np.argsort(distances)
    attached = 0
    target_count = max(8, 2 * int(params.nearest))
    max_distance = max(params.connect_radius, params.display_diff * np.sqrt(2.0))
    for target in order:
        target = int(target)
        if target == int(endpoint_idx) or distances[target] > max_distance:
            continue
        if graph_add_edge(graph, nodes, endpoint_idx, target, obstacle_union, params):
            attached += 1
            if attached >= target_count:
                break
    return attached


def stitch_visible_components(graph, nodes, obstacle_union, params, required_nodes=()):
    added = 0
    required_nodes = tuple(int(node) for node in required_nodes or ())
    max_iterations = max(1, min(nodes.shape[0], 64 if len(required_nodes) >= 2 else nodes.shape[0]))
    for _ in range(max_iterations):
        components, component_index = graph_components(graph, nodes.shape[0])
        if len(components) <= 1:
            break
        if len(required_nodes) >= 2:
            start_component = int(component_index[required_nodes[0]])
            goal_component = int(component_index[required_nodes[1]])
            if start_component == goal_component:
                break

        best = None
        if len(required_nodes) >= 2:
            active_component = components[start_component]
            goal_point = nodes[required_nodes[1]]
            for comp_b_idx, comp_b in enumerate(components):
                if comp_b_idx == start_component:
                    continue
                for a in active_component:
                    deltas = nodes[comp_b] - nodes[a]
                    distances = np.linalg.norm(deltas, axis=1)
                    order = np.argsort(distances)[: min(12, len(comp_b))]
                    for order_idx in order:
                        b = comp_b[int(order_idx)]
                        distance = float(distances[int(order_idx)])
                        if distance > max(
                            params.display_diff * np.sqrt(2.0), params.connect_radius
                        ):
                            continue
                        goal_bias = 0.1 * float(np.linalg.norm(nodes[b] - goal_point))
                        score = distance + goal_bias
                        if best is not None and score >= best[0]:
                            continue
                        if segment_is_free(nodes[a], nodes[b], obstacle_union):
                            best = (score, int(a), int(b))
                            break
        else:
            for comp_a_idx, comp_a in enumerate(components):
                for comp_b_idx in range(comp_a_idx + 1, len(components)):
                    comp_b = components[comp_b_idx]
                    for a in comp_a:
                        deltas = nodes[comp_b] - nodes[a]
                        distances = np.linalg.norm(deltas, axis=1)
                        for order_idx in np.argsort(distances)[: min(12, len(comp_b))]:
                            b = comp_b[int(order_idx)]
                            distance = float(distances[int(order_idx)])
                            if distance > max(
                                params.display_diff * np.sqrt(2.0), params.connect_radius
                            ):
                                continue
                            if best is not None and distance >= best[0]:
                                continue
                            if segment_is_free(nodes[a], nodes[b], obstacle_union):
                                best = (distance, int(a), int(b))
                                break
        if best is None:
            break
        if graph_add_edge(graph, nodes, best[1], best[2], obstacle_union, params):
            added += 1
        else:
            break
    return added


def reachable_node_set(graph, start_idx):
    start_idx = int(start_idx)
    seen = {start_idx}
    stack = [start_idx]
    while stack:
        current = stack.pop()
        for neighbour, _cost, _length in graph.get(current, []):
            neighbour = int(neighbour)
            if neighbour in seen:
                continue
            seen.add(neighbour)
            stack.append(neighbour)
    return seen


def strengthen_roadmap_connectivity(
    graph, nodes, start_idx, goal_idx, obstacle_union, params
):
    start_attachments = attach_endpoint_to_roadmap(
        graph,
        nodes,
        start_idx,
        obstacle_union,
        params,
    )
    goal_attachments = attach_endpoint_to_roadmap(
        graph,
        nodes,
        goal_idx,
        obstacle_union,
        params,
    )
    stitched = stitch_visible_components(
        graph,
        nodes,
        obstacle_union,
        params,
        required_nodes=(start_idx, goal_idx),
    )
    components, component_index = graph_components(graph, nodes.shape[0])
    reachable = reachable_node_set(graph, start_idx)
    return {
        "components": len(components),
        "reachable_nodes": len(reachable),
        "pruned_nodes": int(nodes.shape[0] - len(reachable)),
        "start_component": int(component_index[int(start_idx)]),
        "goal_component": int(component_index[int(goal_idx)]),
        "start_degree": len(graph[int(start_idx)]),
        "goal_degree": len(graph[int(goal_idx)]),
        "start_attachments": start_attachments,
        "goal_attachments": goal_attachments,
        "stitched_edges": stitched,
    }


def astar_roadmap(
    nodes,
    graph,
    start_idx,
    goal_idx,
    edge_penalty=None,
    disabled_edges=None,
    use_length_cost=False,
):
    edge_penalty = edge_penalty or {}
    disabled_edges = disabled_edges or set()
    frontier = []
    heappush(frontier, (0.0, 0.0, start_idx))
    came_from = {}
    cost_so_far = {start_idx: 0.0}

    def heuristic(idx):
        return float(np.linalg.norm(nodes[idx] - nodes[goal_idx]))

    while frontier:
        _priority, current_cost, current = heappop(frontier)
        if current_cost > cost_so_far.get(current, float("inf")) + 1.0e-9:
            continue
        if current == goal_idx:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, current_cost

        for nxt, edge_cost, edge_length in graph.get(current, []):
            edge = tuple(sorted((current, nxt)))
            if edge in disabled_edges:
                continue
            base_cost = float(edge_length) if use_length_cost else float(edge_cost)
            new_cost = current_cost + base_cost + float(edge_penalty.get(edge, 0.0))
            if new_cost < cost_so_far.get(nxt, float("inf")):
                cost_so_far[nxt] = new_cost
                came_from[nxt] = current
                heappush(frontier, (new_cost + heuristic(nxt), new_cost, nxt))
    return None, float("inf")


def path_edges(path):
    return [tuple(sorted((a, b))) for a, b in zip(path[:-1], path[1:])]


def path_has_loop(path):
    return len(set(path)) != len(path)


def route_has_self_intersection(route):
    route = dedupe_points(route)
    if len(route) < 4:
        return False
    line = LineString(route)
    return not line.is_simple


def route_node_path(nodes, node_path):
    return dedupe_points([nodes[idx] for idx in node_path])


def astar_roadmap_via(nodes, graph, waypoints, edge_penalty=None, use_length_cost=True):
    full_path = []
    total_cost = 0.0
    for source, target in zip(waypoints[:-1], waypoints[1:]):
        segment, cost = astar_roadmap(
            nodes,
            graph,
            source,
            target,
            edge_penalty=edge_penalty,
            use_length_cost=use_length_cost,
        )
        if segment is None or len(segment) < 2:
            return None, float("inf")
        if full_path:
            full_path.extend(segment[1:])
        else:
            full_path.extend(segment)
        total_cost += float(cost)
    if path_has_loop(full_path):
        return None, float("inf")
    return full_path, total_cost


def route_signature(route, centroids):
    signature = {}
    points = np.asarray(route, dtype=float)
    if points.shape[0] < 2:
        return tuple()
    for obstacle_id, centroid in centroids:
        value = 0
        ray_y = float(centroid[1])
        ray_x = float(centroid[0])
        for a, b in zip(points[:-1], points[1:]):
            ay = float(a[1])
            by = float(b[1])
            if (ay <= ray_y < by) or (by <= ray_y < ay):
                t = (ray_y - ay) / max(by - ay, 1.0e-12)
                cross_x = float(a[0] + t * (b[0] - a[0]))
                if cross_x >= ray_x:
                    value += 1 if by > ay else -1
        if value:
            signature[int(obstacle_id)] = int(value)
    return tuple(sorted(signature.items()))


def mean_route_separation(route, other_route):
    samples = polyline_sample(route, spacing=0.5)
    if samples.ndim != 2 or samples.shape[0] == 0:
        return 0.0
    other = LineString(dedupe_points(other_route))
    return float(
        np.mean([other.distance(Point(float(p[0]), float(p[1]))) for p in samples])
    )


def edge_overlap(candidate_edges, accepted_edge_sets):
    if not candidate_edges or not accepted_edge_sets:
        return 0.0
    candidate = set(candidate_edges)
    overlaps = []
    for accepted in accepted_edge_sets:
        denom = max(1, min(len(candidate), len(accepted)))
        overlaps.append(len(candidate & accepted) / float(denom))
    return float(max(overlaps)) if overlaps else 0.0


def path_is_diverse(route, signature, edges, accepted, params):
    if not accepted:
        return True
    accepted_signatures = {item["signature"] for item in accepted}
    if signature not in accepted_signatures:
        return True
    overlap = edge_overlap(edges, [item["edges"] for item in accepted])
    if overlap < params.max_overlap:
        return True
    separation = max(mean_route_separation(route, item["route"]) for item in accepted)
    return separation >= params.separation


def direct_line_basis(start_xy, goal_xy, fallback_heading):
    start = np.asarray(start_xy, dtype=float)[:2]
    goal = np.asarray(goal_xy, dtype=float)[:2]
    delta = goal - start
    length = float(np.linalg.norm(delta))
    if length > 1.0e-9:
        forward = delta / length
    else:
        forward = np.asarray(
            [np.cos(float(fallback_heading)), np.sin(float(fallback_heading))],
            dtype=float,
        )
    normal = np.asarray([-forward[1], forward[0]], dtype=float)
    return start, goal, forward, normal, length


def point_line_progress_and_offset(point, start, goal, forward, normal):
    point = np.asarray(point, dtype=float)[:2]
    return float((point - start) @ forward), float((point - start) @ normal)


def local_anchor_points(start_state, goal_xy, obstacle_union, params):
    start, goal, forward, normal, distance = direct_line_basis(
        start_state[:2],
        goal_xy,
        start_state[ActorStateEnum.THETA],
    )
    heading = float(start_state[ActorStateEnum.THETA])
    heading_forward = np.asarray([np.cos(heading), np.sin(heading)], dtype=float)
    min_turn_radius = params.vehicle_length / max(np.tan(params.max_steer), 1.0e-6)
    distances = [
        max(
            2.5 * params.vehicle_length,
            0.65 * min_turn_radius,
            1.2 * params.scene_scale,
        ),
        max(
            4.0 * params.vehicle_length,
            1.15 * min_turn_radius,
            2.0 * params.scene_scale,
        ),
    ]
    angles = [
        0.0,
        0.35 * params.max_steer,
        -0.35 * params.max_steer,
        0.75 * params.max_steer,
        -0.75 * params.max_steer,
    ]
    anchors = []
    for distance_scale in distances:
        for angle in angles:
            direction_heading = heading + float(angle)
            direction = np.asarray(
                [np.cos(direction_heading), np.sin(direction_heading)],
                dtype=float,
            )
            if float(direction @ heading_forward) <= 0.0:
                continue
            point = start + distance_scale * direction
            if np.linalg.norm(goal - point) >= max(
                0.4 * params.scene_scale, 0.15 * distance
            ):
                if point_is_free(point, obstacle_union, params):
                    anchors.append(point)
    return anchors


def select_global_anchor_indices(nodes, start_idx, goal_idx, params, max_anchors):
    start, goal, forward, normal, distance = direct_line_basis(
        nodes[start_idx],
        nodes[goal_idx],
        0.0,
    )
    if distance <= 1.0e-6:
        return []

    candidates = []
    min_endpoint_distance = max(params.connect_radius * 0.35, params.vehicle_width)
    for idx, point in enumerate(nodes):
        if idx in (start_idx, goal_idx):
            continue
        progress, offset = point_line_progress_and_offset(
            point,
            start,
            goal,
            forward,
            normal,
        )
        if progress <= 0.08 * distance or progress >= 0.94 * distance:
            continue
        endpoint_distance = min(
            float(np.linalg.norm(point - start)),
            float(np.linalg.norm(point - goal)),
        )
        if endpoint_distance < min_endpoint_distance:
            continue
        score = abs(offset) + 0.15 * endpoint_distance
        side = -1 if offset < 0 else 1
        candidates.append((score, side, idx))

    candidates.sort(reverse=True)
    selected = []
    side_counts = {-1: 0, 1: 0}
    spacing = max(params.separation, params.connect_radius * 0.35)
    for _score, side, idx in candidates:
        if side_counts[side] > side_counts[-side] + 1:
            continue
        if any(
            float(np.linalg.norm(nodes[idx] - nodes[other])) < spacing
            for other in selected
        ):
            continue
        selected.append(idx)
        side_counts[side] += 1
        if len(selected) >= int(max_anchors):
            break
    return selected


def add_route_candidate(
    *,
    nodes,
    node_path,
    cost,
    centroids,
    accepted,
    raw_seen,
    params,
    reason,
):
    if node_path is None or len(node_path) < 2 or path_has_loop(node_path):
        return False
    key = tuple(node_path)
    if key in raw_seen:
        return False
    raw_seen.add(key)
    route = route_node_path(nodes, node_path)
    if route_has_self_intersection(route):
        return False
    edges = path_edges(node_path)
    signature = route_signature(route, centroids)
    if not path_is_diverse(route, signature, edges, accepted, params):
        return False
    accepted.append(
        {
            "route": route,
            "signature": signature,
            "edges": set(edges),
            "cost": float(cost),
            "reason": reason,
        }
    )
    return True


def roadmap_debug_payload(
    nodes,
    graph,
    accepted,
    *,
    start_idx,
    goal_idx,
    local_anchor_indices=None,
    global_anchor_indices=None,
    connectivity=None,
):
    reachable = reachable_node_set(graph, start_idx)
    edges = []
    seen = set()
    for source, neighbours in graph.items():
        if int(source) not in reachable:
            continue
        for target, _cost, _length in neighbours:
            if int(target) not in reachable:
                continue
            edge = tuple(sorted((int(source), int(target))))
            if edge in seen:
                continue
            seen.add(edge)
            edges.append(
                [
                    nodes[edge[0], :2].astype(float).tolist(),
                    nodes[edge[1], :2].astype(float).tolist(),
                ]
            )
    return {
        "nodes": [
            nodes[int(idx), :2].astype(float).tolist() for idx in sorted(reachable)
        ],
        "edges": edges,
        "start": nodes[int(start_idx), :2].astype(float).tolist(),
        "goal": nodes[int(goal_idx), :2].astype(float).tolist(),
        "local_anchors": [
            nodes[int(idx), :2].astype(float).tolist()
            for idx in local_anchor_indices or []
            if int(idx) in reachable
        ],
        "global_anchors": [
            nodes[int(idx), :2].astype(float).tolist()
            for idx in global_anchor_indices or []
            if int(idx) in reachable
        ],
        "skeletons": [item["route"] for item in accepted],
        "signatures": [item["signature"] for item in accepted],
        "costs": [float(item["cost"]) for item in accepted],
        "reasons": [item.get("reason", "") for item in accepted],
        "connectivity": connectivity or {},
    }


def generate_route_skeletons(
    start_state, goal_xy, static_polygons, obstacle_union, params
):
    start_xy = np.asarray(start_state, dtype=float)[:2]
    if obstacle_union is None or obstacle_union.is_empty:
        routes = open_space_routes(start_xy, goal_xy, params)
        debug = {
            "nodes": [list(start_xy), np.asarray(goal_xy, dtype=float)[:2].tolist()],
            "edges": [[list(start_xy), np.asarray(goal_xy, dtype=float)[:2].tolist()]],
            "start": list(start_xy),
            "goal": np.asarray(goal_xy, dtype=float)[:2].tolist(),
            "local_anchors": [],
            "global_anchors": [],
            "skeletons": routes,
            "signatures": [],
            "costs": [polyline_length(route) for route in routes],
            "reasons": ["open"] * len(routes),
        }
        return routes, debug

    nodes, start_idx, goal_idx = build_roadmap_nodes(
        start_xy,
        goal_xy,
        static_polygons,
        obstacle_union,
        params,
    )
    graph = build_roadmap_edges(nodes, obstacle_union, params)
    connectivity = strengthen_roadmap_connectivity(
        graph,
        nodes,
        start_idx,
        goal_idx,
        obstacle_union,
        params,
    )
    centroids = obstacle_centroids(static_polygons)

    mutable_nodes = [node.copy() for node in nodes]
    local_anchor_indices = []
    for point in local_anchor_points(start_state, goal_xy, obstacle_union, params):
        idx = add_unique_node(
            mutable_nodes,
            point,
            obstacle_union,
            params,
            tolerance=0.25 * params.grid_resolution,
        )
        if idx is not None and idx >= len(nodes):
            local_anchor_indices.append(idx)
    if local_anchor_indices:
        nodes = np.asarray(mutable_nodes, dtype=float)
        graph = build_roadmap_edges(nodes, obstacle_union, params)
        start_idx = 0
        goal_idx = 1
        connectivity = strengthen_roadmap_connectivity(
            graph,
            nodes,
            start_idx,
            goal_idx,
            obstacle_union,
            params,
        )

    accepted = []
    edge_penalty = {}
    raw_seen = set()

    shortest_path, shortest_cost = astar_roadmap(
        nodes,
        graph,
        start_idx,
        goal_idx,
        use_length_cost=True,
    )
    add_route_candidate(
        nodes=nodes,
        node_path=shortest_path,
        cost=shortest_cost,
        centroids=centroids,
        accepted=accepted,
        raw_seen=raw_seen,
        params=params,
        reason="shortest",
    )

    global_anchor_indices = select_global_anchor_indices(
        nodes,
        start_idx,
        goal_idx,
        params,
        max_anchors=max(2 * params.route_candidates, params.raw_routes),
    )

    for idx in [*local_anchor_indices, *global_anchor_indices]:
        node_path, cost = astar_roadmap_via(
            nodes,
            graph,
            [start_idx, idx, goal_idx],
            edge_penalty=None,
            use_length_cost=True,
        )
        add_route_candidate(
            nodes=nodes,
            node_path=node_path,
            cost=cost,
            centroids=centroids,
            accepted=accepted,
            raw_seen=raw_seen,
            params=params,
            reason="anchor",
        )
        if len(accepted) >= params.route_candidates:
            break

    attempts = max(params.raw_routes, params.route_candidates)
    for attempt in range(attempts):
        path, cost = astar_roadmap(
            nodes, graph, start_idx, goal_idx, edge_penalty=edge_penalty
        )
        if path is None or len(path) < 2:
            break
        edges = path_edges(path)
        add_route_candidate(
            nodes=nodes,
            node_path=path,
            cost=cost,
            centroids=centroids,
            accepted=accepted,
            raw_seen=raw_seen,
            params=params,
            reason="penalized",
        )
        if len(accepted) >= params.route_candidates:
            break

        scale = params.diversity_penalty * (1.0 + 0.15 * attempt)
        for edge in edges:
            edge_penalty[edge] = edge_penalty.get(edge, 0.0) + scale

    if accepted:
        shortest = [item for item in accepted if item.get("reason") == "shortest"]
        alternatives = [item for item in accepted if item.get("reason") != "shortest"]
        alternatives.sort(
            key=lambda item: (
                item["signature"] in {alt["signature"] for alt in shortest},
                item["cost"],
            )
        )
        accepted = [*shortest, *alternatives]
        debug = roadmap_debug_payload(
            nodes,
            graph,
            accepted,
            start_idx=start_idx,
            goal_idx=goal_idx,
            local_anchor_indices=local_anchor_indices,
            global_anchor_indices=global_anchor_indices,
            connectivity=connectivity,
        )
        return [item["route"] for item in accepted], debug

    routes = []
    debug = roadmap_debug_payload(
        nodes,
        graph,
        [],
        start_idx=start_idx,
        goal_idx=goal_idx,
        local_anchor_indices=local_anchor_indices,
        global_anchor_indices=global_anchor_indices,
        connectivity=connectivity,
    )
    debug["skeletons"] = []
    debug["costs"] = []
    debug["reasons"] = ["no_graph_path"]
    return routes, debug


def open_space_routes(start_xy, goal_xy, params):
    start = np.asarray(start_xy, dtype=float)[:2]
    goal = np.asarray(goal_xy, dtype=float)[:2]
    delta = goal - start
    distance = float(np.linalg.norm(delta))
    if distance <= 1.0e-6:
        return [[start.tolist(), goal.tolist()]]
    normal = np.asarray([-delta[1], delta[0]], dtype=float) / distance
    midpoint = 0.5 * (start + goal)
    span = min(
        max(params.vehicle_width * 3.0, params.separation), 0.25 * params.display_diff
    )
    routes = [[start.tolist(), goal.tolist()]]
    for idx in range(1, max(params.route_candidates, params.count)):
        side = -1.0 if idx % 2 else 1.0
        scale = 1.0 + 0.5 * ((idx - 1) // 2)
        waypoint = midpoint + side * scale * span * normal
        if point_in_bounds(waypoint, params):
            routes.append([start.tolist(), waypoint.tolist(), goal.tolist()])
    return routes


def state_in_bounds(state, params):
    return point_in_bounds(np.asarray(state, dtype=float)[:2], params)


def state_collision_free(state, obstacle_union, params):
    if not state_in_bounds(state, params):
        return False
    if obstacle_union is None or obstacle_union.is_empty:
        return True
    footprint = vehicle_footprint_polygon(
        state,
        params.vehicle_length,
        params.vehicle_width,
    )
    return not footprint.intersects(obstacle_union)


def rollout_ackermann(state, steer, distance, sample_distance, params):
    state = np.asarray(state, dtype=float)[:4].copy()
    state[ActorStateEnum.VELOCITY] = params.speed
    steps = max(1, int(np.ceil(float(distance) / max(float(sample_distance), 1.0e-3))))
    ds = float(distance) / float(steps)
    curvature = np.tan(float(steer)) / max(params.vehicle_length, 1.0e-6)
    states = []
    for _ in range(steps):
        theta_mid = state[ActorStateEnum.THETA] + 0.5 * curvature * ds
        state[ActorStateEnum.X] += ds * np.cos(theta_mid)
        state[ActorStateEnum.Y] += ds * np.sin(theta_mid)
        state[ActorStateEnum.THETA] = wrap_angle(
            state[ActorStateEnum.THETA] + curvature * ds
        )
        state[ActorStateEnum.VELOCITY] = params.speed
        states.append(state.copy())
    return states


def primitive_is_valid(states, obstacle_union, corridor, params):
    for state in states:
        if not state_collision_free(state, obstacle_union, params):
            return False
        if corridor is not None and not corridor.covers(
            Point(float(state[0]), float(state[1]))
        ):
            return False
    return True


def state_key(state, params):
    x, y, _v, theta = np.asarray(state, dtype=float)[:4]
    col = int(np.floor((x - params.display_offset[0]) / params.lattice_resolution))
    row = int(np.floor((y - params.display_offset[1]) / params.lattice_resolution))
    return col, row, heading_bin(theta, params.heading_bins)


def state_cell(state, params):
    x, y = np.asarray(state, dtype=float)[:2]
    return (
        int(np.floor((x - params.display_offset[0]) / params.lattice_resolution)),
        int(np.floor((y - params.display_offset[1]) / params.lattice_resolution)),
    )


def route_spatial_cells(route, params):
    samples = polyline_sample(
        route,
        spacing=max(float(params.lattice_resolution), 0.5 * float(params.vehicle_width)),
    )
    if samples.ndim != 2 or samples.shape[0] == 0:
        return []
    origin = np.asarray(params.display_offset, dtype=float)[:2]
    cell_size = max(float(params.lattice_resolution), 1.0e-6)
    cells = []
    previous = None
    for point in samples[:, :2]:
        cell = tuple(np.floor((point - origin) / cell_size).astype(int))
        if cell != previous:
            cells.append(cell)
            previous = cell
    return cells


def reconstruct_states(parent, node_states, goal_key):
    keys = []
    key = goal_key
    while key is not None:
        keys.append(key)
        key = parent[key][0]
    keys.reverse()
    states = [node_states[keys[0]].copy()]
    for key in keys[1:]:
        states.extend(state.copy() for state in parent[key][1])
    return states


def connect_to_goal(current, goal_xy, obstacle_union, corridor, params):
    goal = np.asarray(goal_xy, dtype=float)[:2]
    state = np.asarray(current, dtype=float)[:4].copy()
    states = []
    max_distance = max(
        4.0 * params.motion_step, np.linalg.norm(goal - state[:2]) + params.motion_step
    )
    sample_distance = max(params.motion_step / 5.0, 0.05)
    for _ in range(max(1, int(np.ceil(max_distance / sample_distance)))):
        delta = goal - state[:2]
        distance = float(np.linalg.norm(delta))
        if distance <= params.goal_tolerance:
            return states
        bearing = float(np.arctan2(delta[1], delta[0]))
        heading_error = wrap_angle(bearing - state[ActorStateEnum.THETA])
        if np.cos(heading_error) < -0.05:
            return None
        lookahead = max(distance, params.vehicle_length, sample_distance)
        steer = np.arctan2(
            2.0 * params.vehicle_length * np.sin(heading_error), lookahead
        )
        steer = float(np.clip(steer, -params.max_steer, params.max_steer))
        step_states = rollout_ackermann(
            state,
            steer,
            min(sample_distance, distance),
            sample_distance,
            params,
        )
        if not primitive_is_valid(step_states, obstacle_union, corridor, params):
            return None
        state = step_states[-1]
        states.extend(step_states)
    return states if np.linalg.norm(goal - state[:2]) <= params.goal_tolerance else None


def track_route_with_ackermann(start_state, route, obstacle_union, params):
    route = dedupe_points(route)
    if len(route) < 2:
        return None
    route_line = LineString(route)
    if route_line.length <= 1.0e-9:
        return None
    min_turn_radius = params.vehicle_length / max(np.tan(params.max_steer), 1.0e-6)
    corridor_radius = max(
        params.corridor_radius, 2.0 * min_turn_radius, 2.0 * params.vehicle_width
    )
    corridor = route_line.buffer(corridor_radius)
    state = np.asarray(start_state, dtype=float)[:4].copy()
    state[ActorStateEnum.VELOCITY] = params.speed
    if not primitive_is_valid([state], obstacle_union, corridor, params):
        return None

    step_distance = max(params.speed * params.dt, 0.03)
    lookahead = max(
        2.5 * params.vehicle_length, 2.0 * step_distance, params.lattice_resolution
    )
    states = [state.copy()]
    stagnant_steps = 0
    last_progress = route_line.project(Point(float(state[0]), float(state[1])))
    for _ in range(max(1, int(params.horizon))):
        progress = route_line.project(Point(float(state[0]), float(state[1])))
        target_s = min(route_line.length, progress + lookahead)
        target = route_line.interpolate(target_s)
        target_xy = np.asarray([target.x, target.y], dtype=float)
        delta = target_xy - state[:2]
        distance = float(np.linalg.norm(delta))
        if distance <= 1.0e-6:
            target_s = min(route_line.length, progress + 0.5 * lookahead)
            target = route_line.interpolate(target_s)
            target_xy = np.asarray([target.x, target.y], dtype=float)
            delta = target_xy - state[:2]
            distance = float(np.linalg.norm(delta))
            if distance <= 1.0e-6:
                break

        bearing = float(np.arctan2(delta[1], delta[0]))
        heading_error = wrap_angle(bearing - state[ActorStateEnum.THETA])
        if np.cos(heading_error) < -0.15:
            return None
        steer = np.arctan2(
            2.0 * params.vehicle_length * np.sin(heading_error),
            max(distance, lookahead),
        )
        steer = float(np.clip(steer, -params.max_steer, params.max_steer))
        primitive = rollout_ackermann(
            state,
            steer,
            step_distance,
            step_distance,
            params,
        )
        if not primitive_is_valid(primitive, obstacle_union, corridor, params):
            return None
        state = primitive[-1]
        states.extend(primitive)

        new_progress = route_line.project(Point(float(state[0]), float(state[1])))
        progress_slack = max(0.5 * step_distance, 0.5 * params.lattice_resolution)
        if new_progress < last_progress - progress_slack:
            return None
        if new_progress <= last_progress + 1.0e-4:
            stagnant_steps += 1
        else:
            stagnant_steps = 0
        last_progress = new_progress
        if stagnant_steps >= 8:
            return None
        if route_line.length - new_progress <= params.goal_tolerance:
            break

    return states if len(states) >= 2 else None


def lattice_plan_for_route(
    start_state, goal_xy, route, obstacle_union, params, used_cells=None
):
    route = dedupe_points(route)
    route_line = LineString(route)
    min_turn_radius = params.vehicle_length / max(np.tan(params.max_steer), 1.0e-6)
    corridor_radius = max(
        params.corridor_radius, 2.0 * min_turn_radius, 2.0 * params.vehicle_width
    )
    corridor = route_line.buffer(corridor_radius)
    start_state = np.asarray(start_state, dtype=float)[:4].copy()
    start_state[ActorStateEnum.VELOCITY] = params.speed
    goal = np.asarray(goal_xy, dtype=float)[:2]
    used_cells = used_cells or {}

    if not state_collision_free(start_state, obstacle_union, params):
        return None

    steer_values = np.asarray(
        [
            -params.max_steer,
            -0.55 * params.max_steer,
            0.0,
            0.55 * params.max_steer,
            params.max_steer,
        ],
        dtype=float,
    )
    sample_distance = max(params.motion_step / 5.0, 0.05)

    def heuristic(state):
        delta = goal - np.asarray(state, dtype=float)[:2]
        distance = float(np.linalg.norm(delta))
        if distance <= 1.0e-9:
            return 0.0
        bearing = float(np.arctan2(delta[1], delta[0]))
        heading_error = abs(wrap_angle(bearing - state[ActorStateEnum.THETA]))
        return distance + 0.2 * min_turn_radius * heading_error

    start_key = state_key(start_state, params)
    node_states = {start_key: start_state}
    parent = {start_key: (None, [])}
    cost_so_far = {start_key: 0.0}
    frontier = []
    counter = 0
    heappush(frontier, (heuristic(start_state), 0.0, counter, start_key))
    expansions = 0
    started = perf_counter()

    while frontier and expansions < params.max_expansions:
        if (
            params.time_budget_ms > 0.0
            and (perf_counter() - started) * 1000.0 > params.time_budget_ms
        ):
            break
        _priority, current_cost, _counter, current_key = heappop(frontier)
        if current_cost > cost_so_far.get(current_key, float("inf")) + 1.0e-9:
            continue
        current = node_states[current_key]
        expansions += 1

        if np.linalg.norm(goal - current[:2]) <= params.goal_tolerance:
            return reconstruct_states(parent, node_states, current_key)

        if np.linalg.norm(goal - current[:2]) <= 4.0 * params.motion_step:
            connector = connect_to_goal(current, goal, obstacle_union, corridor, params)
            if connector is not None:
                goal_key = ("goal", expansions)
                node_states[goal_key] = connector[-1] if connector else current.copy()
                parent[goal_key] = (current_key, connector)
                return reconstruct_states(parent, node_states, goal_key)

        current_progress = route_line.project(
            Point(float(current[0]), float(current[1]))
        )
        for steer in steer_values:
            primitive = rollout_ackermann(
                current,
                steer,
                params.motion_step,
                sample_distance,
                params,
            )
            if not primitive_is_valid(primitive, obstacle_union, corridor, params):
                continue
            next_state = primitive[-1]
            next_progress = route_line.project(
                Point(float(next_state[0]), float(next_state[1]))
            )
            if next_progress < current_progress - params.lattice_resolution:
                continue
            next_key = state_key(next_state, params)
            if next_key == current_key:
                continue
            next_cell = state_cell(next_state, params)
            route_distance = point_to_polyline_distance(next_state[:2], route_line)
            steer_cost = 0.18 * abs(float(steer)) / max(params.max_steer, 1.0e-6)
            reuse_cost = params.diversity_penalty * float(used_cells.get(next_cell, 0))
            new_cost = (
                current_cost
                + params.motion_step * (1.0 + steer_cost)
                + 0.12 * route_distance * route_distance
                + reuse_cost
            )
            if new_cost >= cost_so_far.get(next_key, float("inf")):
                continue
            cost_so_far[next_key] = new_cost
            node_states[next_key] = next_state
            parent[next_key] = (current_key, primitive)
            counter += 1
            heappush(
                frontier,
                (new_cost + heuristic(next_state), new_cost, counter, next_key),
            )
    return None


def states_to_frenet_path(states, params):
    states = np.asarray(states, dtype=float)
    path = Frenet_path()
    if states.ndim != 2 or states.shape[0] == 0:
        return path
    states = states[: max(1, int(params.horizon) + 1)]
    path.x = states[:, ActorStateEnum.X].astype(float).tolist()
    path.y = states[:, ActorStateEnum.Y].astype(float).tolist()
    path.yaw = states[:, ActorStateEnum.THETA].astype(float).tolist()
    path.s_d = [float(params.speed)] * states.shape[0]
    path.t = [idx * float(params.dt) for idx in range(states.shape[0])]
    if states.shape[0] >= 2:
        ds = np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1)
        path.ds = ds.astype(float).tolist()
        path.s = [0.0]
        path.s.extend(np.cumsum(ds).astype(float).tolist())
        path.c = [
            (
                0.0
                if segment <= 1.0e-9
                else wrap_angle(path.yaw[idx + 1] - path.yaw[idx]) / float(segment)
            )
            for idx, segment in enumerate(ds)
        ]
    else:
        path.ds = []
        path.s = [0.0]
        path.c = []
    path.s_dd = [0.0] * states.shape[0]
    path.s_ddd = [0.0] * states.shape[0]
    path.d = [0.0] * states.shape[0]
    path.d_d = [0.0] * states.shape[0]
    path.d_dd = [0.0] * states.shape[0]
    path.d_ddd = [0.0] * states.shape[0]
    return path


def states_to_route(states, min_spacing):
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[0] == 0:
        return []
    route = [states[0, :2].astype(float).tolist()]
    last = states[0, :2]
    for state in states[1:]:
        point = state[:2]
        if np.linalg.norm(point - last) >= min_spacing:
            route.append(point.astype(float).tolist())
            last = point
    if np.linalg.norm(states[-1, :2] - np.asarray(route[-1])) > 1.0e-6:
        route.append(states[-1, :2].astype(float).tolist())
    return dedupe_points(route)


def states_have_self_intersection(states):
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[0] < 4:
        return False
    points = dedupe_points(states[:, :2], tolerance=1.0e-5)
    if len(points) < 4:
        return False
    return not LineString(points).is_simple


def states_revisit_spatial_cell(states, params):
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[0] < 3:
        return False
    cell_size = max(float(params.lattice_resolution), 0.5 * float(params.vehicle_width))
    visited = {}
    previous_cell = None
    origin = np.asarray(params.display_offset, dtype=float)[:2]
    for index, state in enumerate(states):
        xy = np.asarray(state, dtype=float)[:2]
        cell = tuple(np.floor((xy - origin) / max(cell_size, 1.0e-6)).astype(int))
        if cell == previous_cell:
            continue
        if cell in visited:
            return True
        visited[cell] = index
        previous_cell = cell
    return False


def states_make_monotonic_route_progress(states, route, params):
    states = np.asarray(states, dtype=float)
    route = dedupe_points(route)
    if states.ndim != 2 or states.shape[0] < 2 or len(route) < 2:
        return True
    route_line = LineString(route)
    if route_line.length <= 1.0e-9:
        return True
    progress = np.asarray(
        [
            route_line.project(Point(float(state[0]), float(state[1])))
            for state in states
        ],
        dtype=float,
    )
    allowed_backtrack = max(
        float(params.lattice_resolution), 0.5 * float(params.vehicle_length)
    )
    return bool(np.all(np.diff(progress) >= -allowed_backtrack))


def states_are_loop_free(states, route, params):
    if states_have_self_intersection(states):
        return False, "self_intersection"
    if states_revisit_spatial_cell(states, params):
        return False, "revisited_cell"
    if not states_make_monotonic_route_progress(states, route, params):
        return False, "backtrack"
    return True, ""


def trajectory_respects_kinematics(path, params):
    max_curvature = np.tan(params.max_steer) / max(params.vehicle_length, 1.0e-6)
    allowed = max_curvature * (1.0 + params.curvature_tolerance)
    curvatures = np.asarray(getattr(path, "c", []), dtype=float)
    if curvatures.size == 0:
        return True
    finite = curvatures[np.isfinite(curvatures)]
    return bool(np.all(np.abs(finite) <= allowed + 1.0e-9))


def fallback_forward_candidate(start_state, obstacle_union, params):
    start = np.asarray(start_state, dtype=float)[:4].copy()
    start[ActorStateEnum.VELOCITY] = params.speed
    best_states = [start.copy()]
    for distance in (
        params.motion_step,
        0.5 * params.motion_step,
        params.grid_resolution,
    ):
        states = rollout_ackermann(
            start, 0.0, distance, max(distance / 4.0, 0.05), params
        )
        if primitive_is_valid(states, obstacle_union, None, params):
            best_states.extend(states)
            break
    path = states_to_frenet_path(best_states, params)
    return {
        "path": path,
        "route": states_to_route(
            best_states, min_spacing=max(params.grid_resolution, 0.1)
        ),
        "length": polyline_length([state[:2] for state in best_states]),
        "signature": tuple(),
        "generator": "specialk",
        "fallback": True,
    }


def build_params(
    args,
    display_offset,
    display_diff,
    vehicle_length,
    vehicle_width,
    vehicle_scale,
    max_steer,
    resolution,
):
    scene_scale = max(float(vehicle_scale or 1.0), 1.0e-6)
    min_turn_radius = float(vehicle_length) / max(np.tan(float(max_steer)), 1.0e-6)
    grid_resolution = float(resolution or GRID_RESOLUTION)
    density = max(
        0.0,
        float(getattr(args, "specialk_roadmap_samples_density", 1.0)),
    )
    display_diff = float(display_diff)
    scene_area_m2 = (display_diff / scene_scale) ** 2
    roadmap_samples = int(np.ceil(density * scene_area_m2)) if density > 0.0 else 0
    density_spacing = (
        scene_scale / np.sqrt(density)
        if density > 0.0
        else max(scene_scale, grid_resolution)
    )
    connect_radius = float(
        getattr(args, "specialk_connect_radius", None)
        or max(2.5 * density_spacing, 6.0 * grid_resolution, 4.0 * min_turn_radius)
    )
    lattice_resolution = float(
        getattr(args, "specialk_lattice_resolution", None)
        or max(grid_resolution, 0.35 * scene_scale, min_turn_radius / 4.0)
    )
    obstacle_clearance = getattr(args, "specialk_obstacle_clearance", None)
    if obstacle_clearance is None:
        obstacle_clearance = STATIC_PLANNER_OBSTACLE_CLEARANCE * scene_scale

    return SpecialKParams(
        count=max(1, int(getattr(args, "trajectory_count", 1))),
        speed=float(getattr(args, "robot_speed", 0.5)),
        dt=float(getattr(args, "tick_time", 0.01)),
        horizon=int(getattr(args, "horizon", 1)),
        vehicle_length=float(vehicle_length),
        vehicle_width=float(vehicle_width),
        max_steer=float(max_steer),
        display_offset=np.asarray(display_offset, dtype=float)[:2],
        display_diff=display_diff,
        scene_scale=scene_scale,
        grid_resolution=grid_resolution,
        heading_bins=max(8, int(getattr(args, "specialk_heading_bins", 16))),
        roadmap_samples_density=density,
        roadmap_samples=roadmap_samples,
        roadmap_grid_step=float(
            getattr(args, "specialk_roadmap_grid_step", None)
            or max(2.0 * grid_resolution, density_spacing)
        ),
        obstacle_edge_step=float(
            getattr(args, "specialk_obstacle_edge_step", None)
            or max(2.0 * grid_resolution, min(density_spacing, 1.5 * scene_scale))
        ),
        nearest=max(3, int(getattr(args, "specialk_nearest", 8))),
        connect_radius=connect_radius,
        raw_routes=max(
            1,
            int(
                getattr(args, "specialk_raw_routes", 0)
                or max(3 * int(getattr(args, "trajectory_count", 1)), 12)
            ),
        ),
        route_candidates=max(
            1,
            int(
                getattr(args, "specialk_route_candidates", 0)
                or max(2 * int(getattr(args, "trajectory_count", 1)), 6)
            ),
        ),
        max_overlap=float(getattr(args, "specialk_max_overlap", 0.6)),
        separation=float(
            getattr(args, "specialk_separation", None)
            or max(2.0 * float(vehicle_width), density_spacing, 2.0 * grid_resolution)
        ),
        clearance_weight=float(getattr(args, "specialk_clearance_weight", 0.25)),
        diversity_penalty=float(getattr(args, "specialk_diversity_penalty", 1.2)),
        corridor_radius=float(
            getattr(args, "specialk_corridor_radius", None)
            or max(2.0 * min_turn_radius, 2.5 * float(vehicle_width))
        ),
        lattice_resolution=lattice_resolution,
        motion_step=float(
            getattr(args, "specialk_motion_step", None)
            or max(1.5 * lattice_resolution, 0.75 * min_turn_radius)
        ),
        goal_tolerance=float(
            getattr(args, "specialk_goal_tolerance", None)
            or max(1.5 * lattice_resolution, 0.5 * scene_scale)
        ),
        max_expansions=max(250, int(getattr(args, "specialk_max_expansions", 1200))),
        time_budget_ms=float(getattr(args, "specialk_time_budget_ms", 0.0)),
        curvature_tolerance=float(getattr(args, "kpaths_curvature_tolerance", 0.25)),
        obstacle_clearance=max(0.0, float(obstacle_clearance)),
        debug=bool(
            getattr(args, "debug_paths", False)
            or getattr(args, "debug_steering", False)
        ),
        show_roadmap=bool(getattr(args, "debug_paths", False)),
    )


def generate_specialk_trajectories(
    start,
    end,
    args,
    *,
    static_polygons=None,
    display_offset=None,
    display_diff=None,
    vehicle_length=0.7,
    vehicle_width=0.7,
    vehicle_scale=1.0,
    max_steer=np.deg2rad(30.0),
    resolution=None,
):
    params = build_params(
        args,
        display_offset,
        display_diff,
        vehicle_length,
        vehicle_width,
        vehicle_scale,
        max_steer,
        resolution,
    )
    start_state = np.asarray(start, dtype=float)[:4]
    goal_xy = np.asarray(end, dtype=float)[:2]
    obstacle_union = buffered_obstacle_union(
        static_polygons,
        clearance=params.obstacle_clearance,
    )

    started = perf_counter()
    routes, roadmap_debug = generate_route_skeletons(
        start_state,
        goal_xy,
        static_polygons,
        obstacle_union,
        params,
    )
    if params.show_roadmap:
        args._last_specialk_debug = roadmap_debug
    else:
        args._last_specialk_debug = None

    accepted = []
    used_cells = {}
    reject_counts = {"lattice": 0, "kinematics": 0, "loop": 0, "duplicate": 0}
    seen_route_cells = []
    centroids = obstacle_centroids(static_polygons)

    for route in routes:
        route_cells = route_spatial_cells(route, params)
        states = track_route_with_ackermann(start_state, route, obstacle_union, params)
        source = "tracker"
        if states is None:
            states = lattice_plan_for_route(
                start_state,
                goal_xy,
                route,
                obstacle_union,
                params,
                used_cells=used_cells,
            )
            source = "lattice"
        if states is None or len(states) < 2:
            reject_counts["lattice"] += 1
            continue
        loop_free, loop_reason = states_are_loop_free(states, route, params)
        if not loop_free:
            reject_counts["loop"] += 1
            if params.debug:
                print(
                    "[specialk] rejected phase-two trajectory "
                    f"reason={loop_reason} source={source}"
                )
            continue
        path = states_to_frenet_path(states, params)
        if not trajectory_respects_kinematics(path, params):
            reject_counts["kinematics"] += 1
            continue
        cells = [state_cell(state, params) for state in states]
        if any(
            edge_overlap(route_cells, [existing]) > 0.98
            for existing in seen_route_cells
        ):
            reject_counts["duplicate"] += 1
            continue
        tracked_route = states_to_route(
            states, min_spacing=max(params.grid_resolution, 0.1)
        )
        accepted.append(
            {
                "path": path,
                "route": dedupe_points(route),
                "tracked_route": tracked_route,
                "roadmap_route": route,
                "cells": cells,
                "length": polyline_length(route),
                "signature": route_signature(route, centroids),
                "generator": "specialk",
                "source": source,
            }
        )
        seen_route_cells.append(set(route_cells))
        for cell in set(cells):
            used_cells[cell] = used_cells.get(cell, 0) + 1
        if len(accepted) >= params.count:
            break

    if not accepted:
        accepted.append(fallback_forward_candidate(start_state, obstacle_union, params))

    if params.debug:
        elapsed_ms = (perf_counter() - started) * 1000.0
        lengths = [round(item["length"], 3) for item in accepted]
        print(
            "[specialk] "
            f"accepted={len(accepted)}/{params.count} "
            f"routes={len(routes)} lengths={lengths} "
            f"roadmap_density={params.roadmap_samples_density:.3f} "
            f"roadmap_samples={params.roadmap_samples} "
            f"lattice_resolution={params.lattice_resolution:.3f} "
            f"motion_step={params.motion_step:.3f} "
            f"obstacle_clearance={params.obstacle_clearance:.3f} "
            f"elapsed_ms={elapsed_ms:.1f} "
            f"rejects={reject_counts}"
        )

    return accepted[: params.count]
