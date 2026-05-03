# Homotopy-Aware K-Shortest Route Generation with State-Lattice Ackermann Feasibility

## Overview

This document describes a detailed engineering plan for:

> **Homotopy-aware k-shortest route generation on a geometric roadmap, followed by state-lattice planning with Dubins motion primitives to enforce Ackermann feasibility, optimized for frequent replanning.**

The core idea is to split planning into two stages:

```text
Stage 1: Fast geometric/topological route generation
         → generate k diverse route skeletons/corridors

Stage 2: Ackermann-feasible trajectory generation
         → refine each route using a state lattice with Dubins primitives
```

The geometric roadmap gives route diversity and speed. The state lattice gives vehicle feasibility.

---

## 1. Problem Setup

Assume the environment is:

```text
Workspace:      R²
Obstacles:      inflated polygons -- inflate if necessary by robot dimensions
Robot model:    bicycle / Ackermann
Start:          q_start = (x, y, theta)
Goal:           goal pose or goal region
Task:           generate k diverse feasible trajectories to goal
```

Because obstacles are already inflated, route generation can treat the robot as a point in 2D free space. Ackermann feasibility is enforced later.

The desired output is:

```text
Trajectory set:
    τ₁, τ₂, ..., τₖ

where:
    τ₁ is the shortest or near-shortest feasible trajectory
    τ₂...τₖ are diverse alternatives
    each τᵢ satisfies collision, curvature, and vehicle motion constraints
```

---

## 2. High-Level Architecture

```text
Preprocessing:
    1. Store inflated polygonal map.
    2. Build a reusable geometric roadmap.
    3. Assign topological signatures to roadmap edges.
    4. Precompute nearest-neighbor and visibility structures.
    5. Precompute Dubins/state-lattice motion primitives.

Online replanning:
    1. Insert current robot state into roadmap.
    2. Insert goal into roadmap.
    3. Run Yen’s k-shortest path search on the roadmap.
    4. Filter or cluster paths by homotopy signature.
    5. Convert each geometric route into a corridor.
    6. Run bounded state-lattice planning inside each corridor.
    7. Smooth/refine feasible candidates.
    8. Evaluate trajectories using the exploration metric.
    9. Return ranked trajectory set.
```

For speed, the static environment work should be done offline or incrementally. The online loop should avoid rebuilding the roadmap.

---

## 3. Geometric Roadmap Design

The geometric roadmap is a graph:

```text
G = (V, E)
```

where each vertex is a point in free space and each edge is a collision-free geometric connection between two vertices.

Each edge stores:

```text
edge = {
    u, v,
    length,
    clearance,
    polyline or segment geometry,
    homotopy contribution,
    optional Dubins-feasibility hint,
    nearby obstacles,
    cached collision status
}
```

The roadmap should support:

1. **Shortest route generation**
2. **Topologically diverse route generation**

It does not need to fully encode heading or curvature yet, but it should avoid producing pathological zig-zag routes that are impossible for the vehicle to follow.

---

## 3.1 Recommended Roadmap Type

For polygonal inflated obstacles, use a hybrid roadmap:

```text
Roadmap = visibility graph backbone
        + generalized Voronoi / medial-axis samples
        + sparse random / Halton samples
        + start/goal connection nodes
```

This is more robust than using only one roadmap family.

### Visibility Graph Component

A visibility graph connects mutually visible obstacle vertices and start/goal points. Since obstacles are inflated, this gives efficient near-shortest geometric paths.

Useful nodes:

```text
- obstacle polygon vertices
- offset or buffered obstacle vertices
- reflex/corner points
- start and goal attachment points
```

Edges are straight line segments that do not intersect inflated obstacles.

Advantages:

```text
- shortest geometric routes tend to appear here
- good at finding paths around polygon corners
- compact for polygonal maps
```

Disadvantages:

```text
- paths often graze obstacle corners
- low clearance
- can produce sharp turns
```

To reduce corner-grazing, use **clearance-aware vertex offsets** rather than raw polygon vertices.

For each obstacle vertex, create one or more nearby free-space samples slightly outside the obstacle corner along the angular bisector of the free region. Because the polygons are already inflated, this does not need to be large.

### Voronoi / Medial-Axis Component

A generalized Voronoi diagram provides high-clearance routes through free space.

Advantages:

```text
- produces safe, central corridors
- good for exploration
- gives qualitatively different routes than visibility graph paths
```

Disadvantages:

```text
- paths may be longer than necessary
- medial-axis construction can be expensive
- route quality depends on discretization if approximated
```

For speed, you do not need an exact continuous Voronoi diagram. A grid-based distance transform with skeletonization is often sufficient if your environment is large or updated frequently.

### Sparse Sampling Component

Add a sparse PRM-like set of samples in free space.

Recommended sampling methods:

```text
- Halton sequence
- Poisson-disk sampling
- obstacle-biased samples
- narrow-passage samples
```

Use a deterministic low-discrepancy sequence, such as Halton, rather than purely random samples. This makes replanning reproducible and allows incremental roadmap refinement.

Connect samples to nearby neighbors using visibility checks.

---

## 3.2 Roadmap Node Categories

Use typed nodes:

```text
NodeType:
    OBSTACLE_CORNER
    CORNER_OFFSET
    VORONOI
    FREE_SPACE_SAMPLE
    NARROW_PASSAGE
    START
    GOAL
```

This helps tune edge costs. For example, you may penalize raw corner-grazing edges but favor high-clearance Voronoi edges when generating exploration alternatives.

---

## 3.3 Roadmap Edge Generation

For each node, connect to a limited number of neighbors.

Possible rules:

```text
Connect if:
    distance(u, v) <= r_connect
    segment(u, v) is collision-free
    edge clearance >= clearance_min
```

Also add:

```text
- k nearest visible neighbors
- same-cell or adjacent-cell neighbors
- Voronoi graph adjacency edges
- visibility graph edges near obstacle corners
```

A practical connection strategy:

```text
For each node:
    1. Query k nearest neighbors using KD-tree.
    2. Keep candidates within radius r_connect.
    3. Run segment-polygon collision check.
    4. Store valid edges.
```

For speed, keep `k` modest:

```text
k_nearest = 8 to 20
```

If the roadmap is too sparse, increase samples, not necessarily connectivity. Dense connectivity makes Yen’s algorithm expensive.

---

## 3.4 Edge Cost

The basic geometric edge cost is length:

```text
c(e) = ||v - u||
```

For useful route corridors, use a weighted cost:

```text
c(e) =
    w_len       * length(e)
  + w_clear     * clearance_penalty(e)
  + w_turn      * heading_change_proxy(e)
  + w_type      * edge_type_penalty(e)
```

Example clearance penalty:

```text
clearance_penalty(e) = 1 / (epsilon + clearance(e))
```

Example heading-change proxy:

```text
heading_change_proxy at node v =
    angle between previous edge and next edge
```

Since the turn cost depends on the previous edge, you can either:

1. Ignore it in the geometric roadmap and handle it in the lattice stage, or
2. Use an expanded graph state `(previous_node, current_node)`.

For speed, start with edge length plus clearance penalty, then let the lattice planner handle turn feasibility.

Recommended initial cost:

```text
c(e) = length(e) * (1 + α / (ε + clearance(e)))
```

Use a small clearance penalty so the shortest path is not distorted too much:

```text
α = 0.1 to 1.0
```

---

## 4. Homotopy-Aware Path Representation

You need a way to decide whether two routes are meaningfully different.

In a planar polygonal environment, routes are homotopically distinct if they wind around obstacles differently.

Instead of expensive exact homotopy reasoning, use a practical signature.

---

## 4.1 Obstacle Reference Points

Assign each obstacle a representative point:

```text
z_j = obstacle centroid or interior reference point
```

For every path, compute how the path winds around each obstacle.

For polygonal obstacles:

```text
signature(path) = [h₁, h₂, ..., h_m]
```

where `h_j` describes the route’s relationship to obstacle `j`.

Options:

```text
1. Winding number
2. H-signature using complex coordinates
3. Left/right crossing signature against reference rays
4. Obstacle-side sequence
```

For implementation speed, use a **crossing-based H-signature approximation**.

---

## 4.2 Crossing-Based Signature

For each obstacle, define a ray from its centroid to infinity, for example in the positive x direction.

For each directed edge in the path, test whether it crosses that ray.

If it crosses, increment or decrement the obstacle’s signature component based on crossing direction.

```text
h_j += +1 if edge crosses obstacle j's reference ray upward
h_j += -1 if edge crosses obstacle j's reference ray downward
```

The full path signature is:

```text
H(path) = tuple(h_1, ..., h_m)
```

Two paths with different `H(path)` values are considered topologically different.

This is fast and easy to cache because each roadmap edge can store its own homotopy contribution:

```text
H(path) = sum over edges e in path of H(e)
```

During path extraction:

```text
path_signature = edge_signature[e₁]
               + edge_signature[e₂]
               + ...
               + edge_signature[e_n]
```

---

## 4.3 Local Obstacle Filtering

If there are many obstacles, storing a full vector over all obstacles may be too expensive.

Use only relevant obstacles:

```text
Relevant obstacles =
    obstacles near the route corridor
    obstacles within bounding box(start, goal) expanded by margin
    obstacles whose reference rays intersect roadmap edges near candidate paths
```

For speed, store sparse signatures:

```text
H(path) = {
    obstacle_id_7: +1,
    obstacle_id_12: -1,
    obstacle_id_19: +1
}
```

Normalize by removing zero entries.

---

## 4.4 Diversity Criteria

Use both topological and geometric diversity.

Two paths are considered redundant if:

```text
same homotopy signature
and high edge overlap
and small average spatial separation
```

Useful metrics:

```text
edge_overlap(P, Q) =
    shared_edge_length / min(length(P), length(Q))

mean_separation(P, Q) =
    average distance from samples on P to nearest point on Q

signature_distance(P, Q) =
    ||H(P) - H(Q)||₁
```

Accept a new path if:

```text
signature is new
or edge_overlap < overlap_threshold
or mean_separation > separation_threshold
```

Typical thresholds:

```text
overlap_threshold     = 0.5 to 0.8
separation_threshold  = 1 to 3 robot widths
```

---

## 5. Yen’s K-Shortest Route Generation

Yen’s algorithm finds the k shortest loopless paths in a graph.

Use it on the geometric roadmap.

Input:

```text
G = geometric roadmap
s = start attachment node
g = goal attachment node
K_raw = number of raw paths to search
```

Output:

```text
candidate route skeletons P₁, P₂, ..., P_K
```

Because some paths will be topologically redundant or later dynamically infeasible, request more raw paths than you need.

```text
K_raw = 3K to 10K
```

If you ultimately want 5 diverse trajectories, generate perhaps 30 to 50 raw roadmap paths.

---

## 5.1 Basic Yen’s Algorithm Loop

```text
A = []  # accepted shortest paths
B = priority queue of deviations

P0 = shortest_path(G, s, g)
A.append(P0)

for k in 1...K_raw-1:
    for i in 0...len(A[k-1])-2:
        spur_node = A[k-1][i]
        root_path = A[k-1][0:i]

        Temporarily remove edges/nodes that would duplicate previous roots.
        spur_path = shortest_path(G_modified, spur_node, g)

        if spur_path exists:
            total_path = root_path + spur_path
            B.push(total_path)

    A.append(B.pop_min())
```

Use A* instead of Dijkstra for the shortest-path subroutine.

The heuristic is Euclidean distance to goal:

```text
h(v) = ||v - goal||
```

Since edge costs are at least Euclidean length, this remains admissible if the cost is pure length. If the cost includes penalties, use Euclidean distance as a lower bound.

---

## 5.2 Homotopy-Aware Modification

Plain Yen’s algorithm produces the k shortest graph paths, many of which may be minor variants.

Add a filter after each candidate path is generated:

```text
if is_diverse(candidate, accepted_diverse_paths):
    accepted_diverse_paths.append(candidate)
```

Do not stop Yen’s algorithm when you have one path per homotopy class unless you already have enough dynamically feasible candidates. Some homotopy classes may fail in the lattice stage.

Recommended structure:

```text
raw_paths = Yen(G, s, g, K_raw)

diverse_routes = []

for P in raw_paths:
    H = homotopy_signature(P)

    if passes_diversity_filter(P, H, diverse_routes):
        diverse_routes.append(P)

    if len(diverse_routes) >= K_route:
        break
```

Use:

```text
K_route = 2K to 4K
```

because the state-lattice stage may reject some.

---

## 5.3 Cost Perturbation for Diversity

In addition to filtering, you can encourage Yen’s algorithm to produce route alternatives by adding edge penalties for already-used edges.

After accepting a diverse path, increase the cost of its edges:

```text
c'(e) = c(e) + λ_used * used_count(e)
```

or:

```text
c'(e) = c(e) * (1 + λ_overlap * used_count(e))
```

This turns the search into a soft-diversity generator.

Use this carefully. If you need the first path to be the shortest, run ordinary shortest path first before applying penalties.

Recommended:

```text
1. Compute shortest path with original costs.
2. Store it as required candidate.
3. For alternatives, use Yen’s algorithm with diversity filtering and optional overlap penalties.
```

---

## 6. Start and Goal Insertion

Because replanning occurs frequently, the roadmap should be static except for start and goal.

At each replanning step:

```text
1. Remove previous START and GOAL temporary nodes.
2. Insert current robot position as START.
3. Insert goal position or goal boundary samples.
4. Connect START and GOAL to nearby visible roadmap nodes.
```

For the start, the robot has heading. The geometric roadmap ignores heading, but the state lattice will enforce it. Still, do not connect the start to nodes that require immediate impossible motion.

Start connection rule:

```text
Candidate node v is allowed if:
    visible(start_xy, v)
    distance <= r_start_connect
    heading alignment is plausible
```

Heading alignment test:

```text
angle_diff(robot_theta, atan2(v_y - y, v_x - x)) <= theta_connect_max
```

If reverse motion is allowed, this can be relaxed.

For goal insertion:

```text
If goal is a region:
    sample several goal boundary/interior points
If goal is a pose:
    insert one goal node plus nearby approach nodes
```

For an Ackermann robot, it is better to represent the goal as a small set of allowed terminal poses:

```text
goal_states = {
    (x_g, y_g, theta_g),
    nearby approach poses
}
```

---

## 7. Route Corridor Construction

Each geometric route from Yen’s algorithm is a polyline:

```text
P = [p₀, p₁, ..., p_n]
```

Convert it into a corridor for the state-lattice planner.

The corridor is a tube around the route:

```text
Corridor(P, r) = union of disks or capsules around route segments
```

Use variable corridor width:

```text
r_corridor(s) = min(
    r_max,
    β * local_clearance(s)
)
```

Recommended:

```text
r_min = 2 to 3 robot widths
r_max = environment-dependent
β     = 0.5 to 0.8
```

The corridor should be wide enough for the state lattice to make turns. For Ackermann vehicles, ensure:

```text
corridor_width >= 2 * minimum_turning_radius
```

near sharp route bends where possible.

If a route corridor is too narrow for the vehicle’s turning radius, reject early or widen the corridor locally.

---

## 8. State-Lattice Planning with Dubins Primitives

The second stage takes a route corridor and computes a feasible trajectory.

State:

```text
q = (x, y, theta)
```

Controls for bicycle/Ackermann:

```text
v
steering angle δ
```

Kinematic bicycle model:

```text
x_dot     = v cos(theta)
y_dot     = v sin(theta)
theta_dot = v / L * tan(delta)
```

Minimum turning radius:

```text
R_min = L / tan(delta_max)
```

Dubins primitives are appropriate if the vehicle only moves forward.

If reverse is allowed, use Reeds-Shepp primitives instead.

---

## 8.1 Lattice Discretization

Discretize:

```text
x, y      on grid
theta     into N_theta bins
```

Typical values:

```text
grid_resolution = 0.25 to 1.0 m
N_theta         = 16, 32, or 48
```

For frequent replanning, use the coarsest resolution that still gives executable paths.

A good starting point:

```text
grid_resolution = R_min / 4 to R_min / 8
N_theta = 16 or 32
```

---

## 8.2 Motion Primitives

Use precomputed Dubins-like primitives.

Each primitive starts at:

```text
(0, 0, theta_i)
```

and ends at one of several neighboring lattice states.

Primitive types:

```text
S      straight
L      constant-left-turn arc
R      constant-right-turn arc
LS     left arc + straight
RS     right arc + straight
SL     straight + left arc
SR     straight + right arc
LR     compound turns if useful
RL
```

Classical Dubins shortest paths use:

```text
LSL, RSR, LSR, RSL, RLR, LRL
```

For a state lattice, you usually precompute short executable primitives rather than solving a global Dubins path for every edge.

Each primitive stores:

```text
primitive = {
    start_heading_bin,
    delta_x,
    delta_y,
    delta_theta_bin,
    sampled_points,
    length,
    curvature,
    control_sequence,
    swept_volume,
    cost
}
```

Generate primitives by integrating the bicycle model using a small set of steering commands:

```text
δ ∈ {-δ_max, -δ_mid, 0, δ_mid, δ_max}
```

and fixed arc lengths:

```text
s ∈ {s_short, s_medium, s_long}
```

Then snap endpoints to lattice cells.

---

## 8.3 Lattice Graph Search

For each route corridor, run A* or weighted A* in the lattice state space.

Search state:

```text
(x_idx, y_idx, theta_idx)
```

Successors:

```text
apply each valid primitive from current theta_idx
```

Validity checks:

```text
1. primitive endpoint inside map
2. primitive swept path collision-free
3. primitive samples stay inside route corridor
4. primitive respects curvature and steering constraints
```

Cost:

```text
g +=
    w_len       * primitive_length
  + w_turn      * steering_cost
  + w_reverse   * reverse_penalty, if allowed
  + w_clear     * obstacle_clearance_penalty
  + w_corridor  * distance_from_route_centerline
  + w_smooth    * steering_change_penalty
```

Because you are using Dubins primitives, curvature feasibility is already encoded.

---

## 8.4 Heuristic

Use a strong heuristic for speed.

For forward-only Ackermann motion:

```text
h(q) = DubinsDistance(q, goal_pose)
```

You can combine this with a 2D obstacle-aware cost-to-go precomputed from the goal:

```text
h(q) = max(
    Dubins_lower_bound(q, goal),
    grid_2d_distance_to_goal(x, y)
)
```

The 2D grid distance can be precomputed once per goal using Dijkstra/Fast Marching on an occupancy grid.

For speed, this is valuable.

---

## 8.5 Corridor-Biased Search

The lattice planner should not search the whole map for every candidate. Restrict it to the route corridor.

For each lattice cell, precompute or compute on demand:

```text
distance_to_route_centerline
inside_corridor
local_corridor_progress
```

Reject states outside the corridor:

```text
if distance_to_route > r_corridor:
    invalid
```

Cost states farther from the route centerline:

```text
corridor_cost = w_corridor * distance_to_route_centerline²
```

Also encourage progress along the route:

```text
progress_index = projection of state onto route polyline
```

Prevent excessive backtracking with:

```text
if progress decreases too much:
    add penalty or reject
```

Do not make this too strict, because Ackermann vehicles may need to swing wide near corners.

---

## 9. Ensuring the Shortest Path Is Included

There are two notions of shortest path:

```text
1. shortest geometric roadmap path
2. shortest Ackermann-feasible trajectory
```

These are not necessarily the same.

A roadmap shortest path may be too sharp for the vehicle, so the first lattice-refined candidate might not be the true shortest feasible trajectory.

To handle this:

```text
1. Always include the shortest geometric route from Yen’s algorithm.
2. Run lattice planning on it first.
3. Also run a global state-lattice planner without a tight corridor, or with a very wide corridor around the shortest route.
4. Treat this result as the shortest feasible baseline under your lattice discretization.
```

Recommended:

```text
Candidate 0:
    full-map or wide-corridor weighted A* lattice plan

Candidates 1...K:
    homotopy-diverse corridor-constrained lattice plans
```

This gives a defensible “shortest feasible” candidate.

If speed is critical, only run full-map lattice planning when:

```text
- the previous shortest path is invalid
- the robot has moved far from the previous plan
- the environment or goal changed
- the route set changed substantially
```

Otherwise, reuse and repair the previous shortest trajectory.

---

## 10. Frequent Replanning Strategy

Frequent replanning changes the design priorities. You want to avoid doing expensive global work every cycle.

Use this division:

```text
Offline / slow update:
    build roadmap
    build spatial index
    precompute edge signatures
    precompute primitive library
    precompute obstacle distance field

Online / every replan:
    connect start/goal
    run bounded k-shortest search
    run bounded lattice refinement
    score candidates
```

---

## 10.1 Cache Everything Static

Precompute and cache:

```text
- roadmap nodes
- roadmap edges
- edge lengths
- edge clearances
- edge homotopy signatures
- edge collision status
- KD-tree over roadmap nodes
- obstacle spatial index
- distance transform / clearance field
- lattice primitive library
- primitive swept-volume masks
```

For static inflated polygons, edge collision checks should almost never be repeated.

---

## 10.2 Dynamic Start/Goal Connection Cache

At every cycle, the start changes slightly. Instead of rebuilding connections from scratch:

```text
1. Query nearest roadmap nodes from KD-tree.
2. Collision-check only the small set of candidate edges.
3. Cache successful local connections for nearby start cells.
```

You can quantize start positions:

```text
start_cache_key = (
    floor(x / cache_res),
    floor(y / cache_res),
    floor(theta / theta_cache_res)
)
```

Then reuse local connections when the robot remains in the same cache cell.

---

## 10.3 Use Previous Solution as a Seed

At replan step `t`, you have previous candidate trajectories.

For each previous trajectory:

```text
1. Trim the already-executed prefix.
2. Check whether the suffix is still valid.
3. Use it as candidate without replanning if valid.
4. Use it as a warm-start corridor for the lattice planner.
```

The previous best path should often remain valid for several cycles.

This allows the planner to spend most computation on alternatives and exploration value rather than recomputing the obvious route.

---

## 10.4 Limit Yen’s Search

Yen’s algorithm can become expensive if the graph is large and `K_raw` is high.

Use bounds:

```text
max_raw_paths
max_spur_nodes_per_path
max_shortest_path_expansions
max_cost_ratio
max_runtime_ms
```

Example:

```text
K_desired = 5
K_route   = 12
K_raw     = 40

max_cost_ratio = 2.0
```

Reject geometric paths whose cost exceeds:

```text
cost(P) > max_cost_ratio * cost(shortest_path)
```

This prevents wasting time on extreme detours.

---

## 10.5 Use Lazy Evaluation

Do not lattice-plan every candidate route immediately.

Instead:

```text
1. Generate route candidates.
2. Score cheap geometric proxies.
3. Select the most promising M routes.
4. Run lattice planning only for those.
```

Cheap proxy score:

```text
proxy_score(P) =
    a * geometric_length
  - b * expected_information_gain
  + c * low_clearance_penalty
  + d * overlap_with_existing_paths
  + e * estimated_turning_difficulty
```

Then choose:

```text
M = 2K to 3K
```

for lattice refinement.

---

## 10.6 Anytime Behavior

Use an anytime structure:

```text
1. Return previous valid plan immediately.
2. Compute shortest updated route.
3. Refine one candidate at a time.
4. Keep best feasible set found so far.
```

Even within one planning call, order candidates by priority:

```text
1. previous best trajectory repair
2. shortest route
3. topologically distinct high-value alternatives
4. longer exploratory alternatives
```

This is important if your planner has a strict runtime budget.

---

## 11. Speed-Oriented Planner Design

A practical online planner might look like this:

```text
function replan(start_pose, goal, map, previous_solutions):

    valid_old = validate_and_trim(previous_solutions)

    start_node = connect_start_to_roadmap(start_pose)
    goal_nodes = connect_goal_to_roadmap(goal)

    shortest_route = astar_roadmap(start_node, goal_nodes)

    raw_routes = yen_k_shortest(
        graph,
        start_node,
        goal_nodes,
        K_raw,
        cost_bound = rho * cost(shortest_route),
        time_budget = T_yen
    )

    diverse_routes = homotopy_filter(raw_routes, K_route)

    route_queue = rank_by_proxy_metric(diverse_routes)

    feasible_trajectories = valid_old

    for route in route_queue:
        if time_remaining() < T_min:
            break

        corridor = build_corridor(route)

        trajectory = lattice_plan(
            start_pose,
            goal,
            corridor,
            time_budget = T_lattice_per_route
        )

        if trajectory is feasible:
            feasible_trajectories.append(trajectory)

        if len(feasible_trajectories) >= K_final:
            break

    scored = evaluate_task_metric(feasible_trajectories)

    return top_k(scored, K_final)
```

---

## 12. Dubins Primitives in the Roadmap Versus Lattice

Roadmap state:

```text
(x, y)
```

Yen’s paths are geometric.

Dubins primitives are used in the lattice stage.

Pros:

```text
- faster graph search
- simpler homotopy signatures
- easier to generate diverse routes
- smaller graph
```

Cons:

```text
- some geometric paths may fail later
```

---

## 13. Curvature Feasibility Prefilter

Before running expensive lattice planning, reject geometric routes that are obviously impossible or difficult.

For each internal waypoint `p_i`, compute the turn angle:

```text
φ_i = angle between p_i - p_{i-1} and p_{i+1} - p_i
```

A sharp turn requires local space. Estimate whether it is feasible using minimum turning radius.

A simple proxy:

```text
required_turning_space_i ≈ R_min * tan(|φ_i| / 2)
```

If nearby clearance is too small, mark the route as difficult or infeasible.

Proxy penalty:

```text
turning_difficulty(P) =
    sum_i max(0, required_turning_space_i - local_clearance_i)
```

Use this in route ranking:

```text
proxy_score(P) += w_turn_feasibility * turning_difficulty(P)
```

Do not reject too aggressively; the lattice planner may find a feasible swing-around maneuver inside the corridor.

---

## 14. Collision Checking Acceleration

Collision checking dominates runtime if not handled carefully.

Use multiple layers.

### 14.1 Static Edge Collision Cache

For roadmap edges:

```text
edge_collision_free[e] = true/false
```

computed once.

### 14.2 Spatial Index Over Polygons

Use an R-tree or bounding volume hierarchy.

For a segment or primitive:

```text
1. Query candidate polygons by bounding box.
2. Check exact intersection only against those polygons.
```

### 14.3 Occupancy Grid for Lattice Validation

For the state lattice, continuous polygon collision is expensive. Use a high-resolution occupancy/distance grid derived from inflated polygons.

Then primitive validation becomes:

```text
for sampled point in primitive:
    if occupancy_grid[point] occupied:
        invalid
```

Because obstacles are already inflated, point checking is acceptable. If you want extra conservatism, also require:

```text
distance_field[point] >= safety_margin
```

### 14.4 Primitive Swept Masks

For each primitive and heading bin, precompute the relative cells it touches.

Then validation is:

```text
for cell_offset in primitive_mask:
    if occupied[current_cell + rotated_offset]:
        invalid
```

This is much faster than sampling continuous geometry every time.

---

## 15. Homotopy Filtering in Practice

A practical route-selection loop:

```text
diverse = []

for P in raw_paths:

    H = compute_signature(P)
    overlap_ok = true
    separation_ok = true

    for Q in diverse:
        if H == H(Q):
            if edge_overlap(P, Q) > overlap_threshold:
                overlap_ok = false
                break

            if mean_separation(P, Q) < separation_threshold:
                separation_ok = false
                break

    if H is new:
        accept P
    else if overlap_ok and separation_ok:
        accept P
```

This allows multiple routes in the same homotopy class if they are geometrically different enough, which can matter in large open spaces.

---

## 16. Ranking Geometric Route Candidates

Before lattice planning, compute a cheap score.

For exploration, you likely care about information gain or visibility. You can approximate this on the route skeleton first.

Example:

```text
route_proxy_score(P) =
    w_len       * normalized_length(P)
  - w_info      * expected_visibility_gain(P)
  + w_clear     * low_clearance_penalty(P)
  + w_turn      * turning_difficulty(P)
  + w_overlap   * overlap_with_selected_routes(P)
```

Then lattice-plan the best few.

This is important because state-lattice planning is the expensive stage.

---

## 17. Final Trajectory Scoring

After feasible trajectories are generated, use your actual metric.

Example final score:

```text
J(τ) =
    α * travel_time(τ)
  + β * path_length(τ)
  + γ * risk(τ)
  - η * information_gain(τ)
  + λ * control_effort(τ)
  + μ * negative_clearance_penalty(τ)
```

You may also want to preserve diversity in the final selected set.

Do not simply take the top `K` by individual score, because they may all be similar. Use a set objective:

```text
Selected = argmax over trajectory sets:
    sum individual utility
  + diversity_bonus
```

Greedy selection is usually sufficient:

```text
S = []
while len(S) < K:
    choose τ maximizing:
        utility(τ) + λ_div * min_distance_to_set(τ, S)
```

---

## 18. Suggested Implementation Phases

### Phase 1: Basic Geometric Roadmap

Implement:

```text
- inflated polygon input
- roadmap nodes from obstacle corners and free-space samples
- KD-tree connection
- segment collision checking
- A* shortest path
```

Goal:

```text
* produce one shortest geometric route
* produce a debug representation showing the completed roadmap
```



### Phase 2: Yen’s K-Shortest Paths

Implement:

```text
- Yen’s algorithm
- path deduplication
- cost bound
- max candidate limit
```

Goal:

```text
produce 20 to 50 route skeletons quickly
```

### Phase 3: Homotopy Signatures

Implement:

```text
- obstacle reference rays
- edge signature cache
- path signature accumulation
- route filtering by signature and overlap
```

Goal:

```text
produce 5 to 15 genuinely distinct route skeletons
```

### Phase 4: Corridor Generation

Implement:

```text
- route polyline tube/capsule corridor
- distance-to-route lookup
- local route progress
- corridor visualization
```

Goal:

```text
produce bounded planning regions for each candidate route
```

### Phase 5: State Lattice with Dubins Primitives

Implement:

```text
- heading discretization
- primitive library
- primitive collision checking
- A* in (x, y, theta)
- Dubins-distance heuristic
- goal pose/region handling
```

Goal:

```text
produce feasible Ackermann trajectories inside corridors
```

### Phase 6: Speed Optimization

Add:

```text
- static roadmap cache
- edge collision cache
- edge signature cache
- KD-tree
- distance field
- primitive masks
- previous-plan repair
- lazy candidate refinement
- anytime replanning
```

Goal:

```text
fit within online replanning budget
```

---

## 19. Recommended Parameter Defaults

Initial values to try:

```text
K_final                 = 5
K_route                 = 12 to 20
K_raw_yen               = 40 to 100

roadmap_samples_density = 0.5 to 3 samples / m^2 depending on map complexity
k_nearest               = 8 to 20
r_connect               = 5 to 20 m, environment-dependent

homotopy_overlap_thresh = 0.6
separation_thresh       = 2 robot widths
max_cost_ratio          = 1.5 to 2.5

lattice_resolution      = R_min / 4 to R_min / 8
N_theta                 = 16 or 32
primitive_lengths       = [0.5R_min, 1.0R_min, 2.0R_min]
corridor_min_width      = 2 to 3 robot widths
corridor_max_width      = environment-dependent

Yen time budget         = 5% to 20% of planning cycle
lattice time budget     = remaining budget, split by candidate priority
```

---

## 20. Important Design Choices

### Use 2D Yen, Not Full SE(2) Yen, for Speed

For frequent replanning, keep Yen’s algorithm on a compact 2D roadmap.

Use the lattice planner to enforce Ackermann feasibility.

### Always Over-Generate Geometric Routes

Some candidates will be redundant or infeasible.

If you need 5 final trajectories, generate many more geometric candidates:

```text
raw Yen paths:        40 to 100
diverse route paths:  10 to 20
lattice attempts:     5 to 15
final trajectories:   3 to 5
```

### Treat the Previous Solution as Candidate Zero

In frequent replanning, the fastest plan is often the one you already have.

Always validate and reuse previous trajectories before planning from scratch.

### Use Homotopy Diversity Before Metric Evaluation

Your exploration metric may prefer one region repeatedly. Homotopy filtering ensures the candidate set contains structurally different choices before scoring.

---

## 21. Final Recommended Pipeline

The final system should look like this:

```text
Offline:
    Build hybrid geometric roadmap:
        visibility/corner nodes
        Voronoi/clearance nodes
        sparse PRM samples

    Precompute:
        edge collision
        edge length
        edge clearance
        edge homotopy signature
        KD-tree
        distance field
        Dubins/state-lattice primitives

Online replan:
    1. Validate and trim previous trajectories.
    2. Insert start and goal into roadmap.
    3. Compute shortest geometric path.
    4. Run bounded Yen’s k-shortest path search.
    5. Homotopy-filter and overlap-filter route skeletons.
    6. Rank route skeletons using cheap proxy score.
    7. For selected routes:
           build corridor
           run bounded A* state-lattice planner using Dubins primitives
           validate trajectory
    8. Score feasible trajectories using final exploration metric.
    9. Return shortest feasible candidate plus diverse high-value alternatives.
```

The key engineering principle is:

> **Use the roadmap for topological diversity, and use the state lattice only where necessary.**

That is what keeps the method fast enough for frequent replanning while still producing Ackermann-feasible alternatives.
