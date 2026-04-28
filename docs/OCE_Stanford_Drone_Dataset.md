# Specification: OCE Evaluation on the Stanford Drone Dataset

## 1. Objective

The objective is to evaluate **Occlusion-Conditioned Entropy (OCE)** as an information-aware trajectory-selection criterion for a mobile observing robot operating in real-world pedestrian scenes.

The core research question is:

> Can a mobile robot improve prediction of a pedestrian’s future route by planning informative viewpoints under occlusion, rather than simply maximizing visibility or following a nominal navigation policy?

The task is **active pedestrian route disambiguation under occlusion**. The robot is not solving general crowd navigation. Instead, it moves through a real-world overhead scene while maintaining a belief over a selected target pedestrian’s future location and route.

The contribution is:

> OCE provides an information-aware objective for receding-horizon planning, allowing a robot to select trajectories that reduce future uncertainty about a target under occlusion.

---

## 2. Datasets

### 2.1 Primary Dataset: Constrained / Annotated Stanford Drone Dataset

Use the constrained / annotated Stanford Drone Dataset as the primary experimental dataset.  All tracks appear to be pedestrian, restricted to walkable surfaces.  If necessary, tracks can be flagged or filtered by speed in normalized scene units.

The required artifacts are:

- Pedestrian tracks
- Scene reference images
- Building polygons
- Obstacle polygons
- Offroad or non-walkable polygons
- Walkable regions, if explicitly provided
- Coordinate-frame and display-scaling metadata
- Scene identifiers
- Agent identifiers
- Frame indices

The constrained / annotated dataset should be sufficient for the main OCE experiments because it provides the three key inputs required by the planner:

1. **Target motion data**
   Pedestrian tracks provide real target trajectories for learning Markov transition models and for held-out test rollouts.

2. **Occlusion geometry**
   Buildings and obstacles provide static occluders for computing robot-to-target line-of-sight visibility.

3. **Traversability / scene topology**
   Walkable regions can either be used directly, if provided, or derived from the complement of buildings, obstacles, and offroad regions.

The raw video files are not required for algorithm development or quantitative evaluation.

### 2.2 Optional Dataset: Original Stanford Drone Video

The original Stanford Drone videos are optional.

They may be useful later for:

- Presentation figures
- Qualitative overlays
- Demonstration videos
- Sanity-checking unusual trajectories
- Showing representative success and failure cases

However, the videos are not part of the minimum viable experiment.

Required only if needed for presentation:

- A small representative subset of raw scene videos
- Corresponding scene reference images
- Frame-to-trajectory alignment metadata

### 2.3 Derived Scene Representation

For each selected scene, derive:

\[
\mathcal{W} = \text{walkable region}
\]

and

\[
\mathcal{O} = \text{static occluder set}
\]

where:

\[
\mathcal{O} = \text{buildings} \cup \text{obstacles}
\]

and, if needed:

\[
\mathcal{W} = \text{scene area} \setminus
(\text{buildings} \cup \text{obstacles} \cup \text{offroad})
\]

The precise treatment of offroad regions should be scene-dependent:

- If offroad regions are physically non-traversable but visually transparent, they should restrict robot/pedestrian motion but not block visibility.
- If offroad regions correspond to bushes, walls, trees, or other visual barriers, they may be included in the occluder set.
- If uncertain, keep offroad as non-walkable but not occluding for the first implementation.

### 2.4 Data Products Required for OCE

The final preprocessed dataset should provide, for each scene:

- `scene_image`
- `walkable_polygon`
- `obstacle_polygons`
- `building_polygons`
- `occluder_polygons`
- `pedestrian_tracks`
- `agent_ids`
- `frame_ids`
- `train_val_test_split`
- `state_space`
- `state_transition_data`
- `route_or_destination_labels`
- `visibility_query_function`

These are sufficient to implement the complete OCE benchmark without using video.

---

## 3. Data Preprocessing

### 3.1 Establish Coordinate Frames and Scene Bounds

All trajectory, image, polygon, and robot-planning coordinates must be expressed in a common scene coordinate frame.

Ground-truth metric measurements are not available in the constrained-SDD artifacts. Therefore, each scene should be converted from source image coordinates into normalized scene units.

For each scene, define a coordinate normalizer such that the longest side of the scene has length `10`:

\[
k_{\text{scene}} =
\frac{10}{\max(x_{\max}^{\text{src}} - x_{\min}^{\text{src}},\;
y_{\max}^{\text{src}} - y_{\min}^{\text{src}})}
\]

The scene frame uses the lower-left environment corner as the origin, `x` right, and `y` up:

\[
x_{\text{scene}} = (x_{\text{src}} - x_{\min}^{\text{src}}) k_{\text{scene}}
\]

\[
y_{\text{scene}} = (y_{\max}^{\text{src}} - y_{\text{src}}) k_{\text{scene}}
\]

For display on a square `[0,1] x [0,1]` surface, use one additional scalar display factor:

\[
k_{\text{display}} =
\frac{1}{\max(x_{\max} - x_{\min},\; y_{\max} - y_{\min})}
\]

and:

\[
x_{\text{display}} = (x - x_{\min}) k_{\text{display}}
\]

\[
y_{\text{display}} = (y - y_{\min}) k_{\text{display}}
\]

One display axis will reach `1.0`; the other may be less than `1.0`.

Each scene is recorded at `30 FPS`. Store:

\[
\Delta t = \frac{1}{30}
\]

Use this `dt` for speed calculations in normalized scene units per second.

Required outputs:

- Source-to-scene transform
- Scene origin
- Scene bounds
- Trajectory positions in scene/planning units
- Semantic polygons in scene/planning units
- Scene coordinate normalizer
- Display normalization factor
- Display bounds
- Frame-rate metadata
- `dt` metadata

Tasks:

1. Parse SDD trajectory annotations.
2. Extract agent center positions.
3. Parse semantic polygons.
4. Filter to pedestrian agents.  All agents should be filtered to pedestrians already; however, flag any that have a speed above a threshold. Use `2 normalized scene units/s` as the default until metric calibration is available.
5. Remove short tracks whose source-frame endpoint displacement is below a configurable pixel threshold. Use Euclidean distance between the first and last trajectory point, with `10 px` as the default.
6. Use the image extent, tracks, and polygons to establish the environment bounds.
7. Use the lower-left corner of the environment bounds as the scene origin.
8. Normalize trajectories and polygons with the scene coordinate normalizer.
9. Store `FPS = 30` and `dt = 1/30` in scene metadata.
10. Estimate `scene_units_per_meter` from the mean per-track median speed divided by an assumed walking speed of `1.4 m/s` by default. This avoids overweighting long stopped or slow tracks when calibrating actor display dimensions. Store this local actor-dimension multiplier in scene metadata. Consumers that need stable cross-scene display can use a dataset-level percentile of the local multipliers; the pedestrian simulator defaults to the 75th percentile.
11. Store a single display normalization factor for aspect-preserving rendering on a square `[0,1] x [0,1]` surface.
12. Validate by overlaying trajectories and polygons on scene images.
13. Write functions to load and save scene data as a JSON format. Prioritize readability of the JSON output.

### 3.2 Select Usable Scenes

Not all scenes will be useful for OCE evaluation. Select scenes with:

- Sufficient pedestrian density
- Clear pedestrian route structure
- Static obstacles or buildings
- Meaningful occlusion geometry
- Repeated pedestrian routes
- Visible branch points, entrances, exits, or intersections

Avoid scenes where:

- Pedestrian motion is mostly straight-line
- There is little or no occlusion
- There are too few repeated trajectories to learn transitions
- The geometry does not create meaningful viewpoint choices

### 3.3 Clean Pedestrian Trajectories

For each pedestrian trajectory:

- Remove tracks shorter than a minimum duration.
- Remove tracks with implausible jumps.
- Smooth positions if needed.
- Resample to a fixed planning timestep.
- Map positions to walkable regions.
- Discard trajectories that spend too much time outside annotated walkable space.

Suggested parameters:

| Parameter | Suggested Value |
|---|---:|
| Minimum track duration | 4–6 s |
| Planning timestep | 0.4–1.0 s |
| Minimum endpoint displacement | 10 source pixels, configurable |
| Maximum jump threshold | Based on percentile speed |
| Agent class | Pedestrian |
| Assumed walking speed | 1.4 m/s |

---

### 3.4 Pedestrian Simulator Integration

The existing pedestrian simulator should support SDD as an alternate scenario provider, not as a replacement for the ETH-style track loader.

Implementation plan:

1. Add a scenario abstraction consumed by `Simulation`.
2. Represent scenario data as pedestrian tracks, display bounds, static environment polygons, and metadata.
3. Keep the existing ETH text-track loader as the default legacy data source.
4. Add an SDD data source that reads processed SDD scene artifacts from `outputs/sdd_processed/scene_XXX`.
5. Convert SDD y-up scene coordinates into the simulator's y-down display convention on load.
6. Display SDD `Building`, `Obstacle`, `Object`, and `Offroad` polygons as static environment objects.
7. Apply the SDD actor-dimension multiplier so robot and pedestrian metric dimensions are rendered in scene units.
8. Treat `Building`, `Obstacle`, `Object`, `Offroad`, and pedestrians as sensor-blocking polygons for visibility and scan integration.
9. Union overlapping blockers before constructing VisiLibity environments so dense pedestrian scenes do not produce invalid intersecting obstacle boundaries.
10. Verify with a display-only render test before adding static polygons to the planning costmap.

The implementation notes and command-line display test live in `src/python/pedestrian/pedestrian/docs/sdd_scenario_integration.md`.

---

## 4. Scene Representation

### 4.1 Build Walkable Region

Construct the walkable region:

\[
\mathcal{W} = \text{scene area} \setminus (\text{building} \cup \text{obstacle} \cup \text{offroad})
\]

Both pedestrians and the robot may be constrained to this region, unless a scene requires separate pedestrian-only and robot-only regions.

### 4.2 Build Discrete State Space

Construct a discrete state space over the walkable topology.

Two state representations are possible.

#### Option A: Uniform Grid over Walkable Space

Create a grid over the scene and retain only cells whose centers lie in walkable space.  Use a fixed grid size of 0.25m (configurable).


State:

\[
s_t \in \{1,\ldots,M\}
\]

Advantages:

- Easiest to implement
- Directly compatible with existing discrete OCE
- Supports visibility masks as diagonal matrices

Disadvantages:

- Less semantic
- Can be large
- Transition matrices may be sparse but high-dimensional

#### Option B: Topological Graph over Pedestrian Routes

Cluster historical pedestrian trajectories into route corridors and decision nodes.

State:

\[
s_t \in \{\text{path segment}, \text{junction}, \text{entrance}, \text{exit}, \text{plaza zone}\}
\]

Advantages:

- More behaviorally meaningful
- Smaller state space
- Easier to interpret
- Better suited to route-intent entropy

Disadvantages:

- More preprocessing
- Requires trajectory clustering or manual graph construction

### 4.3 Recommended Initial Approach

Start with:

1. A grid state space for implementation speed.
2. Destination or route labels for class entropy.
3. Later, replace or augment the grid with a topological route graph.

---

## 5. Pedestrian Transition Model

### 5.1 Define Latent Classes

Candidate class definitions:

1. **Destination class**
   The target exits through one of several scene exits.

2. **Route class**
   The target follows one of several common path clusters.

Use the destination class to group pedestrians by their end position.   Use a maximum cluster size (ball) of 2m (configurable).

### 5.2 Learn Markov Transitions

Using the grid state space, and for each scene, learn pedestrian motion transitions from training trajectories.

For each class \(c\), estimate:

\[
P_c(s_{t+1} \mid s_t)
\]

using only trajectories assigned to that class, where \(c\) is a latent destination or route class.


### 5.3 Estimate Class Priors

For a target observed up to time \(t_0\), initialize:

\[
b_0(s,c) = P(s_{t_0}=s, c \mid y_{0:t_0})
\]

Simple initialization:

- Current target state is known exactly.
- Class prior is proportional to historical route frequency conditioned on current state and heading.

More realistic initialization:

- Use a short observed prefix.
- Infer posterior route probabilities from prefix likelihood.
- Initialize class entropy from this posterior.

---

## 6. Observation and Occlusion Model

### 6.1 Robot Sensor Model

The simulated robot observes the target if:

1. The target is within sensor range.
2. The target is inside the field of view.
3. The line segment from robot to target is not blocked by static occluders.
4. Optionally, the line segment is not blocked by other pedestrians.

Observation:

\[
y_t =
\begin{cases}
s_t, & \text{if visible} \\
\text{occ}, & \text{if occluded}
\end{cases}
\]

### 6.2 Static Occlusion

Static occlusion is computed from:

- Building polygons
- Obstacle polygons
- Offroad polygons, if treated as visual barriers
- Manually added occluders where needed

For each candidate robot pose \(r_t\), compute a visibility mask:

\[
V^{\text{occ}}_t(r_t)
\]

where each diagonal entry indicates whether a target state is occluded from the robot pose.

### 6.3 Dynamic Occlusion

Dynamic occlusion is an optional extension.

Possible implementation:

- Other pedestrians are modeled as circular or elliptical occluders.
- Crowds are approximated as density-based occlusion fields.
- Occlusion probability increases with local pedestrian density.

This should be treated as a second-stage experiment, not part of the minimum viable version.

### 6.4 Noisy Sensor Extension

The initial experiment can assume perfect visible-state observations.

A later experiment should include distance-dependent sensor noise:

\[
\sigma(d) = \min(\sigma_{\max}, \sigma_{\min} + c d)
\]

Visible observations are then distributed by a discrete Gaussian kernel around the true target position.

This allows the experiment to test whether OCE remains useful when visible observations do not collapse the belief to a point mass.

---

## 7. Robot Model

### 7.1 Robot State

Full robot state:

\[
r_t = (x_t, y_t, v_t, \theta_t)
\]

### 7.2 Robot Dynamics

Unicycle dynamics for MPC-style evaluation.  Make use of the existing MPPI planner

### 7.3 Robot Task

Each episode defines:

- Robot start state \(r_0\)
- Robot goal state \(g\)
- Navigation constraints
- Selected target pedestrian
- Planning horizon \(H\)
- Replanning interval \(R\)

The robot must reach the goal while minimizing uncertainty about the target.

---

## 8. Planner Formulations

### 8.1 Candidate Trajectory Generation

At each replanning step, generate a set of feasible candidate robot trajectories.

Candidate generation options:

1. K-shortest paths to goal
2. Motion primitive rollout
3. Lattice planner
4. MPC trajectory samples
5. RRT or informed sampling
6. Monte Carlo Tree Search (MCTS)

Recommended initial implementation:

> Use K-shortest or penalty-diverse paths.

Each candidate trajectory is evaluated over a fixed planning horizon, and the robot executes only the first few steps before replanning.

---

## 8.2 Nominal Planner

The nominal planner ignores the target and optimizes only navigation.

Cost:

\[
J_{\text{nominal}}(\pi)
=
J_{\text{nav}}(\pi)
\]

where \(J_{\text{nav}}\) includes:

- Goal progress
- Path length
- Obstacle avoidance
- Smoothness or control effort

Purpose:

> Tests what happens when the robot does not actively gather information.

---

## 8.3 Visibility-Max Planner

The visibility-max planner selects the candidate trajectory that minimizes expected occlusion probability.

Cost:

\[
J_{\text{vis}}(\pi)
=
\frac{1}{H}
\sum_{t=1}^{H}
P(\text{target occluded at }t \mid \pi)
\]

Equivalent objective:

\[
\max_\pi
\sum_{t=1}^{H}
P(\text{target visible at }t \mid \pi)
\]

Purpose:

> Tests whether simply seeing the target more often is sufficient.

---

## 8.4 Greedy Information Planner

The greedy information planner chooses actions that maximize immediate expected entropy reduction.

One-step information gain:

\[
\Delta H_t
=
H(b_t) - \mathbb{E}[H(b_{t+1})]
\]

Purpose:

> Tests whether OCE provides value beyond generic one-step information gain.

This baseline is important because OCE should be shown to reason over future occlusions and belief evolution, not merely immediate entropy reduction.

---

## 8.5 OCE Planner

For each candidate robot trajectory \(\pi\), compute:

\[
J_{\text{OCE}}(\pi)
=
\alpha J_{\text{state}}(\pi)
+
\beta J_{\text{class}}(\pi)
+
\gamma J_{\text{occ}}(\pi)
+
\lambda J_{\text{nav}}(\pi)
\]

where:

- \(J_{\text{state}}\): expected target state entropy
- \(J_{\text{class}}\): expected route/class entropy
- \(J_{\text{occ}}\): expected occlusion probability
- \(J_{\text{nav}}\): path length, goal progress, smoothness, or control effort

The OCE planner selects:

\[
\pi^*
=
\arg\min_\pi J_{\text{OCE}}(\pi)
\]

The key distinction is that OCE does not merely seek visibility. It seeks trajectories that minimize uncertainty after accounting for how observations and occlusions affect the target belief.

---

## 8.6 Monte Carlo Belief-Space Baseline

For small scenes only, approximate the full belief-space objective by sampling observation histories.

Purpose:

> Validates OCE against a sampled approximation of the full information-gathering objective.

This baseline may be too expensive for full-scale evaluation but is useful for small controlled episodes.

---

## 8.7 Oracle / Hindsight Baseline

Optional upper bound.

The oracle planner knows the true future target trajectory and selects the best robot trajectory accordingly.

Purpose:

> Provides a performance ceiling and contextualizes the gap between OCE and perfect foresight.

---

## 9. Benchmark Episode Construction

### 9.1 Mine Ambiguous Pedestrian Episodes

Evaluate only on episodes where OCE has a meaningful opportunity to help.

An episode is valid if:

1. The target pedestrian has an observed prefix.
2. Multiple future route classes remain plausible.
3. The target approaches a branch point or occluded region.
4. The nominal robot path does not reveal all relevant future states.
5. At least one feasible robot deviation changes future visibility.
6. The true future route is known from held-out SDD data.

This avoids diluting results with trivial straight-line pedestrian tracks.

### 9.2 Episode Contents

Each episode should store:

- Scene ID
- Target pedestrian ID
- Observed prefix frames
- Target future trajectory
- Target initial belief
- Robot start
- Robot goal
- Static occluder map
- Candidate planning horizon
- Valid route classes
- Train/test split metadata

### 9.3 Train/Test Protocol

For each scene:

1. Split trajectories into training, validation, and test sets.
2. Learn transition models on training trajectories.
3. Tune planner weights on validation episodes.
4. Report final results on test episodes.

Important constraint:

> Do not use the target’s future test trajectory to construct the transition matrix for that same episode.

---

## 10. Evaluation Metrics

### 10.1 Belief Quality Metrics

Primary metrics:

| Metric | Meaning |
|---|---|
| Mean state entropy | Average uncertainty over target location |
| Final state entropy | Residual uncertainty at horizon end |
| Mean class entropy | Average uncertainty over route/intent |
| Final class entropy | Whether route ambiguity remains unresolved |
| Time to 95% route confidence | Speed of disambiguation |
| True-route posterior | Probability assigned to the actual route |
| Negative log-likelihood | Calibration of belief over true future states |

Recommended headline metrics:

1. Mean class entropy
2. Time to correct route confidence
3. Final state entropy
4. Negative log-likelihood of true future states

### 10.2 Visibility Metrics

Report visibility metrics, but do not use them as the only success criterion.

| Metric | Meaning |
|---|---|
| Percentage target visible | How often the target is directly seen |
| Longest occlusion duration | Worst continuous loss of target |
| Expected occlusion probability | Visibility-baseline objective |
| Number of reacquisitions | How often target is recovered after occlusion |

These metrics help demonstrate the distinction:

> OCE may not maximize raw visibility, but it should produce more informative visibility.

### 10.3 Robot Navigation Metrics

| Metric | Meaning |
|---|---|
| Path length | Cost of information gathering |
| Time to goal | Navigation delay |
| Minimum obstacle distance | Safety feasibility |
| Control effort / smoothness | Practicality |
| Planning latency | Runtime feasibility |

### 10.4 Statistical Reporting

For each method, report:

- Mean
- Standard deviation
- 95% confidence interval
- Paired comparisons across identical episodes
- Success/failure counts

Use paired tests because all methods should evaluate the same target episode, robot start/goal, and target realization.

---

## 11. Ablation Studies

### 11.1 Cost-Term Ablation

Evaluate:

1. State entropy only
2. Class entropy only
3. State + class entropy
4. State + class + occlusion penalty
5. Risk-weighted state/class entropy

Purpose:

> Determine whether performance comes from route disambiguation, state localization, or generic occlusion avoidance.

### 11.2 Sensor-Noise Ablation

Evaluate:

1. Perfect observations
2. Distance-dependent Gaussian noise
3. False negative/dropout probability
4. Reduced field of view
5. Reduced sensor range

Purpose:

> Determine whether OCE remains useful when visible observations are imperfect.

### 11.3 State-Space Ablation

Compare:

1. Uniform grid
2. Adaptive grid
3. Route graph
4. Hybrid grid + route class

Purpose:

> Determine whether OCE performance depends on state representation or generalizes across discretizations.

### 11.4 Horizon Ablation

Evaluate different planning horizons:

- Short horizon
- Medium horizon
- Long horizon

Expected result:

> OCE should be strongest when informative viewpoints require reasoning beyond immediate visibility.

### 11.5 Occlusion Ablation

Compare:

1. No occlusion
2. Static occlusion only
3. Dynamic pedestrian occlusion only
4. Static + dynamic occlusion

Purpose:

> Determine whether OCE’s advantage is specifically tied to occlusion-aware belief evolution.

---

## 12. Expected Qualitative Results

The final evaluation should include qualitative examples where:

1. **Nominal planner** takes the shortest path and loses target information.
2. **Visibility-max planner** chooses a viewpoint with high immediate visibility but low route-disambiguation value.
3. **Greedy information planner** selects a locally informative viewpoint but misses future occlusion effects.
4. **OCE planner** moves toward a viewpoint that reveals the branch point, exit, or reappearance region.
5. OCE may accept short-term occlusion if doing so reduces future ambiguity.

Central qualitative claim:

> Visibility is not equivalent to information.

OCE should be shown choosing views where different route hypotheses separate, not merely where the target is easiest to see.

---

## 13. Implementation Checklist

### Phase 1 — Data Setup

- [ ] Download constrained / annotated SDD.
- [ ] Parse pedestrian tracks.
- [ ] Parse building polygons.
- [ ] Parse obstacle polygons.
- [ ] Parse offroad / non-walkable polygons.
- [ ] Derive walkable regions if not explicitly provided.
- [ ] Derive occluder regions from buildings and obstacles.
- [ ] Align all data in a common scene coordinate frame.
- [ ] Visualize tracks and polygons over scene reference images.
- [ ] Filter pedestrian tracks.
- [ ] Select initial scenes.

Optional:

- [ ] Download representative raw SDD videos for presentation only.

### Phase 2 — State-Space Construction

- [ ] Build walkable mask.
- [ ] Create grid or graph state space.
- [ ] Map trajectories to states.
- [ ] Identify exits, branch points, and route clusters.
- [ ] Assign route/destination labels.
- [ ] Split trajectories into train/validation/test.

### Phase 3 — Transition Learning

- [ ] Estimate global transition matrix.
- [ ] Estimate class-conditioned transition matrices.
- [ ] Smooth sparse transitions.
- [ ] Validate transition rollout against held-out tracks.
- [ ] Compute route priors from observed prefixes.

### Phase 4 — Visibility and Occlusion

- [ ] Implement line-of-sight ray casting.
- [ ] Generate visibility masks for robot poses.
- [ ] Add field-of-view constraints.
- [ ] Add sensor range constraints.
- [ ] Validate visibility overlays.
- [ ] Add optional dynamic occluders.

### Phase 5 — Planner Implementation

- [ ] Implement robot motion model.
- [ ] Generate candidate trajectories.
- [ ] Implement nominal planner.
- [ ] Implement visibility-max planner.
- [ ] Implement greedy information planner.
- [ ] Implement OCE planner.
- [ ] Implement receding-horizon replanning.

### Phase 6 — Episode Mining

- [ ] Detect ambiguous route-choice prefixes.
- [ ] Detect occlusion-relevant episodes.
- [ ] Generate robot start-goal pairs.
- [ ] Store standardized benchmark episodes.
- [ ] Verify that all baselines receive identical initial beliefs.

### Phase 7 — Evaluation

- [ ] Run all planners on validation episodes.
- [ ] Tune OCE weights.
- [ ] Run final test experiments.
- [ ] Compute belief metrics.
- [ ] Compute visibility metrics.
- [ ] Compute navigation metrics.
- [ ] Generate paired statistical comparisons.
- [ ] Produce qualitative plots and videos.

### Phase 8 — Ablations

- [ ] Cost-term ablation.
- [ ] Sensor-noise ablation.
- [ ] Horizon ablation.
- [ ] State-space ablation.
- [ ] Static vs dynamic occlusion ablation.

---

## 14. Minimum Viable Experiment

The smallest useful version is:

1. Use the constrained / annotated SDD only.
2. Use pedestrian tracks as target trajectories.
3. Use buildings and obstacles as static occluders.
4. Derive walkable regions from annotations.
5. Build a grid or graph over walkable space.
6. Learn destination- or route-conditioned Markov transition matrices.
7. Simulate one robot observer.
8. Use static occlusion only.
9. Compare:
   - nominal planner
   - visibility-max planner
   - OCE planner
10. Evaluate on mined ambiguous route-choice episodes.
11. Report:
   - mean state entropy
   - mean class entropy
   - time to route confidence
   - true-route posterior
   - visibility percentage
   - path-length overhead

---

## 15. Preferred Final Paper Claim

The final claim should be carefully scoped:

> We demonstrate that OCE improves active pedestrian route disambiguation in real-world overhead scenes by selecting robot trajectories that reduce future belief uncertainty under occlusion, rather than merely maximizing target visibility.

Avoid claiming:

> OCE solves crowd navigation.

Better phrasing:

> OCE provides an information-aware objective that can be integrated into a receding-horizon navigation planner.

---

## 16. Summary

The Stanford Drone evaluation should not be framed as generic crowd navigation or pure target following. The strongest framing is:

> Active pedestrian route disambiguation under occlusion.

The robot has its own task, but it may deviate from its nominal path to preserve useful information about a selected pedestrian. OCE is evaluated as a trajectory-scoring function that trades navigation efficiency against belief quality.

The main comparison is:

- **Nominal planner:** “How do I reach my goal?”
- **Visibility-max planner:** “Where can I see the target most often?”
- **OCE planner:** “Which trajectory leaves me least uncertain about the target after future occlusions?”

That final question is the central contribution.
