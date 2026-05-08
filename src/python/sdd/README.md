# OCE SDD Tools

Project-local wrappers and preprocessing tools for the constrained Stanford Drone Dataset.

The third-party dataset/code under `src/thirdParty/constrained-sdd` is intentionally not modified.

## Scene reviewer

From the repository root:

```bash
PYTHONPATH=src/python/sdd /home/bjgilhul/miniconda3/envs/ppo/bin/python -m oce_sdd.review_scenes \
  --data-root src/thirdParty/sdd/data \
  --models-root outputs/sdd_models \
  --out outputs/sdd_scene_review
```

Then open:

```text
outputs/sdd_scene_review/index.html
```

The downloaded constrained-SDD trajectory pickles contain track IDs and 2D point sequences, but not original SDD agent labels or global frame IDs. When `--models-root` points at generated transition-model outputs, the reviewer colors tracks by learned destination class. Otherwise, it falls back to optional agent labels from a CSV/JSON map; missing labels are marked as `unknown`.

The reviewer also exports scene-aligned Markov transition-count heat maps to browser JSON for the selected scene and destination class.

Expected CSV class-map columns:

```text
scene_id,track_id,agent_class
```

Launch with:
```text
python -m http.server 8000 --directory outputs/sdd_scene_review
```

## Scenario designer

Generate a browser-based scenario designer from the same constrained-SDD artifacts:

```bash
PYTHONPATH=src/python/sdd /home/bjgilhul/miniconda3/envs/ppo/bin/python -m oce_sdd.scenario_designer \
  --data-root src/thirdParty/sdd/data \
  --models-root outputs/sdd_models \
  --out outputs/sdd_scenario_designer
```

Launch with:

```text
python -m http.server 8000 --directory outputs/sdd_scenario_designer
```

Then open:

```text
http://localhost:8000/
```

The designer exports `oce_sdd_scenario_design.v1` JSON in normalized scene coordinates when scene transform metadata is available. Older image-pixel exports are still identified by `coordinateFrame: constrained_sdd_image_pixels`. Each design contains:

- `robotStartZone`: a rectangular polygon
- `targetGoalZones`: user-drawn polygons for alternate target-destination clustering
- `selectedTrackIds`: tracks eligible to spawn as targets or moving occlusions
- `targetsOfInterestTrackIds`: selected tracks whose class-identification uncertainty is logged/tracked for experiments
- `robotGoal`: a point in the same coordinate frame
- `sourceSceneJson`: the generated scene-data JSON used by the browser

Designs are autosaved in browser local storage per scene, and can be exported/imported as JSON from the page.

The designer includes a time slider/playback bar for scrubbing tracks from start to end. In `Tracks` mode, click toggles a single track and dragging a rectangle selects all visible tracks whose current time-bar position is inside the rectangle. Selected tracks can then be marked as tracked targets of interest in the inspector. Use the `Viewer` link in the header to open the scene-reviewer view for the current scene from the same generated site.


## Coordinate frames and bounds

Section 3.1 preprocessing converts each scene into an explicit normalized Cartesian frame:

- source pixel frame: origin at top-left, x right, y down, units pixels
- OCE scene frame: origin at the bottom-left image corner, x right, y up, longest scene side is `10`
- display frame: one scalar factor maps the longest normalized scene axis to `1.0`
- timing: `fps = 30`, `dt = 1/30` seconds by default

Run from the repository root:

```bash
PYTHONPATH=src/python/sdd /home/bjgilhul/miniconda3/envs/ppo/bin/python -m oce_sdd.preprocess \
  --data-root src/thirdParty/sdd/data \
  --out outputs/sdd_processed \
  --min-track-displacement 10 \
  --assumed-walking-speed 1.4
```

Each scene directory contains:

- `metadata.json`: origin, axes, source-to-scene transform, scene bounds, `dt`, display factor, short-track filtering, top-level `scene_scale` in scene-units per meter, validation metrics
- `trajectories_scene.npz`: scene-frame trajectory arrays keyed by `track_<id>`
- `polygons_scene.json`: scene-frame semantic polygons
- `alignment_overlay.png`: image-space round-trip overlay for alignment checks

## Grid state space and transition models

Build section 4.2 Option A grid states and section 5 destination-conditioned Markov models:

```bash
PYTHONPATH=src/python/sdd /home/bjgilhul/miniconda3/envs/ppo/bin/python -m oce_sdd.modeling \
  --processed-root outputs/sdd_processed \
  --out outputs/sdd_models \
  --grid-size 0.25 \
  --destination-radius-meters 2.0 \
  --destination-min-samples 3 \
  --transition-min-support 10 \
  --global-goal-tau-meters 1.0 \
  --map-goal-tau-meters 1.0 \
  --endpoint-snap-distance 0.25
```

`--grid-size` and `--destination-radius-meters` use meters and are converted per scene with `metadata.json`'s `scene_scale`. `--endpoint-snap-distance` uses normalized scene units. Destination classes use complete-link clustering over walkable shortest-path distances, so `--destination-radius-meters` is the maximum within-class endpoint diameter in meters. Use `--destination-radius` only if you want to pass the radius directly in normalized scene units.
Endpoints that fall just outside the walkable grid can be snapped to a nearby walkable state with `--endpoint-snap-distance`; by default this is the grid cell size.
Transitions use a three-layer goal-conditioned Markov estimator. `--transition-min-support` controls the visit-count smoothing threshold, while `--global-goal-tau-meters` and `--map-goal-tau-meters` control the global-flow goal penalty and map-prior goal temperature.

Each model scene directory contains:

- `state_space.npz` and `state_space.json`
- `trajectory_states.npz`
- `splits.json`
- `destination_classes.json`
- `transitions/global_transition.npz`
- `transitions/class_<id>_transition.npz`
- diagnostic transition layers under `transitions/class_<id>_*_transition.npz`

## Class-identification experiment logs

When running the pedestrian simulator with discrete OCE and SDD models, pass:

```bash
--experiment-log-dir results/experiment_logs/run_001
```

The simulator writes:

- `target_beliefs.csv`: one row per tracked target per step, including ground-truth position/state/class, visibility, state entropy, mode entropy, true-class probability, and `mode_prob_<class_id>` columns.
- `uncertainty_summary.csv`: one row per step, including tracked/visible counts, summed and mean state/mode entropy, `total_uncertainty` as summed mode entropy, and mean true-class probability.

Only tracks listed in `targetsOfInterestTrackIds` are logged/tracked when that field is present. Older scenario designs without the field preserve legacy behavior by tracking every loaded actor.

Generate standard plots with:

```bash
python processing/plot_experiment_uncertainty.py results/experiment_logs/run_001
```

The script writes `total_uncertainty.png`, `per_target_entropy.png`, `visibility_timeline.png`, and `true_class_probability.png` under `results/experiment_logs/run_001/plots`.
