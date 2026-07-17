# SDD Scenario Integration Plan

## Goal

Extend the pedestrian simulator so ETH text tracks remain supported while processed Stanford Drone Dataset scenes can be selected as an alternate scenario provider.

The first implementation slice is display-only: load one processed SDD scene, draw its pedestrian tracks through the existing actor display path, and draw static environment polygons for buildings, obstacles, objects, and undrivable/offroad areas.

## Scenario Abstraction

`Simulation` should consume a `Scenario` rather than knowing every dataset format directly.

The scenario contract contains:

- `tracks`: simulator-ready pedestrian tracks keyed by track id, with points stored as `[x, y, frame]`
- `display_offset`: lower-left display origin used by existing normalized robot start/goal logic
- `display_diff`: square display side length in scenario units
- `static_polygons`: environment objects such as buildings, obstacles, objects, and offroad regions
- `metadata`: provider-specific provenance and coordinate metadata

Static polygons contain:

- `polygon_class`
- `points`
- `blocking`
- optional metadata

## Data Sources

### ETH

The current ETH text loader is preserved as `load_eth_scenario(path)`.

It keeps the existing behavior:

- parse `frame id x y`
- interpolate missing frames per track
- compute square display bounds from track extents
- provide no static polygons

`--tracks` continues to work as before. Internally, it now routes through the ETH data source.

### SDD

`load_sdd_scenario(processed_root, scene_id)` reads processed SDD artifacts:

- `metadata.json`
- `trajectories_scene.npz`
- `polygons_scene.json`

The SDD provider uses the normalized scene coordinate frame produced by `oce_sdd.preprocess`. The display square is derived from scene bounds, preserving aspect ratio with the existing `display_offset` and `display_diff` logic.

Processed SDD coordinates are stored in a Cartesian scene frame with y increasing upward. The existing pygame simulator display uses y increasing downward, so the SDD provider flips y on load:

```text
y_sim = y_min + y_max - y_scene
```

This conversion is applied to both trajectories and static polygons. ETH data remains unchanged.

Actor dimensions are converted from metric defaults into SDD scene units with the preprocessing calibration:

```text
scene_units_per_meter = mean_track_median_speed_scene_units_per_s / assumed_walking_speed_mps
```

The default assumed walking speed is `1.4 m/s`. The preprocessing uses one typical speed per track, then averages across tracks, so long stopped or slow tracks do not dominate the display calibration.

For display in the pedestrian simulator, SDD uses a shared dataset-level actor scale by default: the 75th percentile of corrected per-scene actor multipliers. This avoids scene-to-scene visual jumps from noisy local speed calibration. The percentile is configurable with `--sdd-actor-scale-percentile`; use `-1` to keep each scene's local scale. `Simulation` applies the selected multiplier to robot and pedestrian geometry, images, extents, and collision footprints. ETH and random scenarios keep a multiplier of `1.0`.

Blocking static polygon classes:

- `Building`
- `Obstacle`
- `Object`
- `Offroad`

`Entrance` polygons are loaded and displayed, but are not blocking.

## Simulation Integration

`Simulation` accepts:

- `scenario`
- or `data_source`
- or legacy `tracks`

Selection order:

1. Explicit `scenario`
2. Explicit `data_source`
3. Legacy `tracks` as ETH
4. Random generated scenario

This keeps current ETH and random behavior intact.

## Display-Only Test

The display test is:

```bash
PYTHONPATH=src/python/pedestrian/pedestrian \
  /home/bjgilhul/miniconda3/envs/ppo/bin/python \
  src/python/pedestrian/pedestrian/display_sdd_scene.py \
  --sdd-processed-root outputs/sdd_processed \
  --sdd-scene-id 12
```

Expected output:

```text
outputs/pedestrian_sdd_display/scene_012.png
```

The rendered frame should show:

- robot start and goal
- SDD pedestrians active at frame zero
- SDD static polygons under the actors
- square display bounds using the existing simulation display surface

## Next Slice

The first perception integration is now in place:

- `Simulation.sensor_blocking_polygons()` combines live pedestrian actor polygons with blocking SDD static polygons.
- `Simulation.calculate_visibility()` unions overlapping blockers before passing them to VisiLibity. This is required because dense SDD pedestrian tracks can overlap, and VisiLibity rejects intersecting obstacle boundaries.
- `Simulation._calculate_scan()` passes the same actor and static blocker set into the fake scanner when the scanner backend is available.
- The existing render path draws the visibility polygon overlay with `_draw_visibility()`.

Blocking SDD classes:

- `Building`
- `Obstacle`
- `Object`
- `Offroad`

Pedestrians are also sensor blockers.

Remaining perception/planning work:

- build the local MPPI costmap from SDD blocking polygons
