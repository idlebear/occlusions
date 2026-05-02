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
  --endpoint-snap-distance 0.25
```

`--grid-size` and `--destination-radius-meters` use meters and are converted per scene with `metadata.json`'s `scene_scale`. `--endpoint-snap-distance` uses normalized scene units. Destination classes use complete-link clustering over walkable shortest-path distances, so `--destination-radius-meters` is the maximum within-class endpoint diameter in meters. Use `--destination-radius` only if you want to pass the radius directly in normalized scene units.
Endpoints that fall just outside the walkable grid can be snapped to a nearby walkable state with `--endpoint-snap-distance`; by default this is the grid cell size.

Each model scene directory contains:

- `state_space.npz` and `state_space.json`
- `trajectory_states.npz`
- `splits.json`
- `destination_classes.json`
- `transitions/global_transition.npz`
- `transitions/class_<id>_transition.npz`
