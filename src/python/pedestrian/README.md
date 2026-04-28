# Pedestrian

Initial basis for a pedestrian environment for reinforcement learning -- training 'n' pedestrians to
interact without collisions.  One of those "pedestrians" may or may not be a robot.

## Trajectory Predictions

Integration with a trajectory prediction library [Trajectron++](https://github.com/idlebear/Trajectron-plus-plus) to make dynamic predictions of the other agents in the scene.

## Scenario data sources

The simulator supports scenario providers for pedestrian tracks and static environment objects.

- ETH text tracks remain available through `--tracks`.
- Processed SDD scenes can be loaded with `--data-source sdd --sdd-scene-id <id>`.

See `pedestrian/docs/sdd_scenario_integration.md` for the integration plan and display-only test.

Render an SDD scene without running the controller:

```bash
PYTHONPATH=src/python/pedestrian/pedestrian \
  /home/bjgilhul/miniconda3/envs/ppo/bin/python \
  src/python/pedestrian/pedestrian/display_sdd_scene.py \
  --sdd-processed-root outputs/sdd_processed \
  --sdd-scene-id 12
```


