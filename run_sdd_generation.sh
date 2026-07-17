#!\bin/bash

PYTHONPATH=src/python/sdd

python -m oce_sdd.preprocess   \
    --data-root src/thirdParty/sdd/data \
    --out outputs/sdd_processed  \
    --min-track-displacement 10  \
    --assumed-walking-speed 1.4

python -m oce_sdd.modeling  \
    --processed-root outputs/sdd_processed \
    --out outputs/sdd_models   \
    --grid-size 1.0   \
    --destination-radius-meters 4.0   \
    --destination-min-samples 3 \
    --endpoint-snap-distance 1.0

# set up scene review and scenario designer sites for manual inspection of generated models and scenes
python -m oce_sdd.review_scenes  \
    --data-root src/thirdParty/sdd/data   \
    --models-root outputs/sdd_models   \
    --out outputs/sdd_scene_review

python -m oce_sdd.scenario_designer \
  --data-root src/thirdParty/sdd/data \
  --models-root outputs/sdd_models \
  --out outputs/sdd_scenario_designer
