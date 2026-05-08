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
    --grid-size 0.50   \
    --destination-radius-meters 4.0   \
    --destination-min-samples 3 \
    --endpoint-snap-distance 0.25

python -m oce_sdd.review_scenes  \
    --data-root src/thirdParty/sdd/data   \
    --models-root outputs/sdd_models   \
    --out outputs/sdd_scene_review
