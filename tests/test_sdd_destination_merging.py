import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "python" / "sdd"))

from oce_sdd.modeling import (  # noqa: E402
    DestinationClasses,
    destination_class_merge_components,
    merge_adjacent_destination_classes,
)


def destination_classes_fixture() -> DestinationClasses:
    endpoints = (
        np.asarray([[0.0, -0.2], [0.0, 0.2]], dtype=float),
        np.asarray([[1.0, -0.2], [1.0, 0.2]], dtype=float),
        np.asarray([[2.0, -0.2], [2.0, 0.2]], dtype=float),
        np.asarray([[5.0, -0.2], [5.0, 0.2]], dtype=float),
    )
    centers = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [5.0, 0.0]], dtype=float)
    return DestinationClasses(
        radius=0.5,
        centers=centers,
        train_counts=np.asarray([2, 2, 2, 2], dtype=np.int64),
        member_endpoints=endpoints,
        member_state_ids=tuple(
            np.asarray([2 * i, 2 * i + 1], dtype=np.int64)
            for i in range(len(endpoints))
        ),
        class_radii=np.asarray([0.2, 0.2, 0.2, 0.2], dtype=float),
        track_to_class={10: 0, 11: 1, 12: 2, 13: 3, 99: -1},
        pre_merge_class_count=4,
    )


def test_destination_class_merge_components_uses_connected_components():
    components = destination_class_merge_components(
        centers=np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [5.0, 0.0]]),
        radii=np.asarray([0.3, 0.3, 0.3, 0.3]),
        max_radius=0.5,
        epsilon=0.4,
    )

    assert components == [[0, 1, 2], [3]]


def test_merge_adjacent_destination_classes_remaps_tracks_and_members():
    merged = merge_adjacent_destination_classes(
        destination_classes_fixture(),
        epsilon=0.6,
        epsilon_meters=0.5,
    )

    assert merged.pre_merge_class_count == 4
    assert merged.merge_epsilon == 0.6
    assert merged.merge_epsilon_meters == 0.5
    assert merged.centers.shape == (2, 2)
    assert np.allclose(merged.centers, [[1.0, 0.0], [5.0, 0.0]])
    assert np.array_equal(merged.train_counts, [6, 2])
    assert merged.member_endpoints[0].shape == (6, 2)
    assert np.array_equal(merged.member_state_ids[0], [0, 1, 2, 3, 4, 5])
    assert merged.track_to_class == {10: 0, 11: 0, 12: 0, 13: 1, 99: -1}
    assert np.allclose(merged.class_radii, [1.019803902718557, 0.2])
