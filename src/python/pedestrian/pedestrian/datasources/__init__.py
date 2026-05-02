from datasources.eth import load_eth_scenario
from datasources.scenario import Scenario, StaticPolygon
from datasources.sdd import load_sdd_scenario


def load_scenario(
    data_source,
    *,
    tracks=None,
    sdd_processed_root="outputs/sdd_processed",
    sdd_scene_id=None,
):
    if data_source in (None, "random"):
        return None
    if data_source == "eth":
        if tracks is None:
            raise ValueError("ETH data source requires a --tracks file")
        return load_eth_scenario(tracks)
    if data_source == "sdd":
        if sdd_scene_id is None:
            raise ValueError("SDD data source requires --sdd-scene-id")
        return load_sdd_scenario(sdd_processed_root, sdd_scene_id)
    raise ValueError(f"Unknown data source: {data_source}")
