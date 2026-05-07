from datasources.eth import load_eth_scenario
from datasources.scenario import Scenario, StaticPolygon
from datasources.sdd import load_sdd_scenario


def load_scenario(data_source, *, tracks=None, **data_args):
    if data_source in (None, "random"):
        return None
    if data_source == "eth":
        if tracks is None:
            raise ValueError("ETH data source requires a --tracks file")
        return load_eth_scenario(tracks)
    if data_source == "sdd":
        if data_args.get("sdd_scene_id") is None:
            raise ValueError("SDD data source requires --sdd-scene-id")
        return load_sdd_scenario(
            processed_root=data_args.get("sdd_processed_root"),
            scene_id=data_args.get("sdd_scene_id"),
            scenario_config=data_args.get("sdd_scenario_config"),
        )
    raise ValueError(f"Unknown data source: {data_source}")
