import csv
import importlib.util
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
PEDESTRIAN_ROOT = ROOT / "src" / "python" / "pedestrian" / "pedestrian"
WARP_ROOT = ROOT / "src" / "thirdParty" / "warp_mppi"
sys.path.insert(0, str(PEDESTRIAN_ROOT))
sys.path.insert(0, str(WARP_ROOT))


def _install_main_import_stubs():
    pygame_stub = types.ModuleType("pygame")
    pygame_stub.image = types.SimpleNamespace(load=lambda *_args, **_kwargs: object())
    pygame_stub.transform = types.SimpleNamespace(
        scale=lambda image, _size: image,
        rotate=lambda image, _angle: image,
    )
    sys.modules.setdefault("pygame", pygame_stub)
    sys.modules.setdefault("visilibity", types.ModuleType("visilibity"))
    shapely_stub = types.ModuleType("shapely")
    shapely_geometry_stub = types.ModuleType("shapely.geometry")
    shapely_geometry_stub.LineString = object
    shapely_geometry_stub.MultiPolygon = object
    shapely_geometry_stub.Point = object
    shapely_geometry_stub.Polygon = object
    shapely_ops_stub = types.ModuleType("shapely.ops")
    shapely_ops_stub.unary_union = lambda polygons: None
    sys.modules.setdefault("shapely", shapely_stub)
    sys.modules.setdefault("shapely.geometry", shapely_geometry_stub)
    sys.modules.setdefault("shapely.ops", shapely_ops_stub)
    skimage_stub = types.ModuleType("skimage")
    skimage_morphology_stub = types.ModuleType("skimage.morphology")
    skimage_morphology_stub.binary_dilation = lambda *args, **kwargs: None
    skimage_morphology_stub.binary_opening = lambda *args, **kwargs: None
    sys.modules.setdefault("skimage", skimage_stub)
    sys.modules.setdefault("skimage.morphology", skimage_morphology_stub)
    velocity_prediction_stub = types.ModuleType("controller.velocity_prediction")
    velocity_prediction_stub.ControlVariations = object
    sys.modules.setdefault("controller.velocity_prediction", velocity_prediction_stub)
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.__version__ = "2.0.0"
    sys.modules.setdefault("pandas", pandas_stub)
    polycheck_stub = types.ModuleType("polycheck")
    polycheck_stub.load_polycheck = lambda *args, **kwargs: None
    polycheck_stub.faux_scan = None
    polycheck_stub.visibility_from_region = lambda *args, **kwargs: None
    sys.modules.setdefault("polycheck", polycheck_stub)
    warp_mppi_stub = types.ModuleType("warp_mppi")
    legacy_stub = types.ModuleType("warp_mppi.legacy")
    legacy_stub.PyCudaMPPI = object
    legacy_stub.evaluate_trajectories_by_entropy_gpu = lambda *args, **kwargs: None
    legacy_stub.evaluate_discrete_oce_gpu = lambda *args, **kwargs: None
    warp_mppi_stub.legacy = legacy_stub
    sys.modules.setdefault("warp_mppi", warp_mppi_stub)
    sys.modules.setdefault("warp_mppi.legacy", legacy_stub)


_install_main_import_stubs()
import main as pedestrian_main  # noqa: E402
from datasources.scenario import Scenario  # noqa: E402
from simulation import Simulation  # noqa: E402


def _fake_sdd_models():
    transitions = np.eye(2, dtype=float)[np.newaxis, :, :]
    return {
        "state_space_metadata": {
            "bounds": {"min_x": 0.0, "min_y": 0.0, "max_y": 2.0},
            "rows": 1,
            "cols": 2,
            "state_count": 2,
            "cell_size": 1.0,
        },
        "state_space": {
            "grid_to_state": np.asarray([[0, 1]], dtype=np.int64),
            "walkable_mask": np.asarray([[True, True]]),
            "grid_indices": np.asarray([[0, 0], [0, 1]], dtype=np.int64),
            "centers": np.asarray([[0.5, 0.5], [1.5, 0.5]], dtype=float),
        },
        "classes": [{"id": 7, "prob": 1.0}],
        "models": {
            7: {"transitions": sparse.csr_matrix(transitions[0])},
            "global": {"transitions": sparse.csr_matrix(transitions[0])},
        },
    }


def test_simulation_actor_state_includes_tracked_status():
    scenario = Scenario(
        name="tracked-test",
        data_source="sdd",
        tracks={
            2: [[1.0, 1.0, 0.0], [1.1, 1.0, 1.0]],
            3: [[2.0, 1.0, 0.0], [2.1, 1.0, 1.0]],
        },
        display_offset=[0.0, 0.0],
        display_diff=4.0,
        metadata={
            "scene_scale": 1.0,
            "dt": 1.0,
            "grid": {"width": 4.0, "height": 4.0, "resolution": 1.0},
            "targets_of_interest_track_ids": [2],
            "robot_start_zone": np.asarray([0.0, 0.0], dtype=float),
            "robot_goal": np.asarray([3.0, 3.0], dtype=float),
        },
    )
    sim = Simulation(
        scenario=scenario,
        generator_name="uniform",
        generator_args={"seed": 1},
        ego_start=[0.0, 0.0],
        ego_goal=[3.0, 3.0],
        enable_scan=False,
    )

    sim._generate_new_agents()
    states = {actor["id"]: actor for actor in sim._get_info()["actors"]}

    assert states["2"]["tracked"] is True
    assert states["3"]["tracked"] is False


def _scenario_with_tracks(track_ids, targets_of_interest):
    return Scenario(
        name="limit-test",
        data_source="sdd",
        tracks={track_id: [[float(track_id), 0.0, 0.0]] for track_id in track_ids},
        display_offset=[0.0, 0.0],
        display_diff=4.0,
        metadata={
            "scene_scale": 1.0,
            "dt": 1.0,
            "grid": {"width": 4.0, "height": 4.0, "resolution": 1.0},
            "targets_of_interest_track_ids": list(targets_of_interest),
        },
    )


def test_limited_scenario_tracks_selects_targets_of_interest_first():
    scenario = _scenario_with_tracks([1, 2, 3, 4, 5], targets_of_interest=[4, 5])
    sim = object.__new__(Simulation)

    limited = Simulation._limited_scenario_tracks(sim, scenario, limit_tracks=3, seed=2)

    assert {4, 5}.issubset(set(limited.tracks))
    assert len(limited.tracks) == 3
    assert limited.metadata["targets_of_interest_track_ids"] == [4, 5]
    assert {4, 5}.issubset(set(limited.metadata["track_limit_selected_ids"]))


def test_limited_scenario_tracks_keeps_all_targets_when_limit_is_smaller():
    scenario = _scenario_with_tracks([1, 2, 3, 4], targets_of_interest=[3, 4])
    sim = object.__new__(Simulation)

    limited = Simulation._limited_scenario_tracks(sim, scenario, limit_tracks=1, seed=2)

    assert set(limited.tracks) == {3, 4}
    assert limited.metadata["track_limit_selected_count"] == 2
    assert limited.metadata["targets_of_interest_track_ids"] == [3, 4]


def test_limited_scenario_tracked_targets_caps_interest_set_before_track_limit():
    scenario = _scenario_with_tracks([1, 2, 3, 4, 5], targets_of_interest=[3, 4, 5])
    sim = object.__new__(Simulation)

    limited_targets = Simulation._limited_scenario_tracked_targets(
        sim, scenario, limit_tracked_targets=1, seed=2
    )
    limited_tracks = Simulation._limited_scenario_tracks(
        sim, limited_targets, limit_tracks=3, seed=2
    )

    tracked_ids = set(limited_tracks.metadata["targets_of_interest_track_ids"])
    assert len(tracked_ids) == 1
    assert tracked_ids.issubset({3, 4, 5})
    assert tracked_ids.issubset(set(limited_tracks.tracks))
    assert len(limited_tracks.tracks) == 3


def test_limited_scenario_tracked_targets_uses_all_tracks_when_interest_missing():
    scenario = _scenario_with_tracks([1, 2, 3, 4], targets_of_interest=[])
    scenario = replace(
        scenario,
        metadata={
            key: value
            for key, value in scenario.metadata.items()
            if key != "targets_of_interest_track_ids"
        },
    )
    sim = object.__new__(Simulation)

    limited = Simulation._limited_scenario_tracked_targets(
        sim, scenario, limit_tracked_targets=2, seed=2
    )

    tracked_ids = set(limited.metadata["targets_of_interest_track_ids"])
    assert len(tracked_ids) == 2
    assert tracked_ids.issubset(set(scenario.tracks))


def test_discrete_oce_tracker_ignores_untracked_visible_actors():
    tracker = pedestrian_main.DiscreteOCETracker(_fake_sdd_models())

    tracker.update(
        [
            {"id": "1", "tracked": False, "visible": True, "pos": [0.5, 1.5]},
            {"id": "2", "tracked": True, "visible": True, "pos": [1.5, 1.5]},
        ],
        tick=1,
    )

    assert "1" not in tracker.agent_hmms
    assert "2" in tracker.agent_hmms
    assert np.allclose(tracker.agent_hmms["2"].state_distribution, [0.0, 1.0])


def test_append_experiment_logs_writes_target_and_summary_csv(tmp_path):
    tracker = pedestrian_main.DiscreteOCETracker(_fake_sdd_models())
    actors = [{"id": "2", "tracked": True, "visible": True, "pos": [1.5, 1.5, 0.3, 0.0]}]
    tracker.update(actors, tick=1)

    pedestrian_main.append_experiment_logs(
        tmp_path,
        tick=1,
        time_s=0.5,
        actors=actors,
        tracker=tracker,
        sdd_models={"track_to_class": {"2": 7}},
    )

    with (tmp_path / "target_beliefs.csv").open(newline="") as csv_file:
        target_rows = list(csv.DictReader(csv_file))
    with (tmp_path / "uncertainty_summary.csv").open(newline="") as csv_file:
        summary_rows = list(csv.DictReader(csv_file))

    assert target_rows[0]["track_id"] == "2"
    assert float(target_rows[0]["mode_prob_7"]) == 1.0
    assert float(target_rows[0]["true_class_probability"]) == 1.0
    assert int(summary_rows[0]["tracked_count"]) == 1
    assert float(summary_rows[0]["total_uncertainty"]) == 0.0


def test_experiment_plot_script_creates_pngs(tmp_path):
    target_csv = tmp_path / "target_beliefs.csv"
    summary_csv = tmp_path / "uncertainty_summary.csv"
    target_csv.write_text(
        "tick,time_s,track_id,tracked,visible,x,y,theta,speed,true_state_id,true_class_id,"
        "state_entropy,mode_entropy,true_class_probability,mode_prob_7\n"
        "1,0.5,2,True,True,1.5,1.5,0,0.3,1,7,0.0,0.0,1.0,1.0\n"
    )
    summary_csv.write_text(
        "tick,time_s,tracked_count,visible_tracked_count,sum_state_entropy,"
        "mean_state_entropy,sum_mode_entropy,mean_mode_entropy,total_uncertainty,"
        "mean_true_class_probability\n"
        "1,0.5,1,1,0.0,0.0,0.0,0.0,0.0,1.0\n"
    )
    module_path = ROOT / "processing" / "plot_experiment_uncertainty.py"
    spec = importlib.util.spec_from_file_location("plot_experiment_uncertainty", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_dir = tmp_path / "plots"
    module.plot_total_uncertainty(module.read_csv_rows(summary_csv), output_dir)
    module.plot_per_target_entropy(module.read_csv_rows(target_csv), output_dir)
    module.plot_visibility(module.read_csv_rows(target_csv), output_dir)
    module.plot_true_class_probability(module.read_csv_rows(target_csv), output_dir)

    assert (output_dir / "total_uncertainty.png").exists()
    assert (output_dir / "per_target_entropy.png").exists()
    assert (output_dir / "visibility_timeline.png").exists()
    assert (output_dir / "true_class_probability.png").exists()
