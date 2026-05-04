from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy import sparse

from oce_sdd.data import (
    POLYGON_CLASS_ORDER,
    SDDData,
    load_agent_class_map,
    load_constrained_sdd_data,
    scene_summary,
)


def main() -> None:
    args = parse_args()
    data = load_constrained_sdd_data(args.data_root, dequantized=not args.quantized)
    class_map = load_agent_class_map(args.agent_classes)
    write_review_site(
        data,
        output_root=Path(args.out),
        class_map=class_map,
        default_agent_class=args.default_agent_class,
        trajectory_stride=args.trajectory_stride,
        models_root=Path(args.models_root) if args.models_root else None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a browser-based scene reviewer for constrained SDD data."
    )
    parser.add_argument(
        "--data-root",
        default="src/thirdParty/sdd/data",
        help="Path containing constrained-SDD pickle artifacts.",
    )
    parser.add_argument(
        "--out",
        default="outputs/sdd_scene_review",
        help="Output directory for the static review site.",
    )
    parser.add_argument(
        "--agent-classes",
        default=None,
        help="Optional CSV/JSON mapping scene_id,track_id to agent_class.",
    )
    parser.add_argument(
        "--default-agent-class",
        default="unknown",
        help="Class label used when no class map entry exists.",
    )
    parser.add_argument(
        "--trajectory-stride",
        type=int,
        default=5,
        help="Keep every Nth trajectory point in reviewer JSON.",
    )
    parser.add_argument(
        "--quantized",
        action="store_true",
        help="Use trajectories.pkl instead of trajectories_dequantized.pkl.",
    )
    parser.add_argument(
        "--models-root",
        default="outputs/sdd_models",
        help="Optional model output root containing destination_classes.json per scene.",
    )
    return parser.parse_args()


def write_review_site(
    data: SDDData,
    *,
    output_root: Path,
    class_map: dict[tuple[int, int], str],
    default_agent_class: str,
    trajectory_stride: int,
    models_root: Path | None,
) -> None:
    if trajectory_stride < 1:
        raise ValueError("--trajectory-stride must be >= 1")

    images_root = output_root / "images"
    scenes_root = output_root / "scenes"
    models_out_root = output_root / "models"
    images_root.mkdir(parents=True, exist_ok=True)
    scenes_root.mkdir(parents=True, exist_ok=True)
    models_out_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for scene_id in data.scene_ids:
        destination_labels = load_destination_labels(models_root, scene_id)
        markov_model_rel = write_markov_model_json(models_root, models_out_root, scene_id)
        image_rel = f"images/scene_{scene_id:03d}.png"
        scene_rel = f"scenes/scene_{scene_id:03d}.json"
        _write_scene_image(data.images[scene_id], output_root / image_rel)
        _write_scene_json(
            data,
            scene_id,
            output_root / scene_rel,
            image_rel=image_rel,
            class_map=class_map,
            default_agent_class=default_agent_class,
            trajectory_stride=trajectory_stride,
            destination_labels=destination_labels,
            markov_model_rel=markov_model_rel,
        )

        summary = scene_summary(data, scene_id, class_map, default_agent_class)
        summary["destination_class_counts"] = count_destination_classes(destination_labels)
        summary["scene_json"] = scene_rel
        summary["image"] = image_rel
        summaries.append(summary)

    _write_index_json(summaries, scenes_root / "index.json")
    _write_summary_csv(summaries, output_root / "scene_summary.csv")
    (output_root / "index.html").write_text(_review_html(), encoding="utf-8")

    print(f"Wrote {len(summaries)} scenes to {output_root}")
    print(f"Open {output_root / 'index.html'} in a browser")


def _write_scene_image(image: np.ndarray, path: Path) -> None:
    if image.ndim == 3 and image.shape[2] == 3:
        out = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        out = image
    if not cv2.imwrite(str(path), out):
        raise IOError(f"Could not write scene image: {path}")


def _write_scene_json(
    data: SDDData,
    scene_id: int,
    path: Path,
    *,
    image_rel: str,
    class_map: dict[tuple[int, int], str],
    default_agent_class: str,
    trajectory_stride: int,
    destination_labels: dict[int, int],
    markov_model_rel: str | None,
) -> None:
    image = data.images[scene_id]
    tracks = data.trajectories[scene_id]
    polygons = data.polygons[scene_id]
    height, width = image.shape[:2]

    track_records = []
    max_frame = 0
    for track_id, points in sorted(tracks.items()):
        sampled = points[::trajectory_stride]
        if sampled.shape[0] == 0:
            continue
        track_id_int = int(track_id)
        destination_class = destination_labels.get(track_id_int)
        if destination_class is None:
            destination_status = "not_modeled"
        elif destination_class < 0:
            destination_status = "unassigned"
        else:
            destination_status = "assigned"
        max_frame = max(max_frame, sampled.shape[0] - 1)
        track_records.append(
            {
                "trackId": track_id_int,
                "agentClass": class_map.get((scene_id, track_id_int), default_agent_class),
                "destinationClass": int(destination_class) if destination_class is not None else None,
                "destinationStatus": destination_status,
                "pointStride": trajectory_stride,
                "points": np.round(sampled, 3).tolist(),
            }
        )

    polygon_records = []
    for poly_class in _ordered_polygon_classes(polygons):
        for vertices in polygons.get(poly_class, []):
            polygon_records.append(
                {
                    "polygonClass": poly_class,
                    "vertices": np.round(vertices, 3).tolist(),
                }
            )

    scene = {
        "sceneId": scene_id,
        "image": image_rel,
        "imageWidth": int(width),
        "imageHeight": int(height),
        "maxFrame": int(max_frame),
        "timeMode": "track_point_index",
        "timeNote": (
            "The constrained-SDD trajectory pickle contains local point sequences, "
            "not original global frame IDs."
        ),
        "classMode": "destination_class" if destination_labels else "agent_class",
        "markovModel": {"index": markov_model_rel} if markov_model_rel else None,
        "tracks": track_records,
        "polygons": polygon_records,
    }
    path.write_text(json.dumps(scene, separators=(",", ":")), encoding="utf-8")


def write_markov_model_json(
    models_root: Path | None,
    models_out_root: Path,
    scene_id: int,
) -> str | None:
    if models_root is None:
        return None

    scene_model_root = models_root / f"scene_{scene_id:03d}"
    transitions_root = scene_model_root / "transitions"
    destination_path = scene_model_root / "destination_classes.json"
    state_space_path = scene_model_root / "state_space.json"
    state_space_npz_path = scene_model_root / "state_space.npz"
    global_counts_path = transitions_root / "global_counts.npz"
    if (
        not global_counts_path.exists()
        or not destination_path.exists()
        or not state_space_path.exists()
        or not state_space_npz_path.exists()
    ):
        return None

    scene_out_root = models_out_root / f"scene_{scene_id:03d}"
    scene_out_root.mkdir(parents=True, exist_ok=True)
    for pattern in ("class_*_heatmap.json", "class_*_transition.json"):
        for stale_path in scene_out_root.glob(pattern):
            stale_path.unlink()
    destination_data = json.loads(destination_path.read_text())
    state_space = json.loads(state_space_path.read_text())

    global_heatmap_rel = f"models/scene_{scene_id:03d}/global_heatmap.json"
    entries = [
        {
            "id": "global",
            "label": "Global",
            "heatmap": global_heatmap_rel,
        }
    ]
    write_transition_activity_heatmap(
        global_counts_path,
        state_space_npz_path,
        state_space,
        scene_out_root / "global_heatmap.json",
    )

    for class_record in destination_data.get("classes", []):
        class_id = int(class_record["class_id"])
        counts_path = transitions_root / f"class_{class_id:03d}_counts.npz"
        if not counts_path.exists():
            continue
        rel_path = f"models/scene_{scene_id:03d}/class_{class_id:03d}_heatmap.json"
        write_transition_activity_heatmap(
            counts_path,
            state_space_npz_path,
            state_space,
            scene_out_root / f"class_{class_id:03d}_heatmap.json",
        )
        entries.append(
            {
                "id": class_id,
                "label": f"Destination {class_id}",
                "trainCount": int(class_record.get("train_count", 0)),
                "center": class_record.get("center"),
                "heatmap": rel_path,
            }
        )

    index = {
        "sceneId": scene_id,
        "stateCount": int(state_space.get("state_count", 0)),
        "rows": int(state_space.get("rows", 0)),
        "cols": int(state_space.get("cols", 0)),
        "cellSize": float(state_space.get("cell_size", 0.0)),
        "models": entries,
    }
    (scene_out_root / "index.json").write_text(json.dumps(index, separators=(",", ":")), encoding="utf-8")
    return f"models/scene_{scene_id:03d}/index.json"


def write_transition_activity_heatmap(
    counts_path: Path,
    state_space_npz_path: Path,
    state_space: dict[str, Any],
    output_path: Path,
) -> None:
    counts = sparse.load_npz(counts_path).tocsr()
    state_arrays = np.load(state_space_npz_path)
    grid_to_state = np.asarray(state_arrays["grid_to_state"], dtype=int)
    walkable_mask = np.asarray(state_arrays["walkable_mask"], dtype=bool)
    rows, cols = grid_to_state.shape

    source_counts = np.asarray(counts.sum(axis=1)).reshape(-1)
    values = np.zeros((rows, cols), dtype=float)
    valid = grid_to_state >= 0
    values[valid] = source_counts[grid_to_state[valid]]

    cell_size = float(state_space.get("cell_size", 0.0))
    bounds = dict(state_space.get("bounds", {}))
    grid_extent = {
        "min_x": float(bounds.get("min_x", 0.0)),
        "min_y": float(bounds.get("min_y", 0.0)),
        "max_x": float(bounds.get("min_x", 0.0)) + cols * cell_size,
        "max_y": float(bounds.get("min_y", 0.0)) + rows * cell_size,
    }
    payload = {
        "kind": "source_transition_counts",
        "rows": int(rows),
        "cols": int(cols),
        "cellSize": cell_size,
        "bounds": bounds,
        "gridExtent": grid_extent,
        "stateCount": int(state_space.get("state_count", counts.shape[0])),
        "totalTransitions": float(values.sum()),
        "activeCells": int(np.count_nonzero(values)),
        "maxValue": float(values.max()) if values.size else 0.0,
        "values": np.round(values.reshape(-1), 6).tolist(),
        "walkable": walkable_mask.astype(int).reshape(-1).tolist(),
    }
    output_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def load_destination_labels(models_root: Path | None, scene_id: int) -> dict[int, int]:
    if models_root is None:
        return {}
    path = models_root / f"scene_{scene_id:03d}" / "destination_classes.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {
        int(track_id): int(class_id)
        for track_id, class_id in data.get("track_to_class", {}).items()
    }


def count_destination_classes(destination_labels: dict[int, int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for class_id in destination_labels.values():
        key = str(class_id)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _ordered_polygon_classes(polygons: dict[str, list[np.ndarray]]) -> list[str]:
    ordered = [poly_class for poly_class in POLYGON_CLASS_ORDER if poly_class in polygons]
    ordered.extend(sorted(set(polygons) - set(ordered)))
    return ordered


def _write_index_json(summaries: list[dict[str, Any]], path: Path) -> None:
    index_records = []
    for row in summaries:
        class_counts = row["agent_class_counts"]
        polygon_counts = row["polygon_counts"]
        index_records.append(
            {
                "sceneId": row["scene_id"],
                "sceneJson": row["scene_json"],
                "image": row["image"],
                "imageWidth": row["image_width"],
                "imageHeight": row["image_height"],
                "trackCount": row["track_count"],
                "maxTrackLength": row["max_track_length"],
                "medianTrackLength": row["median_track_length"],
                "agentClassCounts": class_counts,
                "destinationClassCounts": row.get("destination_class_counts", {}),
                "polygonCounts": polygon_counts,
            }
        )
    path.write_text(json.dumps({"scenes": index_records}, indent=2), encoding="utf-8")


def _write_summary_csv(summaries: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "scene_id",
        "image_width",
        "image_height",
        "track_count",
        "max_track_length",
        "median_track_length",
        "agent_class_counts",
        "destination_class_counts",
        "polygon_counts",
        "scene_json",
        "image",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(
                {
                    **row,
                    "agent_class_counts": json.dumps(row["agent_class_counts"], sort_keys=True),
                    "destination_class_counts": json.dumps(
                        row.get("destination_class_counts", {}),
                        sort_keys=True,
                    ),
                    "polygon_counts": json.dumps(row["polygon_counts"], sort_keys=True),
                }
            )


def _review_html() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>OCE SDD Scene Reviewer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {
      color-scheme: light;
      font-family: Arial, Helvetica, sans-serif;
      --border: #c9ced6;
      --text: #17202a;
      --muted: #5d6d7e;
      --panel: #f5f7fa;
    }
    body {
      margin: 0;
      color: var(--text);
      background: white;
    }
    header {
      display: grid;
      grid-template-columns: 1fr auto;
      align-items: center;
      gap: 16px;
      padding: 12px 16px;
      border-bottom: 1px solid var(--border);
      background: var(--panel);
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 700;
    }
    main {
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr);
      min-height: calc(100vh - 58px);
    }
    aside {
      border-right: 1px solid var(--border);
      padding: 12px;
      overflow: auto;
    }
    .viewer {
      display: grid;
      grid-template-rows: auto minmax(260px, 1fr) 300px;
      min-width: 0;
    }
    .controls {
      display: grid;
      gap: 10px;
      padding: 12px;
      border-bottom: 1px solid var(--border);
      background: #fff;
    }
    .row {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
    }
    button, select, input[type="range"] {
      font: inherit;
    }
    button {
      border: 1px solid var(--border);
      background: white;
      border-radius: 4px;
      padding: 6px 9px;
      cursor: pointer;
    }
    button:hover {
      background: #eef2f6;
    }
    select {
      max-width: 100%;
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 5px;
      background: white;
    }
    canvas {
      width: 100%;
      height: 100%;
      display: block;
      background: #e8edf3;
    }
    .canvas-wrap {
      min-height: 0;
      overflow: hidden;
    }
    .metric {
      margin: 0 0 8px;
      font-size: 13px;
      line-height: 1.35;
    }
    .muted {
      color: var(--muted);
    }
    .class-list label {
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 6px 0;
      font-size: 13px;
    }
    .swatch {
      width: 12px;
      height: 12px;
      border-radius: 2px;
      border: 1px solid rgba(0,0,0,0.25);
      display: inline-block;
      flex: 0 0 auto;
    }
    .legend {
      margin-top: 16px;
      padding-top: 12px;
      border-top: 1px solid var(--border);
    }
    .model-panel {
      border-top: 1px solid var(--border);
      padding: 10px 12px 12px;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 8px;
      min-height: 0;
      background: #fff;
    }
    .model-panel canvas {
      border: 1px solid var(--border);
      background: white;
    }
    @media (max-width: 820px) {
      main {
        grid-template-columns: 1fr;
      }
      aside {
        border-right: 0;
        border-bottom: 1px solid var(--border);
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>OCE SDD Scene Reviewer</h1>
    <div class="muted" id="status">Loading...</div>
  </header>
  <main>
    <aside>
      <p class="metric"><strong>Scene</strong></p>
      <select id="sceneSelect"></select>
      <div class="legend">
        <p class="metric"><strong>Scene Stats</strong></p>
        <div id="sceneStats" class="metric muted"></div>
      </div>
      <div class="legend">
        <p class="metric"><strong id="classFilterTitle">Destination Classes</strong></p>
        <div id="classFilters" class="class-list"></div>
      </div>
      <div class="legend">
        <p class="metric"><strong>Polygon Classes</strong></p>
        <div id="polygonLegend" class="class-list"></div>
      </div>
      <div class="legend">
        <p class="metric"><strong>Time</strong></p>
        <p class="metric muted" id="timeNote"></p>
      </div>
    </aside>
    <section class="viewer">
      <div class="controls">
        <div class="row">
          <button id="prevScene">Prev Scene</button>
          <button id="nextScene">Next Scene</button>
          <button id="prevFrame">Back</button>
          <button id="playPause">Play</button>
          <button id="nextFrame">Forward</button>
          <span id="frameLabel" class="muted"></span>
          <label><input id="endpointOnly" type="checkbox" checked> Endpoints only</label>
        </div>
        <div class="row">
          <input id="frameSlider" type="range" min="0" max="0" value="0" style="flex:1 1 260px">
        </div>
      </div>
      <div class="canvas-wrap">
        <canvas id="sceneCanvas"></canvas>
      </div>
      <div class="model-panel">
        <div class="row">
          <strong>Markov Spatial Heat Map</strong>
          <select id="modelClassSelect"></select>
          <span id="modelStats" class="muted"></span>
        </div>
        <canvas id="modelCanvas"></canvas>
      </div>
    </section>
  </main>
  <script>
    const classColors = [
      "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
      "#56B4E9", "#F0E442", "#332288", "#88CCEE", "#44AA99",
      "#117733", "#999933", "#DDCC77", "#CC6677", "#882255",
      "#AA4499", "#DDDDDD", "#000000", "#6699CC", "#661100"
    ];
    const polygonColors = {
      Building: "rgba(44, 62, 80, 0.42)",
      Obstacle: "rgba(192, 57, 43, 0.42)",
      Object: "rgba(211, 84, 0, 0.42)",
      Offroad: "rgba(39, 174, 96, 0.32)",
      Entrance: "rgba(41, 128, 185, 0.50)"
    };
    const state = {
      index: null,
      sceneIndex: 0,
      scene: null,
      image: null,
      markovIndex: null,
      markovHeatmap: null,
      frame: 0,
      playing: false,
      endpointOnly: true,
      classVisible: new Map()
    };

    const el = {
      status: document.getElementById("status"),
      sceneSelect: document.getElementById("sceneSelect"),
      sceneStats: document.getElementById("sceneStats"),
      classFilterTitle: document.getElementById("classFilterTitle"),
      classFilters: document.getElementById("classFilters"),
      polygonLegend: document.getElementById("polygonLegend"),
      timeNote: document.getElementById("timeNote"),
      prevScene: document.getElementById("prevScene"),
      nextScene: document.getElementById("nextScene"),
      prevFrame: document.getElementById("prevFrame"),
      nextFrame: document.getElementById("nextFrame"),
      playPause: document.getElementById("playPause"),
      endpointOnly: document.getElementById("endpointOnly"),
      frameSlider: document.getElementById("frameSlider"),
      frameLabel: document.getElementById("frameLabel"),
      canvas: document.getElementById("sceneCanvas"),
      modelClassSelect: document.getElementById("modelClassSelect"),
      modelStats: document.getElementById("modelStats"),
      modelCanvas: document.getElementById("modelCanvas")
    };
    const ctx = el.canvas.getContext("2d");
    const modelCtx = el.modelCanvas.getContext("2d");

    async function init() {
      const response = await fetch("scenes/index.json");
      state.index = await response.json();
      populateSceneSelect();
      bindControls();
      await loadScene(0);
      requestAnimationFrame(animationTick);
    }

    function populateSceneSelect() {
      el.sceneSelect.innerHTML = "";
      state.index.scenes.forEach((scene, i) => {
        const option = document.createElement("option");
        option.value = String(i);
        option.textContent = `Scene ${scene.sceneId} (${scene.trackCount} tracks)`;
        el.sceneSelect.appendChild(option);
      });
    }

    function bindControls() {
      el.sceneSelect.addEventListener("change", event => loadScene(Number(event.target.value)));
      el.prevScene.addEventListener("click", () => loadScene(Math.max(0, state.sceneIndex - 1)));
      el.nextScene.addEventListener("click", () => loadScene(Math.min(state.index.scenes.length - 1, state.sceneIndex + 1)));
      el.prevFrame.addEventListener("click", () => setFrame(state.frame - 1));
      el.nextFrame.addEventListener("click", () => setFrame(state.frame + 1));
      el.playPause.addEventListener("click", () => {
        state.playing = !state.playing;
        el.playPause.textContent = state.playing ? "Pause" : "Play";
      });
      el.frameSlider.addEventListener("input", event => setFrame(Number(event.target.value)));
      el.endpointOnly.addEventListener("change", event => {
        state.endpointOnly = event.target.checked;
        draw();
      });
      el.modelClassSelect.addEventListener("change", () => loadSelectedMarkovHeatmap());
      window.addEventListener("resize", draw);
    }

    async function loadScene(sceneIndex) {
      state.sceneIndex = sceneIndex;
      el.sceneSelect.value = String(sceneIndex);
      const indexRecord = state.index.scenes[sceneIndex];
      el.status.textContent = `Loading scene ${indexRecord.sceneId}...`;
      const sceneResponse = await fetch(indexRecord.sceneJson);
      state.scene = await sceneResponse.json();
      state.image = await loadImage(state.scene.image);
      state.markovIndex = null;
      state.markovHeatmap = null;
      state.frame = 0;
      state.classVisible = new Map(classesForScene().map(item => [item.key, true]));
      el.frameSlider.max = String(state.scene.maxFrame);
      el.frameSlider.value = "0";
      el.status.textContent = `Scene ${state.scene.sceneId}`;
      renderSidebar(indexRecord);
      await loadMarkovIndex();
      draw();
    }

    function loadImage(src) {
      return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = src;
      });
    }

    function renderSidebar(indexRecord) {
      el.sceneStats.innerHTML = [
        `Image: ${indexRecord.imageWidth} x ${indexRecord.imageHeight}`,
        `Tracks: ${indexRecord.trackCount}`,
        `Destination classes: ${destinationClassesForScene().length}`,
        `Max points: ${indexRecord.maxTrackLength}`,
        `Median points: ${indexRecord.medianTrackLength.toFixed(1)}`
      ].join("<br>");

      el.classFilterTitle.textContent = state.scene.classMode === "destination_class"
        ? "Destination Classes"
        : "Agent Classes";
      el.classFilters.innerHTML = "";
      classesForScene().forEach((classItem, i) => {
        const label = document.createElement("label");
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = true;
        checkbox.addEventListener("change", () => {
          state.classVisible.set(classItem.key, checkbox.checked);
          draw();
        });
        const swatch = document.createElement("span");
        swatch.className = "swatch";
        swatch.style.background = colorForClass(classItem.key, i);
        const count = state.scene.tracks.filter(track => classKeyForTrack(track) === classItem.key).length;
        label.append(checkbox, swatch, `${classItem.label} (${count})`);
        el.classFilters.appendChild(label);
      });

      el.polygonLegend.innerHTML = "";
      const polygonClasses = [...new Set(state.scene.polygons.map(poly => poly.polygonClass))];
      polygonClasses.forEach(className => {
        const label = document.createElement("label");
        const swatch = document.createElement("span");
        swatch.className = "swatch";
        swatch.style.background = polygonColors[className] || "rgba(90,90,90,0.35)";
        const count = state.scene.polygons.filter(poly => poly.polygonClass === className).length;
        label.append(swatch, `${className} (${count})`);
        el.polygonLegend.appendChild(label);
      });

      el.timeNote.textContent = state.scene.timeNote;
      updateFrameLabel();
    }

    async function loadMarkovIndex() {
      el.modelClassSelect.innerHTML = "";
      state.markovIndex = null;
      state.markovHeatmap = null;
      if (!state.scene.markovModel || !state.scene.markovModel.index) {
        el.modelStats.textContent = "No model data";
        drawModelHeatmap();
        return;
      }

      const response = await fetch(state.scene.markovModel.index);
      state.markovIndex = await response.json();
      state.markovIndex.models.forEach((model, i) => {
        const option = document.createElement("option");
        option.value = String(i);
        option.textContent = model.trainCount === undefined
          ? model.label
          : `${model.label} (${model.trainCount})`;
        el.modelClassSelect.appendChild(option);
      });
      el.modelClassSelect.value = "0";
      await loadSelectedMarkovHeatmap();
    }

    async function loadSelectedMarkovHeatmap() {
      if (!state.markovIndex || !state.markovIndex.models.length) {
        state.markovHeatmap = null;
        drawModelHeatmap();
        return;
      }
      const model = state.markovIndex.models[Number(el.modelClassSelect.value)];
      el.modelStats.textContent = "Loading...";
      const response = await fetch(model.heatmap);
      state.markovHeatmap = await response.json();
      state.markovHeatmap.label = model.label;
      state.markovHeatmap.trainCount = model.trainCount;
      drawModelHeatmap();
    }

    function classesForScene() {
      if (state.scene.classMode === "destination_class") {
        const keys = [...new Set(state.scene.tracks.map(track => classKeyForTrack(track)))];
        return keys.sort(classKeyCompare).map(key => ({ key, label: labelForClassKey(key) }));
      }
      return [...new Set(state.scene.tracks.map(track => String(track.agentClass)))]
        .sort()
        .map(key => ({ key, label: key }));
    }

    function destinationClassesForScene() {
      return [...new Set(state.scene.tracks
        .map(track => track.destinationClass)
        .filter(classId => Number.isInteger(classId) && classId >= 0))]
        .sort((a, b) => a - b);
    }

    function classKeyForTrack(track) {
      if (state.scene.classMode === "destination_class") {
        if (track.destinationStatus === "not_modeled") return "destination:not-modeled";
        if (track.destinationStatus === "unassigned") return "destination:unassigned";
        return Number.isInteger(track.destinationClass) && track.destinationClass >= 0
          ? `destination:${track.destinationClass}`
          : "destination:unassigned";
      }
      return String(track.agentClass);
    }

    function classKeyCompare(a, b) {
      if (a === "destination:not-modeled") return 1;
      if (b === "destination:not-modeled") return -1;
      if (a === "destination:unassigned") return 1;
      if (b === "destination:unassigned") return -1;
      const aNum = Number(a.split(":")[1]);
      const bNum = Number(b.split(":")[1]);
      if (Number.isFinite(aNum) && Number.isFinite(bNum)) return aNum - bNum;
      return a.localeCompare(b);
    }

    function labelForClassKey(key) {
      if (key === "destination:unassigned") return "Unassigned";
      if (key === "destination:not-modeled") return "Not modeled";
      if (key.startsWith("destination:")) return `Destination ${key.split(":")[1]}`;
      return key;
    }

    function colorForClass(className, fallbackIndex = 0) {
      if (className === "destination:unassigned") return "#777777";
      if (className === "destination:not-modeled") return "#bbbbbb";
      const classes = classesForScene();
      const index = classes.findIndex(item => item.key === className);
      return classColors[(index >= 0 ? index : fallbackIndex) % classColors.length];
    }

    function setFrame(frame) {
      const maxFrame = state.scene ? state.scene.maxFrame : 0;
      state.frame = Math.max(0, Math.min(maxFrame, frame));
      el.frameSlider.value = String(state.frame);
      updateFrameLabel();
      draw();
    }

    function updateFrameLabel() {
      const stride = state.scene?.tracks?.[0]?.pointStride || 1;
      el.frameLabel.textContent = `Point ${state.frame} / ${state.scene.maxFrame} (source index approx. ${state.frame * stride})`;
    }

    function animationTick() {
      if (state.playing && state.scene) {
        const next = state.frame >= state.scene.maxFrame ? 0 : state.frame + 1;
        setFrame(next);
      }
      setTimeout(() => requestAnimationFrame(animationTick), 75);
    }

    function draw() {
      if (!state.scene || !state.image) return;
      const wrap = el.canvas.parentElement;
      const dpr = window.devicePixelRatio || 1;
      el.canvas.width = Math.max(1, Math.floor(wrap.clientWidth * dpr));
      el.canvas.height = Math.max(1, Math.floor(wrap.clientHeight * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const fit = fitRect(wrap.clientWidth, wrap.clientHeight, state.scene.imageWidth, state.scene.imageHeight);
      ctx.clearRect(0, 0, wrap.clientWidth, wrap.clientHeight);
      ctx.save();
      ctx.translate(fit.x, fit.y);
      ctx.scale(fit.scale, fit.scale);
      ctx.drawImage(state.image, 0, 0);
      drawPolygons();
      drawTracks();
      ctx.restore();
      drawModelHeatmap();
    }

    function fitRect(dstW, dstH, srcW, srcH) {
      const scale = Math.min(dstW / srcW, dstH / srcH);
      return {
        scale,
        x: (dstW - srcW * scale) / 2,
        y: (dstH - srcH * scale) / 2
      };
    }

    function drawPolygons() {
      state.scene.polygons.forEach(poly => {
        if (poly.vertices.length < 2) return;
        ctx.beginPath();
        poly.vertices.forEach(([x, y], i) => {
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.fillStyle = polygonColors[poly.polygonClass] || "rgba(90,90,90,0.35)";
        ctx.strokeStyle = "rgba(0,0,0,0.45)";
        ctx.lineWidth = 1.25;
        ctx.fill();
        ctx.stroke();
      });
    }

    function drawTracks() {
      const classes = classesForScene();
      state.scene.tracks.forEach(track => {
        const classKey = classKeyForTrack(track);
        if (!state.classVisible.get(classKey)) return;
        const classIndex = classes.findIndex(item => item.key === classKey);
        const color = colorForClass(classKey, classIndex);
        const end = state.endpointOnly
          ? track.points.length - 1
          : Math.min(state.frame, track.points.length - 1);
        if (end < 0) return;
        if (!state.endpointOnly) {
          drawPath(track.points, track.points.length - 1, "rgba(20,20,20,0.14)", 1);
          drawPath(track.points, end, color, 2.2);
        }
        const [x, y] = track.points[end];
        ctx.beginPath();
        ctx.arc(x, y, state.endpointOnly ? 4.5 : 3.5, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.strokeStyle = "white";
        ctx.lineWidth = 1.25;
        ctx.fill();
        ctx.stroke();
      });
    }

    function drawPath(points, end, strokeStyle, lineWidth) {
      if (!points.length) return;
      ctx.beginPath();
      ctx.moveTo(points[0][0], points[0][1]);
      for (let i = 1; i <= end; i += 1) {
        ctx.lineTo(points[i][0], points[i][1]);
      }
      ctx.strokeStyle = strokeStyle;
      ctx.lineWidth = lineWidth;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.stroke();
    }

    function drawModelHeatmap() {
      const canvas = el.modelCanvas;
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      modelCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const width = rect.width;
      const height = rect.height;
      modelCtx.clearRect(0, 0, width, height);
      modelCtx.fillStyle = "white";
      modelCtx.fillRect(0, 0, width, height);

      if (!state.markovHeatmap) {
        modelCtx.fillStyle = "#5d6d7e";
        modelCtx.font = "13px Arial, Helvetica, sans-serif";
        modelCtx.fillText("No Markov model selected", 12, 24);
        return;
      }

      const heatmap = state.markovHeatmap;
      const extent = heatmap.gridExtent || heatmap.bounds || {};
      const minX = Number.isFinite(extent.min_x) ? extent.min_x : 0;
      const minY = Number.isFinite(extent.min_y) ? extent.min_y : 0;
      const maxX = Number.isFinite(extent.max_x) ? extent.max_x : heatmap.cols * heatmap.cellSize;
      const maxY = Number.isFinite(extent.max_y) ? extent.max_y : heatmap.rows * heatmap.cellSize;
      const sceneW = Math.max(1e-9, maxX - minX);
      const sceneH = Math.max(1e-9, maxY - minY);
      const marginLeft = 42;
      const marginRight = width >= 520 ? 86 : 58;
      const marginTop = 14;
      const marginBottom = 30;
      const availableW = Math.max(1, width - marginLeft - marginRight);
      const availableH = Math.max(1, height - marginTop - marginBottom);
      const scale = Math.max(1e-9, Math.min(availableW / sceneW, availableH / sceneH));
      const plotW = sceneW * scale;
      const plotH = sceneH * scale;
      const x0 = marginLeft + (availableW - plotW) / 2;
      const y0 = marginTop + (availableH - plotH) / 2;
      const cellW = heatmap.cellSize * scale;
      const cellH = heatmap.cellSize * scale;
      const maxValue = Math.max(heatmap.maxValue || 0, 1e-12);

      modelCtx.fillStyle = "#f7f8fa";
      modelCtx.fillRect(x0, y0, plotW, plotH);
      for (let row = 0; row < heatmap.rows; row += 1) {
        for (let col = 0; col < heatmap.cols; col += 1) {
          const idx = row * heatmap.cols + col;
          const walkable = heatmap.walkable[idx] === 1;
          const value = heatmap.values[idx] || 0;
          const x = x0 + col * cellW;
          const y = y0 + plotH - (row + 1) * cellH;
          if (!walkable) {
            modelCtx.fillStyle = "#e4e7eb";
          } else if (value > 0) {
            const rgba = viridis(Math.sqrt(value / maxValue));
            modelCtx.fillStyle = `rgba(${rgba[0]},${rgba[1]},${rgba[2]},0.88)`;
          } else {
            modelCtx.fillStyle = "#fbfcfd";
          }
          modelCtx.fillRect(x, y, Math.ceil(cellW) + 0.5, Math.ceil(cellH) + 0.5);
        }
      }

      modelCtx.strokeStyle = "#17202a";
      modelCtx.lineWidth = 1;
      modelCtx.strokeRect(x0, y0, plotW, plotH);
      if (cellW >= 5 && cellH >= 5) {
        drawGridLines(x0, y0, plotW, plotH, cellW, cellH, heatmap.rows, heatmap.cols);
      }

      modelCtx.fillStyle = "#17202a";
      modelCtx.font = "12px Arial, Helvetica, sans-serif";
      modelCtx.fillText("scene x", x0 + Math.max(0, plotW / 2 - 20), y0 + plotH + 22);
      modelCtx.save();
      modelCtx.translate(14, y0 + Math.max(0, plotH / 2 + 20));
      modelCtx.rotate(-Math.PI / 2);
      modelCtx.fillText("scene y", 0, 0);
      modelCtx.restore();

      const trainText = heatmap.trainCount === undefined ? "" : `, train ${heatmap.trainCount}`;
      el.modelStats.textContent = `${heatmap.label}: source-state transition counts, active ${heatmap.activeCells}, max ${formatNumber(heatmap.maxValue)}${trainText}`;
      drawHeatLegend(x0 + plotW + 12, y0, Math.max(40, width - x0 - plotW - 18), plotH, maxValue);
    }

    function drawGridLines(x0, y0, plotW, plotH, cellW, cellH, rows, cols) {
      modelCtx.beginPath();
      for (let col = 1; col < cols; col += 1) {
        const x = x0 + col * cellW;
        modelCtx.moveTo(x, y0);
        modelCtx.lineTo(x, y0 + plotH);
      }
      for (let row = 1; row < rows; row += 1) {
        const y = y0 + row * cellH;
        modelCtx.moveTo(x0, y);
        modelCtx.lineTo(x0 + plotW, y);
      }
      modelCtx.strokeStyle = "rgba(23,32,42,0.10)";
      modelCtx.lineWidth = 1;
      modelCtx.stroke();
    }

    function drawHeatLegend(x, y, width, height, maxValue) {
      if (width < 28) return;
      const legendWidth = 12;
      const legendHeight = Math.max(60, Math.round(height));
      const image = modelCtx.createImageData(legendWidth, legendHeight);
      for (let yy = 0; yy < legendHeight; yy += 1) {
        const t = 1 - yy / Math.max(1, legendHeight - 1);
        const rgba = viridis(t);
        for (let xx = 0; xx < legendWidth; xx += 1) {
          const idx = (yy * legendWidth + xx) * 4;
          image.data[idx] = rgba[0];
          image.data[idx + 1] = rgba[1];
          image.data[idx + 2] = rgba[2];
          image.data[idx + 3] = rgba[3];
        }
      }
      modelCtx.putImageData(image, x, y);
      modelCtx.strokeStyle = "#17202a";
      modelCtx.strokeRect(x, y, legendWidth, legendHeight);
      modelCtx.fillStyle = "#17202a";
      modelCtx.font = "11px Arial, Helvetica, sans-serif";
      modelCtx.fillText(formatNumber(maxValue), x + 16, y + 10);
      modelCtx.fillText("0", x + 16, y + legendHeight);
    }

    function formatNumber(value) {
      if (!Number.isFinite(value)) return "0";
      if (Math.abs(value) >= 1000) return value.toFixed(0);
      if (Math.abs(value) >= 10) return value.toFixed(1);
      return value.toFixed(2);
    }

    function viridis(t) {
      const stops = [
        [68, 1, 84],
        [59, 82, 139],
        [33, 145, 140],
        [94, 201, 98],
        [253, 231, 37]
      ];
      const clamped = Math.max(0, Math.min(1, t));
      const scaled = clamped * (stops.length - 1);
      const i = Math.min(stops.length - 2, Math.floor(scaled));
      const local = scaled - i;
      const a = stops[i];
      const b = stops[i + 1];
      return [
        Math.round(a[0] + (b[0] - a[0]) * local),
        Math.round(a[1] + (b[1] - a[1]) * local),
        Math.round(a[2] + (b[2] - a[2]) * local),
        255
      ];
    }

    init().catch(error => {
      console.error(error);
      el.status.textContent = `Error: ${error.message}`;
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
