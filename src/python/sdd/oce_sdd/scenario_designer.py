from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from oce_sdd.coordinates import make_scene_transform
from oce_sdd.data import (
    SDDData,
    load_agent_class_map,
    load_constrained_sdd_data,
    scene_summary,
)
from oce_sdd.review_scenes import (
    _review_html,
    _write_scene_image,
    _write_scene_json,
    count_destination_classes,
    load_destination_labels,
    write_markov_model_json,
)


def main() -> None:
    args = parse_args()
    data = load_constrained_sdd_data(args.data_root, dequantized=not args.quantized)
    class_map = load_agent_class_map(args.agent_classes)
    write_scenario_designer(
        data,
        output_root=Path(args.out),
        class_map=class_map,
        default_agent_class=args.default_agent_class,
        trajectory_stride=args.trajectory_stride,
        models_root=Path(args.models_root) if args.models_root else None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a browser-based scenario designer for constrained SDD data."
    )
    parser.add_argument(
        "--data-root",
        default="src/thirdParty/sdd/data",
        help="Path containing constrained-SDD pickle artifacts.",
    )
    parser.add_argument(
        "--out",
        default="outputs/sdd_scenario_designer",
        help="Output directory for the static designer site.",
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
        help="Keep every Nth trajectory point in designer JSON.",
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


def write_scenario_designer(
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
        transform = make_scene_transform(scene_id, data.images[scene_id].shape)
        summary["transform"] = transform.to_metadata()
        summary["destination_class_counts"] = count_destination_classes(destination_labels)
        summary["scene_json"] = scene_rel
        summary["image"] = image_rel
        summaries.append(summary)

    _write_index_json(summaries, scenes_root / "index.json")
    (output_root / "index.html").write_text(_designer_html(), encoding="utf-8")
    (output_root / "viewer.html").write_text(_viewer_html(), encoding="utf-8")

    print(f"Wrote {len(summaries)} scenes to {output_root}")
    print(f"Open {output_root / 'index.html'} in a browser")


def _write_index_json(summaries: list[dict[str, Any]], path: Path) -> None:
    records = []
    for row in summaries:
        records.append(
            {
                "sceneId": row["scene_id"],
                "sceneJson": row["scene_json"],
                "image": row["image"],
                "imageWidth": row["image_width"],
                "imageHeight": row["image_height"],
                "trackCount": row["track_count"],
                "maxTrackLength": row["max_track_length"],
                "medianTrackLength": row["median_track_length"],
                "agentClassCounts": row["agent_class_counts"],
                "transform": row.get("transform"),
                "destinationClassCounts": row.get("destination_class_counts", {}),
                "polygonCounts": row["polygon_counts"],
            }
        )
    path.write_text(json.dumps({"scenes": records}, indent=2), encoding="utf-8")

    csv_path = path.parent.parent / "scene_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        for row in summaries:
            csv_row = {
                **row,
                "agent_class_counts": json.dumps(row["agent_class_counts"], sort_keys=True),
                "destination_class_counts": json.dumps(
                    row.get("destination_class_counts", {}),
                    sort_keys=True,
                ),
                "polygon_counts": json.dumps(row["polygon_counts"], sort_keys=True),
            }
            csv_row.pop("transform", None)
            writer.writerow(csv_row)


def _viewer_html() -> str:
    return _review_html().replace(
        '<div class="muted" id="status">Loading...</div>',
        '<div class="row"><a class="muted" href="index.html">Designer</a>'
        '<div class="muted" id="status">Loading...</div></div>',
        1,
    )


def _designer_html() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>OCE SDD Scenario Designer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {
      color-scheme: light;
      font-family: Arial, Helvetica, sans-serif;
      --border: #c9ced6;
      --text: #17202a;
      --muted: #5d6d7e;
      --panel: #f5f7fa;
      --active: #1f77b4;
      --goal: #d55e00;
      --target: #009e73;
      --start: #cc79a7;
    }
    * { box-sizing: border-box; }
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
    header a {
      color: var(--muted);
      text-decoration: none;
      font-size: 13px;
    }
    header a:hover { color: var(--text); }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 700;
    }
    main {
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr) 330px;
      min-height: calc(100vh - 58px);
    }
    aside, .inspector {
      padding: 12px;
      overflow: auto;
      min-width: 0;
    }
    aside { border-right: 1px solid var(--border); }
    .inspector { border-left: 1px solid var(--border); }
    .viewer {
      display: grid;
      grid-template-rows: auto minmax(320px, 1fr);
      min-width: 0;
      min-height: 0;
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
    .mode-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    button, select, input {
      font: inherit;
    }
    button {
      border: 1px solid var(--border);
      background: white;
      border-radius: 4px;
      padding: 6px 9px;
      cursor: pointer;
      min-height: 31px;
    }
    button:hover { background: #eef2f6; }
    button.active {
      border-color: var(--active);
      color: white;
      background: var(--active);
    }
    button.danger { color: #a93226; }
    select, input[type="text"] {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 6px;
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
      position: relative;
    }
    .metric {
      margin: 0 0 8px;
      font-size: 13px;
      line-height: 1.35;
    }
    .muted { color: var(--muted); }
    .section {
      margin-top: 16px;
      padding-top: 12px;
      border-top: 1px solid var(--border);
    }
    .track-list, .zone-list {
      display: grid;
      gap: 6px;
      max-height: 220px;
      overflow: auto;
    }
    .track-item, .zone-item {
      display: grid;
      grid-template-columns: auto 1fr auto;
      align-items: center;
      gap: 8px;
      padding: 6px;
      border: 1px solid var(--border);
      border-radius: 4px;
      background: white;
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
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      min-height: 22px;
      padding: 2px 7px;
      border: 1px solid var(--border);
      border-radius: 999px;
      background: #fff;
      font-size: 12px;
    }
    .json-preview {
      width: 100%;
      min-height: 190px;
      resize: vertical;
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 8px;
      font: 12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      color: var(--text);
      background: #fbfcfd;
    }
    @media (max-width: 1080px) {
      main { grid-template-columns: 280px minmax(0, 1fr); }
      .inspector {
        grid-column: 1 / -1;
        border-left: 0;
        border-top: 1px solid var(--border);
      }
    }
    @media (max-width: 760px) {
      main { grid-template-columns: 1fr; }
      aside {
        border-right: 0;
        border-bottom: 1px solid var(--border);
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>OCE SDD Scenario Designer</h1>
    <div class="row">
      <a href="viewer.html" id="viewerLink">Viewer</a>
      <div class="muted" id="status">Loading...</div>
    </div>
  </header>
  <main>
    <aside>
      <p class="metric"><strong>Scene</strong></p>
      <select id="sceneSelect"></select>
      <div class="section">
        <p class="metric"><strong>Mode</strong></p>
        <div class="mode-grid">
          <button data-mode="tracks" class="active">Tracks</button>
          <button data-mode="start">Start Zone</button>
          <button data-mode="target">Target Zones</button>
          <button data-mode="goal">Robot Goal</button>
        </div>
      </div>
      <div class="section">
        <p class="metric"><strong>Scene Stats</strong></p>
        <div id="sceneStats" class="metric muted"></div>
      </div>
      <div class="section">
        <p class="metric"><strong>Track Classes</strong></p>
        <div id="classFilters"></div>
      </div>
      <div class="section">
        <p class="metric"><strong>Display</strong></p>
        <label class="metric"><input id="showAllPaths" type="checkbox" checked> Full paths</label>
        <label class="metric"><input id="showStaticPolygons" type="checkbox" checked> Static polygons</label>
      </div>
    </aside>
    <section class="viewer">
      <div class="controls">
        <div class="row">
          <button id="prevScene">Prev Scene</button>
          <button id="nextScene">Next Scene</button>
          <button id="finishPolygon">Finish Polygon</button>
          <button id="undoVertex">Undo Vertex</button>
          <button id="clearDraft">Clear Draft</button>
          <span id="pointerLabel" class="muted"></span>
        </div>
        <div class="row">
          <button id="selectVisible">Select Visible Tracks</button>
          <button id="clearTracks">Clear Track Selection</button>
          <button id="clearStart">Clear Start</button>
          <button id="clearGoal">Clear Goal</button>
          <button id="clearZones">Clear Target Zones</button>
        </div>
        <div class="row">
          <button id="prevFrame">Back</button>
          <button id="playPause">Play</button>
          <button id="nextFrame">Forward</button>
          <span id="frameLabel" class="muted"></span>
        </div>
        <div class="row">
          <input id="frameSlider" type="range" min="0" max="0" value="0" style="flex:1 1 280px">
        </div>
      </div>
      <div class="canvas-wrap">
        <canvas id="sceneCanvas"></canvas>
      </div>
    </section>
    <section class="inspector">
      <p class="metric"><strong>Scenario Design</strong></p>
      <input id="scenarioName" type="text" placeholder="Scenario name">
      <div class="section">
        <p class="metric">
          <span class="pill"><span class="swatch" style="background:var(--start)"></span>Start</span>
          <span id="startStatus" class="muted"></span>
        </p>
        <p class="metric">
          <span class="pill"><span class="swatch" style="background:var(--goal)"></span>Robot Goal</span>
          <span id="goalStatus" class="muted"></span>
        </p>
        <p class="metric">
          <span class="pill"><span class="swatch" style="background:var(--target)"></span>Target Zones</span>
          <span id="zoneStatus" class="muted"></span>
        </p>
      </div>
      <div class="section">
        <p class="metric"><strong>Selected Tracks</strong> <span id="trackStatus" class="muted"></span></p>
        <div id="selectedTracks" class="track-list"></div>
      </div>
      <div class="section">
        <p class="metric"><strong>Target Goal Zones</strong></p>
        <div id="zoneList" class="zone-list"></div>
      </div>
      <div class="section">
        <div class="row">
          <button id="exportJson">Export JSON</button>
          <button id="copyJson">Copy JSON</button>
          <button id="downloadSceneData">Scene Data</button>
        </div>
        <p class="metric muted" id="saveStatus"></p>
        <textarea id="jsonPreview" class="json-preview" readonly></textarea>
      </div>
      <div class="section">
        <p class="metric"><strong>Import Design</strong></p>
        <input id="importJson" type="file" accept="application/json,.json">
      </div>
    </section>
  </main>
  <script>
    const classColors = [
      "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
      "#56B4E9", "#F0E442", "#332288", "#88CCEE", "#44AA99",
      "#117733", "#999933", "#DDCC77", "#CC6677", "#882255",
      "#AA4499", "#777777", "#000000", "#6699CC", "#661100"
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
      mode: "tracks",
      design: null,
      classVisible: new Map(),
      showAllPaths: true,
      showStaticPolygons: true,
      frame: 0,
      playing: false,
      draft: [],
      dragStart: null,
      dragRect: null,
      dragMode: null,
      suppressNextClick: false,
      hoveredTrackId: null,
      currentFit: null
    };
    const el = {
      status: document.getElementById("status"),
      viewerLink: document.getElementById("viewerLink"),
      sceneSelect: document.getElementById("sceneSelect"),
      sceneStats: document.getElementById("sceneStats"),
      classFilters: document.getElementById("classFilters"),
      prevScene: document.getElementById("prevScene"),
      nextScene: document.getElementById("nextScene"),
      finishPolygon: document.getElementById("finishPolygon"),
      undoVertex: document.getElementById("undoVertex"),
      clearDraft: document.getElementById("clearDraft"),
      pointerLabel: document.getElementById("pointerLabel"),
      selectVisible: document.getElementById("selectVisible"),
      clearTracks: document.getElementById("clearTracks"),
      clearStart: document.getElementById("clearStart"),
      clearGoal: document.getElementById("clearGoal"),
      clearZones: document.getElementById("clearZones"),
      prevFrame: document.getElementById("prevFrame"),
      playPause: document.getElementById("playPause"),
      nextFrame: document.getElementById("nextFrame"),
      frameSlider: document.getElementById("frameSlider"),
      frameLabel: document.getElementById("frameLabel"),
      showAllPaths: document.getElementById("showAllPaths"),
      showStaticPolygons: document.getElementById("showStaticPolygons"),
      canvas: document.getElementById("sceneCanvas"),
      scenarioName: document.getElementById("scenarioName"),
      startStatus: document.getElementById("startStatus"),
      goalStatus: document.getElementById("goalStatus"),
      zoneStatus: document.getElementById("zoneStatus"),
      trackStatus: document.getElementById("trackStatus"),
      selectedTracks: document.getElementById("selectedTracks"),
      zoneList: document.getElementById("zoneList"),
      exportJson: document.getElementById("exportJson"),
      copyJson: document.getElementById("copyJson"),
      downloadSceneData: document.getElementById("downloadSceneData"),
      saveStatus: document.getElementById("saveStatus"),
      jsonPreview: document.getElementById("jsonPreview"),
      importJson: document.getElementById("importJson")
    };
    const ctx = el.canvas.getContext("2d");

    async function init() {
      const response = await fetch("scenes/index.json");
      state.index = await response.json();
      populateSceneSelect();
      bindControls();
      await loadScene(0);
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
      document.querySelectorAll("[data-mode]").forEach(button => {
        button.addEventListener("click", () => setMode(button.dataset.mode));
      });
      el.finishPolygon.addEventListener("click", finishTargetZone);
      el.undoVertex.addEventListener("click", () => {
        state.draft.pop();
        draw();
        updateInspector();
      });
      el.clearDraft.addEventListener("click", () => {
        state.draft = [];
        state.dragRect = null;
        draw();
        updateInspector();
      });
      el.selectVisible.addEventListener("click", selectVisibleTracks);
      el.clearTracks.addEventListener("click", () => {
        state.design.selectedTrackIds = [];
        persistDesign();
      });
      el.clearStart.addEventListener("click", () => {
        state.design.robotStartZone = null;
        persistDesign();
      });
      el.clearGoal.addEventListener("click", () => {
        state.design.robotGoal = null;
        persistDesign();
      });
      el.clearZones.addEventListener("click", () => {
        state.design.targetGoalZones = [];
        state.draft = [];
        persistDesign();
      });
      el.prevFrame.addEventListener("click", () => setFrame(state.frame - 1));
      el.nextFrame.addEventListener("click", () => setFrame(state.frame + 1));
      el.playPause.addEventListener("click", () => {
        state.playing = !state.playing;
        el.playPause.textContent = state.playing ? "Pause" : "Play";
      });
      el.frameSlider.addEventListener("input", event => setFrame(Number(event.target.value)));
      el.showAllPaths.addEventListener("change", event => {
        state.showAllPaths = event.target.checked;
        draw();
      });
      el.showStaticPolygons.addEventListener("change", event => {
        state.showStaticPolygons = event.target.checked;
        draw();
      });
      el.scenarioName.addEventListener("input", () => {
        state.design.name = el.scenarioName.value;
        persistDesign();
      });
      el.exportJson.addEventListener("click", () => downloadJson(currentDesignPayload(), designFilename()));
      el.copyJson.addEventListener("click", copyDesignJson);
      el.downloadSceneData.addEventListener("click", () => downloadJson(state.scene, `scene_${padSceneId(state.scene.sceneId)}_data.json`));
      el.importJson.addEventListener("change", importDesignFile);
      el.canvas.addEventListener("mousedown", onCanvasMouseDown);
      el.canvas.addEventListener("mousemove", onCanvasMouseMove);
      el.canvas.addEventListener("mouseup", onCanvasMouseUp);
      el.canvas.addEventListener("mouseleave", () => {
        state.hoveredTrackId = null;
        el.pointerLabel.textContent = "";
        draw();
      });
      el.canvas.addEventListener("click", onCanvasClick);
      window.addEventListener("resize", draw);
      requestAnimationFrame(animationTick);
    }

    async function loadScene(sceneIndex) {
      saveDesignToLocalStorage();
      state.sceneIndex = sceneIndex;
      el.sceneSelect.value = String(sceneIndex);
      const indexRecord = state.index.scenes[sceneIndex];
      el.status.textContent = `Loading scene ${indexRecord.sceneId}...`;
      const sceneResponse = await fetch(indexRecord.sceneJson);
      state.scene = await sceneResponse.json();
      state.image = await loadImage(state.scene.image);
      state.design = loadDesignFromLocalStorage() || emptyDesign();
      state.frame = 0;
      state.playing = false;
      state.draft = [];
      state.dragStart = null;
      state.dragRect = null;
      state.dragMode = null;
      state.suppressNextClick = false;
      state.classVisible = new Map(classesForScene().map(item => [item.key, true]));
      el.frameSlider.max = String(state.scene.maxFrame);
      el.frameSlider.value = "0";
      el.playPause.textContent = "Play";
      el.viewerLink.href = `viewer.html?scene=${state.scene.sceneId}`;
      el.scenarioName.value = state.design.name;
      el.status.textContent = `Scene ${state.scene.sceneId}`;
      renderSidebar(indexRecord);
      updateInspector();
      updateFrameLabel();
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

    function emptyDesign() {
      return {
        name: `sdd_scene_${padSceneId(state.scene.sceneId)}_scenario`,
        robotStartZone: null,
        targetGoalZones: [],
        selectedTrackIds: [],
        robotGoal: null
      };
    }

    function renderSidebar(indexRecord) {
      el.sceneStats.innerHTML = [
        `Image: ${indexRecord.imageWidth} x ${indexRecord.imageHeight}`,
        `Tracks: ${indexRecord.trackCount}`,
        `Polygons: ${state.scene.polygons.length}`,
        `Max points: ${indexRecord.maxTrackLength}`,
        `Median points: ${indexRecord.medianTrackLength.toFixed(1)}`
      ].join("<br>");

      el.classFilters.innerHTML = "";
      classesForScene().forEach((classItem, i) => {
        const label = document.createElement("label");
        label.className = "metric";
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
        label.append(checkbox, swatch, ` ${classItem.label} (${count})`);
        el.classFilters.appendChild(label);
      });
    }

    function setMode(mode) {
      state.mode = mode;
      state.draft = [];
      state.dragRect = null;
      state.dragMode = null;
      state.suppressNextClick = false;
      document.querySelectorAll("[data-mode]").forEach(button => {
        button.classList.toggle("active", button.dataset.mode === mode);
      });
      draw();
      updateInspector();
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

    function visibleTracks() {
      return state.scene.tracks.filter(track => state.classVisible.get(classKeyForTrack(track)));
    }

    function selectedTrackSet() {
      return new Set(state.design.selectedTrackIds.map(Number));
    }

    function selectVisibleTracks() {
      const ids = new Set(state.design.selectedTrackIds.map(Number));
      visibleTracks().forEach(track => ids.add(Number(track.trackId)));
      state.design.selectedTrackIds = [...ids].sort((a, b) => a - b);
      persistDesign();
    }

    function selectTracksInsideRect(rect) {
      const bounds = rectBounds(rect);
      const ids = new Set(state.design.selectedTrackIds.map(Number));
      visibleTracks().forEach(track => {
        const point = trackPointAtFrame(track, state.frame);
        if (!point) return;
        if (
          point[0] >= bounds.minX &&
          point[0] <= bounds.maxX &&
          point[1] >= bounds.minY &&
          point[1] <= bounds.maxY
        ) {
          ids.add(Number(track.trackId));
        }
      });
      state.design.selectedTrackIds = [...ids].sort((a, b) => a - b);
      persistDesign();
    }

    function setFrame(frame) {
      const maxFrame = state.scene ? state.scene.maxFrame : 0;
      state.frame = Math.max(0, Math.min(maxFrame, Math.round(frame)));
      el.frameSlider.value = String(state.frame);
      updateFrameLabel();
      draw();
    }

    function updateFrameLabel() {
      if (!state.scene) return;
      const stride = state.scene.tracks[0]?.pointStride || 1;
      el.frameLabel.textContent = `Point ${state.frame} / ${state.scene.maxFrame} (source index approx. ${state.frame * stride})`;
    }

    function animationTick() {
      if (state.playing && state.scene) {
        const next = state.frame >= state.scene.maxFrame ? 0 : state.frame + 1;
        setFrame(next);
      }
      setTimeout(() => requestAnimationFrame(animationTick), 75);
    }

    function onCanvasMouseDown(event) {
      if (state.mode !== "start" && state.mode !== "tracks") return;
      const point = canvasToScene(event);
      if (!point) return;
      state.dragStart = point;
      state.dragRect = rectangleFromPoints(point, point);
      state.dragMode = state.mode === "tracks" ? "track_select" : "start";
      state.suppressNextClick = false;
    }

    function onCanvasMouseMove(event) {
      const point = canvasToScene(event);
      if (!point) return;
      const nearest = nearestTrack(point);
      state.hoveredTrackId = nearest && nearest.distance <= 10 / Math.max(state.currentFit?.scale || 1, 1e-9)
        ? nearest.track.trackId
        : null;
      el.pointerLabel.textContent = `${point.x.toFixed(1)}, ${point.y.toFixed(1)}${state.hoveredTrackId !== null ? ` | track ${state.hoveredTrackId}` : ""}`;
      if (state.dragStart && (state.dragMode === "start" || state.dragMode === "track_select")) {
        state.dragRect = rectangleFromPoints(state.dragStart, point);
      }
      draw();
    }

    function onCanvasMouseUp(event) {
      if (!state.dragStart || !state.dragMode) return;
      const point = canvasToScene(event);
      if (point) {
        const rect = rectangleFromPoints(state.dragStart, point);
        if (state.dragMode === "start" && polygonArea(rect.vertices) > 4) {
          state.design.robotStartZone = rect;
          persistDesign();
        } else if (state.dragMode === "track_select" && polygonArea(rect.vertices) > 4) {
          selectTracksInsideRect(rect);
          state.suppressNextClick = true;
        }
      }
      state.dragStart = null;
      state.dragRect = null;
      state.dragMode = null;
      draw();
    }

    function onCanvasClick(event) {
      if (state.suppressNextClick) {
        state.suppressNextClick = false;
        return;
      }
      const point = canvasToScene(event);
      if (!point) return;
      if (state.mode === "tracks") {
        const nearest = nearestTrack(point);
        const tolerance = 10 / Math.max(state.currentFit?.scale || 1, 1e-9);
        if (nearest && nearest.distance <= tolerance) {
          toggleTrack(nearest.track.trackId);
        }
      } else if (state.mode === "target") {
        state.draft.push([round3(point.x), round3(point.y)]);
        draw();
        updateInspector();
      } else if (state.mode === "goal") {
        state.design.robotGoal = { x: round3(point.x), y: round3(point.y) };
        persistDesign();
      }
    }

    function toggleTrack(trackId) {
      const id = Number(trackId);
      const ids = new Set(state.design.selectedTrackIds.map(Number));
      if (ids.has(id)) ids.delete(id);
      else ids.add(id);
      state.design.selectedTrackIds = [...ids].sort((a, b) => a - b);
      persistDesign();
    }

    function finishTargetZone() {
      if (state.draft.length < 3) return;
      const zoneId = nextZoneId();
      state.design.targetGoalZones.push({
        id: zoneId,
        name: `target_goal_zone_${zoneId}`,
        vertices: state.draft.map(([x, y]) => [round3(x), round3(y)])
      });
      state.draft = [];
      persistDesign();
    }

    function nextZoneId() {
      const ids = state.design.targetGoalZones.map(zone => Number(zone.id)).filter(Number.isFinite);
      return ids.length ? Math.max(...ids) + 1 : 1;
    }

    function rectangleFromPoints(a, b) {
      const minX = Math.min(a.x, b.x);
      const maxX = Math.max(a.x, b.x);
      const minY = Math.min(a.y, b.y);
      const maxY = Math.max(a.y, b.y);
      return {
        type: "rectangle",
        vertices: [
          [round3(minX), round3(minY)],
          [round3(maxX), round3(minY)],
          [round3(maxX), round3(maxY)],
          [round3(minX), round3(maxY)]
        ]
      };
    }

    function rectBounds(rect) {
      const xs = rect.vertices.map(([x]) => x);
      const ys = rect.vertices.map(([, y]) => y);
      return {
        minX: Math.min(...xs),
        maxX: Math.max(...xs),
        minY: Math.min(...ys),
        maxY: Math.max(...ys)
      };
    }

    function trackPointAtFrame(track, frame) {
      if (!track.points.length) return null;
      return track.points[Math.min(Math.max(0, Math.round(frame)), track.points.length - 1)];
    }

    function nearestTrack(point) {
      let best = null;
      visibleTracks().forEach(track => {
        for (let i = 0; i < track.points.length; i += 1) {
          const [x, y] = track.points[i];
          const dx = point.x - x;
          const dy = point.y - y;
          const distance = Math.hypot(dx, dy);
          if (!best || distance < best.distance) {
            best = { track, distance };
          }
        }
      });
      return best;
    }

    function canvasToScene(event) {
      if (!state.currentFit) return null;
      const rect = el.canvas.getBoundingClientRect();
      const x = (event.clientX - rect.left - state.currentFit.x) / state.currentFit.scale;
      const y = (event.clientY - rect.top - state.currentFit.y) / state.currentFit.scale;
      if (x < 0 || y < 0 || x > state.scene.imageWidth || y > state.scene.imageHeight) return null;
      return { x, y };
    }

    function draw() {
      if (!state.scene || !state.image) return;
      const wrap = el.canvas.parentElement;
      const dpr = window.devicePixelRatio || 1;
      el.canvas.width = Math.max(1, Math.floor(wrap.clientWidth * dpr));
      el.canvas.height = Math.max(1, Math.floor(wrap.clientHeight * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const fit = fitRect(wrap.clientWidth, wrap.clientHeight, state.scene.imageWidth, state.scene.imageHeight);
      state.currentFit = fit;
      ctx.clearRect(0, 0, wrap.clientWidth, wrap.clientHeight);
      ctx.save();
      ctx.translate(fit.x, fit.y);
      ctx.scale(fit.scale, fit.scale);
      ctx.drawImage(state.image, 0, 0);
      if (state.showStaticPolygons) drawStaticPolygons();
      drawTracks();
      drawScenarioOverlays();
      ctx.restore();
    }

    function fitRect(dstW, dstH, srcW, srcH) {
      const scale = Math.min(dstW / srcW, dstH / srcH);
      return {
        scale,
        x: (dstW - srcW * scale) / 2,
        y: (dstH - srcH * scale) / 2
      };
    }

    function drawStaticPolygons() {
      state.scene.polygons.forEach(poly => {
        drawClosedPolygon(poly.vertices, {
          fill: polygonColors[poly.polygonClass] || "rgba(90,90,90,0.35)",
          stroke: "rgba(0,0,0,0.45)",
          width: 1.25
        });
      });
    }

    function drawTracks() {
      const classes = classesForScene();
      const selected = selectedTrackSet();
      state.scene.tracks.forEach(track => {
        const classKey = classKeyForTrack(track);
        if (!state.classVisible.get(classKey)) return;
        const classIndex = classes.findIndex(item => item.key === classKey);
        const baseColor = colorForClass(classKey, classIndex);
        const isSelected = selected.has(Number(track.trackId));
        const isHovered = Number(state.hoveredTrackId) === Number(track.trackId);
        const stroke = isSelected ? "#111111" : baseColor;
        const currentIndex = Math.min(state.frame, track.points.length - 1);
        if (currentIndex < 0) return;
        if (state.showAllPaths) {
          drawPath(track.points, stroke, isSelected ? 2.3 : 1.2, isSelected ? 0.28 : 0.16);
        }
        drawPath(track.points, stroke, isSelected ? 3.1 : 1.8, isSelected ? 0.86 : 0.48, currentIndex);
        const end = track.points[currentIndex];
        ctx.beginPath();
        ctx.arc(end[0], end[1], isSelected || isHovered ? 5.5 : 3.8, 0, Math.PI * 2);
        ctx.fillStyle = isSelected ? "#111111" : baseColor;
        ctx.strokeStyle = isSelected ? "#ffffff" : "rgba(255,255,255,0.85)";
        ctx.lineWidth = isSelected || isHovered ? 1.8 : 1.1;
        ctx.fill();
        ctx.stroke();
      });
    }

    function drawScenarioOverlays() {
      if (state.design.robotStartZone) {
        drawClosedPolygon(state.design.robotStartZone.vertices, {
          fill: "rgba(204, 121, 167, 0.26)",
          stroke: "#cc79a7",
          width: 3
        });
      }
      if (state.dragRect) {
        const isTrackSelection = state.dragMode === "track_select";
        drawClosedPolygon(state.dragRect.vertices, {
          fill: isTrackSelection ? "rgba(31, 119, 180, 0.18)" : "rgba(204, 121, 167, 0.18)",
          stroke: isTrackSelection ? "#1f77b4" : "#cc79a7",
          width: 2
        });
      }
      state.design.targetGoalZones.forEach(zone => {
        drawClosedPolygon(zone.vertices, {
          fill: "rgba(0, 158, 115, 0.24)",
          stroke: "#009e73",
          width: 2.5
        });
        drawTextAtCentroid(zone.vertices, String(zone.id), "#006b4f");
      });
      if (state.draft.length) {
        drawOpenPolygon(state.draft, "#009e73");
      }
      if (state.design.robotGoal) {
        drawGoalMarker(state.design.robotGoal.x, state.design.robotGoal.y);
      }
    }

    function drawClosedPolygon(vertices, style) {
      if (!vertices || vertices.length < 2) return;
      ctx.beginPath();
      vertices.forEach(([x, y], i) => {
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.closePath();
      if (style.fill) {
        ctx.fillStyle = style.fill;
        ctx.fill();
      }
      ctx.strokeStyle = style.stroke;
      ctx.lineWidth = style.width;
      ctx.stroke();
    }

    function drawOpenPolygon(vertices, color) {
      ctx.beginPath();
      vertices.forEach(([x, y], i) => {
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.setLineDash([6, 5]);
      ctx.stroke();
      ctx.setLineDash([]);
      vertices.forEach(([x, y]) => {
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      });
    }

    function drawPath(points, strokeStyle, lineWidth, alpha, endIndex = null) {
      if (!points.length) return;
      const end = endIndex === null ? points.length - 1 : Math.min(endIndex, points.length - 1);
      if (end < 0) return;
      ctx.save();
      ctx.globalAlpha = alpha;
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
      ctx.restore();
    }

    function drawGoalMarker(x, y) {
      const radius = 8;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(213, 94, 0, 0.25)";
      ctx.strokeStyle = "#d55e00";
      ctx.lineWidth = 3;
      ctx.fill();
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x - radius - 5, y);
      ctx.lineTo(x + radius + 5, y);
      ctx.moveTo(x, y - radius - 5);
      ctx.lineTo(x, y + radius + 5);
      ctx.stroke();
    }

    function drawTextAtCentroid(vertices, text, color) {
      const centroid = polygonCentroid(vertices);
      ctx.fillStyle = color;
      ctx.font = "bold 16px Arial, Helvetica, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(text, centroid.x, centroid.y);
    }

    function updateInspector() {
      if (!state.design) return;
      const start = state.design.robotStartZone;
      const goal = state.design.robotGoal;
      el.startStatus.textContent = start ? `${Math.round(polygonArea(start.vertices))} px²` : "not set";
      el.goalStatus.textContent = goal ? `${goal.x.toFixed(1)}, ${goal.y.toFixed(1)}` : "not set";
      el.zoneStatus.textContent = `${state.design.targetGoalZones.length}`;
      el.trackStatus.textContent = `${state.design.selectedTrackIds.length}`;
      el.selectedTracks.innerHTML = "";
      const selected = new Set(state.design.selectedTrackIds.map(Number));
      state.scene.tracks
        .filter(track => selected.has(Number(track.trackId)))
        .sort((a, b) => Number(a.trackId) - Number(b.trackId))
        .forEach(track => {
          const item = document.createElement("div");
          item.className = "track-item";
          const swatch = document.createElement("span");
          swatch.className = "swatch";
          swatch.style.background = colorForClass(classKeyForTrack(track));
          const remove = document.createElement("button");
          remove.textContent = "Remove";
          remove.addEventListener("click", () => toggleTrack(track.trackId));
          item.append(swatch, `Track ${track.trackId}`, remove);
          el.selectedTracks.appendChild(item);
        });
      el.zoneList.innerHTML = "";
      state.design.targetGoalZones.forEach(zone => {
        const item = document.createElement("div");
        item.className = "zone-item";
        const swatch = document.createElement("span");
        swatch.className = "swatch";
        swatch.style.background = "var(--target)";
        const remove = document.createElement("button");
        remove.textContent = "Remove";
        remove.addEventListener("click", () => {
          state.design.targetGoalZones = state.design.targetGoalZones.filter(item => item.id !== zone.id);
          persistDesign();
        });
        item.append(swatch, `${zone.name} (${zone.vertices.length})`, remove);
        el.zoneList.appendChild(item);
      });
      el.jsonPreview.value = JSON.stringify(currentDesignPayload(), null, 2);
    }

    function currentDesignPayload() {
      const transform = transformForCurrentScene();

      function transformPoint(p) {
        if (!transform) return p;
        const [ox, oy] = transform.source_origin_xy;
        const scale = transform.source_to_scene_scale;
        return [
          round3((p[0] - ox) * scale),
          round3((oy - p[1]) * scale)
        ];
      }

      function transformZone(zone) {
        if (!zone || !zone.vertices) return zone;
        return {
          ...zone,
          vertices: zone.vertices.map(transformPoint)
        };
      }

      function transformGoal(goal) {
        if (!goal) return goal;
        const [x, y] = transformPoint([goal.x, goal.y]);
        return { x, y };
      }

      return {
        schema: "oce_sdd_scenario_design.v1",
        name: state.design.name,
        sceneId: state.scene.sceneId,
        coordinateFrame: transform ? "normalized_scene_coordinates" : "constrained_sdd_image_pixels",
        image: {
          path: state.scene.image,
          width: state.scene.imageWidth,
          height: state.scene.imageHeight
        },
        timeMode: state.scene.timeMode,
        pointStride: state.scene.tracks[0]?.pointStride || 1,
        robotStartZone: transformZone(state.design.robotStartZone),
        targetGoalZones: state.design.targetGoalZones.map(transformZone),
        selectedTrackIds: state.design.selectedTrackIds,
        robotGoal: transformGoal(state.design.robotGoal),
        sourceSceneJson: `scenes/scene_${padSceneId(state.scene.sceneId)}.json`,
        updatedAt: new Date().toISOString(),
        createdBy: "oce_sdd.scenario_designer"
      };
    }

    function transformForCurrentScene() {
      const sceneRecord = state.index.scenes.find(s => s.sceneId === state.scene.sceneId);
      if (sceneRecord?.transform) return sceneRecord.transform;
      if (!state.scene?.imageWidth || !state.scene?.imageHeight) return null;
      return {
        source_origin_xy: [0, Number(state.scene.imageHeight)],
        source_to_scene_scale: 10 / Math.max(Number(state.scene.imageWidth), Number(state.scene.imageHeight))
      };
    }

    function persistDesign() {
      saveDesignToLocalStorage();
      updateInspector();
      draw();
      el.saveStatus.textContent = "Saved in this browser";
    }

    function saveDesignToLocalStorage() {
      if (!state.scene || !state.design) return;
      localStorage.setItem(storageKey(), JSON.stringify(state.design));
    }

    function loadDesignFromLocalStorage() {
      if (!state.scene) return null;
      const raw = localStorage.getItem(storageKey());
      if (!raw) return null;
      try {
        return normalizeDesign(JSON.parse(raw));
      } catch {
        return null;
      }
    }

    function normalizeDesign(raw) {
      const fallback = emptyDesign();
      return {
        name: String(raw.name || fallback.name),
        robotStartZone: raw.robotStartZone || null,
        targetGoalZones: Array.isArray(raw.targetGoalZones) ? raw.targetGoalZones : [],
        selectedTrackIds: Array.isArray(raw.selectedTrackIds)
          ? raw.selectedTrackIds.map(Number).filter(Number.isFinite).sort((a, b) => a - b)
          : [],
        robotGoal: raw.robotGoal || null
      };
    }

    function storageKey() {
      return `oce_sdd_scenario_design_scene_${padSceneId(state.scene.sceneId)}`;
    }

    function copyDesignJson() {
      const text = JSON.stringify(currentDesignPayload(), null, 2);
      navigator.clipboard.writeText(text).then(() => {
        el.saveStatus.textContent = "Copied JSON";
      }).catch(() => {
        el.saveStatus.textContent = "Clipboard unavailable";
      });
    }

    function importDesignFile(event) {
      const file = event.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const payload = JSON.parse(String(reader.result));
          if (Number(payload.sceneId) !== Number(state.scene.sceneId)) {
            el.saveStatus.textContent = `Import scene ${payload.sceneId} does not match scene ${state.scene.sceneId}`;
            return;
          }
          state.design = normalizeDesign(payload);
          el.scenarioName.value = state.design.name;
          persistDesign();
          el.saveStatus.textContent = "Imported design";
        } catch (error) {
          el.saveStatus.textContent = `Import failed: ${error.message}`;
        }
      };
      reader.readAsText(file);
      event.target.value = "";
    }

    function downloadJson(data, filename) {
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      el.saveStatus.textContent = `Exported ${filename}`;
    }

    function designFilename() {
      const safeName = state.design.name.replace(/[^a-zA-Z0-9_.-]+/g, "_").replace(/^_+|_+$/g, "");
      return `${safeName || `scene_${padSceneId(state.scene.sceneId)}_scenario`}.json`;
    }

    function padSceneId(sceneId) {
      return String(sceneId).padStart(3, "0");
    }

    function round3(value) {
      return Math.round(value * 1000) / 1000;
    }

    function polygonArea(vertices) {
      if (!vertices || vertices.length < 3) return 0;
      let area = 0;
      for (let i = 0; i < vertices.length; i += 1) {
        const [x1, y1] = vertices[i];
        const [x2, y2] = vertices[(i + 1) % vertices.length];
        area += x1 * y2 - x2 * y1;
      }
      return Math.abs(area) / 2;
    }

    function polygonCentroid(vertices) {
      if (!vertices.length) return { x: 0, y: 0 };
      const sum = vertices.reduce((acc, [x, y]) => ({ x: acc.x + x, y: acc.y + y }), { x: 0, y: 0 });
      return { x: sum.x / vertices.length, y: sum.y / vertices.length };
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
