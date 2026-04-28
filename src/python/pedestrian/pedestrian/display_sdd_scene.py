from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from config import GENERATOR_ARGS, SCREEN_HEIGHT, SCREEN_MARGIN, SCREEN_WIDTH, TICK_TIME
from datasources import load_scenario
from simulation import Simulation


def main() -> None:
    args = parse_args()
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pygame.init()
    pygame.font.init()
    surface = pygame.Surface((args.width, args.height), pygame.SRCALPHA)

    generator_args = dict(GENERATOR_ARGS)
    generator_args["seed"] = args.seed

    scenario = load_scenario(
        "sdd",
        sdd_processed_root=args.sdd_processed_root,
        sdd_scene_id=args.sdd_scene_id,
        sdd_actor_scale_percentile=args.sdd_actor_scale_percentile,
    )
    sim = Simulation(
        screen=surface,
        scenario=scenario,
        generator_name="uniform",
        generator_args=generator_args,
        tick_time=args.tick_time,
        margin=args.margin,
        screen_width=args.width,
        screen_height=args.height,
        ego_start=args.ego_start,
        ego_goal=args.ego_goal,
        enable_scan=False,
    )
    sim.render_display_only(str(output_path))
    print(
        f"Wrote {output_path} with {len(scenario.tracks)} tracks and "
        f"{len(scenario.static_polygons)} static polygons"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an SDD scene through the pedestrian Simulation display path."
    )
    parser.add_argument("--sdd-processed-root", default="outputs/sdd_processed")
    parser.add_argument("--sdd-scene-id", type=int, required=True)
    parser.add_argument(
        "--sdd-actor-scale-percentile",
        type=float,
        default=75.0,
        help="Dataset percentile for shared SDD actor scale. Use -1 for local scale.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG. Defaults to outputs/pedestrian_sdd_display/scene_<id>.png.",
    )
    parser.add_argument("--width", default=SCREEN_WIDTH, type=int)
    parser.add_argument("--height", default=SCREEN_HEIGHT, type=int)
    parser.add_argument("--margin", default=SCREEN_MARGIN, type=int)
    parser.add_argument("--tick-time", default=TICK_TIME, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--ego-start", nargs=2, type=float, default=[0.10, 0.10])
    parser.add_argument("--ego-goal", nargs=2, type=float, default=[0.90, 0.90])
    args = parser.parse_args()
    if args.out is None:
        args.out = f"outputs/pedestrian_sdd_display/scene_{args.sdd_scene_id:03d}.png"
    return args


if __name__ == "__main__":
    main()
