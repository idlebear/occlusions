from copy import deepcopy
from dataclasses import replace
from importlib import import_module
from math import sqrt, exp
from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pygame
import random as random_module
from random import random, expovariate, seed

import visilibity as vis
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from Grid.OccupancyGrid import OccupancyGrid
from util.xkcdColour import XKCD_ColourPicker
from datasources import load_eth_scenario, load_scenario

# local functions/imports
from Actor import DeliveryBot, Pedestrian, Vehicle, STATE
from config import *

faux_scan = None
visibility_from_region = None
POLYCHECK_IMPORT_ERROR = None


def load_polycheck():
    global faux_scan, visibility_from_region, POLYCHECK_IMPORT_ERROR
    if faux_scan is not None or POLYCHECK_IMPORT_ERROR is not None:
        return
    try:
        from polycheck import faux_scan as imported_faux_scan
        from polycheck import visibility_from_region as imported_visibility_from_region
    except Exception as exc:
        POLYCHECK_IMPORT_ERROR = exc
        return
    faux_scan = imported_faux_scan
    visibility_from_region = imported_visibility_from_region


DEBUG = 0
FORECAST_COUNT = 5
FORECAST_INTERVAL = 0.1

# rewards - a high penalty for colliding with anything, a small penalty for
# deviating from the desired velocity, a slightly smaller one for deviating from
# the desired Y position, and a positive reward for moving forward
REWARD_COLLISION = -100000  # note that this includes leaving the road surface!
REWARD_DEVIATION_V = -100.0
REWARD_DEVIATION_Y = -100.0
REWARD_FORWARD_MOTION = 0.01  # a small positive reward for not dying


MAX_V = 10.0  # define a max v for scaling the observation output to keep it in the
# range [0,1]


def scenario_grid_metadata(scenario):
    if scenario is None:
        return {}
    metadata = getattr(scenario, "metadata", {}) or {}
    grid = metadata.get("grid") or metadata.get("simulation_grid") or {}
    return grid if isinstance(grid, dict) else {}


def finite_positive(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) and value > 0 else None


def resolve_scenario_grid(scenario):
    grid = scenario_grid_metadata(scenario)
    width = finite_positive(grid.get("width"))
    height = finite_positive(grid.get("height"))
    resolution = finite_positive(grid.get("resolution"))

    if scenario is not None:
        display_diff = finite_positive(getattr(scenario, "display_diff", None))
        metadata = getattr(scenario, "metadata", {}) or {}
        scale = finite_positive(metadata.get("scene_scale"))
        if width is None and display_diff is not None and scale is not None:
            width = display_diff
        if height is None and display_diff is not None and scale is not None:
            height = display_diff
        if resolution is None and scale is not None:
            resolution = GRID_RESOLUTION * scale

    return {
        "width": width or GRID_WIDTH,
        "height": height or GRID_HEIGHT,
        "resolution": resolution or GRID_RESOLUTION,
    }


def get_location(origin, location):
    return [location[0] - origin[0], 1 - (location[1] - origin[1])]


class Window:
    def __init__(
        self,
        screen,
        screen_width,
        screen_height,
        margin,
        display_origin=(0, 0),
        display_size=10.0,
    ):
        self.screen = screen

        self.sim_time_text = pygame.font.SysFont("dejavuserif", 15)
        self.elapsed_time_text = pygame.font.SysFont("dejavuserif", 10)
        self.status_font = pygame.font.SysFont("roboto", STATUS_FONT_SIZE)

        self._xmargin = margin * 0.5
        self._ymargin = margin * 0.5
        self._screen_width = screen_width
        self._screen_height = screen_height
        self._env_size = min(screen_height, screen_width) - margin
        self._scale = self._env_size / display_size  # pixels per meter
        self._display_size = display_size
        self._border_offset = 10
        self._origin = display_origin

        self.tmp_screen = pygame.Surface(
            (self.screen.get_width(), self.screen.get_height()), flags=pygame.SRCALPHA
        )

    # def _get_location_on_screen(self, origin, location):
    #     return [
    #         int(self._xmargin + (location[0]-origin[0] - EGO_X_OFFSET)*self._env_size),
    #         int(self._ymargin + (location[1]-origin[1] - EGO_Y_OFFSET)*self._env_size)
    #     ]

    def clear(self):
        self.screen.fill(SCREEN_BACKGROUND_COLOUR)

    def get_drawing_scale(self):
        return self._scale

    def draw_polyline(self, points, colour, width=2):
        start = points[0]
        for end in points[1:]:
            self.draw_line(start, end, colour, width)
            start = end

    def draw_line(self, start, end, colour, width=2):
        sx = self._xmargin + int((start[0] - self._origin[0]) * self._scale)
        sy = self._ymargin + int((start[1] - self._origin[1]) * self._scale)
        ex = self._xmargin + int((end[0] - self._origin[0]) * self._scale)
        ey = self._ymargin + int((end[1] - self._origin[1]) * self._scale)

        pygame.draw.line(
            self.screen, color=colour, start_pos=(sx, sy), end_pos=(ex, ey), width=width
        )

    def draw_circle(self, center, colour, radius=2):
        cx = self._xmargin + int((center[0] - self._origin[0]) * self._scale)
        cy = self._ymargin + int((center[1] - self._origin[1]) * self._scale)
        radius = int(radius * self._scale)

        pygame.draw.circle(self.screen, color=colour, center=(cx, cy), radius=radius)

    def draw_rect(self, colour, center, height, width=None):
        if width is None:
            width = height

        pygame.draw.rect(
            self.screen,
            colour,
            (
                self._xmargin
                + int((center[0] - width / 2.0 - self._origin[0]) * self._scale),
                self._ymargin
                + int((center[1] - height / 2.0 - self._origin[1]) * self._scale),
                int(width * self._scale),
                int(height * self._scale),
            ),
        )

    def draw_polygon(
        self,
        outline_colour,
        fill_colour,
        points,
        width=ACTOR_PATH_WIDTH,
        use_transparency=False,
    ):
        points = [
            [
                self._xmargin + int((x - self._origin[0]) * self._scale),
                self._ymargin + int((y - self._origin[1]) * self._scale),
            ]
            for x, y in points
        ]

        if use_transparency:
            self.tmp_screen.fill((0, 0, 0, 0))
            if fill_colour is not None:
                pygame.draw.polygon(self.tmp_screen, fill_colour, points, 0)
            pygame.draw.polygon(self.tmp_screen, outline_colour, points, width=width)
            self.screen.blit(self.tmp_screen, (0, 0))
        else:
            if fill_colour is not None:
                pygame.draw.polygon(self.screen, fill_colour, points, 0)
            pygame.draw.polygon(self.screen, outline_colour, points, width=width)

    # Quick image rotation
    #   https://stackoverflow.com/questions/4183208/how-do-i-rotate-an-image-around-its-center-using-pygame
    def draw_image(self, image, center, orientation):
        center = (
            self._xmargin + int((center[0] - self._origin[0]) * self._scale),
            self._ymargin + int((center[1] - self._origin[1]) * self._scale),
        )
        rotated_image = pygame.transform.rotate(image, np.rad2deg(orientation))
        new_rect = rotated_image.get_rect(center=image.get_rect(center=center).center)
        self.screen.blit(rotated_image, new_rect)

    def draw_status(self, collisions, sim_time):
        #  draw the limits of the environment
        pygame.draw.rect(
            self.screen,
            SCREEN_OUTLINE_COLOUR,
            (
                self._xmargin - self._border_offset,
                self._ymargin - self._border_offset,
                self._env_size + self._border_offset * 2,
                self._env_size + self._border_offset * 2,
            ),
            2,
        )

        collisions_str = f"Collisions: {collisions}"
        time_str = f"Sim Time: {sim_time:.4f}"

        text_width, text_height = self.status_font.size(collisions_str)
        time_text_width, time_text_height = self.status_font.size(time_str)

        x_avg_offset = self._env_size + self._xmargin - text_width - STATUS_XMARGIN * 2
        y_avg_offset = self._env_size + self._ymargin - text_height - STATUS_YMARGIN

        pygame.draw.rect(
            self.screen,
            SCREEN_BACKGROUND_COLOUR,
            (
                x_avg_offset - STATUS_XMARGIN,
                y_avg_offset - STATUS_YMARGIN,
                text_width + STATUS_XMARGIN * 2,
                text_height + STATUS_YMARGIN,
            ),
            0,
        )
        pygame.draw.rect(
            self.screen,
            SCREEN_OUTLINE_COLOUR,
            (
                x_avg_offset - STATUS_XMARGIN,
                y_avg_offset - STATUS_YMARGIN,
                text_width + STATUS_XMARGIN * 2,
                text_height + STATUS_YMARGIN,
            ),
            2,
        )
        text = self.status_font.render(collisions_str, False, STATUS_FONT_COLOUR)
        self.screen.blit(text, (x_avg_offset + STATUS_XMARGIN / 3, y_avg_offset))

        pygame.draw.rect(
            self.screen,
            SCREEN_BACKGROUND_COLOUR,
            (
                self._xmargin + STATUS_XMARGIN / 2,
                self._xmargin + STATUS_YMARGIN,
                time_text_width + STATUS_XMARGIN,
                time_text_height + STATUS_YMARGIN,
            ),
            0,
        )

        text = self.status_font.render(time_str, False, STATUS_FONT_COLOUR)
        self.screen.blit(
            text, (self._xmargin + STATUS_XMARGIN, self._ymargin + STATUS_YMARGIN)
        )

    def save_screen(self, path):
        pygame.image.save(self.screen, path)


class Simulation:
    def __init__(
        self,
        generator_name="uniform",
        generator_args=None,
        num_actors=1,
        tracks=None,
        data_source=None,
        scenario=None,
        sdd_processed_root="outputs/sdd_processed",
        sdd_scene_id=None,
        limit_tracks=None,
        ego_start=None,
        ego_heading=0,
        ego_goal=None,
        pois_lambda=0.01,
        screen=None,
        speed=ACTOR_SPEED,
        margin=SCREEN_MARGIN,
        screen_width=SCREEN_WIDTH,
        screen_height=SCREEN_HEIGHT,
        tick_time=TICK_TIME,
        record_data=False,
        enable_scan=True,
    ):
        self.num_actors = num_actors
        self.actor_target_speed = speed
        self.pois_lambda = pois_lambda
        self.scenario = scenario
        if self.scenario is None and data_source is not None:
            self.scenario = load_scenario(
                data_source,
                tracks=tracks,
                sdd_processed_root=sdd_processed_root,
                sdd_scene_id=sdd_scene_id,
            )
        elif self.scenario is None and tracks is not None:
            self.scenario = load_eth_scenario(tracks)

        if self.scenario is not None:
            self.scenario = self._limited_scenario_tracks(
                self.scenario,
                limit_tracks,
                seed=generator_args.get("seed") if generator_args else None,
            )

        if self.scenario is not None:
            self.track_data = self.scenario.tracks
            self.display_offset = self.scenario.display_offset
            self.display_diff = self.scenario.display_diff
            self.static_polygons = self.scenario.static_polygons
            self.scene_scale = self.scenario.metadata.get("scene_scale", 1.0)
        else:
            self.track_data = None
            self.tracks = None
            self.display_diff = DEFAULT_DISPLAY_SIZE  # meters
            self.display_offset = [0, 0]
            self.static_polygons = []
            self.scene_scale = 1.0

        self.record_data = record_data
        self.tick_time = self._resolve_tick_time(tick_time)
        self.enable_scan = enable_scan
        self.debug_tick_timing = False
        self.last_tick_timing = {}
        self.last_scan_timing = {}
        self.last_observation_timing = {}
        grid = resolve_scenario_grid(self.scenario)
        self.grid_width = grid["width"]
        self.grid_height = grid["height"]
        self.grid_resolution = grid["resolution"]

        if screen is not None or record_data:
            self.window = Window(
                screen=screen,
                screen_width=screen_width,
                screen_height=screen_height,
                margin=margin,
                display_size=self.display_diff,
                display_origin=self.display_offset,
            )
            self.image_scale = self.window.get_drawing_scale()
        else:
            self.window = None
            self.image_scale = 1.0

        # load the draw method
        self.load_generator(
            generator_name=generator_name, generator_args=generator_args
        )

        self.observation_shape = [
            int(np.ceil(self.grid_height / self.grid_resolution)),
            int(np.ceil(self.grid_width / self.grid_resolution)),
        ]

        self.maps = None
        self.ig_images = None
        self.ig_val_images = None

        # Construct the occupancy grid
        self.obs = OccupancyGrid(
            dim=max(self.grid_width, self.grid_height),
            resolution=self.grid_resolution,
            origin=(0, 0),
        )

        self.ego_start = ego_start
        self.ego_goal = ego_goal
        self.ego_heading = ego_heading

        colours = XKCD_ColourPicker()
        self.colours = colours.values(30, "red")

        # if DEBUG:
        #     self.maps = []
        #     num_plots = FORECAST_COUNT + 1
        #     self.map_fig, self.map_ax = plt.subplots(num_plots, 1, figsize=(5, 15))
        #     H,W,D = self.grid.get_grid_size()
        #     for i in range(num_plots):
        #         self.maps.append( self.map_ax[i].imshow(np.zeros([H, W, 3], dtype=np.uint8)) )

        #     plt.show(block=False)

        self.reset()

    def _limited_scenario_tracks(self, scenario, limit_tracks, seed=None):
        if limit_tracks is None:
            return scenario

        limit_tracks = int(limit_tracks)
        if limit_tracks < 0:
            raise ValueError("--limit_tracks must be >= 0")

        track_items = list(scenario.tracks.items())
        raw_track_count = len(track_items)
        if limit_tracks >= raw_track_count:
            selected_items = track_items
        elif limit_tracks == 0:
            selected_items = []
        else:
            rng = random_module.Random(seed) if seed is not None else random_module
            selected_items = rng.sample(track_items, limit_tracks)

        selected_items = sorted(selected_items, key=lambda item: str(item[0]))
        metadata = dict(scenario.metadata)
        metadata["track_limit"] = limit_tracks
        metadata["track_limit_raw_count"] = raw_track_count
        metadata["track_limit_selected_count"] = len(selected_items)
        metadata["track_limit_selected_ids"] = [
            track_id for track_id, _ in selected_items
        ]

        print(
            "Loaded "
            f"{len(selected_items)} / {raw_track_count} tracks"
            f" for scenario {scenario.name}"
        )
        return replace(
            scenario,
            tracks=dict(selected_items),
            metadata=metadata,
        )

    def _resolve_tick_time(self, requested_tick_time):
        if self.scenario is None or self.scenario.data_source != "sdd":
            return requested_tick_time

        dt = self.scenario.metadata.get("dt")
        if dt is None:
            return requested_tick_time

        dt = float(dt)
        if dt <= 0:
            raise ValueError(f"Invalid SDD scene dt: {dt}")
        return dt

    def load_tracks(self, tracks):
        scenario = load_eth_scenario(tracks)
        return scenario.tracks, scenario.display_offset, scenario.display_diff

    def _random_unit(self):
        return float(np.asarray(self.generator.random(n=1)).reshape(-1)[0])

    def _sample_axis_value(self, spec, offset):
        if isinstance(spec, (float, int, np.floating, np.integer)):
            return self.display_diff * float(spec) + offset, False

        lo, hi = spec[0], spec[1]
        value = (
            offset
            + (float(lo) + (float(hi) - float(lo)) * self._random_unit())
            * self.display_diff
        )
        return value, True

    def _sample_location(self, spec):
        if spec is None:
            return (
                np.asarray(self.generator.random(n=2), dtype=float) * self.display_diff
                + self.display_offset
            ), True

        x, x_random = self._sample_axis_value(spec[0], self.display_offset[0])
        y, y_random = self._sample_axis_value(spec[1], self.display_offset[1])
        return np.asarray([x, y], dtype=float), bool(x_random or y_random)

    def _blocking_polygon_union(self):
        polygons = []
        for points in self.blocking_static_polygon_points():
            polygon = Polygon(points)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if not polygon.is_empty:
                polygons.append(polygon)
        if not polygons:
            return None
        return unary_union(polygons)

    def _is_free_start_location(self, point, blocking_union):
        if blocking_union is None:
            return True
        radius = ROBOT_RADIUS * self.scene_scale + MIN_SEPARATION * self.scene_scale
        return not blocking_union.buffer(radius).covers(
            Point(float(point[0]), float(point[1]))
        )

    def _ego_static_collision(self):
        if not self.static_polygons:
            return None

        ego_polygon = Polygon(self.ego.get_poly())
        if not ego_polygon.is_valid:
            ego_polygon = ego_polygon.buffer(0)
        if ego_polygon.is_empty:
            return None

        for static_polygon in self.static_polygons:
            if not static_polygon.blocking or len(static_polygon.points) < 3:
                continue
            polygon = Polygon(static_polygon.points)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if not polygon.is_empty and ego_polygon.intersects(polygon):
                return static_polygon
        return None

    def _sample_free_location(self, spec, label, max_attempts=250):
        blocking_union = self._blocking_polygon_union()
        candidate, is_random = self._sample_location(spec)
        if not is_random or self._is_free_start_location(candidate, blocking_union):
            return candidate

        for _ in range(max_attempts - 1):
            candidate, _ = self._sample_location(spec)
            if self._is_free_start_location(candidate, blocking_union):
                return candidate

        raise RuntimeError(
            f"Unable to sample a collision-free random {label} after {max_attempts} attempts."
        )

    def reset(self):
        # reset the random number generator
        self.generator.reset()

        sx, sy = self._sample_free_location(self.ego_start, "ego start")
        gx, gy = self._sample_free_location(self.ego_goal, "ego goal")

        self.ego = DeliveryBot(
            id=0,
            x=np.array([sx, sy, 0, self.ego_heading, 0]),
            goal=[gx, gy],
            colour="red",
            outline_colour="darkred",
            resolution=self.grid_resolution,
            image_name="robot",
            image_scale=self.image_scale,
            size_scale=self.scene_scale,
        )
        self.ego.set_visible(True)

        self.actor_list = []
        if self.track_data is not None:
            self.tracks = deepcopy(self.track_data)
        else:
            self.tracks = None

        self.sim_time = 0.0
        self.next_time = 0.0
        self.sim_start_time = 0.0

        self.next_agent_x = -np.inf

        self.ticks = 0
        self.collisions = 0

        self.obs.reset()

        return self._get_next_observation(self._calculate_scan())

    def translate_coordinates(self, x, y):
        """
        Translate the x, y coordinates from the percentage of the sim ( range 0.0-1.0) to world coordinates
        """
        return [
            x * self.display_diff + self.display_offset[0],
            y * self.display_diff + self.display_offset[1],
        ]

    def load_generator(self, generator_name, generator_args):
        # load the generator
        self.generator_name = generator_name
        self.generator_args = generator_args
        gen_mod = import_module("." + self.generator_name, package="generators")
        generator_fn = gen_mod.get_generator_fn()
        self.generator = generator_fn(**self.generator_args)

    ############################################################################
    # Plotting and drawing functions
    ############################################################################

    def _draw_actor(self, actor, draw_extent=True):
        actor_image = actor.get_image()
        if draw_extent:
            # draw collision radius
            self.window.draw_circle(
                actor.x[:2],
                colour=actor.colour,
                radius=actor.get_extent() + MIN_SEPARATION * self.scene_scale,
            )
        if actor_image is not None:
            actor_pos = actor.get_pos()
            # drawing with y inverted reverse the rotation to correct the display
            self.window.draw_image(
                image=actor_image, center=actor_pos, orientation=-actor.x[STATE.THETA]
            )
        else:
            actor_poly = actor.get_poly()
            self.window.draw_polygon(
                outline_colour=actor.outline_colour,
                fill_colour=actor.colour,
                points=actor_poly,
            )

    def _draw_ego(self):
        self._draw_actor(self.ego, draw_extent=False)

    def _draw_static_polygons(self):
        colours = {
            "Building": ((44, 62, 80, 255), (44, 62, 80, 95)),
            "Obstacle": ((146, 43, 33, 255), (146, 43, 33, 95)),
            "Object": ((175, 96, 26, 255), (175, 96, 26, 85)),
            "Offroad": ((88, 120, 92, 255), (88, 120, 92, 70)),
            "Entrance": ((41, 128, 185, 255), None),
        }
        for polygon in self.static_polygons:
            if len(polygon.points) < 3:
                continue
            outline_colour, fill_colour = colours.get(
                polygon.polygon_class,
                ((90, 90, 90, 255), (90, 90, 90, 65)),
            )
            self.window.draw_polygon(
                outline_colour=outline_colour,
                fill_colour=fill_colour,
                points=polygon.points,
                width=2,
                use_transparency=fill_colour is not None,
            )

    def _path_colours(self):
        return [
            (52, 152, 219, 220),
            (231, 76, 60, 220),
            (46, 204, 113, 220),
            (155, 89, 182, 220),
            (241, 196, 15, 220),
            (230, 126, 34, 220),
            (26, 188, 156, 220),
            (149, 165, 166, 220),
            (52, 73, 94, 220),
            (192, 57, 43, 220),
        ]

    def _draw_path(self, path, colours=None, selected_index=None):
        if path is None:
            return
        colours = colours or self._path_colours()
        if type(path) == list:
            for i, p in enumerate(path):
                colour = colours[i % len(colours)]
                radius = 0.05
                width = 2
                if selected_index is not None and i == selected_index:
                    colour = (0, 180, 80, 255)
                    radius = 0.075
                    width = 4
                points = [pos[:2] for pos in zip(p.x, p.y)]
                if len(points) >= 2:
                    self.window.draw_polyline(points, colour=colour, width=width)
                point_stride = max(1, len(points) // 12)
                for pos in points[::point_stride]:
                    self.window.draw_circle(pos[:2], colour=colour, radius=radius)
        else:
            points = [pos[:2] for pos in zip(path.x, path.y)]
            if len(points) >= 2:
                self.window.draw_polyline(points, colour=colours[0], width=2)
            for pos in points:
                self.window.draw_circle(pos[:2], colour=colours[0], radius=0.05)

    def _draw_debug_routes(self, routes, selected_index=None):
        if not routes:
            return
        colours = self._path_colours()
        for index, route in enumerate(routes):
            if route is None or len(route) < 2:
                continue
            selected = selected_index is not None and index == selected_index
            colour = (0, 180, 80, 255) if selected else colours[index % len(colours)]
            width = 4 if selected else 2
            self.window.draw_polyline(route, colour=colour, width=width)
            for point in route:
                self.window.draw_circle(
                    point[:2], colour=colour, radius=0.06 if selected else 0.04
                )

    def _draw_debug_roadmap(self, roadmap):
        if not roadmap:
            return
        edge_colour = (82, 104, 118, 65)
        node_colour = (82, 104, 118, 115)
        skeleton_colour = (52, 73, 94, 150)
        local_anchor_colour = (41, 128, 185, 220)
        global_anchor_colour = (155, 89, 182, 210)
        start_colour = (39, 174, 96, 240)
        goal_colour = (192, 57, 43, 240)

        for edge in roadmap.get("edges", []):
            if len(edge) >= 2:
                self.window.draw_polyline(edge[:2], colour=edge_colour, width=1)
        for node in roadmap.get("nodes", []):
            self.window.draw_circle(node[:2], colour=node_colour, radius=0.025)
        for skeleton in roadmap.get("skeletons", []):
            if skeleton is not None and len(skeleton) >= 2:
                self.window.draw_polyline(
                    skeleton,
                    colour=skeleton_colour,
                    width=2,
                )
        for point in roadmap.get("global_anchors", []):
            self.window.draw_circle(point[:2], colour=global_anchor_colour, radius=0.07)
        for point in roadmap.get("local_anchors", []):
            self.window.draw_circle(point[:2], colour=local_anchor_colour, radius=0.08)
        if roadmap.get("start") is not None:
            self.window.draw_circle(
                roadmap["start"][:2], colour=start_colour, radius=0.09
            )
        if roadmap.get("goal") is not None:
            self.window.draw_circle(
                roadmap["goal"][:2], colour=goal_colour, radius=0.09
            )

    def _draw_nominal_path(self, path):
        if path is None or len(path) < 2:
            return
        colour = (33, 102, 172, 255)
        self.window.draw_polyline(path, colour=colour, width=4)
        for point in path:
            self.window.draw_circle(point[:2], colour=colour, radius=0.07)

    def _draw_status(self):
        self.window.draw_status(self.collisions, self.sim_time)

    def _draw_visibility(self):
        vis_poly = self.calculate_visibility()
        if vis_poly is not None:
            pts = []
            for i in range(vis_poly.n()):
                pt = vis_poly[i]
                pts.append([pt.x(), pt.y()])

            self.window.draw_polygon(
                outline_colour="black",
                fill_colour=None,
                points=pts,
            )

    def draw_polyline(self, points, colour, width=2):
        self.window.draw_polyline(points, colour, width)

    def blocking_static_polygon_points(self):
        return [
            polygon.points
            for polygon in self.static_polygons
            if polygon.blocking and len(polygon.points) >= 3
        ]

    def sensor_blocking_polygons(self):
        polygons = [actor.get_poly() for actor in self.actor_list]
        polygons.extend(self.blocking_static_polygon_points())
        return polygons

    def visibility_polygon_from_points(self, points):
        points = np.asarray(points, dtype=float)
        if len(points) > 1 and np.allclose(points[0], points[-1]):
            points = points[:-1]
        if len(points) < 3:
            return None
        if polygon_signed_area(points) > 0:
            points = points[::-1]
        return vis.Polygon([vis.Point(point[0], point[1]) for point in points])

    def visibility_blocking_polygons(self):
        polygons = []
        for points in self.sensor_blocking_polygons():
            points = np.asarray(points, dtype=float)
            if len(points) > 1 and np.allclose(points[0], points[-1]):
                points = points[:-1]
            if len(points) < 3:
                continue
            polygon = Polygon(points)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if not polygon.is_empty:
                polygons.append(polygon)

        if not polygons:
            return []

        merged = unary_union(polygons)
        if isinstance(merged, Polygon):
            merged_polygons = [merged]
        elif isinstance(merged, MultiPolygon):
            merged_polygons = list(merged.geoms)
        else:
            merged_polygons = [
                geom
                for geom in getattr(merged, "geoms", [])
                if isinstance(geom, Polygon)
            ]

        return [
            np.asarray(polygon.exterior.coords, dtype=float)
            for polygon in merged_polygons
        ]

    ##################################################################################
    # Simulator step functions
    ##################################################################################

    def calculate_visibility(self):
        # calculate the visibility polygon
        shapes = []

        # environment poly is counter clockwise and large enough to be off screen
        ox = self.ego.x[0] + self.display_diff
        oy = self.ego.x[1] + self.display_diff
        shapes.append(
            vis.Polygon(
                [
                    vis.Point(ox, oy),
                    vis.Point(ox - self.display_diff * 2.0, oy),
                    vis.Point(
                        ox - self.display_diff * 2.0, oy - self.display_diff * 2.0
                    ),
                    vis.Point(ox, oy - self.display_diff * 2.0),
                ]
            )
        )

        for points in self.visibility_blocking_polygons():
            polygon = self.visibility_polygon_from_points(points)
            if polygon is not None:
                shapes.append(polygon)

        vis_poly = None
        env = vis.Environment(shapes)
        if env.is_valid(EPSILON):
            observer = vis.Point(self.ego.x[0], self.ego.x[1])
            vis_poly = vis.Visibility_Polygon(observer, env, EPSILON)

        return vis_poly

    def _generate_new_agents(self):
        if self.tracks is not None:
            activated = []
            for id, track in self.tracks.items():
                if self.sim_time >= track[0][2] * self.tick_time:
                    self.actor_list.append(
                        Pedestrian(
                            id=(
                                str(int(id))
                                if type(id) is int or type(id) is float
                                else id
                            ),
                            track=track.copy(),
                            image_name="pedestrian",
                            image_scale=self.image_scale,
                            size_scale=self.scene_scale,
                            dt=self.tick_time,
                        )
                    )
                    activated.append(id)
            for id in activated:
                del self.tracks[id]

        else:
            while len(self.actor_list) < self.num_actors:
                rnd = self.generator.uniform()

                x = self.generator.random(n=2) * self.display_diff + self.display_offset
                goal = (
                    self.generator.random(n=2) * self.display_diff + self.display_offset
                )
                heading = np.arctan2(goal[1] - x[1], goal[0] - x[0])
                v = float(0.2 + self.generator.random() * 1.0)

                actor = Pedestrian(
                    id=self.ticks,
                    x=np.array([x[0], x[1], heading, v, 0]),
                    goal=goal,
                    image_name="pedestrian",
                    image_scale=self.image_scale,
                    size_scale=self.scene_scale,
                )

                self.actor_list.append(actor)

    def _get_next_observation(self, scan_data):
        timing_enabled = bool(getattr(self, "debug_tick_timing", False))
        observation_started = perf_counter() if timing_enabled else None
        section_started = observation_started
        observation_timing = {} if timing_enabled else None

        def mark_observation_timing(name):
            nonlocal section_started
            if not timing_enabled:
                return
            now = perf_counter()
            observation_timing[name] = now - section_started
            section_started = now

        # update the observation
        self.obs.move_origin(self.ego.x[0:2])
        mark_observation_timing("move_origin")
        self.obs.update(
            X=[*self.ego.x[0:2], self.ego.x[STATE.THETA]],
            angle_min=SCAN_START_ANGLE,
            angle_inc=SCAN_ANGLE_INCREMENT,
            ranges=scan_data,
            min_range=0,
            max_range=SCAN_RANGE + 1,
        )
        mark_observation_timing("update")
        observation = self.obs.probabilityMap()
        mark_observation_timing("probability_map")
        if timing_enabled:
            observation_timing["total"] = perf_counter() - observation_started
            self.last_observation_timing = observation_timing
        return observation

    def _get_info(self):
        info = {}
        info["ego"] = self.ego.get_state()
        info["goal"] = self.ego.goal
        info["time"] = self.sim_time

        actor_states = []
        for actor in self.actor_list:
            if actor.distance_to(self.ego.x) <= SCAN_RANGE:
                actor_state = actor.get_state()
                poly = actor.get_poly()
                min_angle = np.pi / 2
                min_pt = None
                for pt in poly:
                    angle = abs(
                        np.arctan((pt[1] - self.ego.x[1]) / (pt[0] - self.ego.x[0]))
                    )
                    if angle < min_angle:
                        min_angle = angle
                        min_pt = pt
                actor_state["min_pt"] = min_pt
                actor_states.append(actor_state)

        info["map"] = None  # self._get_map()
        info["actors"] = actor_states
        info["information_gain"] = None  # self.information_gain
        return info

    def _calculate_scan(self):
        timing_enabled = bool(getattr(self, "debug_tick_timing", False))
        scan_started = perf_counter() if timing_enabled else None
        section_started = scan_started
        scan_timing = {} if timing_enabled else None

        def mark_scan_timing(name):
            nonlocal section_started
            if not timing_enabled:
                return
            now = perf_counter()
            scan_timing[name] = now - section_started
            section_started = now

        def finish_scan_timing(**counts):
            if not timing_enabled:
                return
            scan_timing["total"] = perf_counter() - scan_started
            scan_timing.update(counts)
            self.last_scan_timing = scan_timing

        # create the scan of the environment
        if not self.enable_scan:
            finish_scan_timing(
                polygon_count=0,
                vertex_count=0,
                ray_count=SCAN_RAYS,
                fallback=False,
            )
            return np.full(SCAN_RAYS, SCAN_RANGE + 1, dtype=np.float32)

        load_polycheck()
        mark_scan_timing("load_polycheck")
        if faux_scan is None:
            for actor in self.actor_list:
                actor.set_visible(False)
            mark_scan_timing("visible_update")
            finish_scan_timing(
                polygon_count=0,
                vertex_count=0,
                ray_count=SCAN_RAYS,
                fallback=True,
            )
            return np.full(SCAN_RAYS, SCAN_RANGE + 1, dtype=np.float32)

        # build a list of sensor-blocking polygons in the environment
        polygons = self.sensor_blocking_polygons()
        vertex_count = 0
        for polygon in polygons:
            polygon_array = np.asarray(polygon)
            if polygon_array.ndim == 1:
                vertex_count += polygon_array.size // 2
            else:
                vertex_count += len(polygon_array)
        mark_scan_timing("polygons")

        scan_data, indices = faux_scan(
            polygons,
            origin=self.ego.x[0:2],
            angle_start=SCAN_START_ANGLE + self.ego.x[STATE.THETA],
            angle_inc=SCAN_ANGLE_INCREMENT,
            num_rays=SCAN_RAYS,
            max_range=SCAN_RANGE,
            resolution=SCAN_RESOLUTION,
        )
        mark_scan_timing("faux_scan")

        # Actor polygons are packed before static polygons, so only hit indices
        # in actor_list range correspond to dynamic actors.
        visible_actor_indices = {
            int(index) for index in indices if 0 <= int(index) < len(self.actor_list)
        }
        for index, actor in enumerate(self.actor_list):
            actor.set_visible(index in visible_actor_indices)
        mark_scan_timing("visible_update")

        # clear any rays that didn't hit anything
        scan_data[scan_data == -1] = SCAN_RANGE + 1
        scan_data = scan_data.astype(np.float32)
        mark_scan_timing("postprocess")
        finish_scan_timing(
            polygon_count=len(polygons),
            vertex_count=vertex_count,
            ray_count=SCAN_RAYS,
            fallback=False,
        )
        return scan_data

    ##################################################################################
    # Simulator step functions
    ##################################################################################

    def _tick_actor(self, actor, tick_time):
        """step of simulation for each actor

        Args:
            actor_index (_type_): the index of the actor
        """
        actor.tick()

    def tick(self, action):
        """[summary]"""
        timing_enabled = bool(getattr(self, "debug_tick_timing", False))
        tick_started = perf_counter() if timing_enabled else None
        section_started = tick_started
        tick_timing = {} if timing_enabled else None

        def mark_timing(name):
            nonlocal section_started
            if not timing_enabled:
                return
            now = perf_counter()
            tick_timing[name] = now - section_started
            section_started = now

        # one clock tick for the simulation time
        self.sim_time += self.tick_time
        self.ticks += 1
        mark_timing("advance_time")

        # apply the requested action to the ego vehicle
        self.ego.set_control(action)
        self.ego.tick(self.tick_time)
        mark_timing("ego")

        self._generate_new_agents()
        mark_timing("generate_agents")

        # move everyone
        finished_actors = []
        collisions = 0
        static_collision = self._ego_static_collision()
        if static_collision is not None:
            print(
                "COLLISION DETECTED: Robot collided with static "
                f"{static_collision.polygon_class}"
            )
            collisions += 1
            self.ego.set_collided("red")
        mark_timing("static_collision")

        for i, actor in enumerate(self.actor_list[::-1]):
            actor.tick(self.tick_time)

            if actor.at_goal():
                finished_actors.append(actor)

            # Check for collision with ego vehicle
            # Use bounding box intersection as a fast first check
            ego_bbox = self.ego.get_bounding_box()
            actor_bbox = actor.get_bounding_box()

            # Check if bounding boxes overlap
            bbox_overlap = not (
                ego_bbox[2] < actor_bbox[0]  # ego right < actor left
                or ego_bbox[0] > actor_bbox[2]  # ego left > actor right
                or ego_bbox[3] < actor_bbox[1]  # ego top < actor bottom
                or ego_bbox[1] > actor_bbox[3]  # ego bottom > actor top
            )

            if bbox_overlap:
                # More precise collision check using distance to center
                center_distance = self.ego.distance_to(actor.x)
                collision_threshold = max(
                    (ego_bbox[2] - ego_bbox[0] + ego_bbox[3] - ego_bbox[1])
                    / 4,  # ego "radius"
                    (actor_bbox[2] - actor_bbox[0] + actor_bbox[3] - actor_bbox[1])
                    / 4,  # actor "radius"
                )

                if center_distance < collision_threshold:
                    print(
                        f"COLLISION DETECTED: Robot collided with {type(actor).__name__} at distance {center_distance:.3f}"
                    )
                    # collisions += 1
                    actor.set_collided("red")
                    self.ego.set_collided("red")
        mark_timing("actors")

        # clean up
        for actor in finished_actors:
            self.actor_list.remove(actor)
        mark_timing("cleanup")

        # degrade previous sensor information
        self.obs.decay(0.95)
        mark_timing("decay")

        # update the observation
        self.scan_data = self._calculate_scan()
        mark_timing("scan")
        if timing_enabled:
            for key, value in getattr(self, "last_scan_timing", {}).items():
                tick_timing[f"scan_{key}"] = value
        observation = self._get_next_observation(self.scan_data)
        mark_timing("observation")
        if timing_enabled:
            for key, value in getattr(self, "last_observation_timing", {}).items():
                tick_timing[f"observation_{key}"] = value
        info = self._get_info()
        mark_timing("info")

        # calculate the reward
        reward = 0

        # check if this episode is finished
        done = collisions != 0 or self.ego.at_goal()
        mark_timing("done")

        if timing_enabled:
            tick_timing["total"] = perf_counter() - tick_started
            tick_timing["actor_count"] = len(self.actor_list)
            tick_timing["finished_actors"] = len(finished_actors)
            tick_timing["visible_actors"] = sum(
                1 for actor in self.actor_list if actor.visible
            )
            tick_timing["collisions"] = collisions
            tick_timing["scan_enabled"] = bool(self.enable_scan)
            self.last_tick_timing = tick_timing

        return observation, reward, done, info

    def render(
        self,
        actors=None,
        trajectories=None,
        trajectory_weights=None,
        horizon=1,
        path=None,
        debug_routes=None,
        debug_roadmap=None,
        selected_path_index=None,
        prefix_str=None,
    ):
        if self.window is not None:
            self.window.clear()
            self._draw_static_polygons()

            for actor in self.actor_list:
                try:
                    prediction = actors[actor.id]
                    for traj in prediction:
                        colour = self.colours[actor.serial % len(self.colours)]
                        self.draw_polyline(traj, colour=colour)
                    future = actor.project(horizon)
                    if future is not None:
                        self.draw_polyline(future, colour="black")
                except KeyError:
                    pass
                self._draw_actor(actor)

            self._draw_debug_roadmap(debug_roadmap)
            self._draw_debug_routes(debug_routes, selected_path_index)
            self._draw_path(path, selected_index=selected_path_index)

            self._draw_ego()
            if trajectories is not None:
                trajectory_weights = np.asarray(trajectory_weights, dtype=float)
                if trajectory_weights.size:
                    min_weight = np.min(trajectory_weights)
                    range_weight = np.max(trajectory_weights) - min_weight
                    if range_weight > 0:
                        normalized_weights = (
                            trajectory_weights - min_weight
                        ) / range_weight
                    else:
                        normalized_weights = (
                            np.ones_like(trajectory_weights)
                            if np.max(trajectory_weights) > 0
                            else np.zeros_like(trajectory_weights)
                        )

                    for weight, normalized_weight, trajectory in zip(
                        trajectory_weights, normalized_weights, trajectories
                    ):
                        if weight > 0:
                            self.draw_polyline(
                                trajectory,
                                colour=[
                                    *EGO_TRAJECTORY_COLOUR,
                                    int(200 + normalized_weight * 55.0),
                                ],
                            )
                            # for pos in trajectory:
                            #     self.window.draw_circle(
                            #         pos[:2],
                            #         colour=[*EGO_TRAJECTORY_COLOUR, int(200 + normalized_weight * 55.0)],
                            #         radius=self.ego.get_extent(),
                            #     )
            self._draw_visibility()
            self._draw_status()

            # BUGBUG - make screen saving optional
            # if prefix_str is None:
            #     prefix_str = "pedestrian"
            # self.window.save_screen(f"results/{prefix_str}_{self.ticks:05}.png")

        if DEBUG:
            pass
            # # draw the probability and velocity grid
            # for i, (prob, v) in enumerate(forecast):
            #     map_img = Image.fromarray(np.flipud(((1 - prob) * 255.0).astype(np.uint8))).convert("RGB")
            #     self.maps[i].set_data(map_img)

            # self.map_fig.canvas.draw()
            # self.map_fig.canvas.flush_events()

    def futures(self, timesteps=1):
        # get the future states of the actors
        futures = []
        for actor in self.actor_list:
            future = actor.project(timesteps)
            if future is not None:
                futures.append(future)
        return futures

    def render_display_only(self, path, show_visibility=True):
        if self.window is None:
            raise RuntimeError("Simulation must be constructed with a screen to render")

        self._generate_new_agents()
        self.window.clear()
        self._draw_static_polygons()
        for actor in self.actor_list:
            self._draw_actor(actor)
        self._draw_ego()
        if show_visibility:
            self._draw_visibility()
        self._draw_status()
        self.window.save_screen(path)


def polygon_signed_area(points):
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
