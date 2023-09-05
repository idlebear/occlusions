from numpy import pi as PI

DISTANCE_TOLERANCE = 0.005
SIMULATION_SPEED = 100

MAX_TIMESTEPS = 1e6

NUM_ACTORS = 1
NUM_INSTANCES = 12
NUM_EPOCHS = 10

LAMBDA = 0.8
SERVICE_TIME = 0
TICK_TIME = 0.05
BETA = 0.712  # constant for TSP length
PLANNING_HORIZON = 50
WAYPOINT_INTERVAL = 5

EPSILON = 0.000001

WINDOW_SIZE = 50
EGO_X_OFFSET = -WINDOW_SIZE // 2
EGO_Y_OFFSET = -WINDOW_SIZE // 2

LANE_WIDTH = 4
ROAD_WIDTH = 2 * LANE_WIDTH

DEFAULT_POLICY_NAME = "random"
DEFAULT_GENERATOR_NAME = "uniform"
GENERATOR_ARGS = {
    "min": 0,
    "max": 1,
    "seed": None,
    "dim": 2,
    "mix": 0.5,
}

SHOW_SIM = False

ACTOR_SPEED = 6.0
OPPONENT_CAR_SPEED = 6.0
MAX_CAR_SPEED = 9.0
OPPONENT_PEDESTRIAN_SPEED = 1.0
MAX_PEDESTRIAN_SPEED = 2.0

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
SCREEN_MARGIN = 120

SCREEN_OUTLINE_COLOUR = (0, 0, 0, 255)
SCREEN_BACKGROUND_COLOUR = (255, 255, 255, 255)
ACTOR_COMPLETE_PATH_COLOUR = (220, 220, 220, 255)
ACTOR_COLOUR = (100, 100, 240, 255)
ACTOR_PATH_COLOUR = (40, 40, 40, 255)
ACTOR_PATH_WIDTH = 3

STATUS_FONT_COLOUR = (0, 0, 0, 255)
STATUS_FONT_SIZE = 32

ROAD_MARKING_COLOUR = (255, 255, 0, 255)
ROAD_COLOUR = (59, 79, 35, 255)

STATUS_Y_SIZE = 100
STATUS_YMARGIN = 8
STATUS_X_SIZE = 300
STATUS_XMARGIN = 16

# Observations are made/delivered on a occupancy grid
GRID_HEIGHT = 50
GRID_WIDTH = 50
GRID_RESOLUTION = 0.5
GRID_ORIGIN_Y_OFFSET = 0
GRID_ORIGIN_X_OFFSET = 0
GRID_SIZE = int(GRID_WIDTH / GRID_RESOLUTION)

# Fake scanner parameters
SCAN_RANGE = 30
SCAN_RAYS = 800
SCAN_RESOLUTION = 0.05
SCAN_FOV = 2 * PI
SCAN_START_ANGLE = -SCAN_FOV / 2
SCAN_ANGLE_INCREMENT = SCAN_FOV / SCAN_RAYS
SCAN_STDDEV_RANGE = GRID_RESOLUTION

# Constructing a Car lined street (the parkade)
CAR_OFFSET = 1.65
CAR_SPACING = 7.5
CAR_WIGGLE = 1.0
CAR_RATIO = 1.9 / 4.8
CAR_SCALE = 4.8
CAR_ROTATION = PI / 12
