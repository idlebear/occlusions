from numpy import pi as PI

# --------------------------------------------------------------------------------
# DEBUG flags
# --------------------------------------------------------------------------------
DEBUG_INFORMATION_GAIN = False
DEBUG_VISIBILITY = False
DEBUG_TRAJECTORIES = False
DEBUG_MPC = False

# --------------------------------------------------------------------------------
# Figure Constants
# --------------------------------------------------------------------------------
FIG_LIDAR_MAP = 1
FIG_OCCUPANCY_GRID = 2
FIG_DYNAMIC_OCCUPANCY_GRID = 3
FIG_TRAJECTORIES = 4
FIG_VISIBILITY = 5
FIG_VISIBILITY_COSTMAP = 6
FIG_MPC = 7

# --------------------------------------------------------------------------------
# LIDAR Constants
# --------------------------------------------------------------------------------
LIDAR_RANGE = 25.0
LIDAR_RAYS = 2650.0
LIDAR_INCREMENT = (PI * 2.0) / LIDAR_RAYS

# --------------------------------------------------------------------------------
# Grid Constants
# --------------------------------------------------------------------------------
# Observations are made/delivered on a occupancy grid
GRID_WIDTH = 2 * LIDAR_RANGE
GRID_HEIGHT = GRID_WIDTH
GRID_RESOLUTION = 0.2
GRID_SIZE = int(GRID_WIDTH / GRID_RESOLUTION)
GRID_ORIGIN_Y_OFFSET = 0
GRID_ORIGIN_X_OFFSET = 0
OCCUPANCY_THRESHOLD = 0.2


EPSILON = 0.00000001
DISTANCE_TOLERANCE = 0.005
TARGET_TOLERANCE = 1.0
SIMULATION_SPEED = 100
TICK_TIME = 0.01
MAX_TIMESTEPS = 1e5
NUM_INSTANCES = 1

MAX_SIMULATION_TIME = 1000
NUM_ACTORS = 1
BETA = 0.712  # constant for TSP length
ACTOR_SPEED = 1.0

ROBOT_SPEED = 0.5
ROBOT_ACCELERATION = 0.5
CONTROL_LIMITS = [2.0, PI / 3.0]
CONTROL_VARIATION_LIMITS = [2.0, PI / 2.0]

OCC_PROB = 0.65  # default occupancy probability

DEFAULT_POLICY_NAME = "random"

# arguments for the random number generator
DEFAULT_GENERATOR_NAME = "uniform"
GENERATOR_ARGS = {
    "min": 0,
    "max": 1,
    "seed": None,
    "dim": 2,
    "mix": 0.5,
}

SHOW_SIM = False

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
SCREEN_MARGIN = 120
DEFAULT_DISPLAY_SIZE = 20

SCREEN_OUTLINE_COLOUR = (0, 0, 0, 255)
SCREEN_BACKGROUND_COLOUR = (255, 255, 255, 255)
ACTOR_COMPLETE_PATH_COLOUR = (220, 220, 220, 255)
ACTOR_COLOUR = (100, 100, 240, 255)
ACTOR_PATH_COLOUR = (40, 40, 40, 255)
ACTOR_PATH_WIDTH = 3
STATUS_FONT_COLOUR = (0, 0, 0, 255)
STATUS_FONT_SIZE = 32
EGO_TRAJECTORY_COLOUR = [100, 100, 255]  # list to allow unpack and combination with alpha


STATUS_Y_SIZE = 100
STATUS_YMARGIN = 8
STATUS_X_SIZE = 300
STATUS_XMARGIN = 16


# Fake scanner parameters
SCAN_RANGE = 30
SCAN_RAYS = 800
SCAN_RESOLUTION = 0.05
SCAN_FOV = 2 * PI
SCAN_START_ANGLE = -SCAN_FOV / 2
SCAN_ANGLE_INCREMENT = SCAN_FOV / SCAN_RAYS
SCAN_STDDEV_RANGE = GRID_RESOLUTION / 4

# MPPI parameter defaults
ALPHA_FACTOR = 1.0  # allow the costmap to forget about entries over time
DISCOUNT_FACTOR = 1.0  # discount cost over the horizon
X_WEIGHT = 15.0
Y_WEIGHT = 15.0
V_WEIGHT = 10.0
THETA_WEIGHT = 10.0
A_WEIGHT = 0.1
DELTA_WEIGHT = 0.01
DEFAULT_ACCELERATION = 4.0
DEFAULT_METHOD_WEIGHT = 100.0
DEFAULT_LAMBDA = 200.0

FINAL_X_WEIGHT = 1.0
FINAL_Y_WEIGHT = 1.0
FINAL_V_WEIGHT = 0.0
FINAL_THETA_WEIGHT = 0.0
