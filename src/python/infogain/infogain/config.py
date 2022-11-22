DISTANCE_TOLERANCE = 0.005
SIMULATION_SPEED = 100

MAX_SIMULATION_TIME = 1000
MAX_SERVICED_TASKS = 1000
NUM_ACTORS = 1
ACTOR_SPEED = 1
LAMBDA = 0.8
SERVICE_TIME = 0
TICK_TIME = 0.01
BETA = 0.712    # constant for TSP length

EPSILON = 0.000001

EGO_X_OFFSET = -0.15
EGO_Y_OFFSET = -0.5

LANE_WIDTH = 0.06
ROAD_WIDTH = 2 * LANE_WIDTH

DEFAULT_POLICY_NAME = "random"
DEFAULT_GENERATOR_NAME = "uniform"
GENERATOR_ARGS = {
    'min': 0,
    'max': 1,
    'seed': None,
    'dim': 2,
    'mix': 0.5,
}

SHOW_SIM = False

OPPONENT_RISK_SPEED = 0.4
OPPONENT_CAR_SPEED = 0.4
OPPONENT_PEDESTRIAN_SPEED = 0.15

COLLISION_DISTANCE = 0.05

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
GRID_HEIGHT = 0.2
GRID_WIDTH = 1.0
GRID_RESOLUTION = 0.05
GRID_ORIGIN_Y_OFFSET = -GRID_HEIGHT / 2
GRID_ORIGIN_X_OFFSET = EGO_X_OFFSET

