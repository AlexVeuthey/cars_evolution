# display parameters
WIN_SIZE_X = 1200
WIN_SIZE_Y = 600

# moving the car
MAX_STEERING = 5.0
MAX_ACCELERATION = 5.0
MAX_SPEED = 10.0
SPEED_DAMPING = 0.1
TOLERANCE = 1.0

# car basic attributes
FOV = 40.0
MAX_DISTANCE = 100.0
LENGTH = 40.0
WIDTH = 20.0

# neural network
N_INPUT = 3
N_OUTPUT = 2
N_HIDDEN = 10

# genetics
N_DRIVERS = 20
N_BEST_TO_KEEP = int(N_DRIVERS / 100 * 20)
N_COMBINED = int(N_DRIVERS / 100 * 80)
