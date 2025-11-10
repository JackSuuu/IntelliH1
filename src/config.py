import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Groq API setup from environment variables
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("Please set GROQ_API_KEY in your .env file")

# Simulation parameters
SIMULATION_TIMESTEP = 0.002
GRAVITY = [0, 0, -9.81]

# Car parameters
CAR_MASS = 5.0
WHEEL_MASS = 0.2
JOINT_DAMPING = 1.0
FRICTION_LOSS = 0.05

# Perception
LIDAR_RANGE = 10.0
LIDAR_FOV = 180  # degrees
LIDAR_SAMPLES = 90
OBSTACLE_DETECTION_THRESHOLD = 2.0
