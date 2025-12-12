# IntelliH1 ROS2 Package

ROS2 integration for the IntelliH1 cognitive humanoid framework using "Wrap & Bridge" strategy.

## Architecture

The system follows a modular node-based architecture:

```
┌─────────────────────────────────────────────────────────┐
│              🧠 brain_node (1Hz planning)               │
│         LLM + A* Path Planning + Navigation             │
│              Publishes: /cmd_vel, /path                 │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│            🚶 rl_node (50Hz control)                    │
│         Unitree RL Locomotion Controller                │
│       Subscribes: /cmd_vel, /joint_states               │
│            Publishes: /motor_cmd                        │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│           🎮 sim_node (500Hz physics)                   │
│              MuJoCo Physics Engine                      │
│     Subscribes: /motor_cmd                              │
│     Publishes: /joint_states, /tf                       │
└─────────────────────────────────────────────────────────┘
```

## Nodes

### 1. `sim_node` - MuJoCo Simulation
- **Frequency**: 500Hz (physics timestep)
- **Function**: Runs MuJoCo physics simulation
- **Publishes**:
  - `/joint_states` (sensor_msgs/JointState) - Joint positions and velocities
  - `/tf` (tf2_msgs/TFMessage) - Robot base_link transform
- **Subscribes**:
  - `/motor_cmd` (std_msgs/Float64MultiArray) - Motor torques

### 2. `rl_node` - RL Locomotion Controller
- **Frequency**: 50Hz (RL inference)
- **Function**: Executes Unitree RL walking policy
- **Publishes**:
  - `/motor_cmd` (std_msgs/Float64MultiArray) - Motor torques
- **Subscribes**:
  - `/joint_states` (sensor_msgs/JointState) - Robot state
  - `/cmd_vel` (geometry_msgs/Twist) - Velocity commands

### 3. `brain_node` - High-Level Planning
- **Frequency**: 1Hz planning, 10Hz control
- **Function**: LLM-based command parsing, A* path planning, navigation
- **Publishes**:
  - `/cmd_vel` (geometry_msgs/Twist) - Velocity commands
  - `/path` (nav_msgs/Path) - Planned path for visualization
  - `/brain_status` (std_msgs/String) - Status messages
- **Subscribes**:
  - `/navigation_command` (std_msgs/String) - Natural language commands
  - `/tf` - Robot position

## Installation

### Prerequisites

On macOS, use RoboStack with Conda (recommended):

```bash
# Install Miniforge/Mambaforge if not already installed
brew install miniforge

# Create ROS2 environment
conda create -n ros_env python=3.10
conda activate ros_env

# Add RoboStack channels
conda config --add channels conda-forge
conda config --add channels robostack-staging

# Install ROS2 Humble
conda install ros-humble-desktop colcon-common-extensions

# Install Python dependencies
pip install mujoco groq python-dotenv torch numpy scipy
```

### Build Package

```bash
cd ~/ros2_ws
colcon build --packages-select intelli_h1_ros
source install/setup.bash
```

## Usage

### Launch All Nodes

```bash
# Source ROS2 workspace
source ~/ros2_ws/install/setup.bash

# Launch with default parameters
ros2 launch intelli_h1_ros demo_launch.py

# Launch with custom speed
ros2 launch intelli_h1_ros demo_launch.py max_speed:=1.5

# Launch without RViz
ros2 launch intelli_h1_ros demo_launch.py use_rviz:=false
```

### Send Navigation Commands

```bash
# Send navigation command via topic
ros2 topic pub --once /navigation_command std_msgs/String "data: 'go to kitchen'"

# Available locations:
# - kitchen (5.0, 3.0)
# - bedroom (-3.0, 6.0)
# - living_room (0.0, -4.0)
```

### Monitor Topics

```bash
# View joint states
ros2 topic echo /joint_states

# View velocity commands
ros2 topic echo /cmd_vel

# View planned path
ros2 topic echo /path

# View brain status
ros2 topic echo /brain_status
```

### Visualize in RViz

```bash
# Launch RViz separately
rviz2 -d ~/ros2_ws/install/intelli_h1_ros/share/intelli_h1_ros/config/intelli_h1.rviz

# Set Fixed Frame to "map"
# Add displays:
# - TF (to see coordinate frames)
# - Path (topic: /path)
# - RobotModel (requires URDF)
```

## Parameters

### sim_node
- `model_path`: Path to MuJoCo XML model (default: h1/scene_enhanced.xml)
- `update_rate`: Physics update frequency in Hz (default: 500.0)

### rl_node
- `robot_type`: Robot model (h1, h1_2, or g1) (default: h1)
- `max_speed`: Maximum walking speed in m/s (default: 1.0)
- `update_rate`: Control frequency in Hz (default: 50.0)

### brain_node
- `planning_rate`: LLM planning frequency in Hz (default: 1.0)
- `control_rate`: Navigation control frequency in Hz (default: 10.0)
- `goal_tolerance`: Distance threshold to goal in meters (default: 1.2)
- `max_speed`: Maximum speed for navigation (default: 1.0)

## TF Tree

```
map
 └─ base_link (published by sim_node)
     └─ [other robot links from robot_state_publisher]
```

## Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/joint_states` | sensor_msgs/JointState | Robot joint positions/velocities |
| `/tf` | tf2_msgs/TFMessage | Transform tree |
| `/cmd_vel` | geometry_msgs/Twist | Velocity commands |
| `/motor_cmd` | std_msgs/Float64MultiArray | Motor torques |
| `/path` | nav_msgs/Path | Planned path |
| `/navigation_command` | std_msgs/String | Natural language commands |
| `/brain_status` | std_msgs/String | Status messages |

## Troubleshooting

### MuJoCo model not found
Make sure the `extern/unitree_rl_gym` submodule is properly initialized:
```bash
cd /path/to/IntelliH1
git submodule update --init --recursive
```

### TF not available
The sim_node needs to be running and publishing transforms. Check:
```bash
ros2 topic echo /tf
```

### RL policy not loading
Ensure the pre-trained policy file exists:
```bash
ls extern/unitree_rl_gym/deploy/pre_train/h1/motion.pt
```

### LLM commands not working
Set your Groq API key:
```bash
export GROQ_API_KEY=your_key_here
# Or create .env file in project root
```

## Development

### Node Communication Flow

1. User sends command to `/navigation_command`
2. `brain_node` parses command using LLM
3. `brain_node` plans path with A* 
4. `brain_node` publishes `/cmd_vel` to follow path
5. `rl_node` receives `/cmd_vel` and computes motor commands
6. `rl_node` publishes `/motor_cmd` 
7. `sim_node` applies torques and steps physics
8. `sim_node` publishes updated `/joint_states` and `/tf`
9. Loop continues until goal reached

### Adding Custom Messages

If you need custom message types:

1. Create `msg/` directory in package
2. Define `.msg` files
3. Update `package.xml` and `CMakeLists.txt` (if needed)
4. Rebuild package

## Known Limitations

- Robot URDF visualization not yet integrated (robot_state_publisher disabled)
- Perception/LIDAR not yet bridged to ROS2
- No dynamic obstacle avoidance in ROS2 nodes
- Manual coordinate frame conversion between MuJoCo and ROS needed

## Future Work

- [ ] Add URDF publishing for robot visualization
- [ ] Bridge LIDAR/perception data to sensor_msgs/LaserScan
- [ ] Add dynamic obstacle detection
- [ ] Implement action server interface for navigation
- [ ] Add parameter file support (YAML configs)
- [ ] Create Gazebo simulation as alternative to MuJoCo
- [ ] Add teleop control interface

## License

MIT License - See LICENSE file in root directory
