# ROS2 Migration Guide for IntelliH1

This document explains the ROS2 migration using the "Wrap & Bridge" strategy, converting IntelliH1 from a monolithic application to a distributed node-based system.

## Overview

The migration preserves all existing algorithms and wraps them into ROS2 nodes without rewriting core logic. This approach:

- ✅ Maintains existing code quality and testing
- ✅ Enables modular development and debugging
- ✅ Provides standard ROS2 interfaces
- ✅ Allows visualization with RViz2
- ✅ Supports distributed computing

## Architecture Transformation

### Before (Monolithic)
```
test_llm_navigation.py
    ↓
CognitiveController (single process)
    ├─ LLM Planning
    ├─ Perception
    ├─ Path Planning
    └─ RL Controller
    ↓
MuJoCo Simulation
```

### After (ROS2 Node Graph)
```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│ brain_node  │ cmd_vel │   rl_node    │motor_cmd│  sim_node   │
│ (Planning)  ├────────→│ (Controller) ├────────→│ (Physics)   │
│             │         │              │         │             │
└──────┬──────┘         └──────┬───────┘         └──────┬──────┘
       │                       │                        │
       │                       │                        ↓
       │                       │                  /joint_states
       │                       ↓                        │
       ↓                  /cmd_vel                      ↓
    /path                                              /tf
```

## Migration Strategy: "Wrap & Bridge"

### Phase 1: Simulation Node (`sim_node.py`)

**Wrapped Code**: `src/simulation/environment.py`

**What it does**:
- Runs MuJoCo physics engine at 500Hz
- Wraps existing Simulation class
- No changes to physics code

**ROS2 Additions**:
```python
# Publishers
/joint_states (sensor_msgs/JointState)  # Joint angles for RViz
/tf (tf2_msgs/TFMessage)                # Robot pose in world

# Subscribers  
/motor_cmd (std_msgs/Float64MultiArray) # Torques from RL controller

# Timer
500Hz physics update loop
```

**Key Implementation**:
```python
class MuJoCoSimNode(Node):
    def __init__(self):
        # Load MuJoCo model (existing code)
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # ROS2 additions
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(0.002, self.timer_callback)  # 500Hz
    
    def timer_callback(self):
        # Apply motor commands (existing)
        self.data.ctrl[:] = self.motor_commands
        mujoco.mj_step(self.model, self.data)
        
        # Publish state (new)
        self.publish_joint_states()
        self.publish_tf()
```

### Phase 2: RL Controller Node (`rl_node.py`)

**Wrapped Code**: `src/control/unitree_rl_controller.py`

**What it does**:
- Executes Unitree RL walking policy at 50Hz
- Wraps existing UnitreeRLController class
- No changes to RL inference code

**ROS2 Additions**:
```python
# Publishers
/motor_cmd (std_msgs/Float64MultiArray)  # Computed torques

# Subscribers
/joint_states (sensor_msgs/JointState)   # Current robot state
/cmd_vel (geometry_msgs/Twist)           # Velocity commands from brain

# Timer
50Hz control loop
```

**Key Implementation**:
```python
class RLControllerNode(Node):
    def __init__(self):
        # Initialize RL controller (existing code)
        self.controller = UnitreeRLController(
            self.model, self.data, 
            robot_type='h1', max_speed=1.0
        )
        
        # ROS2 additions
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', ...)
        self.motor_cmd_pub = self.create_publisher(Float64MultiArray, '/motor_cmd', ...)
        self.timer = self.create_timer(0.02, self.timer_callback)  # 50Hz
    
    def timer_callback(self):
        # Compute action (existing)
        action = self.controller.compute_action()
        
        # Publish (new)
        msg = Float64MultiArray()
        msg.data = action.tolist()
        self.motor_cmd_pub.publish(msg)
```

### Phase 3: Brain Node (`brain_node.py`)

**Wrapped Code**: 
- `src/llm/navigation_planner.py`
- `src/perception/path_planner.py`
- `src/control/cognitive_controller.py`

**What it does**:
- Natural language command parsing with LLM (1Hz)
- A* path planning with obstacle avoidance
- Navigation control (10Hz)
- No changes to planning algorithms

**ROS2 Additions**:
```python
# Publishers
/cmd_vel (geometry_msgs/Twist)     # Velocity commands to RL
/path (nav_msgs/Path)              # Planned path for RViz
/brain_status (std_msgs/String)    # Status updates

# Subscribers
/navigation_command (std_msgs/String)  # Natural language input
/tf                                    # Robot pose

# Timers
1Hz planning loop (LLM + A*)
10Hz control loop (navigation)
```

**Key Implementation**:
```python
class BrainNode(Node):
    def __init__(self):
        # Initialize planners (existing code)
        self.nav_planner = NavigationPlanner()
        self.path_planner = AStarPlanner()
        
        # ROS2 additions
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/path', 10)
        self.planning_timer = self.create_timer(1.0, self.planning_callback)
        self.control_timer = self.create_timer(0.1, self.control_callback)
    
    def planning_callback(self):
        # A* planning (existing)
        path = self.path_planner.plan(start, goal)
        
        # Publish for visualization (new)
        self.publish_path(path)
    
    def control_callback(self):
        # Navigation logic (existing)
        vx, omega = self.compute_velocity_command()
        
        # Publish (new)
        cmd = Twist()
        cmd.linear.x = vx
        cmd.angular.z = omega
        self.cmd_vel_pub.publish(cmd)
```

## File Structure

```
IntelliH1/
├── src/                          # Original code (unchanged)
│   ├── control/
│   │   ├── unitree_rl_controller.py
│   │   └── cognitive_controller.py
│   ├── llm/
│   │   └── navigation_planner.py
│   ├── perception/
│   │   └── path_planner.py
│   └── simulation/
│       └── environment.py
│
└── ros2_ws/                      # New ROS2 workspace
    └── src/
        └── intelli_h1_ros/       # ROS2 package
            ├── package.xml
            ├── setup.py
            ├── setup.cfg
            ├── README.md
            │
            ├── intelli_h1_ros/   # Python package
            │   ├── __init__.py
            │   ├── sim_node.py   # Wraps simulation/environment.py
            │   ├── rl_node.py    # Wraps control/unitree_rl_controller.py
            │   └── brain_node.py # Wraps cognitive_controller.py + planners
            │
            ├── launch/
            │   └── demo_launch.py
            │
            ├── config/
            │   └── intelli_h1.rviz
            │
            └── resource/
                └── intelli_h1_ros
```

## Installation Steps

### 1. Install ROS2 via RoboStack (macOS/Linux)

```bash
# Install Miniforge
brew install miniforge  # macOS
# OR download from: https://github.com/conda-forge/miniforge

# Create ROS2 environment
conda create -n ros_env python=3.10
conda activate ros_env

# Add channels
conda config --add channels conda-forge
conda config --add channels robostack-staging

# Install ROS2 Humble (most stable)
conda install ros-humble-desktop colcon-common-extensions

# Install dependencies
pip install mujoco groq python-dotenv torch numpy scipy
```

### 2. Build the Package

```bash
cd ~/IntelliH1/ros2_ws
colcon build --packages-select intelli_h1_ros
source install/setup.bash
```

### 3. Verify Installation

```bash
# Check package is found
ros2 pkg list | grep intelli_h1_ros

# Check nodes are available
ros2 pkg executables intelli_h1_ros
# Should output:
#   intelli_h1_ros sim_node
#   intelli_h1_ros rl_node
#   intelli_h1_ros brain_node
```

## Usage

### Launch All Nodes

```bash
# Source workspace
source ~/IntelliH1/ros2_ws/install/setup.bash

# Launch with default parameters
ros2 launch intelli_h1_ros demo_launch.py

# Launch with custom speed (1.5 m/s)
ros2 launch intelli_h1_ros demo_launch.py max_speed:=1.5

# Launch specific robot type
ros2 launch intelli_h1_ros demo_launch.py robot_type:=h1_2
```

### Send Commands

```bash
# Send navigation command
ros2 topic pub --once /navigation_command std_msgs/String \
  "data: 'walk to the kitchen'"

# Send direct velocity command
ros2 topic pub /cmd_vel geometry_msgs/Twist \
  "{linear: {x: 1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

### Monitor System

```bash
# View node graph
rqt_graph

# Monitor all topics
ros2 topic list

# Echo joint states
ros2 topic echo /joint_states

# Echo planned path
ros2 topic echo /path

# View TF tree
ros2 run tf2_tools view_frames
```

### Visualize in RViz

```bash
# Launch RViz with config
rviz2 -d ~/IntelliH1/ros2_ws/install/intelli_h1_ros/share/intelli_h1_ros/config/intelli_h1.rviz

# In RViz:
# 1. Set Fixed Frame to "map"
# 2. Add TF display (see coordinate frames)
# 3. Add Path display (topic: /path, color: green)
# 4. Add Grid (reference plane)
```

## Key Concepts

### TF (Transform) Tree

ROS2 uses TF to represent spatial relationships:

```
map (world frame)
 └─ base_link (robot base, published by sim_node)
     ├─ left_hip
     ├─ right_hip
     └─ ... (other links)
```

**Coordinate Conversion**:
- MuJoCo: Quaternion as [w, x, y, z]
- ROS2: Quaternion as [x, y, z, w]

```python
# MuJoCo -> ROS2
ros_quat = [mujoco_quat[1], mujoco_quat[2], mujoco_quat[3], mujoco_quat[0]]
```

### Message Passing

All communication uses standard ROS2 messages:

| Purpose | Message Type | Topic |
|---------|-------------|-------|
| Joint angles | sensor_msgs/JointState | /joint_states |
| Robot pose | tf2_msgs/TFMessage | /tf |
| Velocity cmd | geometry_msgs/Twist | /cmd_vel |
| Motor torques | std_msgs/Float64MultiArray | /motor_cmd |
| Path | nav_msgs/Path | /path |

### QoS (Quality of Service)

Real-time data uses BEST_EFFORT for low latency:
```python
qos_profile = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)
```

## Advantages of ROS2 Version

1. **Modularity**: Each node can be developed/tested independently
2. **Distributed**: Nodes can run on different machines
3. **Debugging**: Use ROS2 tools (rqt, rostopic, etc.)
4. **Visualization**: RViz2 for real-time visualization
5. **Recording**: rosbag2 for data logging
6. **Standard Interface**: Other ROS2 packages can integrate easily
7. **Language Agnostic**: Can mix Python and C++ nodes

## Known Limitations

1. **URDF Not Integrated**: Robot visualization requires URDF setup
2. **No Perception Bridge**: LIDAR/camera data not yet published to ROS2
3. **Single Machine Only**: Current setup runs all nodes on one computer
4. **Manual TF Setup**: Coordinate transformations need manual configuration

## Future Enhancements

### Short-term
- [ ] Add URDF publishing for robot visualization
- [ ] Bridge perception data (LIDAR → sensor_msgs/LaserScan)
- [ ] Add parameter files (YAML configs)
- [ ] Create simple teleop node

### Medium-term
- [ ] Implement nav2 compatibility
- [ ] Add action server for navigation goals
- [ ] Create custom message types for structured data
- [ ] Add diagnostic publishing

### Long-term
- [ ] Multi-robot support
- [ ] Distributed computing setup
- [ ] Gazebo simulation integration
- [ ] Hardware deployment package

## Troubleshooting

### "Model file not found"
```bash
# Check submodule is initialized
cd ~/IntelliH1
git submodule update --init --recursive
ls extern/unitree_rl_gym/resources/robots/h1/
```

### "TF timeout"
```bash
# Verify sim_node is publishing TF
ros2 topic echo /tf
ros2 run tf2_ros tf2_echo map base_link
```

### "Policy file not found"
```bash
# Check policy exists
ls extern/unitree_rl_gym/deploy/pre_train/h1/motion.pt
```

### "Import error: rclpy not found"
```bash
# Make sure ROS2 environment is sourced
conda activate ros_env
source ~/IntelliH1/ros2_ws/install/setup.bash
```

## References

- [ROS2 Humble Documentation](https://docs.ros.org/en/humble/)
- [RoboStack Installation](https://robostack.github.io/)
- [MuJoCo Python Bindings](https://mujoco.readthedocs.io/en/stable/python.html)
- [Unitree RL Gym](https://github.com/unitreerobotics/unitree_rl_gym)

## Contributing

When adding new features to the ROS2 version:

1. **Preserve existing code**: Keep original algorithms in `src/`
2. **Wrap, don't rewrite**: Create thin ROS2 wrappers in `ros2_ws/`
3. **Follow ROS2 conventions**: Use standard message types
4. **Document thoroughly**: Update README and migration guide
5. **Test both versions**: Ensure original code still works

## License

MIT License - See LICENSE file in root directory
