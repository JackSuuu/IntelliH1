# ROS2 Quick Start Guide

This guide will get you up and running with the IntelliH1 ROS2 integration in under 10 minutes.

## Prerequisites

- macOS or Linux
- Python 3.10+
- Conda/Mamba package manager

## Step 1: Install ROS2 via RoboStack (5 minutes)

```bash
# Install miniforge if you don't have it
brew install miniforge  # macOS
# OR wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && bash Miniforge3-Linux-x86_64.sh  # Linux

# Configure channels FIRST (required before creating environment)
conda config --add channels conda-forge
conda config --add channels robostack-staging
conda config --set channel_priority strict

# Create and activate ROS2 environment
conda create -n ros_env python=3.10 -y
conda activate ros_env

# Install ROS2 Humble (this takes 3-4 minutes)
# For systems with GUI (desktop/laptop):
conda install ros-humble-desktop colcon-common-extensions -y
# For headless servers (no display):
conda install ros-humble-ros-base colcon-common-extensions -y

# Install Python dependencies
pip install mujoco groq python-dotenv torch numpy scipy
```

## Step 2: Setup Workspace (1 minute)

```bash
# Navigate to IntelliH1 repository
cd /path/to/IntelliH1

# Initialize submodules (if not already done)
git submodule update --init --recursive

# Build ROS2 package
cd ros2_ws
colcon build --packages-select intelli_h1_ros

# Source the workspace
source install/setup.bash
```

## Step 3: Verify Installation (30 seconds)

```bash
# Check package is available
ros2 pkg list | grep intelli_h1_ros

# Check executables
ros2 pkg executables intelli_h1_ros
# Should show:
#   intelli_h1_ros brain_node
#   intelli_h1_ros rl_node
#   intelli_h1_ros sim_node

# Check nodes can be found
ros2 run intelli_h1_ros sim_node --help
```

## Step 4: Run Your First Demo (1 minute)

### Option A: Basic Locomotion (no brain)

Test just the simulation and RL controller:

```bash
# Terminal 1: Launch sim + RL controller
ros2 launch intelli_h1_ros simple_demo.launch.py

# Terminal 2: Send velocity command
ros2 topic pub /cmd_vel geometry_msgs/Twist \
  "{linear: {x: 1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" \
  --rate 10
```

### Option B: Full Navigation with LLM

Launch all nodes including the brain:

```bash
# Set your Groq API key first
export GROQ_API_KEY=your_key_here

# Terminal 1: Launch all nodes
ros2 launch intelli_h1_ros demo_launch.py max_speed:=1.0

# Terminal 2: Send navigation command
ros2 topic pub --once /navigation_command std_msgs/String \
  "data: 'walk to the kitchen'"
```

## Step 5: Monitor and Debug

### View all topics
```bash
ros2 topic list
```

### Echo joint states
```bash
ros2 topic echo /joint_states
```

### Monitor velocity commands
```bash
ros2 topic echo /cmd_vel
```

### View node graph
```bash
# Install if not available
conda install ros-humble-rqt-graph -y

# Launch graph viewer
rqt_graph
```

### Check TF tree
```bash
ros2 run tf2_tools view_frames
# Generates frames.pdf showing transform tree
```

## Common Issues

### Issue: "ros2: command not found"
**Solution**: Make sure you activated the conda environment and sourced the workspace
```bash
conda activate ros_env
cd /path/to/IntelliH1/ros2_ws
source install/setup.bash
```

### Issue: "Model file not found"
**Solution**: Initialize git submodules
```bash
cd /path/to/IntelliH1
git submodule update --init --recursive
```

### Issue: "Policy file not found"
**Solution**: Check that the pre-trained policy exists
```bash
ls extern/unitree_rl_gym/deploy/pre_train/h1/motion.pt
```
If missing, you need to download it from the unitree_rl_gym repository.

### Issue: Nodes start but robot doesn't move
**Solution**: Check that all three nodes are running and communicating
```bash
# Check running nodes
ros2 node list

# Check topics are being published
ros2 topic hz /joint_states
ros2 topic hz /cmd_vel
ros2 topic hz /motor_cmd
```

## Next Steps

1. **Visualization**: Set up RViz2 for visual feedback
   ```bash
   rviz2 -d ros2_ws/install/intelli_h1_ros/share/intelli_h1_ros/config/intelli_h1.rviz
   ```

2. **Custom Commands**: Create your own navigation commands
   ```bash
   ros2 topic pub --once /navigation_command std_msgs/String \
     "data: 'go to bedroom'"
   ```

3. **Parameter Tuning**: Adjust speed and behavior
   ```bash
   ros2 launch intelli_h1_ros demo_launch.py max_speed:=1.5
   ```

4. **Recording Data**: Use rosbag2 to record sessions
   ```bash
   ros2 bag record /joint_states /cmd_vel /tf
   ```

## Useful Commands Cheatsheet

```bash
# List all nodes
ros2 node list

# List all topics
ros2 topic list

# Echo a topic
ros2 topic echo /topic_name

# Get topic info
ros2 topic info /topic_name

# Publish to a topic
ros2 topic pub /topic_name msg_type "data"

# Check message type
ros2 interface show msg_type

# List parameters of a node
ros2 param list /node_name

# Get parameter value
ros2 param get /node_name param_name

# Set parameter value
ros2 param set /node_name param_name value

# View node graph
rqt_graph

# View TF tree
ros2 run tf2_tools view_frames
ros2 run tf2_ros tf2_echo frame1 frame2
```

## Getting Help

- Check the main README: `../README.md`
- Read the migration guide: `../ROS2_MIGRATION.md`
- Package documentation: `src/intelli_h1_ros/README.md`
- ROS2 documentation: https://docs.ros.org/en/humble/
- RoboStack docs: https://robostack.github.io/

## Development Tips

1. **Always source workspace after building**:
   ```bash
   cd ros2_ws
   colcon build
   source install/setup.bash
   ```

2. **Use `--symlink-install` for faster iteration**:
   ```bash
   colcon build --symlink-install
   ```

3. **Build specific package only**:
   ```bash
   colcon build --packages-select intelli_h1_ros
   ```

4. **Clean build**:
   ```bash
   rm -rf build/ install/ log/
   colcon build
   ```

5. **Debug with verbose output**:
   ```bash
   ros2 run intelli_h1_ros sim_node --ros-args --log-level debug
   ```

Happy coding! 🤖
