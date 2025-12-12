# Testing Guide for IntelliH1 ROS2

This guide explains how to test the ROS2 integration at different levels.

## Testing Strategy

The ROS2 integration follows a "Wrap & Bridge" strategy, so testing happens at two levels:

1. **Original Code Testing**: Test that wrapped Python modules still work
2. **ROS2 Integration Testing**: Test that nodes communicate correctly

## Level 1: Original Code Testing

Before testing ROS2, verify the original code works:

### Test MuJoCo Simulation

```bash
cd /path/to/IntelliH1

# Test simulation loads
python3 -c "
import sys
sys.path.insert(0, 'src')
from simulation.environment import Simulation
sim = Simulation('extern/unitree_rl_gym/resources/robots/h1/scene_enhanced.xml')
print('✅ Simulation loaded successfully')
print(f'Number of joints: {sim.model.njnt}')
print(f'Number of actuators: {sim.model.nu}')
"
```

### Test RL Controller

```bash
# Test RL controller loads
python3 -c "
import sys
sys.path.insert(0, 'src')
import mujoco
from simulation.environment import Simulation
from control.unitree_rl_controller import UnitreeRLController

sim = Simulation('extern/unitree_rl_gym/resources/robots/h1/scene_enhanced.xml')
controller = UnitreeRLController(sim.model, sim.data, robot_type='h1')
print('✅ RL controller loaded successfully')
action = controller.compute_action()
print(f'Action shape: {action.shape}')
"
```

### Test Path Planner

```bash
# Test A* planner
python3 -c "
import sys
sys.path.insert(0, 'src')
from perception.path_planner import AStarPlanner

planner = AStarPlanner(grid_resolution=0.3, map_size=30)
path = planner.plan((0, 0), (5, 3))
print('✅ Path planner works')
print(f'Path has {len(path)} waypoints')
"
```

### Test LLM Navigation Planner (Optional)

```bash
# Requires GROQ_API_KEY
export GROQ_API_KEY=your_key_here

python3 -c "
import sys
sys.path.insert(0, 'src')
from llm.navigation_planner import NavigationPlanner

planner = NavigationPlanner()
print('✅ Navigation planner initialized')
"
```

## Level 2: ROS2 Node Testing

### Prerequisites

```bash
# Activate ROS2 environment
conda activate ros_env
cd /path/to/IntelliH1/ros2_ws
source install/setup.bash
```

### Test 1: Individual Node Startup

Test each node can start without errors:

```bash
# Terminal 1: Test sim_node
ros2 run intelli_h1_ros sim_node

# Expected output:
# [INFO] [mujoco_sim_node]: Loading MuJoCo model: ...
# [INFO] [mujoco_sim_node]: Simulation node initialized at 500Hz
# [INFO] [mujoco_sim_node]: Number of joints: ...

# Press Ctrl+C to stop
```

```bash
# Terminal 2: Test rl_node
ros2 run intelli_h1_ros rl_node

# Expected output:
# [INFO] [rl_controller_node]: Initializing RL controller: h1
# [INFO] [rl_controller_node]: RL controller node initialized at 50Hz

# Press Ctrl+C to stop
```

```bash
# Terminal 3: Test brain_node
ros2 run intelli_h1_ros brain_node

# Expected output:
# [INFO] [brain_node]: Initializing navigation planner...
# [INFO] [brain_node]: ✅ A* path planner initialized
# [INFO] [brain_node]: Brain node initialized

# Press Ctrl+C to stop
```

### Test 2: Topic Communication

Check that topics are being published:

```bash
# Terminal 1: Launch sim_node
ros2 run intelli_h1_ros sim_node

# Terminal 2: Check topics
ros2 topic list
# Should show:
#   /joint_states
#   /motor_cmd
#   /parameter_events
#   /rosout
#   /tf

# Check joint_states is publishing
ros2 topic hz /joint_states
# Should show ~500Hz

# Echo joint state (should see data)
ros2 topic echo /joint_states --once
```

### Test 3: Node Communication Chain

Test the full communication chain: sim → rl → brain

```bash
# Terminal 1: Launch all nodes
ros2 launch intelli_h1_ros simple_demo.launch.py

# Terminal 2: Monitor topics
watch -n 1 "ros2 topic hz /joint_states /cmd_vel /motor_cmd"

# Terminal 3: Send velocity command
ros2 topic pub /cmd_vel geometry_msgs/Twist \
  "{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" \
  --rate 10

# Expected behavior:
# - /joint_states publishes at ~500Hz
# - /cmd_vel receives your commands at 10Hz
# - /motor_cmd publishes at ~50Hz
```

### Test 4: TF Publishing

Verify coordinate transforms are published:

```bash
# Terminal 1: Launch sim_node
ros2 run intelli_h1_ros sim_node

# Terminal 2: Check TF is publishing
ros2 topic echo /tf

# Should see TransformStamped messages with:
#   header.frame_id: "map"
#   child_frame_id: "base_link"

# Check TF lookup
ros2 run tf2_ros tf2_echo map base_link

# Should show continuous transform updates
```

### Test 5: Brain Navigation

Test high-level navigation:

```bash
# Terminal 1: Launch all nodes
ros2 launch intelli_h1_ros demo_launch.py

# Terminal 2: Send navigation command
ros2 topic pub --once /navigation_command std_msgs/String \
  "data: 'go to kitchen'"

# Terminal 3: Monitor status
ros2 topic echo /brain_status

# Terminal 4: Monitor path
ros2 topic echo /path

# Expected behavior:
# - Brain publishes planned path
# - Brain sends /cmd_vel commands
# - Robot moves toward goal
# - Status shows "Goal reached" when done
```

## Level 3: Integration Testing

### Test with RViz2 Visualization

```bash
# Terminal 1: Launch nodes
ros2 launch intelli_h1_ros demo_launch.py

# Terminal 2: Launch RViz
rviz2 -d install/intelli_h1_ros/share/intelli_h1_ros/config/intelli_h1.rviz

# In RViz:
# 1. Check Fixed Frame is "map"
# 2. Enable TF display - should see base_link transform
# 3. Enable Path display - should see planned path when navigating
# 4. Enable Grid for reference

# Terminal 3: Send navigation command
ros2 topic pub --once /navigation_command std_msgs/String "data: 'bedroom'"

# Expected: Green path line appears in RViz showing planned route
```

### Test Parameter Configuration

```bash
# Test with different speeds
ros2 launch intelli_h1_ros demo_launch.py max_speed:=1.5

# Verify parameter is set
ros2 param get /rl_controller max_speed
# Should show: 1.5

# Test with different robot type
ros2 launch intelli_h1_ros demo_launch.py robot_type:=h1_2

ros2 param get /rl_controller robot_type
# Should show: h1_2
```

## Level 4: Performance Testing

### Test Frequency Rates

```bash
# Terminal 1: Launch nodes
ros2 launch intelli_h1_ros simple_demo.launch.py

# Terminal 2: Check all topic frequencies
ros2 topic hz /joint_states &    # Should be ~500Hz
ros2 topic hz /motor_cmd &       # Should be ~50Hz
ros2 topic hz /cmd_vel &         # Should be ~10Hz (when navigating)

# Let run for 10 seconds, then check results
```

### Test Latency

```bash
# Install ROS2 benchmarking tools
pip install ros2-benchmark

# Measure latency from cmd_vel to motor_cmd
ros2 topic delay /cmd_vel /motor_cmd

# Typical values:
# - Mean latency: < 20ms (50Hz control loop)
# - Max latency: < 50ms
```

### Test CPU Usage

```bash
# Terminal 1: Launch nodes
ros2 launch intelli_h1_ros demo_launch.py

# Terminal 2: Monitor CPU usage
top -p $(pgrep -f sim_node) \
     -p $(pgrep -f rl_node) \
     -p $(pgrep -f brain_node)

# Expected CPU usage:
# - sim_node: 30-50% (physics at 500Hz)
# - rl_node: 10-20% (RL inference at 50Hz)
# - brain_node: 1-5% (planning at 1Hz)
```

## Troubleshooting Tests

### Test Fails: "No module named 'rclpy'"

**Cause**: ROS2 environment not activated

**Fix**:
```bash
conda activate ros_env
source /path/to/IntelliH1/ros2_ws/install/setup.bash
```

### Test Fails: "Model file not found"

**Cause**: Git submodules not initialized

**Fix**:
```bash
cd /path/to/IntelliH1
git submodule update --init --recursive
```

### Test Fails: "Policy file not found"

**Cause**: RL policy file missing

**Fix**:
```bash
# Check if policy exists
ls extern/unitree_rl_gym/deploy/pre_train/h1/motion.pt

# If missing, the submodule may not be fully initialized
git submodule update --init --recursive --remote
```

### Test Fails: Topics not publishing

**Check nodes are running**:
```bash
ros2 node list
# Should show:
#   /mujoco_sim
#   /rl_controller
#   /brain
```

**Check for errors**:
```bash
ros2 topic echo /rosout
# Look for ERROR or WARN messages
```

## Automated Testing

### Create Test Script

```bash
#!/bin/bash
# test_ros2_integration.sh

set -e

echo "Testing IntelliH1 ROS2 Integration..."

# Test 1: Package exists
echo "Test 1: Package found"
ros2 pkg list | grep -q intelli_h1_ros || exit 1

# Test 2: Executables exist
echo "Test 2: Executables found"
ros2 pkg executables intelli_h1_ros | grep -q sim_node || exit 1
ros2 pkg executables intelli_h1_ros | grep -q rl_node || exit 1
ros2 pkg executables intelli_h1_ros | grep -q brain_node || exit 1

# Test 3: Launch files exist
echo "Test 3: Launch files found"
ros2 launch intelli_h1_ros demo_launch.py --show-args || exit 1

# Test 4: Nodes can start
echo "Test 4: Starting sim_node..."
timeout 5 ros2 run intelli_h1_ros sim_node &
NODE_PID=$!
sleep 3
kill $NODE_PID 2>/dev/null || true

echo "✅ All tests passed!"
```

```bash
chmod +x test_ros2_integration.sh
./test_ros2_integration.sh
```

## Continuous Integration

For automated testing in CI/CD:

```yaml
# .github/workflows/ros2_test.yml
name: ROS2 Integration Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          submodules: recursive
      
      - name: Install ROS2
        uses: ros-tooling/setup-ros@v0.6
        with:
          required-ros-distributions: humble
      
      - name: Build package
        run: |
          cd ros2_ws
          colcon build --packages-select intelli_h1_ros
          source install/setup.bash
      
      - name: Run tests
        run: |
          cd ros2_ws
          source install/setup.bash
          ./test_ros2_integration.sh
```

## Test Checklist

Before submitting changes, verify:

- [ ] Original Python code still works (Level 1)
- [ ] All nodes start without errors (Level 2, Test 1)
- [ ] Topics are being published (Level 2, Test 2)
- [ ] Nodes communicate correctly (Level 2, Test 3)
- [ ] TF transforms work (Level 2, Test 4)
- [ ] Navigation commands work (Level 2, Test 5)
- [ ] RViz visualization works (Level 3)
- [ ] Parameters configure correctly (Level 3)
- [ ] Performance meets targets (Level 4)
- [ ] Documentation is updated
- [ ] No new warnings or errors in logs

## Getting Help

If tests fail and you can't resolve:

1. Check logs: `ros2 topic echo /rosout`
2. Verify environment: `echo $ROS_DISTRO` (should be "humble")
3. Check dependencies: `pip list | grep -E "rclpy|mujoco"`
4. Review documentation: See README.md and ROS2_MIGRATION.md
5. Open an issue with:
   - Test that failed
   - Full error output
   - System info (`uname -a`, `python3 --version`)

Happy testing! 🧪
