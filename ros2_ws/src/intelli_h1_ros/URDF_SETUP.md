# URDF Setup Guide for IntelliH1 ROS2

This guide explains how to set up the URDF (Unified Robot Description Format) for visualizing the Unitree H1 robot in RViz2.

## What is URDF?

URDF is an XML format for representing robot models in ROS. It describes:
- Robot links (rigid bodies like torso, limbs)
- Joints (connections between links)
- Visual meshes (3D models for visualization)
- Collision meshes (simplified geometry for collision detection)
- Physical properties (mass, inertia)

## Why Do We Need URDF?

The MuJoCo simulation uses MJCF (MuJoCo XML) format, which is different from URDF. To visualize the robot in RViz2, we need:

1. **URDF file**: Robot description for ROS
2. **robot_state_publisher**: ROS2 node that publishes TF transforms from URDF
3. **Joint state synchronization**: Map MuJoCo joint states to URDF joint names

## Current Status

⚠️ **URDF integration is not yet complete**. The ROS2 package currently:

- ✅ Publishes joint states from MuJoCo simulation
- ✅ Publishes base_link TF transform
- ❌ Does not have H1 URDF file configured
- ❌ robot_state_publisher is disabled in launch files

## Options for Getting H1 URDF

### Option 1: Use Official Unitree URDF (Recommended)

The unitree_rl_gym repository contains robot models:

```bash
# Initialize submodule
cd /path/to/IntelliH1
git submodule update --init --recursive

# Check for URDF or MJCF files
ls extern/unitree_rl_gym/resources/robots/h1/

# If URDF exists:
cp extern/unitree_rl_gym/resources/robots/h1/h1.urdf \
   ros2_ws/src/intelli_h1_ros/urdf/
```

### Option 2: Convert MJCF to URDF

If only MJCF files are available, convert them:

```bash
# Install conversion tool
pip install mujoco-mjcf2urdf

# Convert MJCF to URDF
mjcf2urdf \
  extern/unitree_rl_gym/resources/robots/h1/h1.xml \
  ros2_ws/src/intelli_h1_ros/urdf/h1.urdf
```

**Note**: Automatic conversion may require manual fixes:
- Material properties
- Mesh file paths
- Joint limits
- Collision geometries

### Option 3: Use Community URDF

Check for community-maintained H1 URDF files:

```bash
# Example: fan-ziqi's h1-mujoco-sim
git clone https://github.com/fan-ziqi/h1-mujoco-sim.git /tmp/h1-sim
# Check if it contains URDF files
ls /tmp/h1-sim/
```

### Option 4: Create Simplified URDF

For basic visualization, create a simplified URDF with basic shapes:

```xml
<?xml version="1.0"?>
<robot name="h1_simple">
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </visual>
  </link>
  
  <!-- Add joints and links for legs, arms, etc. -->
</robot>
```

## Integration Steps (Once URDF is Ready)

### Step 1: Add URDF to Package

```bash
cd ros2_ws/src/intelli_h1_ros
mkdir -p urdf meshes

# Place URDF file
# Place mesh files (STL, DAE, OBJ) in meshes/
```

### Step 2: Update package.xml

Add URDF data files:

```xml
<!-- In package.xml -->
<export>
  <build_type>ament_python</build_type>
</export>
```

### Step 3: Update setup.py

Include URDF and mesh files:

```python
# In setup.py, add to data_files:
(os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
(os.path.join('share', package_name, 'meshes'), glob('meshes/*')),
```

### Step 4: Create URDF Publisher Node

```python
# urdf_publisher.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import os

class URDFPublisher(Node):
    def __init__(self):
        super().__init__('urdf_publisher')
        
        # Read URDF file
        urdf_file = 'path/to/h1.urdf'
        with open(urdf_file, 'r') as f:
            urdf_content = f.read()
        
        # Publish robot_description
        self.publisher = self.create_publisher(String, 'robot_description', 10)
        
        msg = String()
        msg.data = urdf_content
        self.publisher.publish(msg)
        
        self.get_logger().info('URDF published to /robot_description')

def main():
    rclpy.init()
    node = URDFPublisher()
    rclpy.spin_once(node)
    rclpy.shutdown()
```

### Step 5: Update Launch File

Enable robot_state_publisher:

```python
# In demo_launch.py

# Read URDF file
urdf_path = PathJoinSubstitution([
    FindPackageShare('intelli_h1_ros'),
    'urdf',
    'h1.urdf'
])

with open(urdf_path, 'r') as f:
    robot_description = f.read()

# Robot State Publisher node
robot_state_pub = Node(
    package='robot_state_publisher',
    executable='robot_state_publisher',
    name='robot_state_publisher',
    output='screen',
    parameters=[{
        'robot_description': robot_description,
        'publish_frequency': 50.0,
    }]
)
```

### Step 6: Synchronize Joint Names

Ensure MuJoCo joint names match URDF joint names:

```python
# In sim_node.py, extract joint names from URDF
# Map MuJoCo indices to URDF joint names

joint_name_mapping = {
    'left_hip_yaw_joint': 0,
    'left_hip_roll_joint': 1,
    # ... etc
}
```

## Coordinate Frame Conventions

### MuJoCo Conventions
- Position: (x, y, z)
- Quaternion: **[w, x, y, z]**
- Right-handed coordinate system
- Z-axis typically up

### ROS2/URDF Conventions
- Position: (x, y, z)
- Quaternion: **[x, y, z, w]**
- Right-handed coordinate system
- Z-axis typically up

### Conversion in sim_node.py

```python
# MuJoCo quaternion: [w, x, y, z]
mujoco_quat = self.data.qpos[3:7]

# Convert to ROS quaternion: [x, y, z, w]
ros_quat = [
    mujoco_quat[1],  # x
    mujoco_quat[2],  # y
    mujoco_quat[3],  # z
    mujoco_quat[0],  # w
]
```

## Testing URDF Integration

### Step 1: Check URDF is Valid

```bash
# Install check tool
conda install ros-humble-urdf-tutorial

# Check URDF syntax
check_urdf urdf/h1.urdf

# Visualize URDF structure
urdf_to_graphiz urdf/h1.urdf
```

### Step 2: Verify robot_state_publisher

```bash
# Launch nodes
ros2 launch intelli_h1_ros demo_launch.py

# Check robot_description topic
ros2 topic echo /robot_description

# Check TF frames
ros2 run tf2_tools view_frames
# This generates frames.pdf showing all transforms

# Check specific transform
ros2 run tf2_ros tf2_echo map base_link
```

### Step 3: Visualize in RViz2

```bash
# Launch RViz
rviz2 -d config/intelli_h1.rviz

# In RViz:
# 1. Set Fixed Frame to "map"
# 2. Add RobotModel display
# 3. Set Description Topic to "/robot_description"
# 4. Enable RobotModel display

# You should see the H1 robot model!
```

## Common Issues

### Issue: "No transform from map to base_link"

**Cause**: sim_node not publishing TF

**Solution**: Check sim_node is running:
```bash
ros2 node list | grep sim
ros2 topic echo /tf
```

### Issue: "robot_description not found"

**Cause**: robot_state_publisher not configured

**Solution**: Enable robot_state_publisher in launch file

### Issue: Robot appears upside down or rotated

**Cause**: Coordinate frame mismatch between MuJoCo and ROS

**Solution**: Apply rotation correction in TF publish:
```python
# Add 180° rotation around X axis if needed
from scipy.spatial.transform import Rotation as R
r = R.from_quat(mujoco_quat) * R.from_euler('x', 180, degrees=True)
corrected_quat = r.as_quat()
```

### Issue: Meshes not loading

**Cause**: Mesh file paths in URDF are incorrect

**Solution**: Update URDF mesh paths:
```xml
<!-- Before -->
<mesh filename="package://unitree_description/meshes/h1/torso.dae"/>

<!-- After -->
<mesh filename="package://intelli_h1_ros/meshes/torso.dae"/>
```

## Alternative: Visualization Without URDF

If URDF setup is too complex, you can still visualize using:

### Option 1: TF Markers

Publish TF transforms for each joint and visualize as axes in RViz:
```python
# Publish TF for each joint
for joint_name, position in joint_positions.items():
    t = TransformStamped()
    t.header.frame_id = 'base_link'
    t.child_frame_id = joint_name
    # ... set position from MuJoCo
    self.tf_broadcaster.sendTransform(t)
```

### Option 2: Marker Array

Publish visualization markers for robot links:
```python
from visualization_msgs.msg import Marker, MarkerArray

marker = Marker()
marker.header.frame_id = "base_link"
marker.type = Marker.CYLINDER
marker.scale.x = 0.1  # radius
marker.scale.y = 0.1
marker.scale.z = 0.3  # height
# ... set position and orientation
```

### Option 3: Point Cloud

Publish robot vertices as a point cloud:
```python
from sensor_msgs.msg import PointCloud2

# Extract vertices from MuJoCo visualization
# Publish as point cloud
```

## Next Steps

1. **Obtain H1 URDF**: Download or convert from MJCF
2. **Validate URDF**: Use check_urdf tool
3. **Update package files**: Add URDF to setup.py
4. **Enable robot_state_publisher**: Update launch file
5. **Test in RViz2**: Verify robot visualization
6. **Fine-tune**: Adjust joint name mapping and coordinate frames

## References

- [ROS2 URDF Tutorial](https://docs.ros.org/en/humble/Tutorials/Intermediate/URDF/URDF-Main.html)
- [robot_state_publisher](https://github.com/ros/robot_state_publisher)
- [URDF XML Specification](http://wiki.ros.org/urdf/XML)
- [MuJoCo MJCF Format](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
- [Unitree Robotics GitHub](https://github.com/unitreerobotics)

## Help Needed?

If you successfully set up URDF for H1, please contribute back:
- Share the URDF file
- Document any conversion steps
- Submit a PR to add URDF support

We welcome contributions! 🤝
