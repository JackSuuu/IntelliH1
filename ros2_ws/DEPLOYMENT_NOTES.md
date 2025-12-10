# Deployment Notes

This file contains important notes for deploying the IntelliH1 ROS2 package to production.

## Pre-Deployment Checklist

### 1. Update Maintainer Information

**Files to update:**
- `ros2_ws/src/intelli_h1_ros/package.xml` (line 8)
- `ros2_ws/src/intelli_h1_ros/setup.py` (line 22)

Replace placeholder email `jack@example.com` with your actual email:

```xml
<!-- package.xml -->
<maintainer email="your-email@your-domain.com">Your Name</maintainer>
```

```python
# setup.py
maintainer='Your Name',
maintainer_email='your-email@your-domain.com',
```

### 2. Configure Model Path (Optional)

The default model path is hardcoded in launch files:
- `launch/demo_launch.py` (line 56)
- `launch/simple_demo.launch.py` (line 39)

**Option A: Use Environment Variable**

Set `INTELLIH1_MODEL_PATH` environment variable:
```bash
export INTELLIH1_MODEL_PATH=/path/to/your/model.xml
```

**Option B: Use Launch Parameter**

Already supported! Override via command line:
```bash
ros2 launch intelli_h1_ros demo_launch.py \
  model_path:=/custom/path/to/model.xml
```

### 3. Tune Navigation Parameters

Default values that may need adjustment:

**In brain_node.py:**

- **Waypoint threshold** (line 256): `0.5` meters
  - How close robot must be to waypoint before advancing
  - Increase for faster navigation, decrease for precision

- **Angular velocity gain** (line 274): `0.8` rad/s
  - Maximum turning rate
  - Increase for faster turns, decrease for stability

- **Goal tolerance** (parameter): `1.2` meters
  - How close to goal before stopping
  - Adjust in launch file or via parameter

**To customize via launch file:**

```python
# In demo_launch.py, modify brain_node parameters:
brain_node = Node(
    package='intelli_h1_ros',
    executable='brain_node',
    parameters=[{
        'goal_tolerance': 0.8,  # Stricter goal tolerance
        'max_speed': 1.0,
        # Add custom parameters as needed
    }]
)
```

### 4. Performance Tuning

**Path discovery depth limit** (utils.py, line 23): `10` iterations

This limits how far up the directory tree the code will search for the IntelliH1 root. If your installation is very deep in the filesystem, you may need to increase this.

**Alternative: Use environment variable**
```bash
export INTELLIH1_ROOT=/path/to/IntelliH1
```

This bypasses path discovery entirely.

## Configuration Parameters

### sim_node
| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_path` | `extern/unitree_rl_gym/...` | Path to MuJoCo XML model |
| `update_rate` | `500.0` | Physics update frequency (Hz) |

### rl_node
| Parameter | Default | Description |
|-----------|---------|-------------|
| `robot_type` | `h1` | Robot model (h1, h1_2, g1) |
| `max_speed` | `1.0` | Maximum walking speed (m/s) |
| `model_path` | `extern/unitree_rl_gym/...` | Path to MuJoCo XML model |
| `policy_path` | `''` | Path to RL policy (empty = use default) |
| `update_rate` | `50.0` | Control update frequency (Hz) |

### brain_node
| Parameter | Default | Description |
|-----------|---------|-------------|
| `planning_rate` | `1.0` | LLM planning frequency (Hz) |
| `control_rate` | `10.0` | Navigation control frequency (Hz) |
| `goal_tolerance` | `1.2` | Goal reached threshold (meters) |
| `max_speed` | `1.0` | Maximum navigation speed (m/s) |

### Example: Custom Configuration

Create a parameter file `config/custom_params.yaml`:

```yaml
/**:
  ros__parameters:
    # Simulation
    sim_node:
      update_rate: 500.0
      model_path: "path/to/custom/model.xml"
    
    # RL Controller
    rl_node:
      robot_type: "h1"
      max_speed: 1.5
      update_rate: 50.0
    
    # Brain/Planning
    brain_node:
      planning_rate: 1.0
      control_rate: 10.0
      goal_tolerance: 0.8
      max_speed: 1.5
```

Load in launch file:
```python
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

config_file = LaunchConfiguration('config_file')

# Add argument
config_arg = DeclareLaunchArgument(
    'config_file',
    default_value='config/custom_params.yaml'
)

# Use in node
brain_node = Node(
    package='intelli_h1_ros',
    executable='brain_node',
    parameters=[config_file]
)
```

## Environment Variables

The following environment variables can be used:

| Variable | Purpose | Example |
|----------|---------|---------|
| `INTELLIH1_ROOT` | Override path discovery | `/opt/IntelliH1` |
| `GROQ_API_KEY` | API key for LLM features | `gsk_...` |
| `ROS_DOMAIN_ID` | ROS2 domain isolation | `42` |

## Production Recommendations

### 1. Use systemd for Auto-start

Create `/etc/systemd/system/intellih1.service`:

```ini
[Unit]
Description=IntelliH1 ROS2 Service
After=network.target

[Service]
Type=simple
User=robot
Environment="ROS_DOMAIN_ID=0"
WorkingDirectory=/opt/IntelliH1/ros2_ws
ExecStart=/bin/bash -c "source install/setup.bash && ros2 launch intelli_h1_ros demo_launch.py"
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable intellih1
sudo systemctl start intellih1
```

### 2. Set Up Logging

Configure ROS2 logging:

```bash
# In your launch file or environment
export RCUTILS_COLORIZED_OUTPUT=0  # Disable colors for log files
export RCUTILS_CONSOLE_OUTPUT_FORMAT='[{severity}] [{name}]: {message}'
```

Or use launch file logging:
```python
from launch.actions import SetEnvironmentVariable

SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1')
```

### 3. Monitor with rqt

For debugging in production:

```bash
# Install monitoring tools
conda install ros-humble-rqt-graph ros-humble-rqt-plot

# Monitor node graph
rqt_graph

# Plot topic data
rqt_plot /joint_states/position[0] /cmd_vel/linear/x
```

### 4. Record Data with rosbag2

Record sessions for analysis:

```bash
# Record all topics
ros2 bag record -a

# Record specific topics
ros2 bag record /joint_states /cmd_vel /path /tf

# Playback
ros2 bag play <bag_file>
```

## Security Considerations

### 1. API Keys

Never commit API keys to version control. Use:

- **Environment variables**: `export GROQ_API_KEY=...`
- **Secrets management**: Kubernetes secrets, HashiCorp Vault
- **Config files**: With restricted permissions (`chmod 600`)

### 2. Network Security

For distributed deployments:

- Use ROS2 security (SROS2)
- Restrict network access
- Use VPN or SSH tunnels for remote nodes

### 3. Input Validation

The brain_node accepts natural language commands. Consider:

- Sanitizing user inputs
- Rate limiting command topic
- Whitelisting allowed commands

## Performance Monitoring

### Key Metrics to Monitor

1. **Topic Frequencies**
   ```bash
   ros2 topic hz /joint_states  # Should be ~500Hz
   ros2 topic hz /motor_cmd     # Should be ~50Hz
   ros2 topic hz /cmd_vel       # Should be ~10Hz when navigating
   ```

2. **Latency**
   ```bash
   ros2 topic delay /cmd_vel /motor_cmd
   ```

3. **CPU Usage**
   ```bash
   top -p $(pgrep -f "sim_node|rl_node|brain_node")
   ```

4. **Memory Usage**
   ```bash
   ps aux | grep -E "sim_node|rl_node|brain_node"
   ```

### Expected Performance

| Node | CPU | Memory | Frequency |
|------|-----|--------|-----------|
| sim_node | 30-50% | ~200MB | 500Hz |
| rl_node | 10-20% | ~500MB | 50Hz |
| brain_node | 1-5% | ~100MB | 1-10Hz |

## Troubleshooting

### Common Issues

1. **"Model file not found"**
   - Ensure `git submodule update --init --recursive`
   - Set `INTELLIH1_ROOT` environment variable

2. **"Policy file not found"**
   - Check `extern/unitree_rl_gym/deploy/pre_train/h1/motion.pt`
   - Provide custom path via `policy_path` parameter

3. **High CPU usage**
   - Reduce `update_rate` parameters
   - Check for infinite loops in custom code
   - Monitor with `top` and `htop`

4. **Topics not publishing**
   - Check node is running: `ros2 node list`
   - Check for errors: `ros2 topic echo /rosout`
   - Verify QoS settings match

## Support

For issues and questions:

1. Check documentation in `ros2_ws/`:
   - QUICKSTART.md
   - TESTING.md
   - URDF_SETUP.md

2. Review main documentation:
   - ROS2_MIGRATION.md
   - README.md

3. Open an issue on GitHub with:
   - System info (`uname -a`, `ros2 --version`)
   - Error messages
   - Steps to reproduce

## Updates and Maintenance

### Updating the Package

```bash
cd /path/to/IntelliH1
git pull
git submodule update --remote
cd ros2_ws
colcon build --packages-select intelli_h1_ros
source install/setup.bash
```

### Backing Up Configuration

```bash
# Backup custom configs
cp -r ros2_ws/src/intelli_h1_ros/config ~/backup/
cp -r ros2_ws/src/intelli_h1_ros/launch ~/backup/

# Backup data
ros2 bag record -a -o ~/backup/$(date +%Y%m%d)
```

## License

MIT License - See LICENSE file in repository root

---

**Last Updated**: Initial release
**Version**: 0.1.0
