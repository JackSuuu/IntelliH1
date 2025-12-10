"""
Utility functions for IntelliH1 ROS2 nodes
Shared functionality across sim, rl, and brain nodes
"""

import os
import sys
from pathlib import Path


def get_intellih1_root():
    """
    Get the IntelliH1 project root directory.
    
    Returns:
        str: Absolute path to IntelliH1 project root
    """
    # Find IntelliH1 root by looking for characteristic directories
    current_file = Path(__file__).resolve()
    
    # Try to find root by looking for extern/ directory
    search_path = current_file
    for _ in range(10):  # Limit search depth
        search_path = search_path.parent
        if (search_path / 'extern' / 'unitree_rl_gym').exists():
            return str(search_path)
        if search_path.parent == search_path:  # Reached filesystem root
            break
    
    # Fallback: use environment variable if set
    if 'INTELLIH1_ROOT' in os.environ:
        return os.environ['INTELLIH1_ROOT']
    
    # Fallback: assume standard relative path
    # ros2_ws/src/intelli_h1_ros/intelli_h1_ros/utils.py -> IntelliH1/
    return str(current_file.parent.parent.parent.parent.parent)


def setup_intellih1_path():
    """
    Add IntelliH1 src directory to Python path.
    
    This allows importing from src/control, src/perception, etc.
    Should be called at the start of each node.
    """
    intellih1_root = get_intellih1_root()
    src_path = os.path.join(intellih1_root, 'src')
    
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    return intellih1_root


def mujoco_to_ros_quaternion(mujoco_quat):
    """
    Convert MuJoCo quaternion to ROS quaternion format.
    
    MuJoCo uses [w, x, y, z] order
    ROS uses [x, y, z, w] order
    
    Args:
        mujoco_quat: Quaternion in MuJoCo format [w, x, y, z]
        
    Returns:
        tuple: Quaternion in ROS format (x, y, z, w)
    """
    # MuJoCo: [w, x, y, z] at indices [0, 1, 2, 3]
    # ROS:    [x, y, z, w] = [1, 2, 3, 0]
    return (
        float(mujoco_quat[1]),  # x
        float(mujoco_quat[2]),  # y
        float(mujoco_quat[3]),  # z
        float(mujoco_quat[0]),  # w
    )


# Default location coordinates
# These can be overridden via ROS parameters
DEFAULT_LOCATIONS = {
    "kitchen": (5.0, 3.0),
    "bedroom": (-3.0, 6.0),
    "living_room": (0.0, -4.0),
    "living room": (0.0, -4.0),  # Allow space in name
}
