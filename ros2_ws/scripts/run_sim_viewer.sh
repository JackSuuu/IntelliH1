#!/bin/bash
# Run MuJoCo simulation with viewer using mjpython (required for macOS)
# This script launches the simulation node with full MuJoCo visualization

set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS2_WS="$(dirname "$SCRIPT_DIR")"
INTELLIH1_ROOT="$(dirname "$ROS2_WS")"

echo "=============================================="
echo "  IntelliH1 ROS2 + MuJoCo Viewer (macOS)"
echo "=============================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install miniconda/miniforge."
    exit 1
fi

# Activate ros_env
echo "Activating ros_env conda environment..."
eval "$(conda shell.bash hook)"
conda activate ros_env

# Source ROS2 workspace
echo "Sourcing ROS2 workspace..."
cd "$ROS2_WS"
source install/setup.bash

# Find mjpython
MJPYTHON="/opt/homebrew/Caskroom/miniconda/base/envs/ros_env/bin/mjpython"

if [ ! -f "$MJPYTHON" ]; then
    # Try to find it dynamically
    MJPYTHON=$(find /opt/homebrew/Caskroom/miniconda/base/envs/ros_env -name "mjpython" -type f 2>/dev/null | head -1)
fi

if [ -z "$MJPYTHON" ] || [ ! -f "$MJPYTHON" ]; then
    echo "ERROR: mjpython not found. Make sure mujoco is installed in ros_env."
    echo "Try: conda activate ros_env && pip install mujoco"
    exit 1
fi

echo "Using mjpython: $MJPYTHON"
echo ""

# Set Python path to include our package
export PYTHONPATH="$ROS2_WS/install/intelli_h1_ros/lib/python3.10/site-packages:$PYTHONPATH"

# Run the simulation node with viewer
echo "Launching MuJoCo simulation with viewer..."
echo ""
exec "$MJPYTHON" "$ROS2_WS/src/intelli_h1_ros/intelli_h1_ros/sim_node_viewer.py"
