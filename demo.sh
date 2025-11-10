#!/bin/bash
# Simplified Demo Script for Unitree H1 Robot
# Supports standing, walking, and navigation modes

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Control Modes:"
    echo "  (default)       Standing balance mode"
    echo "  --walk          Walking mode (continuous forward)"
    echo "  --navigate      Navigation mode (walk to target)"
    echo ""
    echo "Options:"
    echo "  --simple        Use simple PD (default: improved PD with gravity comp)"
    echo "  --target LOC    Navigation target (kitchen/bedroom/livingroom or x,y)"
    echo "  --walk-speed N  Walking speed in m/s (default: 0.3)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Standing mode"
    echo "  $0 --walk --walk-speed 0.3            # Walking mode"
    echo "  $0 --navigate --target kitchen        # Navigate to kitchen"
    echo "  $0 --navigate --target 5.0,3.0        # Navigate to coordinates"
    exit 0
}

# Parse args
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    show_help
fi

# Detect OS and choose correct interpreter
if [[ "$OSTYPE" == "darwin"* ]]; then
    PYTHON_CMD="mjpython"
else
    PYTHON_CMD="python3"
fi

echo "============================================================"
echo "🤖 UNITREE H1 - CONTROL DEMO"
echo "============================================================"

# Run test with all arguments passed through
$PYTHON_CMD src/test/test_h1_scene.py "$@"
