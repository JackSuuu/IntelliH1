#!/bin/bash
# Demo Script for IntelliH1: Cognitive Humanoid Framework
# Supports LLM-driven navigation with C++ radar perception

show_help() {
    echo "============================================================"
    echo "🤖 IntelliH1: Cognitive Humanoid Navigation Demo"
    echo "============================================================"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "🎯 Quick Start Examples:"
    echo "  $0 kitchen              # Navigate to kitchen"
    echo "  $0 bedroom              # Navigate to bedroom"
    echo "  $0 \"go to living room\"  # Natural language command"
    echo "  $0 --speed 1.2 bedroom  # Fast navigation to bedroom"
    echo ""
    echo "📍 Available Destinations:"
    echo "  • kitchen      - (5.0, 3.0)   Kitchen area"
    echo "  • bedroom      - (-3.0, 6.0)  Bedroom with bed"
    echo "  • living_room  - (0.0, -4.0)  Living room with couch"
    echo ""
    echo "⚙️  Options:"
    echo "  --speed N       Walking speed in m/s (default: 1.0, max: 1.2)"
    echo "  --robot TYPE    Robot type: h1, h1_2, g1 (default: h1)"
    echo "  --help, -h      Show this help message"
    echo ""
    echo "🔧 Technical Features:"
    echo "  ✅ LLM Planning (Groq API + Llama 3.3)"
    echo "  ✅ C++ Radar Perception (Real-time point cloud processing)"
    echo "  ✅ A* Path Planning (Obstacle avoidance)"
    echo "  ✅ Unitree RL Locomotion (Official pre-trained policy)"
    echo ""
    exit 0
}

# Parse args
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]] || [[ -z "$1" ]]; then
    show_help
fi

# Detect OS and choose correct interpreter
if [[ "$OSTYPE" == "darwin"* ]]; then
    PYTHON_CMD="mjpython"
else
    PYTHON_CMD="python3"
fi

# Default values
SPEED="1.0"
ROBOT="h1"
COMMAND=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --speed)
            SPEED="$2"
            shift 2
            ;;
        --robot)
            ROBOT="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            ;;
        *)
            # Everything else is the command
            if [[ -z "$COMMAND" ]]; then
                COMMAND="$1"
            else
                COMMAND="$COMMAND $1"
            fi
            shift
            ;;
    esac
done

# Convert shorthand to natural language
case "$COMMAND" in
    kitchen)
        COMMAND="walk to the kitchen"
        ;;
    bedroom)
        COMMAND="walk to the bedroom"
        ;;
    living_room|livingroom)
        COMMAND="walk to the living room"
        ;;
esac

echo "============================================================"
echo "🤖 IntelliH1: Cognitive Humanoid Navigation"
echo "============================================================"
echo "🎯 Command: \"$COMMAND\""
echo "⚡ Speed: ${SPEED} m/s"
echo "🤖 Robot: ${ROBOT}"
echo "============================================================"
echo ""

# Run LLM navigation
$PYTHON_CMD src/test/test_llm_navigation.py \
    --command "$COMMAND" \
    --walk-speed "$SPEED" \
    --robot-type "$ROBOT"
