#!/bin/bash

echo "� Testing Unitree H1 Humanoid Robot"
echo "========================================================"

# Parse arguments
TEST_TYPE=${1:-"scene"}  # scene, manipulator, integrated, or all

# Step 1: Compile C++ modules
echo ""
echo "📦 Step 1: Compiling C++ perception module..."
python setup.py build_ext --inplace

if [ $? -ne 0 ]; then
    echo "❌ Compilation failed!"
    exit 1
fi

echo "✅ Compilation successful!"

# Set PYTHONPATH to include src directory
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Step 2: Run tests based on type
if [ "$TEST_TYPE" == "scene" ] || [ "$TEST_TYPE" == "all" ]; then
    echo ""
    echo "🧪 Step 2: Testing Unitree H1 with enhanced scene..."
    echo ""
    conda run -n neuralnav mjpython src/test/test_h1_scene.py
fi

if [ "$TEST_TYPE" == "manipulator" ] || [ "$TEST_TYPE" == "all" ]; then
    echo ""
    echo "🧪 Step 3: Running manipulator tests..."
    echo ""
    conda run -n neuralnav mjpython src/test/test_manipulator.py
fi

if [ "$TEST_TYPE" == "integrated" ] || [ "$TEST_TYPE" == "all" ]; then
    echo ""
    echo "🧪 Step 4: Running integrated navigation + manipulation demo..."
    echo ""
    conda run -n neuralnav mjpython src/test/test_integrated.py
fi

echo ""
echo "========================================================"
echo "✨ Test completed! Check results above."
echo ""
echo "Usage:"
echo "  ./test_humanoid.sh              # Test H1 scene (default)"
echo "  ./test_humanoid.sh scene        # Test H1 with enhanced scene"
echo "  ./test_humanoid.sh manipulator  # Test arm manipulation"
echo "  ./test_humanoid.sh integrated   # Test integrated tasks"
echo "  ./test_humanoid.sh all          # Run all tests"
