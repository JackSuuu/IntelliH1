#!/bin/bash
# Quick test script for BrainLLM standing control

cd /Users/jacksu/Desktop/ARCHIVE/2025/text2wheel

echo "🧠 BrainLLM - Quick Standing Test"
echo "=================================="
echo ""
echo "Running humanoid standing test with gravity compensation..."
echo ""

conda run -n neuralnav env \
  PYTHONPATH=/Users/jacksu/Desktop/ARCHIVE/2025/text2wheel/src:$PYTHONPATH \
  mjpython src/test/test_h1_scene.py
