#!/bin/bash
# BrainLLM - 依赖安装脚本
# 支持 macOS / Linux

set -e  # 遇到错误立即退出

echo "============================================"
echo "🧠 BrainLLM - Installing Dependencies"
echo "============================================"

# 检测操作系统
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
else
    echo "❌ Unsupported OS: $OSTYPE"
    exit 1
fi

echo "✓ Detected OS: $OS"

# 1. 安装系统依赖（C++库）
echo ""
echo "📦 Step 1/3: Installing system dependencies..."

if [ "$OS" == "macos" ]; then
    # macOS (Homebrew)
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew not found. Please install: https://brew.sh"
        exit 1
    fi
    
    echo "Installing via Homebrew..."
    brew install eigen osqp yaml-cpp pybind11 || true
    
    # Pinocchio (特殊安装)
    brew tap gepetto/homebrew-gepetto || true
    brew install pinocchio || true
    
elif [ "$OS" == "linux" ]; then
    # Linux (APT)
    echo "Installing via apt..."
    sudo apt-get update
    sudo apt-get install -y \
        libeigen3-dev \
        libosqp-dev \
        libyaml-cpp-dev \
        pybind11-dev \
        robotpkg-py310-pinocchio || true
fi

# 2. 安装Python依赖
echo ""
echo "🐍 Step 2/3: Installing Python dependencies..."

# 检查conda环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  Not in conda environment. Installing to system Python..."
    pip install -r requirements.txt
else
    echo "✓ Using conda environment: $CONDA_DEFAULT_ENV"
    conda run -n "$CONDA_DEFAULT_ENV" pip install -r requirements.txt
fi

# 3. 编译C++模块（可选 - 后续Phase）
echo ""
echo "🔧 Step 3/3: Building C++ modules (optional)..."

if [ -d "cpp" ]; then
    echo "C++ source found, building..."
    cd cpp
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
    cd ../..
    echo "✓ C++ modules built successfully"
else
    echo "⏭  Skipping C++ build (not implemented yet)"
fi

echo ""
echo "============================================"
echo "✅ Installation complete!"
echo "============================================"
echo ""
echo "🚀 Quick start:"
echo "   conda activate neuralnav"
echo "   python tests/test_standing.py"
echo ""
echo "📖 Documentation: REFACTOR_PLAN.md"
echo "============================================"
