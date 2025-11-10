# IntelliH1: Cognitive Humanoid Framework

> **Building reliable robot intelligence from first principles**

An intelligent humanoid navigation system combining LLM-based planning (Groq API) with the official Unitree RL walking controller for the Unitree H1 robot.


![Status](https://img.shields.io/badge/status-stable-brightgreen)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MuJoCo 3.3+](https://img.shields.io/badge/MuJoCo-3.3+-green.svg)](https://mujoco.org/)
[![Unitree RL](https://img.shields.io/badge/Unitree-RL_Gym-blue)](https://github.com/unitreerobotics/unitree_rl_gym)

## Demo

https://github.com/user-attachments/assets/b1d41aa8-13d5-47da-9389-0d6525d0fc74

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Technical Architecture](#technical-architecture)
- [Control Modes](#control-modes)
- [Known Issues](#known-issues)
- [Development Notes](#development-notes)
- [References](#references)
- [License](#license)

---

## ✨ Features

### 🎯 Core Capabilities

1. **⭐ Official Unitree RL Walking Controller** (✅ Stable)
   - Pre-trained RL policy from [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)
   - 30+ seconds continuous stable walking
   - Supports H1, H1_2, and G1 robots
   - **Configurable walking speed: 0.5-3.0 m/s** (recommended: 1.0-1.5 m/s)
   - PD tracking with optimized gains
   - Adaptive velocity control based on heading error

2. **🤖 LLM-Driven Navigation** (✅ Production Ready)
   - Natural language command understanding (Groq API + openai/gpt-oss-120b)
   - Automatic target parsing ("walk to the kitchen" → target coordinates)
   - **Real-time navigation with intelligent path following**
   - **Goal-based stopping with 1.2m tolerance**
   - Scene landmarks: Kitchen, Bedroom, Living Room
   - Dynamic waypoint progression tracking

3. **🗺️ A* Path Planning** (✅ Stable)
   - Obstacle-aware global path planning
   - Dynamic occupancy grid mapping
   - Waypoint generation with ~0.3m resolution
   - Real-time path updates every 0.5s
   - Collision avoidance with static obstacles

4. **📡 C++ Radar Perception** (✅ Fully Integrated)
   - **Real-time LIDAR simulation (360° coverage, 10m range)**
   - **C++ optimized point cloud processing (pybind11)**
   - Noise filtering and obstacle detection
   - Occupancy grid generation for path planning
   - ~360 samples per scan with sub-10ms processing time

5. **🗺️ Enhanced Environment Scene** (✅ Available)
   - Kitchen area with counter and cabinet (5.0, 3.0)
   - Bedroom with bed and nightstand (-3.0, 6.0)
   - Living room with couch and coffee table (0.0, -4.0)
   - Manipulable objects (cubes, spheres)

### 🛠️ Tech Stack

- **Physics Engine**: MuJoCo 3.3+
- **Robot Model**: Unitree H1 (URDF/MJCF)
- **Python Version**: 3.10+
- **Core Dependencies**: NumPy, MuJoCo Python bindings
- **AI Integration**: Groq API (Llama 3.3) + LangChain
- **Perception**: OpenCV + C++ optimized point cloud processing
- **Dynamics**: Pinocchio for rigid body dynamics
- **Optimization**: OSQP for quadratic programming

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository with submodules
git clone --recursive https://github.com/JackSuuu/IntelliH1.git
cd IntelliH1

# Create conda environment
conda create -n intellih1 python=3.10
conda activate intellih1

# Install dependencies
pip install mujoco numpy groq python-dotenv
chmod +x scripts/install_deps.sh
./scripts/install_deps.sh

# Setup Groq API for LLM planning
echo "GROQ_API_KEY=your_key_here" > .env

# Compile C++ perception module (optional but recommended)
cd src/perception/cpp
g++ -O3 -Wall -shared -std=c++11 -fPIC \
    $(python3 -m pybind11 --includes) \
    bindings.cpp point_cloud_processor.cpp \
    -o perception_cpp$(python3-config --extension-suffix) \
    $(python3-config --ldflags) \
    -L$(python3-config --prefix)/lib -lpython3.10
cd ../../..
```

### Basic Usage

```bash
# Make demo script executable
chmod +x demo.sh

# 🎯 Quick navigation examples
./demo.sh kitchen                    # Go to kitchen
./demo.sh bedroom                    # Go to bedroom
./demo.sh "walk to living room"      # Natural language

# ⚡ Advanced usage
./demo.sh --speed 1.2 bedroom        # Fast navigation
./demo.sh --speed 0.8 kitchen        # Slower, more stable

# 📋 View all options
./demo.sh --help
```

### Available Locations

| Location | Coordinates | Description |
|----------|-------------|-------------|
| **Kitchen** | (5.0, 3.0) | Kitchen area with counter and cabinet |
| **Bedroom** | (-3.0, 6.0) | Bedroom with bed and nightstand |
| **Living Room** | (0.0, -4.0) | Living room with couch and coffee table |

### Expected Behavior

- **LLM Navigation**: Robot understands natural language commands and plans path to destination
- **C++ Perception**: Real-time LIDAR processing (360 samples) with C++ optimized point cloud filtering
- **Stable Walking**: Robot walks at configurable speed (0.5-3.0 m/s, recommended 1.0-1.5 m/s)
- **Intelligent Control**: Adaptive velocity based on heading error (slows when turning, speeds up when aligned)
- **Goal Detection**: Automatic stopping within 1.2m of destination
- **Path Completion**: Follows 9-waypoint A* path and stops at destination (~6-8m path in 8-12 seconds at 1.0 m/s)

---

## 📁 Project Structure

```
IntelliH1/
├── demo.sh                          # Quick demo launcher
├── COGNITIVE_ARCHITECTURE.md        # Cognitive system documentation
├── reflection.md                    # Development reflections
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package setup
│
├── extern/                          # External dependencies
│   └── unitree_rl_gym/              # Official Unitree RL Gym
│       ├── legged_gym/              # RL training environment
│       │   ├── envs/                # Robot environments (H1, H1_2, G1)
│       │   ├── scripts/             # Training scripts
│       │   └── utils/               # Utilities
│       ├── deploy/                  # Deployment modules
│       │   ├── deploy_mujoco/       # MuJoCo deployment
│       │   ├── deploy_real/         # Real robot deployment
│       │   └── pre_train/           # Pre-trained policies
│       └── resources/               # Robot URDF/MJCF models
│           └── robots/
│               ├── h1/              # H1 robot model
│               ├── h1_2/            # H1_2 robot model
│               └── g1_description/  # G1 robot model
│
├── scripts/
│   └── install_deps.sh              # Dependency installer
│
├── src/                             # Main source code
│   ├── config.py                    # Global configuration
│   │
│   ├── control/                     # Control system (cognitive architecture)
│   │   ├── cognitive_controller.py  # 🧠 Top-level cognitive controller
│   │   │                            #    - LLM integration
│   │   │                            #    - Goal detection (1.2m tolerance)
│   │   │                            #    - Path planning coordination
│   │   └── unitree_rl_controller.py # 🚶 Unitree RL motion controller
│   │                                #    - Walking controller with RL policy
│   │                                #    - Adaptive velocity (heading-based)
│   │                                #    - Navigation control (max 0.8 rad/s)
│   │                                #    - Configurable speed (0.5-3.0 m/s)
│   │
│   ├── robot/
│   │   └── humanoid.py              # Unitree H1 robot interface
│   │
│   ├── perception/                  # Perception system
│   │   ├── multimodal_perception.py # Main perception pipeline
│   │   ├── vision.py                # RGB-D processing
│   │   ├── path_planner.py          # 🗺️ A* path planning (0.3m resolution)
│   │   └── cpp/                     # 📡 C++ optimized modules
│   │       ├── bindings.cpp         #    - pybind11 bindings
│   │       ├── point_cloud_processor.cpp  # LIDAR processing
│   │       └── point_cloud_processor.h    # (360° × 10m range)
│   │
│   ├── llm/                         # 🧠 LLM integration
│   │   ├── groq.py                  # Groq API wrapper
│   │   ├── planner.py               # Task planner
│   │   ├── navigation_planner.py    # Navigation command parser
│   │   ├── rag_system.py            # RAG pipeline
│   │   └── physics_kb.py            # Physics knowledge base
│   │
│   ├── simulation/
│   │   └── environment.py           # MuJoCo simulation wrapper
│   │
│   └── test/                        # Test scripts
│       ├── test_h1_scene.py         # H1 scene tests
│       └── test_llm_navigation.py   # 🎯 LLM navigation demo (main entry)
│
└── .env                             # API keys (create this)
```

### Key Components

| Component | Purpose | Status |
|-----------|---------|--------|
| **cognitive_controller.py** | Top-level cognitive control integrating LLM → Perception → Planning → Motion | ✅ Production |
| **unitree_rl_controller.py** | Unitree RL walking with adaptive velocity and heading correction | ✅ Stable |
| **path_planner.py** | A* global path planning with dynamic obstacle avoidance | ✅ Stable |
| **perception_cpp** | C++ optimized LIDAR processing (360 samples, <10ms) | ✅ Integrated |
| **navigation_planner.py** | LLM-based command parsing and target extraction | ✅ Production |
| **test_llm_navigation.py** | Main demo script for LLM-driven navigation | ✅ Ready |

---

## 🏗️ Technical Architecture

### Control Flow

```text
User Command (demo.sh)
    ↓
test_llm_navigation.py (Main Entry Point)
    ↓
CognitiveController (Top-Level)
    ↓
[4-Layer Cognitive Architecture]
    ├── 🧠 LLM Planning (Groq API + Llama 3.3)
    ├── 👁️ C++ Perception (LIDAR → Point Cloud)
    ├── 🗺️ A* Path Planning (Waypoint Generation)
    └── 🚶 Unitree RL Motion (Walking Control)
    ↓
MuJoCo Physics Engine
    ↓
Viewer (Visualization)
```

### System Architecture

```text
┌─────────────────────────────────────────────────────────┐
│              🧠 LLM Planning Layer                      │
│            Groq API + Llama 3.3 70B                     │
│   Natural Language → Target Position & Speed           │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│           👁️ C++ Perception Layer                      │
│   LIDAR (360° × 10m) + Point Cloud Processing          │
│   C++ pybind11 → Noise Filtering → Occupancy Map       │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│           🗺️ A* Path Planning Layer                    │
│        Grid-based A* + Dynamic Obstacles               │
│   Start + Goal + Map → 9-Waypoint Sequence             │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│           🚶 Unitree RL Motion Control                  │
│   Heading Error Correction + Velocity Scaling          │
│   Waypoints → (vx, ω) Commands → RL Policy → Torques   │
└─────────────────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│              🎮 MuJoCo Physics Engine                   │
│      Unitree H1 (19 DOF) + Enhanced Scene              │
│      Kitchen | Bedroom | Living Room + Obstacles       │
└─────────────────────────────────────────────────────────┘
```

### Control Layer Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Simulation** | MuJoCo 3.3+ | Physics simulation |
| **Dynamics** | Pinocchio 2.6+ | Rigid body dynamics |
| **Optimization** | OSQP 0.6+ | Quadratic programming |
| **Perception** | OpenCV + C++ | Image & point cloud processing |
| **AI** | Groq (Llama 3.3) + LangChain | Task planning & RAG |

---

## 🎮 Control Modes

### 1. Standing Balance Control

```bash
./demo.sh
```

**Implementation Principle**:

- **PD Control**: $\tau = K_p(q_{target} - q) - K_d \dot{q}$
- **Gravity Compensation**: $\tau_{total} = \tau_{PD} + \tau_{gravity}$
- **Balance Strategies**:

```python
ankle_comp = -130 * pos_x - 22 * vel_x   # Ankle strategy
hip_comp = -55 * pos_x - 9 * vel_x       # Hip strategy
torso_comp = -250 * pos_x - 30 * vel_x   # Torso strategy
```

**Controller Parameters**:

| Joint Group | Kp Gain | Kd Gain | Max Torque |
|-------------|---------|---------|-----------|
| Leg Hips | 300-500 | 25-42 | 220-360 Nm |
| Knees | 400-500 | 33-42 | 360 Nm |
| Ankles | 200 | 17 | 75 Nm |
| Torso | 450 | 38 | 200 Nm |
| Arms | 80-120 | 7-10 | 80 Nm |

### 2. Walking Control (Experimental)

```bash
./demo.sh --walk --walk-speed 0.15
```

**Implemented Methods**:

#### Method A: Sine Gait

- Uses sine waves to generate periodic leg motion
- Step frequency: 0.8 Hz
- Step length: 0.05-0.10 m
- Issue: Dynamically unstable, falls within 2-4 seconds

#### Method B: Quasi-Static Walking

- State machine: standing → shift_left → step_right → shift_right → step_left
- Each state lasts 0.5-0.6 seconds
- Issue: Center of mass transfer causes imbalance

#### Method C: ZMP Control (In Development)

```python
# ZMP calculation formula
zmp_x = com_x - (com_z / g) * com_vx
zmp_y = com_y - (com_z / g) * com_vy

# Stability criterion: ZMP must be within support polygon
is_stable = zmp in support_polygon
```

### 3. LLM-Driven Navigation with A* Path Planning

```bash
# Navigate with natural language commands
./demo.sh kitchen                    # Go to kitchen
./demo.sh bedroom                    # Go to bedroom  
./demo.sh "walk to living room"      # Natural language

# Configure speed (0.5-3.0 m/s, recommended: 1.0-1.5 m/s)
./demo.sh --speed 1.0 bedroom        # Balanced speed
./demo.sh --speed 1.5 kitchen        # Fast navigation
./demo.sh --speed 0.8 living_room    # Slower, more stable
```

**Navigation Features**:

- **4-Layer Cognitive Architecture**: LLM → C++ Perception → A* Planning → RL Motion
- **Intelligent Velocity Control**: Slows down when turning (60°+ error → 0.1× speed), speeds up when aligned
- **Goal Detection**: Automatically stops within 1.2m of destination
- **Waypoint Progression**: Tracks progress through 9-waypoint path with real-time feedback
- **Heading Correction**: Proportional control with 0.8 rad/s max turn rate

**Preset Locations**:

- Kitchen: (5.0, 3.0)
- Bedroom: (-3.0, 6.0)
- Living Room: (0.0, -4.0)

---

## ⚠️ Known Issues

### ✅ Recently Fixed

1. **Navigation Speed Configuration** (Fixed ✅)
   - **Issue**: Command-line `--speed` parameter was ignored
   - **Solution**: Propagated `max_speed` through controller hierarchy (CognitiveController → UnitreeRLWalkingController → UnitreeRLController)
   - **Status**: Speed now fully configurable from 0.5-3.0 m/s

2. **Robot Not Stopping at Destination** (Fixed ✅)
   - **Issue**: Robot reached goal area but continued spinning
   - **Solution**:
     - Implemented goal-based stopping (1.2m tolerance around final destination)
     - Added distance-to-goal checking independent of waypoint progression
     - Velocity and target cleared upon goal detection
   - **Status**: Robot now reliably stops at destination

3. **Heading Error Causing Drift** (Fixed ✅)
   - **Issue**: Robot walking in wrong direction with constant omega=0.5 rad/s
   - **Solution**:
     - Adaptive velocity scaling based on heading error (60°+ → 10% speed, 30°+ → 30% speed)
     - Increased turn rate to 0.8 rad/s for faster heading correction
     - Added NAV DEBUG logging every 100 cycles
   - **Status**: Robot now corrects heading before moving forward

### 🔴 Active Issues

1. **Platform Dependencies**
   - macOS requires `mjpython` (auto-detected in demo.sh)
   - Linux and macOS viewer behavior may differ
   - **Workaround**: Auto-detection in demo.sh

2. **C++ Perception Module Compilation**
   - Requires manual compilation with pybind11
   - Platform-specific library paths may need adjustment
   - **Workaround**: Fallback to Python LIDAR simulation if C++ module unavailable

### 🟡 Minor Issues

- Missing comprehensive unit tests
- Navigation tolerance (1.2m) may be too large for precise positioning
- No dynamic obstacle avoidance (only static obstacle map)
- Limited error recovery mechanisms
- Performance metrics not logged systematically

---

## 📝 Development Notes

### Why is Bipedal Walking So Hard?

This project demonstrates the core challenges of bipedal robot control:

#### 1. **Under-Actuated System**

- Robot has 6 DOF floating base (position + orientation)
- Can only control indirectly through joint torques
- Must rely on ground contact forces for motion

#### 2. **Dynamic Instability**

- High center of mass, small support surface
- Any small disturbance accumulates rapidly
- Requires continuous active control to maintain balance

#### 3. **Contact Switching**

- Walking involves transitions: double support → single support → double support
- Each switch is a potential instability point
- Requires precise timing control

#### 4. **Parameter Sensitivity**

```python
# These parameters changing by 10% can cause failure
ankle_gain = 130.0  # ±13 affects stability
knee_stance = 0.25   # ±0.025 affects CoM height
step_frequency = 0.8 # ±0.08 changes gait rhythm
```

### Experimental Records

#### Experiment 1: Basic PD Standing

- **Date**: 2025-11-10
- **Parameters**: Kp=[300,250,400,500,200], Kd=Kp/12
- **Result**: Falls after 6 seconds
- **Issue**: Forward drift 0.01m→0.40m

#### Experiment 2: Adaptive Gains

- **Modification**: `gain = base_gain * (1 + 5*|error|)`
- **Result**: No significant improvement, still falls after 6 seconds
- **Conclusion**: Simply increasing gains insufficient, need different control strategy

#### Experiment 3: Quasi-Static Walking

- **Strategy**: State machine controls center of mass transfer
- **Result**: Immediate imbalance after warmup
- **Cause**: hip_roll compensation conflicts with PD control

#### Experiment 4: Pure PD Walking (Remove Balance Compensation)

- **Modification**: No compensation during walking
- **Result**: Stable for first 6 seconds, falls when starting leg swing
- **Conclusion**: Gait parameters still need significant optimization

### Successful Baseline

The following configuration achieves **stable standing for 4-6 seconds**:

```python
# src/control/pd_standing.py - ImprovedPDController

kp = [300, 250, 400, 500, 200,  # Left leg
      300, 250, 400, 500, 200,  # Right leg
      450,                       # Torso
      120, 120, 80, 80,          # Left arm
      120, 120, 80, 80]          # Right arm

kd = kp / 12.0

# Balance compensation
ankle_comp = -130 * pos_x - 22 * vel_x
hip_comp = -55 * pos_x - 9 * vel_x
torso_comp = -250 * pos_x - 30 * vel_x

# Standing pose
q_target = [
    0.0, 0.0, 0.05, 0.3, -0.15,  # Left leg
    0.0, 0.0, 0.05, 0.3, -0.15,  # Right leg
    0.0,                          # Torso
    0.2, 0.2, 0.0, -0.2,         # Left arm
    0.2, -0.2, 0.0, -0.2         # Right arm
]
```

---

## 📚 References

### Official Resources

- [Unitree Robotics](https://www.unitree.com/)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [Unitree SDK2 Python](https://github.com/unitreerobotics/unitree_sdk2_python)

### Related Projects

- [fan-ziqi/h1-mujoco-sim](https://github.com/fan-ziqi/h1-mujoco-sim) - H1 MuJoCo simulation
- [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) - Unitree RL training environment

### Academic References

- **ZMP Control**: Vukobratović & Borovac (2004) - "Zero-Moment Point"
- **Gait Planning**: Kajita et al. (2003) - "Biped Walking Pattern Generation"
- **PD Control**: Pratt et al. (2001) - "Virtual Model Control"

### ⭐ Known Working Unitree H1 Controllers

**Highly Recommended - Use these as reference!**

1. **Unitree Official**
   - [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) - Official RL training framework
   - [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) - Official SDK with working examples
   - **Status**: ✅ Production ready, hardware-tested

2. **RobotLoco** ⭐⭐⭐
   - [robot_loco](https://github.com/zitongbai/robot_loco) - Locomotion controller for H1
   - Includes: MPC + WBC implementation that actually works!
   - **Status**: ✅ Well-documented, simulation + real robot

3. **H1_MuJoCo_Controller**
   - [h1-mujoco-sim](https://github.com/fan-ziqi/h1-mujoco-sim) - Direct MuJoCo simulation
   - Simple PD controller with gravity compensation
   - **Status**: ✅ Stable standing and walking

4. **LeRobot (Hugging Face)**
   - [lerobot](https://github.com/huggingface/lerobot) - Includes H1 teleoperation
   - Uses Isaac Sim + MuJoCo
   - **Status**: ✅ Active development, well-maintained

5. **Whole-Body MPC**
   - [Crocoddyl + H1](https://github.com/machines-in-motion/mim_robots) - MPC-based controller
   - Research-grade optimal control
   - **Status**: ✅ Academic reference

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

---

## 🙏 Acknowledgments

- **Unitree Robotics** - H1 humanoid robot model
- **MuJoCo Team** - Excellent physics engine
- **Pinocchio** - Efficient dynamics library
- **Groq** - Fast LLM inference
- **Open-source community** - Proven H1 control implementations ([unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym))
