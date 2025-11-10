# IntelliH1: Cognitive Humanoid Framework

> **Building reliable robot intelligence from first principles**

An intelligent humanoid control system combining physics-based whole-body control, multimodal perception, and LLM planning for the Unitree H1 robot.

![](img.png)

![Status](https://img.shields.io/badge/status-experimental-orange)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MuJoCo 3.3+](https://img.shields.io/badge/MuJoCo-3.3+-green.svg)](https://mujoco.org/)

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

### 🎯 Implemented Controllers

1. **PD Standing Control** (✅ Available)
   - Based on official Unitree SDK PD gains
   - Ankle, hip, and torso triple balance strategies
   - Gravity compensation support
   - Adaptive gains (automatically adjusts based on drift)

2. **ZMP Balance Control** (🔬 Experimental)
   - Zero Moment Point calculation
   - Contact force feedback integration
   - Dynamic stability analysis
   - Support polygon constraints

3. **Quasi-Static Walking** (⚠️ Unstable)
   - State machine driven gait generation
   - Center of mass transfer strategies
   - Micro-step walking mode

4. **Navigation Control** (🚧 In Development)
   - Target point navigation
   - Path tracking
   - Scene position presets (kitchen, bedroom, living room)

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
# Clone repository
git clone https://github.com/JackSuuu/brainLLM.git
cd brainLLM

# Create conda environment
conda create -n brainllm python=3.10
conda activate brainllm

# Install dependencies
pip install mujoco numpy
chmod +x scripts/install_deps.sh
./scripts/install_deps.sh

# (Optional) Setup Groq API for LLM planning
echo "GROQ_API_KEY=your_key_here" > .env
```

### Basic Usage

```bash
# 1. Standing balance test (default mode)
./demo.sh

# 2. Using simple PD controller
./demo.sh --simple

# 3. Walking mode (experimental)
./demo.sh --walk --walk-speed 0.15

# 4. Navigation mode (in development)
./demo.sh --navigate --target kitchen
./demo.sh --navigate --target 5.0,3.0

# 5. View help
./demo.sh --help
```

### Expected Behavior

- **Standing Mode**: Robot should maintain upright posture for 4-6 seconds, then fall forward due to accumulated drift
- **Walking Mode**: Robot attempts leg swinging but usually loses balance within 2-4 seconds
- **Best Performance**: Improved PD controller can maintain ~6 seconds of stable standing

---

## 📁 Project Structure

```
brainLLM/
├── demo.sh                          # Quick demo launcher
├── models/
│   └── unitree_h1/
│       ├── h1.urdf                  # Robot URDF
│       ├── h1.xml                   # MuJoCo XML
│       └── scene_enhanced.xml       # Environment scene
├── src/
│   ├── config.py                    # Configuration
│   ├── control/                     # Control algorithms
│   │   ├── pd_standing.py           # PD standing controllers
│   │   ├── walking_controller.py    # Walking controller
│   │   ├── zmp_balance.py           # ZMP balance controller
│   │   ├── quasi_static_walking.py  # Quasi-static walking
│   │   ├── rl_policy_loader.py      # RL policy loader
│   │   └── wbc.py                   # Whole-body controller (experimental)
│   ├── robot/
│   │   └── humanoid.py              # Unitree H1 controller
│   ├── perception/                  # Multimodal perception
│   │   ├── vision.py                # RGB-D processing
│   │   ├── multimodal_perception.py # Perception pipeline
│   │   └── cpp/                     # C++ optimized modules
│   │       ├── bindings.cpp
│   │       └── point_cloud_processor.cpp
│   ├── llm/                         # LLM integration
│   │   ├── groq.py                  # Groq API wrapper
│   │   ├── rag_system.py            # RAG pipeline
│   │   ├── physics_kb.py            # Physics knowledge base
│   │   └── planner.py               # Task planner
│   └── simulation/
│       └── environment.py           # MuJoCo simulation wrapper
├── policies/                        # Pre-trained RL policies
├── scripts/
│   ├── download_pretrained_policy.py
│   └── install_deps.sh
└── test/
    ├── test_h1_scene.py             # Main test script
    └── test_manipulator.py          # Manipulator tests
```

---

## 🏗️ Technical Architecture

### Control Flow

```
User Command (demo.sh)
    ↓
test_h1_scene.py (Main Program)
    ↓
UnitreeH1Controller (Robot Interface)
    ↓
[Controller Selection]
    ├── PDStandingController      (Standing)
    ├── ImprovedPDController      (Standing + Gravity Comp)
    ├── ZMPBalanceController      (ZMP Balance)
    ├── SimpleWalkingController   (Sine Gait)
    ├── QuasiStaticWalkingController (Quasi-Static)
    └── NavigationController      (Navigation)
    ↓
MuJoCo Physics Engine
    ↓
Viewer (Visualization)
```

### System Architecture

```
┌─────────────────────────────────────────┐
│         LLM Planning Layer              │
│   Groq + RAG → Task Understanding       │
└──────────────────┬──────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│      Multimodal Perception              │
│   RGB-D + Point Cloud Processing        │
└──────────────────┬──────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│         Motion Controller               │
│  Simple Walk | DM Control | WBC | RL    │
└──────────────────┬──────────────────────┘
                   ↓
┌─────────────────────────────────────────┐
│         MuJoCo Simulator                │
│      Unitree H1 + Environment           │
└─────────────────────────────────────────┘
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

### 3. Navigation Control

```bash
# Navigate to preset locations
./demo.sh --navigate --target kitchen
./demo.sh --navigate --target bedroom
./demo.sh --navigate --target livingroom

# Navigate to custom coordinates
./demo.sh --navigate --target 5.0,3.0
```

**Preset Locations**:

- Kitchen: (5.0, 3.0)
- Bedroom: (-3.0, 6.0)
- Living Room: (0.0, -4.0)

---

## ⚠️ Known Issues

### 🔴 Critical Issues

1. **Insufficient Standing Stability**
   - **Symptom**: Robot falls forward after 4-6 seconds
   - **Cause**:
     - Accumulated position drift (x-direction: 0.01→0.40m)
     - PD compensation gains insufficient against inertia
     - Lack of predictive control
   - **Mitigation**:
     - Increased compensation gains (adaptive gains implemented)
     - ZMP control (experimental)

2. **Walking Fails Immediately**
   - **Symptom**: Falls within 2-4 seconds after starting walking
   - **Cause**:
     - Gait generation conflicts with balance compensation
     - Insufficient dynamic stability
     - Center of mass transfer too fast
   - **Attempted Solutions**:
     - ✗ Sine gait (too aggressive)
     - ✗ Quasi-static walking (state transitions not smooth)
     - ⏳ ZMP control (needs more debugging)

3. **Corrupted RL Policy Files**
   - `policies/h1_demo_policy.pt` and `models/policies/h1_walk.pt` cannot be loaded
   - RL-assisted control temporarily unavailable

4. **Difficult Parameter Tuning**
   - PD gains require fine adjustment
   - Parameters not interchangeable between controllers
   - Lack of systematic parameter optimization tools

5. **Platform Dependencies**
   - macOS requires `mjpython` (auto-detected)
   - Linux and macOS viewer behavior may differ

### 🟡 Minor Issues

- Missing unit tests
- No parameter configuration system (hardcoded gains)
- Incomplete logging system
- Missing performance analysis tools
- Incomplete documentation

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
- **Open-source community** - Proven H1 control implementations
