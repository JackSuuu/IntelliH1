# Gymnasium RL Integration for Unitree H1

## Overview

This document describes the reinforcement learning integration for the Unitree H1 humanoid robot. The integration uses Gymnasium (formerly OpenAI Gym) for the environment interface and Stable-Baselines3 for training state-of-the-art RL algorithms.

## What Was Added

### 1. Gymnasium Environments (`src/rl/h1_env.py`)

Two custom Gymnasium environments were created:

#### H1StandingEnv
- **Goal**: Train the robot to maintain standing balance at ~1.0m height
- **Episode Length**: 500 steps (5 seconds)
- **Observation Space**: 48 dimensions
  - Joint positions (19)
  - Joint velocities (19)
  - Base orientation quaternion (4)
  - Base linear velocity (3)
  - Base angular velocity (3)
- **Action Space**: 19 continuous actions (normalized joint torques)
- **Reward Function**:
  - Height reward: Gaussian around target height
  - Orientation reward: Stay upright
  - Stability reward: Low angular velocity
  - Energy penalty: Minimize torque usage

#### H1WalkingEnv
- **Goal**: Train the robot to walk forward at target velocity (~0.5 m/s)
- **Episode Length**: 1000 steps (10 seconds)
- **Additional Rewards**:
  - Velocity reward: Match target forward velocity
  - Lateral penalty: Minimize sideways drift
- Inherits all features from H1StandingEnv

### 2. Training Script (`src/rl/train_h1.py`)

A comprehensive training script using PPO (Proximal Policy Optimization):

**Features:**
- Multi-environment parallel training for faster convergence
- Automatic checkpointing every 50k steps
- Periodic evaluation with best model saving
- TensorBoard integration for training visualization
- Configurable hyperparameters via command-line arguments
- GPU support for accelerated training

**Usage:**
```bash
# Basic standing training
python src/rl/train_h1.py --task standing --timesteps 1000000

# Advanced training with GPU
python src/rl/train_h1.py --task walking --timesteps 2000000 --n-envs 8 --device cuda
```

### 3. Evaluation Script (`src/rl/evaluate_h1.py`)

Script to test trained policies:

**Features:**
- Load and evaluate saved models
- Deterministic or stochastic action selection
- Multiple episode evaluation with statistics
- Optional visualization
- Success rate calculation

**Usage:**
```bash
python src/rl/evaluate_h1.py --policy models/policies/standing/best_model.zip --task standing --episodes 10
```

### 4. Configuration File (`configs/rl_config.yaml`)

YAML configuration file with:
- Task-specific parameters (standing vs walking)
- Reward function weights
- PPO hyperparameters
- Training settings
- Checkpoint and logging configuration

### 5. Test Script (`examples/test_rl_env.py`)

Environment validation script that checks:
- Gymnasium API compatibility
- Correct observation/action space types
- Environment reset and step functionality
- Standing and walking environment behavior

### 6. Documentation

- **Main README**: Updated with RL section
- **RL Module README** (`src/rl/README.md`): Comprehensive guide with:
  - Quick start instructions
  - Environment details
  - Training parameters
  - Monitoring guide
  - Integration with existing controllers
  - Troubleshooting tips
  - Advanced usage examples

## Architecture

```
┌─────────────────────────────────────┐
│     Gymnasium Environment           │
│  (H1StandingEnv / H1WalkingEnv)    │
│                                     │
│  ┌───────────────────────────────┐ │
│  │   MuJoCo Simulation           │ │
│  │   Unitree H1 Model            │ │
│  └───────────────────────────────┘ │
└─────────────────┬───────────────────┘
                  │
                  ↓
┌─────────────────────────────────────┐
│  Stable-Baselines3 (PPO)            │
│  ┌────────────────────────────────┐ │
│  │  Policy Network (MLP)          │ │
│  │  Value Network (MLP)           │ │
│  └────────────────────────────────┘ │
└─────────────────┬───────────────────┘
                  │
                  ↓
┌─────────────────────────────────────┐
│      Training / Evaluation          │
│  - Parallel environments            │
│  - TensorBoard logging              │
│  - Automatic checkpointing          │
│  - Model evaluation                 │
└─────────────────────────────────────┘
```

## Key Design Decisions

### 1. Reward Function Design

The reward function was carefully designed to encourage stable standing/walking:

- **Gaussian rewards**: Smooth gradients for height and velocity targets
- **Exponential decay**: Quick penalty for deviation from targets
- **Energy penalty**: Prevents excessive torque usage
- **Multiple objectives**: Balances height, orientation, stability, and movement

### 2. Action Space: Joint Torques

Direct torque control was chosen over position control:
- More flexible and natural for RL
- Better for learning dynamic behaviors
- Follows standard practice in humanoid RL

### 3. Observation Space

Includes both proprioceptive and kinematic information:
- Joint states for local control
- Base orientation for balance
- Velocities for dynamics awareness
- No vision (future enhancement)

### 4. PPO Algorithm

PPO was chosen for several reasons:
- State-of-the-art for continuous control
- Sample efficient
- Stable training
- Well-supported by Stable-Baselines3

### 5. Parallel Environments

Multi-environment training provides:
- Faster data collection
- Better gradient estimates
- More diverse experiences
- Reduced training time

## Integration with Existing System

The RL module integrates seamlessly with the existing H1 controller:

1. **Standalone Training**: Train policies independently
2. **Hybrid Control**: Blend RL with PD controllers
3. **Policy Export**: Save trained models as `.zip` files
4. **Easy Loading**: Load policies with RL policy loader

Example hybrid usage:
```bash
python src/test/test_h1_scene.py --rl --rl-weight 0.3 --rl-policy models/policies/standing/best_model.zip
```

## Training Recommendations

### For Standing Task
- **Timesteps**: 500k - 1M
- **Environments**: 4-8
- **Expected Time**: 30-60 minutes (CPU), 5-10 minutes (GPU)
- **Success Criteria**: Episode length > 450 steps

### For Walking Task
- **Timesteps**: 1M - 2M
- **Environments**: 8-16
- **Expected Time**: 1-2 hours (CPU), 15-30 minutes (GPU)
- **Success Criteria**: Distance > 2m, velocity ≈ 0.5 m/s

## Monitoring Training

Use TensorBoard to monitor:
```bash
tensorboard --logdir logs/tensorboard
```

Key metrics:
- `rollout/ep_rew_mean`: Should increase over time
- `rollout/ep_len_mean`: Should increase (longer episodes)
- `train/explained_variance`: Should approach 1.0
- `train/loss`: Should decrease initially

## Future Enhancements

Potential improvements:
1. **Domain Randomization**: Add noise for robustness
2. **Curriculum Learning**: Progressive task difficulty
3. **Vision-Based Control**: Add camera observations
4. **Multi-Task Learning**: Single policy for multiple tasks
5. **Imitation Learning**: Bootstrap from demonstrations
6. **Terrain Adaptation**: Train on varied surfaces
7. **Disturbance Rejection**: Train with external forces

## Performance Expectations

With proper training:

### Standing
- Success rate: 80-90%
- Stable duration: 4-5+ seconds
- Converges in: 200k-500k steps

### Walking
- Success rate: 60-80%
- Walking distance: 2-4 meters
- Forward velocity: 0.4-0.6 m/s
- Converges in: 1M-2M steps

## Dependencies

New dependencies added:
- `gymnasium>=0.29.0`: RL environment interface
- `stable-baselines3>=2.2.0`: RL algorithms
- `tensorboard>=2.15.0`: Training visualization
- `torch>=2.0.0`: Neural network backend (already present)

All dependencies checked for vulnerabilities ✅

## Testing

All components tested:
- ✅ Environment API compatibility
- ✅ Training script functionality
- ✅ Evaluation script functionality
- ✅ Type conversions (bool, float)
- ✅ No security vulnerabilities

## Security Summary

- CodeQL scan: 0 alerts
- No vulnerabilities in new dependencies
- No sensitive data exposure
- Safe file operations with proper path handling

## Conclusion

The RL integration provides a complete framework for training the Unitree H1 robot to stand and walk using deep reinforcement learning. The implementation follows best practices, uses proven algorithms, and provides comprehensive tooling for training, evaluation, and monitoring.

For detailed usage instructions, see `src/rl/README.md`.
