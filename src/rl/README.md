# Reinforcement Learning Module for Unitree H1

This module provides Gymnasium environments and training utilities for teaching the Unitree H1 humanoid robot to stand and walk using deep reinforcement learning.

## Overview

The RL module includes:
- **Gymnasium Environments**: `H1StandingEnv` and `H1WalkingEnv` for training standing balance and walking
- **Training Scripts**: Using Stable-Baselines3 PPO algorithm
- **Evaluation Scripts**: For testing trained policies
- **Configuration Files**: YAML configs for easy parameter tuning

## Quick Start

### 1. Install Dependencies

```bash
pip install gymnasium stable-baselines3 tensorboard
```

### 2. Train a Standing Policy

```bash
# Basic training (1M steps, ~30-60 minutes on CPU)
python src/rl/train_h1.py --task standing

# Fast training with GPU (if available)
python src/rl/train_h1.py --task standing --device cuda --n-envs 8

# Short test run
python src/rl/train_h1.py --task standing --timesteps 10000
```

### 3. Evaluate the Trained Policy

```bash
# Evaluate with visualization
python src/rl/evaluate_h1.py --policy models/policies/standing/h1_standing_final.zip --task standing

# Quick evaluation (3 episodes, no rendering)
python src/rl/evaluate_h1.py --policy models/policies/standing/best_model.zip --task standing --episodes 3 --no-render
```

### 4. Train a Walking Policy

```bash
# Train walking (2M steps, ~1-2 hours on CPU)
python src/rl/train_h1.py --task walking --timesteps 2000000

# With custom parameters
python src/rl/train_h1.py --task walking --timesteps 2000000 --n-envs 8 --lr 0.0003 --device cuda
```

## Environment Details

### H1StandingEnv

**Goal**: Keep the robot standing upright at ~1.0m height

**Observation Space** (48 dimensions):
- Joint positions (19)
- Joint velocities (19)
- Base orientation quaternion (4)
- Base linear velocity (3)
- Base angular velocity (3)

**Action Space** (19 dimensions):
- Normalized joint torques [-1, 1] for all actuated joints

**Reward Function**:
- Height reward: Gaussian penalty for deviation from target height (1.0m)
- Orientation reward: Penalty for tilting away from upright
- Stability reward: Bonus for low angular velocity
- Energy penalty: Penalty for excessive torque usage

**Episode Termination**:
- Robot falls (pelvis height < 0.5m)
- Max steps reached (500 steps = 5 seconds)

### H1WalkingEnv

**Goal**: Walk forward at target velocity (~0.5 m/s) while maintaining balance

**Observation/Action Space**: Same as H1StandingEnv

**Additional Rewards**:
- Velocity reward: Gaussian penalty for deviation from target forward velocity
- Lateral stability: Penalty for sideways drift

**Episode Termination**:
- Robot falls (pelvis height < 0.5m)
- Max steps reached (1000 steps = 10 seconds)

## Training Parameters

Key hyperparameters for PPO training:

```python
# Learning parameters
learning_rate = 3e-4          # Adam learning rate
batch_size = 256              # Mini-batch size
n_epochs = 10                 # Epochs per policy update

# PPO parameters
gamma = 0.99                  # Discount factor
gae_lambda = 0.95            # GAE parameter
clip_range = 0.2             # PPO clipping parameter
ent_coef = 0.0               # Entropy coefficient (0 = deterministic)

# Training setup
n_envs = 4                   # Parallel environments
total_timesteps = 1_000_000  # Total training steps
```

See `configs/rl_config.yaml` for complete configuration options.

## Monitoring Training

### TensorBoard

Monitor training progress in real-time:

```bash
tensorboard --logdir logs/tensorboard
```

Then open http://localhost:6006 in your browser.

**Key metrics to watch**:
- `rollout/ep_rew_mean`: Average episode reward (should increase)
- `rollout/ep_len_mean`: Average episode length (should increase)
- `train/loss`: Training loss (should decrease)
- `train/explained_variance`: How well value function predicts returns (closer to 1 is better)

### Checkpoints

Models are automatically saved during training:
- **Checkpoints**: Every 50k steps → `models/policies/{task}/h1_{task}_checkpoint_*`
- **Best model**: Based on evaluation → `models/policies/{task}/best_model.zip`
- **Final model**: End of training → `models/policies/{task}/h1_{task}_final.zip`

## Integration with Existing Controllers

The trained RL policies can be used with the existing H1 controller:

```bash
# Use RL policy for standing
python src/test/test_h1_scene.py --rl --rl-policy models/policies/standing/best_model.zip

# Blend RL with PD controller (safer)
python src/test/test_h1_scene.py --rl --rl-weight 0.3 --rl-policy models/policies/standing/best_model.zip
```

The `--rl-weight` parameter controls the blend:
- `0.0`: Pure PD controller
- `1.0`: Pure RL policy
- `0.3`: Recommended blend (30% RL, 70% PD)

## Training Tips

### For Better Results

1. **Start Simple**: Train standing before walking
2. **Use GPU**: Training is 5-10x faster on GPU (`--device cuda`)
3. **Parallel Environments**: More environments = faster training, but requires more RAM
4. **Monitor Training**: Use TensorBoard to detect problems early
5. **Patience**: Good policies need time (1M steps ≈ 30-60 min on CPU)

### Troubleshooting

**Robot falls immediately**:
- Check initial pose in `reset()` function
- Increase height reward weight
- Add more stability reward

**Training not improving**:
- Try different learning rate (1e-4 to 1e-3)
- Increase number of environments
- Adjust reward function weights

**Training too slow**:
- Use GPU (`--device cuda`)
- Increase number of environments
- Reduce evaluation frequency

**Not walking forward**:
- Increase velocity reward weight
- Reduce energy penalty
- Increase episode length (max_steps)

## Advanced Usage

### Custom Reward Function

Edit `src/rl/h1_env.py` to modify reward calculation:

```python
def _compute_reward(self, action, info):
    # Your custom reward logic here
    height_reward = ...
    orientation_reward = ...
    # ...
    return total_reward
```

### Curriculum Learning

Train progressively harder tasks:

```python
# 1. Train standing
python src/rl/train_h1.py --task standing --timesteps 500000

# 2. Train walking starting from standing policy
python src/rl/train_h1.py --task walking --timesteps 1000000 --pretrained models/policies/standing/best_model.zip
```

(Note: `--pretrained` flag not yet implemented, but can be added)

### Hyperparameter Tuning

Use different parameters for experimentation:

```bash
python src/rl/train_h1.py \
    --task standing \
    --timesteps 500000 \
    --lr 0.0001 \
    --batch-size 128 \
    --n-envs 16
```

## File Structure

```
src/rl/
├── __init__.py           # Module exports
├── h1_env.py            # Gymnasium environments
├── train_h1.py          # Training script
├── evaluate_h1.py       # Evaluation script
└── README.md            # This file

configs/
└── rl_config.yaml       # Training configuration

models/policies/         # Saved models (created during training)
├── standing/
│   ├── best_model.zip
│   ├── h1_standing_final.zip
│   └── checkpoints/
└── walking/
    ├── best_model.zip
    └── h1_walking_final.zip

logs/tensorboard/        # Training logs (created during training)
├── standing/
└── walking/
```

## Performance Expectations

### Standing Task

- **Convergence**: ~200k-500k steps
- **Success Criteria**: Episode length > 450 steps (4.5 seconds)
- **Expected Performance**: 80-90% success rate

### Walking Task

- **Convergence**: ~1M-2M steps
- **Success Criteria**: 
  - Episode length > 800 steps (8 seconds)
  - Distance traveled > 2 meters
  - Forward velocity ≈ 0.5 m/s
- **Expected Performance**: 60-80% success rate

## References

- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
- **Gymnasium**: https://gymnasium.farama.org/
- **PPO Paper**: https://arxiv.org/abs/1707.06347
- **MuJoCo**: https://mujoco.readthedocs.io/

## Future Improvements

Potential enhancements:
- [ ] Add domain randomization for robustness
- [ ] Implement vision-based observations
- [ ] Add multi-task learning (standing + walking)
- [ ] Support for different walking speeds and directions
- [ ] Add obstacle avoidance
- [ ] Implement curriculum learning
- [ ] Add pre-trained policies for quick start
