# 🚀 Quick Start: RL Training for Unitree H1

Get started with reinforcement learning for the H1 robot in 5 minutes!

## 📦 Installation

```bash
# Install RL dependencies
pip install gymnasium stable-baselines3 tensorboard

# Verify installation
python examples/test_rl_env.py
```

## 🎯 Train Your First Policy (Standing)

```bash
# Quick test (1 minute)
python src/rl/train_h1.py --task standing --timesteps 10000 --n-envs 2

# Full training (30-60 minutes on CPU)
python src/rl/train_h1.py --task standing --timesteps 1000000 --n-envs 4

# Fast training with GPU (5-10 minutes)
python src/rl/train_h1.py --task standing --timesteps 1000000 --n-envs 8 --device cuda
```

## 📊 Monitor Training

Open TensorBoard in a new terminal:

```bash
tensorboard --logdir logs/tensorboard
```

Then open http://localhost:6006 in your browser to watch training progress in real-time!

## 🎮 Test Your Trained Policy

```bash
# Evaluate the trained policy
python src/rl/evaluate_h1.py \
    --policy models/policies/standing/best_model.zip \
    --task standing \
    --episodes 10

# Quick test (3 episodes, no visualization)
python src/rl/evaluate_h1.py \
    --policy models/policies/standing/best_model.zip \
    --task standing \
    --episodes 3 \
    --no-render
```

## 🚶 Train Walking Policy

```bash
# Train walking (1-2 hours on CPU)
python src/rl/train_h1.py --task walking --timesteps 2000000 --n-envs 8

# Evaluate walking
python src/rl/evaluate_h1.py \
    --policy models/policies/walking/best_model.zip \
    --task walking \
    --episodes 10
```

## 🔧 Customize Training

Edit `configs/rl_config.yaml` to adjust:
- Reward function weights
- Training hyperparameters
- Environment parameters

## 💡 Pro Tips

1. **Start with standing** before attempting walking
2. **Use GPU** for 5-10x speedup: `--device cuda`
3. **Increase parallel envs** for faster training: `--n-envs 16`
4. **Monitor progress** with TensorBoard to catch issues early
5. **Save checkpoints** are automatic - no need to babysit training!

## 📚 Learn More

- **Detailed Guide**: `src/rl/README.md`
- **Integration Docs**: `docs/RL_INTEGRATION.md`
- **Configuration**: `configs/rl_config.yaml`

## 🆘 Troubleshooting

**Robot falls immediately?**
- Training just started - give it time (100k+ steps)
- Check reward in TensorBoard - should increase over time

**Training too slow?**
- Use GPU: `--device cuda`
- Increase parallel environments: `--n-envs 8`

**Out of memory?**
- Reduce parallel environments: `--n-envs 2`
- Reduce batch size: `--batch-size 128`

## 🎉 Success Metrics

### Standing
- ✅ Episode length > 450 steps (4.5 seconds)
- ✅ Success rate > 80%
- ✅ Converges in 200k-500k steps

### Walking
- ✅ Distance traveled > 2 meters
- ✅ Forward velocity ≈ 0.5 m/s
- ✅ Success rate > 60%
- ✅ Converges in 1M-2M steps

---

**Happy Training! 🤖🚀**
