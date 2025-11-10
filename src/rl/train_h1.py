"""
Training script for Unitree H1 robot using Stable-Baselines3.

This script trains an RL agent to control the H1 robot for standing or walking tasks.
Uses PPO (Proximal Policy Optimization) algorithm for training.
"""

import argparse
import os
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import torch

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from rl.h1_env import H1StandingEnv, H1WalkingEnv


def make_env(env_id, rank, seed=0):
    """
    Create a single environment instance.
    
    Args:
        env_id: Environment ID ('standing' or 'walking')
        rank: Process rank for parallel environments
        seed: Random seed
        
    Returns:
        Function that creates the environment
    """
    def _init():
        if env_id == 'standing':
            env = H1StandingEnv(render_mode=None, max_steps=500)
        elif env_id == 'walking':
            env = H1WalkingEnv(render_mode=None, max_steps=1000, target_velocity=0.5)
        else:
            raise ValueError(f"Unknown environment ID: {env_id}")
        
        env.reset(seed=seed + rank)
        return env
    
    return _init


def train_h1(
    task='standing',
    total_timesteps=1_000_000,
    n_envs=4,
    batch_size=256,
    learning_rate=3e-4,
    save_dir='models/policies',
    log_dir='logs/tensorboard',
    checkpoint_freq=50_000,
    eval_freq=10_000,
    n_eval_episodes=5,
    seed=42,
    device='auto',
):
    """
    Train H1 robot using PPO algorithm.
    
    Args:
        task: Task to train ('standing' or 'walking')
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        save_dir: Directory to save trained models
        log_dir: Directory for tensorboard logs
        checkpoint_freq: Frequency to save checkpoints
        eval_freq: Frequency to evaluate model
        n_eval_episodes: Number of episodes for evaluation
        seed: Random seed
        device: Device to use ('auto', 'cpu', or 'cuda')
    """
    # Create directories
    save_dir = Path(save_dir)
    log_dir = Path(log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(f"🤖 Training Unitree H1 - {task.upper()} task")
    print("=" * 70)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {device}")
    print(f"Save directory: {save_dir}")
    print(f"Log directory: {log_dir}")
    print("=" * 70 + "\n")
    
    # Create vectorized training environments
    print(f"Creating {n_envs} parallel training environments...")
    train_env = SubprocVecEnv([make_env(task, i, seed) for i in range(n_envs)])
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env(task, 0, seed + 1000)])
    
    # Create model
    print("\nInitializing PPO model...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=2048 // n_envs,  # Steps per environment per update
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=str(log_dir),
        device=device,
        seed=seed,
    )
    
    print(f"\n📊 Model architecture:")
    print(f"Policy: MLP")
    print(f"Observation space: {train_env.observation_space.shape}")
    print(f"Action space: {train_env.action_space.shape}")
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq // n_envs,
        save_path=str(save_dir / task),
        name_prefix=f"h1_{task}_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / task),
        log_path=str(log_dir / task),
        eval_freq=eval_freq // n_envs,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )
    
    # Train the model
    print("\n🚀 Starting training...")
    print("Monitor progress with: tensorboard --logdir=" + str(log_dir))
    print("=" * 70 + "\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )
    
    # Save final model
    final_model_path = save_dir / task / f"h1_{task}_final.zip"
    model.save(str(final_model_path))
    print(f"\n✅ Training complete! Final model saved to: {final_model_path}")
    
    # Clean up
    train_env.close()
    eval_env.close()
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train Unitree H1 robot with RL')
    
    # Task selection
    parser.add_argument('--task', type=str, default='standing',
                        choices=['standing', 'walking'],
                        help='Task to train (default: standing)')
    
    # Training parameters
    parser.add_argument('--timesteps', type=int, default=1_000_000,
                        help='Total training timesteps (default: 1M)')
    parser.add_argument('--n-envs', type=int, default=4,
                        help='Number of parallel environments (default: 4)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for training (default: 256)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    
    # Save/load parameters
    parser.add_argument('--save-dir', type=str, default='models/policies',
                        help='Directory to save models (default: models/policies)')
    parser.add_argument('--log-dir', type=str, default='logs/tensorboard',
                        help='Directory for tensorboard logs (default: logs/tensorboard)')
    parser.add_argument('--checkpoint-freq', type=int, default=50_000,
                        help='Checkpoint save frequency (default: 50k)')
    parser.add_argument('--eval-freq', type=int, default=10_000,
                        help='Evaluation frequency (default: 10k)')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use (default: auto)')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n🖥️  Using device: {args.device}")
    if args.device == 'cpu':
        print("💡 Tip: Training on GPU is much faster! Use --device cuda if available.\n")
    
    # Train the model
    model = train_h1(
        task=args.task,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        seed=args.seed,
        device=args.device,
    )
    
    print("\n" + "=" * 70)
    print("🎉 Training finished successfully!")
    print("=" * 70)
    print("\nNext steps:")
    print(f"1. Evaluate the model: python src/rl/evaluate_h1.py --task {args.task}")
    print(f"2. View training logs: tensorboard --logdir {args.log_dir}")
    print(f"3. Test in simulation: python src/test/test_h1_scene.py --rl --rl-policy {args.save_dir}/{args.task}/h1_{args.task}_final.zip")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
