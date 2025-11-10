"""
Evaluation script for trained H1 policies.

This script loads a trained policy and evaluates it in the simulation environment.
"""

import argparse
from pathlib import Path
import numpy as np
import time
from stable_baselines3 import PPO
import sys

sys.path.append(str(Path(__file__).parent.parent))

from rl.h1_env import H1StandingEnv, H1WalkingEnv


def evaluate_policy(
    policy_path,
    task='standing',
    n_episodes=10,
    render=True,
    deterministic=True,
    save_video=False,
):
    """
    Evaluate a trained policy.
    
    Args:
        policy_path: Path to trained policy (.zip file)
        task: Task type ('standing' or 'walking')
        n_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        deterministic: Whether to use deterministic actions
        save_video: Whether to save video (not implemented yet)
        
    Returns:
        dict: Evaluation results including mean/std reward and episode length
    """
    print("=" * 70)
    print(f"🤖 Evaluating H1 {task.upper()} Policy")
    print("=" * 70)
    print(f"Policy: {policy_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Render: {render}")
    print(f"Deterministic: {deterministic}")
    print("=" * 70 + "\n")
    
    # Load policy
    print("Loading policy...")
    model = PPO.load(policy_path)
    print("✅ Policy loaded successfully!\n")
    
    # Create environment
    print("Creating environment...")
    render_mode = "human" if render else None
    
    if task == 'standing':
        env = H1StandingEnv(render_mode=render_mode, max_steps=500)
    elif task == 'walking':
        env = H1WalkingEnv(render_mode=render_mode, max_steps=1000, target_velocity=0.5)
    else:
        raise ValueError(f"Unknown task: {task}")
    
    print(f"✅ Environment created: {task}\n")
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    # Run episodes
    print("Starting evaluation...")
    print("-" * 70)
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        
        while not done:
            # Get action from policy
            action, _states = model.predict(obs, deterministic=deterministic)
            
            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            # Render if requested
            if render:
                env.render()
                time.sleep(0.01)  # Small delay for visualization
            
            # Print periodic updates
            if episode_length % 100 == 0:
                height = info.get('pelvis_height', 0)
                print(f"  Step {episode_length}: Height={height:.3f}m, Reward={episode_reward:.2f}")
        
        # Episode finished
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Check success criteria
        height = info.get('pelvis_height', 0)
        if height > 0.5:  # Still standing
            success_count += 1
            result = "✅ SUCCESS"
        else:
            result = "❌ FELL"
        
        print(f"  Episode finished: {result}")
        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Episode length: {episode_length} steps")
        print(f"  Final height: {height:.3f}m")
        
        if task == 'walking' and 'distance_traveled' in info:
            dist = info['distance_traveled']
            vel = info.get('forward_velocity', 0)
            print(f"  Distance traveled: {dist:.3f}m")
            print(f"  Final velocity: {vel:.3f}m/s")
    
    # Compute statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    success_rate = success_count / n_episodes
    
    print("\n" + "=" * 70)
    print("📊 EVALUATION RESULTS")
    print("=" * 70)
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean episode length: {mean_length:.1f} ± {std_length:.1f} steps")
    print(f"Success rate: {success_rate:.1%} ({success_count}/{n_episodes})")
    print("=" * 70 + "\n")
    
    env.close()
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_length': mean_length,
        'std_length': std_length,
        'success_rate': success_rate,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained H1 policy')
    
    parser.add_argument('--policy', type=str, required=True,
                        help='Path to trained policy (.zip file)')
    parser.add_argument('--task', type=str, default='standing',
                        choices=['standing', 'walking'],
                        help='Task type (default: standing)')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering')
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic actions (default: deterministic)')
    parser.add_argument('--save-video', action='store_true',
                        help='Save video of evaluation (not implemented)')
    
    args = parser.parse_args()
    
    # Check if policy exists
    policy_path = Path(args.policy)
    if not policy_path.exists():
        print(f"❌ Error: Policy file not found: {policy_path}")
        print("\nAvailable policies:")
        policies_dir = Path("models/policies")
        if policies_dir.exists():
            for policy_file in policies_dir.rglob("*.zip"):
                print(f"  - {policy_file}")
        return
    
    # Run evaluation
    results = evaluate_policy(
        policy_path=str(policy_path),
        task=args.task,
        n_episodes=args.episodes,
        render=not args.no_render,
        deterministic=not args.stochastic,
        save_video=args.save_video,
    )
    
    print("✅ Evaluation complete!")


if __name__ == '__main__':
    main()
