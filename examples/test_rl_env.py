"""
Simple test script for H1 Gymnasium environments.

This script demonstrates how to use the H1 RL environments and tests
that they work correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rl.h1_env import H1StandingEnv, H1WalkingEnv
import numpy as np


def test_standing_env():
    """Test H1 standing environment with random actions."""
    print("=" * 70)
    print("Testing H1 Standing Environment")
    print("=" * 70 + "\n")
    
    # Create environment
    env = H1StandingEnv(render_mode=None, max_steps=100)
    
    print(f"✅ Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print()
    
    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"✅ Environment reset successfully!")
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial pelvis height: {info['pelvis_height']:.3f}m")
    print()
    
    # Run a few steps with random actions
    print("Running 10 steps with random actions...")
    total_reward = 0
    
    for step in range(10):
        # Sample random action
        action = env.action_space.sample()
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"  Step {step + 1}: Height={info['pelvis_height']:.3f}m, Reward={reward:.2f}")
        
        if terminated or truncated:
            print(f"  Episode ended at step {step + 1}")
            break
    
    print(f"\n✅ Test completed!")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final height: {info['pelvis_height']:.3f}m")
    
    env.close()
    print("\n" + "=" * 70 + "\n")


def test_walking_env():
    """Test H1 walking environment with random actions."""
    print("=" * 70)
    print("Testing H1 Walking Environment")
    print("=" * 70 + "\n")
    
    # Create environment
    env = H1WalkingEnv(render_mode=None, max_steps=100, target_velocity=0.5)
    
    print(f"✅ Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Target velocity: {env.target_velocity} m/s")
    print()
    
    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"✅ Environment reset successfully!")
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial pelvis height: {info['pelvis_height']:.3f}m")
    print()
    
    # Run a few steps with random actions
    print("Running 10 steps with random actions...")
    total_reward = 0
    
    for step in range(10):
        # Sample random action
        action = env.action_space.sample()
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"  Step {step + 1}: Height={info['pelvis_height']:.3f}m, "
              f"Dist={info.get('distance_traveled', 0):.3f}m, Reward={reward:.2f}")
        
        if terminated or truncated:
            print(f"  Episode ended at step {step + 1}")
            break
    
    print(f"\n✅ Test completed!")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final height: {info['pelvis_height']:.3f}m")
    print(f"Distance traveled: {info.get('distance_traveled', 0):.3f}m")
    
    env.close()
    print("\n" + "=" * 70 + "\n")


def test_env_compatibility():
    """Test that environments follow Gymnasium API."""
    print("=" * 70)
    print("Testing Gymnasium API Compatibility")
    print("=" * 70 + "\n")
    
    env = H1StandingEnv()
    
    # Test reset
    obs, info = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape, "Observation shape mismatch"
    assert isinstance(info, dict), "Info should be a dict"
    print("✅ reset() returns correct types")
    
    # Test step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == env.observation_space.shape, "Observation shape mismatch"
    assert isinstance(reward, (int, float)), "Reward should be a number"
    assert isinstance(terminated, bool), "Terminated should be bool"
    assert isinstance(truncated, bool), "Truncated should be bool"
    assert isinstance(info, dict), "Info should be a dict"
    print("✅ step() returns correct types")
    
    # Test action space
    for _ in range(5):
        action = env.action_space.sample()
        assert env.action_space.contains(action), "Sampled action not in action space"
    print("✅ action_space.sample() works correctly")
    
    # Test observation space
    obs, _ = env.reset()
    assert env.observation_space.contains(obs), "Observation not in observation space"
    print("✅ Observations are in observation space")
    
    env.close()
    
    print("\n✅ All compatibility tests passed!")
    print("\n" + "=" * 70 + "\n")


def main():
    """Run all tests."""
    print("\n" + "🤖 H1 Gymnasium Environment Test Suite" + "\n")
    
    try:
        # Test API compatibility
        test_env_compatibility()
        
        # Test standing environment
        test_standing_env()
        
        # Test walking environment
        test_walking_env()
        
        print("🎉 All tests passed successfully!")
        print("\nThe RL environments are ready to use for training!")
        print("\nNext steps:")
        print("  1. Train a policy: python src/rl/train_h1.py --task standing --timesteps 10000")
        print("  2. See documentation: src/rl/README.md")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
