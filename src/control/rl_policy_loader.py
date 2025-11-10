"""
Simple RL Policy Loader for Unitree H1 Standing
Loads pre-trained PyTorch policy and provides torque predictions
"""

import torch
import numpy as np
from pathlib import Path

class RLPolicyLoader:
    """Minimal RL policy loader for H1 balance assistance"""
    
    def __init__(self, policy_path, device='cpu'):
        """
        Initialize policy loader
        
        Args:
            policy_path: Path to .pt policy file
            device: 'cpu' or 'cuda'
        """
        self.device = device
        
        # Load policy model
        policy_path = Path(policy_path)
        if not policy_path.exists():
            raise FileNotFoundError(f"Policy not found: {policy_path}")
        
        print(f"[RLPolicy] Loading policy from: {policy_path}")
        self.policy = torch.jit.load(str(policy_path), map_location=device)
        self.policy.eval()
        
        print(f"[RLPolicy] ✅ Policy loaded successfully")
    
    def predict(self, observation):
        """
        Get policy action for given observation
        
        Args:
            observation: numpy array of robot state
            
        Returns:
            action: numpy array of joint torques
        """
        with torch.no_grad():
            obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device)
            if obs_tensor.ndim == 1:
                obs_tensor = obs_tensor.unsqueeze(0)  # Add batch dimension
            
            action_tensor = self.policy(obs_tensor)
            action = action_tensor.cpu().numpy().flatten()
        
        return action
    
    def get_observation(self, data, nu):
        """
        Build observation vector from MuJoCo data
        
        Standard observation for humanoid:
        - Joint positions (excluding floating base)
        - Joint velocities
        - Base orientation (quaternion or euler)
        - Base linear velocity
        - Base angular velocity
        
        Args:
            data: MuJoCo data object
            nu: Number of actuators
            
        Returns:
            observation: numpy array
        """
        # Joint states (excluding floating base)
        q = data.qpos[7:7+nu]   # Joint positions
        dq = data.qvel[6:6+nu]  # Joint velocities
        
        # Base orientation (quaternion)
        quat = data.qpos[3:7]
        
        # Base velocities
        base_lin_vel = data.qvel[0:3]  # Linear velocity
        base_ang_vel = data.qvel[3:6]  # Angular velocity
        
        # Concatenate observation
        obs = np.concatenate([
            q,              # Joint positions
            dq,             # Joint velocities
            quat,           # Base orientation
            base_lin_vel,   # Base linear velocity
            base_ang_vel    # Base angular velocity
        ])
        
        return obs
