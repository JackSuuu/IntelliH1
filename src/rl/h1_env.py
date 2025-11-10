"""
Gymnasium Environment for Unitree H1 Robot

This environment wraps the MuJoCo simulation of the Unitree H1 humanoid robot
for reinforcement learning tasks including standing balance and walking.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from pathlib import Path


class H1StandingEnv(gym.Env):
    """
    Gymnasium environment for training H1 robot to stand upright.
    
    Observation Space:
        - Joint positions (19 joints)
        - Joint velocities (19 joints)
        - Base orientation (quaternion, 4 values)
        - Base linear velocity (3 values)
        - Base angular velocity (3 values)
        Total: 19 + 19 + 4 + 3 + 3 = 48 dimensions
    
    Action Space:
        - Joint torques for 19 actuated joints
        - Continuous action space: [-1, 1] normalized to actual torque limits
    
    Reward:
        - Height penalty: reward for keeping pelvis at target height (~1.0m)
        - Orientation penalty: reward for keeping upright
        - Stability bonus: bonus for maintaining balance
        - Energy penalty: penalty for excessive torque usage
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(self, render_mode=None, max_steps=500):
        """
        Initialize H1 Standing environment.
        
        Args:
            render_mode: Mode for rendering ("human", "rgb_array", or None)
            max_steps: Maximum number of steps per episode
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        
        # Load MuJoCo model
        model_path = Path(__file__).parent.parent.parent / "models" / "unitree_h1" / "scene_enhanced.xml"
        if not model_path.exists():
            model_path = Path(__file__).parent.parent.parent / "models" / "unitree_h1" / "scene.xml"
        
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        
        # Number of actuators (joints)
        self.nu = self.model.nu  # Should be 19 for H1
        
        # Define observation space
        # 19 joint pos + 19 joint vel + 4 quat + 3 lin_vel + 3 ang_vel = 48
        obs_dim = self.nu * 2 + 4 + 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Define action space: normalized joint torques [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.nu,), dtype=np.float32
        )
        
        # Get torque limits from model
        self.torque_limits = self.model.actuator_ctrlrange.copy()
        
        # Target standing pose (default neutral stance)
        self.target_height = 1.0
        self.target_qpos = np.array([
            0.0, 0.0, 0.05, 0.3, -0.15,  # Left leg
            0.0, 0.0, 0.05, 0.3, -0.15,  # Right leg
            0.0,                          # Torso
            0.2, 0.2, 0.0, -0.2,         # Left arm
            0.2, -0.2, 0.0, -0.2         # Right arm
        ])
        
        # Viewer for rendering
        self.viewer = None
        
        print(f"[H1StandingEnv] Initialized with {self.nu} actuators")
        print(f"[H1StandingEnv] Observation space: {self.observation_space.shape}")
        print(f"[H1StandingEnv] Action space: {self.action_space.shape}")
    
    def _get_obs(self):
        """Get current observation from MuJoCo data."""
        # Joint positions (excluding floating base)
        q = self.data.qpos[7:7+self.nu]
        
        # Joint velocities (excluding floating base)
        dq = self.data.qvel[6:6+self.nu]
        
        # Base orientation (quaternion)
        quat = self.data.qpos[3:7]
        
        # Base velocities
        base_lin_vel = self.data.qvel[0:3]
        base_ang_vel = self.data.qvel[3:6]
        
        # Concatenate observation
        obs = np.concatenate([q, dq, quat, base_lin_vel, base_ang_vel])
        
        return obs.astype(np.float32)
    
    def _get_info(self):
        """Get additional info about current state."""
        pelvis_height = self.data.body('pelvis').xpos[2]
        pelvis_pos = self.data.body('pelvis').xpos.copy()
        
        return {
            "pelvis_height": pelvis_height,
            "pelvis_position": pelvis_pos,
            "step": self.current_step
        }
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            observation: Initial observation
            info: Additional information dict
        """
        super().reset(seed=seed)
        
        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial joint positions to standing pose
        self.data.qpos[7:7+self.nu] = self.target_qpos
        
        # Add small random perturbation for robustness
        if seed is not None:
            np.random.seed(seed)
        self.data.qpos[7:7+self.nu] += np.random.uniform(-0.05, 0.05, self.nu)
        
        # Forward kinematics to update positions
        mujoco.mj_forward(self.model, self.data)
        
        self.current_step = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Normalized joint torques in range [-1, 1]
            
        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Whether episode is done (fell)
            truncated: Whether episode is truncated (max steps)
            info: Additional information
        """
        # Convert normalized action to actual torques
        action = np.clip(action, -1.0, 1.0)
        torques = action * (self.torque_limits[:, 1] - self.torque_limits[:, 0]) / 2
        
        # Apply torques
        self.data.ctrl[:] = torques
        
        # Step simulation (multiple substeps for stability)
        for _ in range(5):  # 5 substeps = 10ms at 2ms timestep
            mujoco.mj_step(self.model, self.data)
        
        self.current_step += 1
        
        # Get new observation
        observation = self._get_obs()
        info = self._get_info()
        
        # Calculate reward
        reward = self._compute_reward(action, info)
        
        # Check termination conditions
        pelvis_height = info["pelvis_height"]
        terminated = bool(pelvis_height < 0.5)  # Fell down
        truncated = bool(self.current_step >= self.max_steps)
        
        return observation, reward, terminated, truncated, info
    
    def _compute_reward(self, action, info):
        """
        Compute reward for current state and action.
        
        Reward components:
        1. Height reward: Keep pelvis at target height
        2. Orientation reward: Stay upright (quaternion near [0,0,0,1])
        3. Stability reward: Low velocity
        4. Energy penalty: Minimize torque usage
        """
        pelvis_height = info["pelvis_height"]
        
        # 1. Height reward (gaussian around target height)
        height_error = abs(pelvis_height - self.target_height)
        height_reward = np.exp(-10 * height_error**2)
        
        # 2. Orientation reward (upright orientation)
        quat = self.data.qpos[3:7]
        # Target quaternion is [0, 0, 0, 1] (no rotation)
        orientation_error = np.linalg.norm(quat - np.array([0, 0, 0, 1]))
        orientation_reward = np.exp(-5 * orientation_error**2)
        
        # 3. Stability reward (low angular velocity)
        ang_vel = self.data.qvel[3:6]
        stability_reward = np.exp(-0.1 * np.linalg.norm(ang_vel)**2)
        
        # 4. Energy penalty (minimize torque)
        energy_penalty = -0.001 * np.sum(action**2)
        
        # Combine rewards
        reward = (
            5.0 * height_reward +
            2.0 * orientation_reward +
            1.0 * stability_reward +
            energy_penalty
        )
        
        return float(reward)
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            if self.viewer.is_running():
                self.viewer.sync()
        elif self.render_mode == "rgb_array":
            # TODO: Implement offscreen rendering
            pass
    
    def close(self):
        """Close the environment and cleanup."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class H1WalkingEnv(H1StandingEnv):
    """
    Gymnasium environment for training H1 robot to walk forward.
    
    Extends H1StandingEnv with additional rewards for forward movement.
    """
    
    def __init__(self, render_mode=None, max_steps=1000, target_velocity=0.5):
        """
        Initialize H1 Walking environment.
        
        Args:
            render_mode: Mode for rendering
            max_steps: Maximum steps per episode
            target_velocity: Target forward velocity in m/s
        """
        super().__init__(render_mode=render_mode, max_steps=max_steps)
        self.target_velocity = target_velocity
        self.initial_pos = None
        
        print(f"[H1WalkingEnv] Target velocity: {target_velocity} m/s")
    
    def reset(self, seed=None, options=None):
        """Reset environment and record initial position."""
        obs, info = super().reset(seed=seed, options=options)
        self.initial_pos = self.data.body('pelvis').xpos.copy()
        return obs, info
    
    def _compute_reward(self, action, info):
        """
        Compute reward for walking task.
        
        Additional components:
        5. Forward velocity reward: Reward for moving forward at target speed
        6. Lateral stability: Penalty for lateral drift
        """
        # Base standing reward
        reward = super()._compute_reward(action, info)
        
        # 5. Forward velocity reward
        forward_vel = self.data.qvel[0]  # X-axis velocity
        vel_error = abs(forward_vel - self.target_velocity)
        velocity_reward = np.exp(-2 * vel_error**2)
        
        # 6. Lateral stability (minimize Y-axis drift)
        lateral_vel = abs(self.data.qvel[1])
        lateral_penalty = -0.5 * lateral_vel
        
        # Add walking-specific rewards
        reward += (
            3.0 * velocity_reward +
            lateral_penalty
        )
        
        return float(reward)
    
    def _get_info(self):
        """Get info with additional walking metrics."""
        info = super()._get_info()
        
        if self.initial_pos is not None:
            current_pos = self.data.body('pelvis').xpos
            distance_traveled = current_pos[0] - self.initial_pos[0]
            info["distance_traveled"] = distance_traveled
            info["forward_velocity"] = self.data.qvel[0]
        
        return info


# Register environments with Gymnasium
gym.register(
    id='H1Standing-v0',
    entry_point='rl.h1_env:H1StandingEnv',
    max_episode_steps=500,
)

gym.register(
    id='H1Walking-v0',
    entry_point='rl.h1_env:H1WalkingEnv',
    max_episode_steps=1000,
)
