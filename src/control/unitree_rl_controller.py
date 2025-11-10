"""
Unitree Official RL Controller Adapter
Integrates the official unitree_rl_gym walking policy into our framework
Based on: https://github.com/unitreerobotics/unitree_rl_gym
"""

import numpy as np
import torch
import os
from pathlib import Path


class UnitreeRLController:
    """
    Official Unitree RL walking controller
    Uses pre-trained policy from unitree_rl_gym
    """
    
    def __init__(self, model, data, policy_path=None, robot_type="h1", max_speed=1.0):
        """
        Initialize Unitree RL controller
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            policy_path: Path to policy file (optional, will use default if not provided)
            robot_type: Robot type ("h1", "h1_2", "g1")
            max_speed: Maximum forward velocity in m/s (default: 1.0, max recommended: 1.5)
        """
        self.model = model
        self.data = data
        self.robot_type = robot_type
        self.max_speed = max_speed  # Configurable maximum speed
        
        # Load configuration based on robot type
        self._load_config(robot_type)
        
        # Load policy
        if policy_path is None:
            # Use default pre-trained policy from unitree_rl_gym
            # Find project root by looking for extern/unitree_rl_gym directory
            current_path = Path(__file__).resolve()
            project_root = current_path.parent.parent.parent  # src/control/ -> src/ -> project_root/
            
            # Handle different execution contexts (test scripts may have different cwd)
            policy_relative = f"extern/unitree_rl_gym/deploy/pre_train/{robot_type}/motion.pt"
            policy_path = project_root / policy_relative
            
            # If not found, try from current working directory
            if not policy_path.exists():
                policy_path = Path.cwd() / policy_relative
            
            # If still not found, try absolute path construction
            if not policy_path.exists():
                # Look for the extern directory in common locations
                possible_roots = [
                    Path.cwd(),
                    Path.cwd().parent,
                    Path(__file__).resolve().parent.parent.parent,
                ]
                for root in possible_roots:
                    test_path = root / policy_relative
                    if test_path.exists():
                        policy_path = test_path
                        break
        
        policy_path = Path(policy_path)
        if not policy_path.exists():
            raise FileNotFoundError(
                f"Policy not found: {policy_path}\n"
                f"Please ensure unitree_rl_gym submodule is initialized:\n"
                f"  git submodule update --init --recursive"
            )
        
        print(f"[UnitreeRL] Loading policy from: {policy_path}")
        self.policy = torch.jit.load(str(policy_path))
        self.policy.eval()
        print(f"[UnitreeRL] ✅ Policy loaded successfully")
        print(f"[UnitreeRL] Max forward speed configured: {self.max_speed} m/s")
        
        # State variables
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.counter = 0
        self.simulation_dt = 0.002  # 2ms timestep
        
        # Command variables (velocity commands)
        self.cmd = np.array([0.8, 0.0, 0.0], dtype=np.float32)  # [vx, vy, omega_z] - increased default speed
        
        print(f"[UnitreeRL] Robot: {robot_type}")
        print(f"[UnitreeRL] DOF: {self.num_actions}")
        print(f"[UnitreeRL] Observation size: {self.num_obs}")
        print(f"[UnitreeRL] Initial command: vx={self.cmd[0]:.2f} m/s")
    
    def _load_config(self, robot_type):
        """Load robot-specific configuration"""
        if robot_type == "h1":
            # H1 configuration (10 DOF: 5 per leg)
            self.kps = np.array([150, 150, 150, 200, 40,  150, 150, 150, 200, 40], dtype=np.float32)
            self.kds = np.array([2, 2, 2, 4, 2,  2, 2, 2, 4, 2], dtype=np.float32)
            self.default_angles = np.array([0, 0.0, -0.1, 0.3, -0.2,
                                           0, 0.0, -0.1, 0.3, -0.2], dtype=np.float32)
            self.num_actions = 10
            self.num_obs = 41
            
        elif robot_type == "h1_2":
            # H1_2 configuration (19 DOF: full body)
            self.kps = np.array([200] * 19, dtype=np.float32)
            self.kds = np.array([5] * 19, dtype=np.float32)
            self.default_angles = np.array([0.0] * 19, dtype=np.float32)
            self.num_actions = 19
            self.num_obs = 66
            
        elif robot_type == "g1":
            # G1 configuration (23 DOF)
            self.kps = np.array([200] * 23, dtype=np.float32)
            self.kds = np.array([5] * 23, dtype=np.float32)
            self.default_angles = np.array([0.0] * 23, dtype=np.float32)
            self.num_actions = 23
            self.num_obs = 76
            
        else:
            raise ValueError(f"Unknown robot type: {robot_type}. Supported: h1, h1_2, g1")
        
        # Common scaling factors
        self.ang_vel_scale = 0.25
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.action_scale = 0.25
        self.cmd_scale = np.array([2.0, 2.0, 0.25], dtype=np.float32)
        self.control_decimation = 10  # Run policy at 50Hz (every 10 steps at 2ms = 20ms)
    
    def set_velocity_command(self, vx, vy=0.0, omega_z=0.0):
        """
        Set desired walking velocity
        
        Args:
            vx: Forward velocity (m/s)
            vy: Lateral velocity (m/s)
            omega_z: Yaw rate (rad/s)
        """
        self.cmd = np.array([vx, vy, omega_z], dtype=np.float32)
        print(f"[UnitreeRL] New command: vx={vx:.2f}, vy={vy:.2f}, omega={omega_z:.2f}")
    
    def get_gravity_orientation(self, quaternion):
        """
        Compute gravity direction in body frame from quaternion
        
        Args:
            quaternion: [qw, qx, qy, qz]
            
        Returns:
            gravity_orientation: 3D vector
        """
        qw, qx, qy, qz = quaternion
        
        gravity_orientation = np.array([
            2 * (-qz * qx + qw * qy),
            -2 * (qz * qy + qw * qx),
            1 - 2 * (qw * qw + qz * qz)
        ], dtype=np.float32)
        
        return gravity_orientation
    
    def _build_observation(self):
        """
        Build observation vector for policy
        
        Observation structure:
        - Angular velocity (3)
        - Gravity orientation (3)
        - Commands (3)
        - Joint positions (num_actions)
        - Joint velocities (num_actions)
        - Previous actions (num_actions)
        - Gait phase (2: sin, cos)
        """
        # Get joint states (skip floating base: qpos[7:], qvel[6:])
        qj = self.data.qpos[7:7+self.num_actions]
        dqj = self.data.qvel[6:6+self.num_actions]
        
        # Get base orientation and angular velocity
        quat = self.data.qpos[3:7]  # [qw, qx, qy, qz]
        omega = self.data.qvel[3:6]  # Angular velocity
        
        # Normalize observations
        qj_normalized = (qj - self.default_angles) * self.dof_pos_scale
        dqj_normalized = dqj * self.dof_vel_scale
        gravity_orientation = self.get_gravity_orientation(quat)
        omega_normalized = omega * self.ang_vel_scale
        
        # Compute gait phase (cyclic signal for rhythm)
        period = 0.8  # seconds
        count = self.counter * self.simulation_dt
        phase = (count % period) / period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)
        
        # Assemble observation
        self.obs[:3] = omega_normalized
        self.obs[3:6] = gravity_orientation
        self.obs[6:9] = self.cmd * self.cmd_scale
        self.obs[9:9+self.num_actions] = qj_normalized
        self.obs[9+self.num_actions:9+2*self.num_actions] = dqj_normalized
        self.obs[9+2*self.num_actions:9+3*self.num_actions] = self.action
        self.obs[9+3*self.num_actions:9+3*self.num_actions+2] = np.array([sin_phase, cos_phase])
        
        return self.obs
    
    def compute_control(self, dt=0.002):
        """
        Compute control torques using RL policy + PD control
        
        Args:
            dt: Control timestep (should be 0.002s for MuJoCo)
            
        Returns:
            tau: Joint torques
        """
        self.counter += 1
        
        # Run policy at lower frequency (50Hz instead of 500Hz)
        if self.counter % self.control_decimation == 0:
            # Build observation
            obs = self._build_observation()
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            
            # Policy inference
            with torch.no_grad():
                action_tensor = self.policy(obs_tensor)
                self.action = action_tensor.detach().numpy().squeeze()
            
            # Transform action to target joint positions
            self.target_dof_pos = self.action * self.action_scale + self.default_angles
        
        # PD control to track target positions
        qj = self.data.qpos[7:7+self.num_actions]
        dqj = self.data.qvel[6:6+self.num_actions]
        
        tau = self.kps * (self.target_dof_pos - qj) - self.kds * dqj
        
        # For full H1 model with 19 joints, pad with zeros for unused actuators
        if self.model.nu > self.num_actions:
            tau_full = np.zeros(self.model.nu, dtype=np.float32)
            tau_full[:self.num_actions] = tau
            return tau_full
        
        return tau
    
    def reset(self):
        """Reset controller state"""
        self.counter = 0
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.cmd = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        print("[UnitreeRL] Controller reset")


class UnitreeRLWalkingController:
    """
    High-level walking controller using Unitree RL policy
    Provides simple interface for common walking tasks
    """
    
    def __init__(self, model, data, robot_type="h1", max_speed=1.0):
        """
        Initialize walking controller
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            robot_type: Robot type ("h1", "h1_2", "g1")
            max_speed: Maximum forward velocity in m/s (default: 1.0)
        """
        self.rl_controller = UnitreeRLController(model, data, robot_type=robot_type, max_speed=max_speed)
        self.model = model
        self.data = data
        self.max_speed = max_speed
        
        # Navigation state
        self.target_position = None
        self.navigation_tolerance = 0.3  # meters
        
    def compute_control(self, dt=0.002):
        """Compute control torques"""
        return self.rl_controller.compute_control(dt)
    
    def set_target_velocity(self, vx, vy=0.0, omega_z=0.0):
        """
        Set walking velocity
        
        Args:
            vx: Forward velocity (m/s), range: [-1.0, 1.0]
            vy: Lateral velocity (m/s), range: [-1.0, 1.0]
            omega_z: Turning rate (rad/s), range: [-1.0, 1.0]
        """
        # Clamp velocities to safe range
        vx = np.clip(vx, -1.0, 1.0)
        vy = np.clip(vy, -1.0, 1.0)
        omega_z = np.clip(omega_z, -1.0, 1.0)
        
        self.rl_controller.set_velocity_command(vx, vy, omega_z)
    
    def set_target(self, target_x, target_y):
        """
        Set navigation target position
        
        Args:
            target_x: Target x coordinate
            target_y: Target y coordinate
        """
        self.target_position = np.array([target_x, target_y])
        print(f"[UnitreeRL] Navigation target set: ({target_x:.2f}, {target_y:.2f})")
    
    def update_navigation(self):
        """
        Update velocity command based on navigation target
        Should be called every control step in navigation mode
        """
        if self.target_position is None:
            return
        
        # Get current position (pelvis)
        current_pos = self.data.qpos[:2]  # [x, y]
        
        # Calculate distance and direction to target
        delta = self.target_position - current_pos
        distance = np.linalg.norm(delta)
        
        # Check if target reached
        if distance < self.navigation_tolerance:
            print(f"[UnitreeRL] 🎯 Target reached! Distance: {distance:.2f}m")
            self.target_position = None
            self.set_target_velocity(0.0, 0.0, 0.0)
            return
        
        # Get current heading
        quat = self.data.qpos[3:7]
        heading = self._quat_to_yaw(quat)
        
        # Calculate desired heading to target
        desired_heading = np.arctan2(delta[1], delta[0])
        heading_error = self._normalize_angle(desired_heading - heading)
        
        # DEBUG: Log navigation state every 100 calls
        if not hasattr(self, '_nav_debug_counter'):
            self._nav_debug_counter = 0
        self._nav_debug_counter += 1
        
        if self._nav_debug_counter % 100 == 0:
            print(f"[NAV DEBUG] Pos: ({current_pos[0]:.2f}, {current_pos[1]:.2f}) → Target: ({self.target_position[0]:.2f}, {self.target_position[1]:.2f})")
            print(f"            Distance: {distance:.2f}m | Heading: {np.degrees(heading):.1f}° | Desired: {np.degrees(desired_heading):.1f}° | Error: {np.degrees(heading_error):.1f}°")
        
        # Simple proportional control - use configured max_speed
        max_forward_vel = self.max_speed  # Use configurable max speed
        max_turn_rate = 0.8    # rad/s - increase for faster turning
        
        # Turning: proportional to heading error
        omega_z = np.clip(heading_error * 3.0, -max_turn_rate, max_turn_rate)  # Increased gain from 2.5 to 3.0
        
        # Forward velocity: reduce speed dramatically when heading error is large
        # Don't walk forward if not facing the right direction!
        if abs(heading_error) > np.pi / 3:  # > 60 degrees off
            # Large heading error - prioritize turning, minimal forward motion
            turn_factor = 0.1
        elif abs(heading_error) > np.pi / 6:  # > 30 degrees off
            # Medium heading error - slow down significantly
            turn_factor = 0.3
        else:
            # Small heading error - normal speed
            turn_factor = max(0.5, 1.0 - abs(heading_error) / (np.pi / 6))
        
        if distance > 2.0:
            vx = max_forward_vel * turn_factor
        elif distance > 0.5:
            vx = max_forward_vel * 0.7 * turn_factor  # 70% speed in medium range
        else:
            vx = max(0.2, distance * 0.5) * turn_factor  # Slow down near target
        
        # Set velocity command
        self.set_target_velocity(vx, 0.0, omega_z)
    
    def _quat_to_yaw(self, quat):
        """Convert quaternion to yaw angle"""
        qw, qx, qy, qz = quat
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        return yaw
    
    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2*np.pi
        while angle < -np.pi:
            angle += 2*np.pi
        return angle
    
    def reset(self):
        """Reset controller"""
        self.rl_controller.reset()
        self.target_position = None
