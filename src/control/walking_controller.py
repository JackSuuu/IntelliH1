"""
Simple Walking Controller for Unitree H1
Based on ZMP (Zero Moment Point) walking and PD control
"""

import numpy as np
import mujoco
from control.pd_standing import PDStandingController, ImprovedPDController


class SimpleWalkingController:
    """
    Simple but functional walking controller using:
    - PD control for joint tracking
    - Sinusoidal gait pattern
    - Center of Mass (CoM) shifting for balance
    """
    
    def __init__(self, model, data):
        """
        Initialize walking controller
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        self.model = model
        self.data = data
        self.nu = model.nu
        
        # Base PD controller for joint tracking
        # Use simple PD controller WITHOUT balance compensation (gait handles balance)
        self.pd_controller = PDStandingController(model, data)
        
        # Override PD gains for walking (higher for dynamic motion)
        self.pd_controller.kp = (self.pd_controller.kp * 1.5).astype(self.pd_controller.kp.dtype)
        self.pd_controller.kd = (self.pd_controller.kd * 1.2).astype(self.pd_controller.kd.dtype)
        
        # Walking parameters (CONSERVATIVE for stability)
        self.step_frequency = 0.8  # Hz (slower = more stable)
        self.step_length = 0.10    # meters (small steps)
        self.step_height = 0.03    # meters (low lift)
        self.step_width = 0.18     # meters (lateral distance between feet)
        
        # Current gait phase (0 to 2π)
        self.phase = 0.0
        
        # Target velocity (will be set by higher-level planner)
        self.target_velocity = np.array([0.0, 0.0])  # [vx, vy] in m/s
        self.target_heading = 0.0  # radians
        
        # Gait state
        self.is_walking = False
        self.warmup_time = 0.0  # Time since walking started
        self.warmup_duration = 2.0  # Gradual ramp-up over 2 seconds
        
        print("[WalkingController] Simple walking controller initialized")
        print(f"  Step frequency: {self.step_frequency} Hz")
        print(f"  Step length: {self.step_length} m")
        print(f"  Step height: {self.step_height} m")
    
    def set_target_velocity(self, vx, vy=0.0):
        """
        Set desired walking velocity
        
        Args:
            vx: Forward velocity (m/s)
            vy: Lateral velocity (m/s)
        """
        self.target_velocity = np.array([vx, vy])
        was_walking = self.is_walking
        self.is_walking = (abs(vx) > 0.01 or abs(vy) > 0.01)
        
        # Reset warmup timer when starting to walk
        if self.is_walking and not was_walking:
            self.warmup_time = 0.0
            print("[WalkingController] Starting walk with warmup...")
    
    def set_target_heading(self, heading):
        """
        Set desired heading direction
        
        Args:
            heading: Target yaw angle in radians
        """
        self.target_heading = heading
    
    def _generate_gait_pattern(self, dt):
        """
        Generate cyclical gait pattern (left-right leg alternation)
        
        Args:
            dt: Control timestep
            
        Returns:
            q_target: Target joint positions for current phase
        """
        # Update phase and warmup
        if self.is_walking:
            self.warmup_time += dt
            self.phase += 2 * np.pi * self.step_frequency * dt
            if self.phase > 2 * np.pi:
                self.phase -= 2 * np.pi
        
        # Base standing pose
        q = self.pd_controller.q_target.copy()
        
        if not self.is_walking:
            return q
        
        # Calculate warmup factor (0 to 1)
        warmup_factor = min(1.0, self.warmup_time / self.warmup_duration)
        
        # Sinusoidal gait: left leg swings when sin(phase) > 0, right leg when sin(phase) < 0
        swing_phase = np.sin(self.phase)  # -1 to 1
        
        # Calculate swing foot trajectory
        # When swing_phase > 0: left leg swings, right leg stance
        # When swing_phase < 0: right leg swings, left leg stance
        
        if swing_phase > 0:
            # Left leg swing phase
            left_swing = abs(swing_phase)
            right_swing = 0.0
        else:
            # Right leg swing phase
            left_swing = 0.0
            right_swing = abs(swing_phase)
        
        # 🐢 MICRO-STEP WALKING: Minimal motion for maximum stability
        # Hip pitch: VERY small forward/backward motion
        hip_forward = 0.08 * warmup_factor  # Reduced from 0.15
        hip_backward = -0.02 * warmup_factor  # Reduced from -0.05
        
        left_hip_pitch = hip_forward * left_swing + hip_backward * (1 - left_swing)
        right_hip_pitch = hip_forward * right_swing + hip_backward * (1 - right_swing)
        
        # Knee: MINIMAL bend during swing (almost straight-leg walking)
        knee_bend = 0.15 * warmup_factor  # Reduced from 0.35
        knee_stance = 0.28  # Keep bent for stability (lower CoM)
        
        left_knee = knee_bend * left_swing + knee_stance * (1 - left_swing)
        right_knee = knee_bend * right_swing + knee_stance * (1 - right_swing)
        
        # Ankle: MINIMAL motion - almost no lifting
        ankle_up = 0.02 * warmup_factor  # Reduced from 0.05
        ankle_down = -0.15  # Keep flat
        
        left_ankle = ankle_up * left_swing + ankle_down * (1 - left_swing)
        right_ankle = ankle_up * right_swing + ankle_down * (1 - right_swing)
        
        # Apply to target pose
        # Joint order: [left_leg(5), right_leg(5), torso(1), left_arm(4), right_arm(4)]
        # Left leg: hip_yaw(0), hip_roll(1), hip_pitch(2), knee(3), ankle(4)
        q[2] = left_hip_pitch   # Left hip pitch
        q[3] = left_knee        # Left knee
        q[4] = left_ankle       # Left ankle
        
        # Right leg
        q[7] = right_hip_pitch  # Right hip pitch
        q[8] = right_knee       # Right knee
        q[9] = right_ankle      # Right ankle
        
        # Torso: keep upright (no lean for stability)
        q[10] = 0.0  # No forward lean
        
        # Arms: minimal swing for stability
        arm_swing = 0.08 * warmup_factor  # Reduced from 0.15
        q[11] = -arm_swing * swing_phase  # Left shoulder pitch
        q[15] = arm_swing * swing_phase   # Right shoulder pitch
        
        return q
    
    def compute_control(self, dt=0.002):
        """
        Compute walking control torques
        
        Args:
            dt: Control timestep
            
        Returns:
            tau: Joint torques
        """
        # Debug: print walking state every 100 steps
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        
        if self._debug_counter % 500 == 0:
            print(f"[WalkingController] is_walking={self.is_walking}, phase={self.phase:.2f}, warmup={self.warmup_time:.2f}s")
        
        # Generate target joint positions for current gait phase
        q_target = self._generate_gait_pattern(dt)
        
        # Update PD controller target
        self.pd_controller.q_target = q_target
        
        # Compute PD control with gravity compensation
        tau = self.pd_controller.compute_control(dt)
        
        return tau
    
    def update_target_pose(self, q_new):
        """Update target pose"""
        self.pd_controller.q_target = q_new


class NavigationController:
    """
    High-level navigation controller that combines walking with path following
    """
    
    def __init__(self, model, data):
        """
        Initialize navigation controller
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        self.model = model
        self.data = data
        
        # Walking controller
        self.walking_controller = SimpleWalkingController(model, data)
        
        # Navigation state
        self.target_position = None  # [x, y] target in world frame
        self.position_threshold = 0.3  # meters (stop when within this distance)
        self.heading_threshold = 0.2  # radians
        
        # Walking speed
        self.walk_speed = 0.3  # m/s (conservative for stability)
        
        print("[NavigationController] Navigation controller initialized")
        print(f"  Walking speed: {self.walk_speed} m/s")
        print(f"  Position threshold: {self.position_threshold} m")
    
    def set_target(self, target_x, target_y):
        """
        Set navigation target
        
        Args:
            target_x: Target x position in world frame
            target_y: Target y position in world frame
        """
        self.target_position = np.array([target_x, target_y])
        print(f"[Navigation] Target set to: ({target_x:.2f}, {target_y:.2f})")
    
    def get_current_position(self):
        """Get robot's current position"""
        return self.data.qpos[0:2]  # [x, y]
    
    def get_current_heading(self):
        """Get robot's current heading (yaw)"""
        # Extract yaw from quaternion
        quat = self.data.qpos[3:7]  # [qw, qx, qy, qz]
        # Convert quaternion to yaw
        # yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2))
        yaw = np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 
                         1 - 2*(quat[2]**2 + quat[3]**2))
        return yaw
    
    def compute_control(self, dt=0.002):
        """
        Compute navigation control
        
        Args:
            dt: Control timestep
            
        Returns:
            tau: Joint torques
            reached: True if target reached
        """
        if self.target_position is None:
            # No target, just stand
            self.walking_controller.set_target_velocity(0.0, 0.0)
            return self.walking_controller.compute_control(dt), False
        
        # Get current state
        current_pos = self.get_current_position()
        current_heading = self.get_current_heading()
        
        # Calculate error
        error_vec = self.target_position - current_pos
        distance = np.linalg.norm(error_vec)
        
        # Check if reached target
        if distance < self.position_threshold:
            print(f"[Navigation] ✅ Target reached! Distance: {distance:.3f}m")
            self.walking_controller.set_target_velocity(0.0, 0.0)
            self.target_position = None
            return self.walking_controller.compute_control(dt), True
        
        # Calculate desired heading (direction to target)
        desired_heading = np.arctan2(error_vec[1], error_vec[0])
        heading_error = desired_heading - current_heading
        
        # Normalize angle to [-π, π]
        while heading_error > np.pi:
            heading_error -= 2*np.pi
        while heading_error < -np.pi:
            heading_error += 2*np.pi
        
        # Simple navigation strategy:
        # If heading error is large, turn in place (slow forward motion)
        # If heading is good, walk forward
        
        if abs(heading_error) > self.heading_threshold:
            # Need to turn
            forward_vel = self.walk_speed * 0.3  # Slow forward while turning
            # TODO: Add turning control (requires hip yaw control)
            # For now, just slow down
        else:
            # Good heading, walk forward
            forward_vel = self.walk_speed
        
        # Scale velocity based on distance (slow down near target)
        if distance < 1.0:
            forward_vel *= distance  # Linear slowdown
        
        # Set walking velocity
        self.walking_controller.set_target_velocity(forward_vel, 0.0)
        self.walking_controller.set_target_heading(desired_heading)
        
        # Compute walking control
        tau = self.walking_controller.compute_control(dt)
        
        return tau, False
    
    def update_target_pose(self, q_new):
        """Update target pose"""
        self.walking_controller.update_target_pose(q_new)
