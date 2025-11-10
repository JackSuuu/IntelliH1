"""
Simple PD Standing Controller for Unitree H1
Based on proven stable standing control principles from:
- fan-ziqi/h1-mujoco-sim
- unitree_rl_gym official examples

Support RL policy augmentation for enhanced balance
"""

import numpy as np
import mujoco
from pathlib import Path

try:
    from control.rl_policy_loader import RLPolicyLoader
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("[WARNING] RL policy loader not available")


class PDStandingController:
    """
    Simple but stable PD controller for standing balance.
    Uses basic PD control with gravity compensation.
    """
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # Get number of actuators
        self.nu = model.nu
        
        # Standing pose (neutral squat position)
        self.q_target = self._get_standing_pose()
        
        # PD Gains - Balanced for stability
        # Strong enough to fight gravity but not so high they cause oscillations
        self.kp = np.array([
            # Legs (HIGH gains for stability)
            300, 250, 400, 500, 200,  # Left leg: yaw, roll, pitch, knee, ankle
            300, 250, 400, 500, 200,  # Right leg
            # Torso (high but not excessive - prevent rotation)
            450,
            # Arms (moderate - less critical)
            120, 120, 80, 80,   # Left arm
            120, 120, 80, 80,   # Right arm
        ])
        
        self.kd = self.kp / 12.0  # Good damping ratio
        
        # Torque limits (conservative for safety)
        self.torque_limit = np.array([
            # Legs
            220, 220, 360, 360, 75,  # Left leg (hip, knee, ankle)
            220, 220, 360, 360, 75,  # Right leg
            # Torso
            200,
            # Arms
            80, 80, 80, 80,
            80, 80, 80, 80,
        ])
        
        print("[PDStanding] Simple PD standing controller initialized")
        print(f"[PDStanding] Actuators: {self.nu}")
    
    def _get_standing_pose(self):
        """
        Return a stable standing configuration.
        Balanced neutral stance with moderate knee bend for stability.
        """
        # Balanced stance - near-vertical with controlled knee bend
        # Let active control handle fine-tuning rather than pre-biasing the pose
        q = np.array([
            # Left leg: hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch
            0.0, 0.0, 0.05, 0.3, -0.15,  # Slightly forward hip, moderate knee
            # Right leg
            0.0, 0.0, 0.05, 0.3, -0.15,
            # Torso (nearly upright)
            0.0,
            # Left arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
            0.2, 0.2, 0.0, -0.2,
            # Right arm
            0.2, -0.2, 0.0, -0.2,
        ])
        return q
    
    def compute_control(self, dt=0.002):
        """
        Compute PD control with ankle strategy for balance.
        
        Returns:
            tau: Joint torques (nu,)
        """
        # Get current joint states
        q = self.data.qpos[7:7+self.nu]  # Skip floating base (7 DOF)
        dq = self.data.qvel[6:6+self.nu]  # Skip floating base (6 DOF)
        
        # Get base position for CoM estimation
        base_pos = self.data.qpos[0:3]  # x, y, z of floating base
        base_vel = self.data.qvel[0:3]  # dx, dy, dz
        
        # 🚀 ADAPTIVE PD CONTROL: Increase gains when drift is large
        # Ankle strategy: adjust ankle angle based on forward/backward drift
        # If leaning forward (x > 0), need to pitch ankles backward (negative)
        # If leaning backward (x < 0), need to pitch ankles forward (positive)
        ankle_base_gain = 130.0
        ankle_adaptive_gain = ankle_base_gain * (1.0 + 5.0 * abs(base_pos[0]))  # 🎯 Stronger response to drift
        ankle_compensation = -ankle_adaptive_gain * base_pos[0] - 25.0 * base_vel[0]  # Increased damping
        
        # Hip strategy: also adjust hip pitch for larger disturbances
        hip_base_gain = 55.0
        hip_adaptive_gain = hip_base_gain * (1.0 + 3.0 * abs(base_pos[0]))
        hip_compensation = -hip_adaptive_gain * base_pos[0] - 12.0 * base_vel[0]
        
        # Torso strategy: keep upper body upright (strong response to prevent cascade failure)
        torso_base_gain = 250.0
        torso_adaptive_gain = torso_base_gain * (1.0 + 4.0 * abs(base_pos[0]))
        torso_compensation = -torso_adaptive_gain * base_pos[0] - 35.0 * base_vel[0]
        
        # Create modified target with balance compensation
        q_target_adjusted = self.q_target.copy()
        q_target_adjusted[2] += hip_compensation    # Left hip pitch
        q_target_adjusted[7] += hip_compensation    # Right hip pitch
        q_target_adjusted[4] += ankle_compensation   # Left ankle pitch
        q_target_adjusted[9] += ankle_compensation   # Right ankle pitch
        q_target_adjusted[10] += torso_compensation  # Torso pitch (keep upright)
        
        # Simple PD control with adjusted target
        tau = self.kp * (q_target_adjusted - q) - self.kd * dq
        
        # Apply torque limits
        tau = np.clip(tau, -self.torque_limit, self.torque_limit)
        
        return tau
    
    def update_target_pose(self, q_new):
        """Update target pose (for future use with walking)"""
        self.q_target = q_new


class ImprovedPDController:
    """
    Improved PD controller with gravity compensation.
    More sophisticated but still simple and reliable.
    """
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.nu = model.nu
        
        # Use same standing pose as simple controller
        self.base_controller = PDStandingController(model, data)
        self.q_target = self.base_controller.q_target
        self.kp = self.base_controller.kp
        self.kd = self.base_controller.kd
        self.torque_limit = self.base_controller.torque_limit
        
        print("[ImprovedPD] PD controller with gravity compensation initialized")
    
    def compute_control(self, dt=0.002):
        """
        Compute PD + gravity compensation + ankle strategy.
        """
        # Get current joint states
        q_full = self.data.qpos.copy()
        dq_full = self.data.qvel.copy()
        
        q = q_full[7:7+self.nu]
        dq = dq_full[6:6+self.nu]
        
        # Get base position for CoM estimation
        base_pos = self.data.qpos[0:3]
        base_vel = self.data.qvel[0:3]
        
        # Ankle strategy: adjust ankle angle based on forward/backward drift
        # If leaning forward (x > 0), need to pitch ankles backward (negative)
        # If leaning backward (x < 0), need to pitch ankles forward (positive)
        ankle_compensation = -130.0 * base_pos[0] - 22.0 * base_vel[0]  # Primary balance - strong
        
        # Hip strategy: also adjust hip pitch for larger disturbances
        hip_compensation = -55.0 * base_pos[0] - 9.0 * base_vel[0]
        
        # Torso strategy: keep upper body upright (moderate to prevent rotation)
        torso_compensation = -250.0 * base_pos[0] - 30.0 * base_vel[0]
        
        # Create modified target with balance compensation
        q_target_adjusted = self.q_target.copy()
        q_target_adjusted[2] += hip_compensation    # Left hip pitch
        q_target_adjusted[7] += hip_compensation    # Right hip pitch
        q_target_adjusted[4] += ankle_compensation   # Left ankle pitch
        q_target_adjusted[9] += ankle_compensation   # Right ankle pitch
        q_target_adjusted[10] += torso_compensation  # Torso pitch (keep upright)
        
        # PD control with adjusted target
        tau_pd = self.kp * (q_target_adjusted - q) - self.kd * dq
        
        # Gravity compensation using MuJoCo's built-in function
        # This is more efficient than computing it manually
        tau_gravity = np.zeros(self.nu)
        
        # MuJoCo provides qfrc_bias which includes gravity and Coriolis
        # Extract just the actuated joints part
        qfrc_bias = self.data.qfrc_bias.copy()
        tau_gravity = qfrc_bias[6:6+self.nu]  # Skip floating base
        
        # Total torque = PD + gravity compensation
        tau = tau_pd + tau_gravity
        
        # Apply torque limits
        tau = np.clip(tau, -self.torque_limit, self.torque_limit)
        
        return tau
    
    def update_target_pose(self, q_new):
        """Update target pose (for future use with walking)"""
        self.q_target = q_new


class RLAssistedPDController:
    """
    🤖 RL-Assisted PD Controller: Best of both worlds!
    - PD provides stable baseline and safety
    - RL provides learned balance strategies for edge cases
    - Blending ensures smooth and safe control
    """
    
    def __init__(self, model, data, policy_path="policies/h1_demo_policy.pt", rl_weight=0.3):
        """
        Args:
            model: MuJoCo model
            data: MuJoCo data
            policy_path: Path to RL policy .pt file
            rl_weight: Weight for RL contribution (0=pure PD, 1=pure RL)
        """
        self.model = model
        self.data = data
        self.nu = model.nu
        
        # Initialize base PD controller (safety baseline)
        self.pd_controller = ImprovedPDController(model, data)
        
        # Try to load RL policy
        if not RL_AVAILABLE:
            print("[RLAssisted] ⚠️ RL not available, falling back to pure PD")
            self.rl_policy = None
            self.rl_weight = 0.0
        else:
            policy_path = Path(policy_path)
            if not policy_path.exists():
                print(f"[RLAssisted] ⚠️ Policy not found: {policy_path}, using PD only")
                self.rl_policy = None
                self.rl_weight = 0.0
            else:
                try:
                    self.rl_policy = RLPolicyLoader(policy_path)
                    self.rl_weight = rl_weight
                    print(f"[RLAssisted] ✅ RL policy loaded! Blend: {(1-rl_weight)*100:.0f}% PD + {rl_weight*100:.0f}% RL")
                except Exception as e:
                    print(f"[RLAssisted] ⚠️ Failed to load RL: {e}, using PD only")
                    self.rl_policy = None
                    self.rl_weight = 0.0
    
    def compute_control(self, dt=0.002):
        """
        Compute blended PD + RL control
        """
        # Always compute PD baseline (safety)
        tau_pd = self.pd_controller.compute_control(dt)
        
        # If RL available, blend in learned policy
        if self.rl_policy is not None and self.rl_weight > 0:
            try:
                # Get observation for RL policy
                obs = self.rl_policy.get_observation(self.data, self.nu)
                
                # Get RL policy action
                tau_rl = self.rl_policy.predict(obs)
                
                # Ensure same dimensions
                if len(tau_rl) != self.nu:
                    print(f"[RLAssisted] ⚠️ RL output size mismatch: {len(tau_rl)} != {self.nu}, using PD only")
                    return tau_pd
                
                # Blend PD (safety) + RL (intelligence)
                tau = (1 - self.rl_weight) * tau_pd + self.rl_weight * tau_rl
                
                # Safety: apply torque limits
                tau = np.clip(tau, -self.pd_controller.torque_limit, self.pd_controller.torque_limit)
                
                return tau
                
            except Exception as e:
                print(f"[RLAssisted] ⚠️ RL inference failed: {e}, falling back to PD")
                return tau_pd
        else:
            # Pure PD mode
            return tau_pd
    
    def update_target_pose(self, q_new):
        """Update target pose (delegates to PD controller)"""
        self.pd_controller.update_target_pose(q_new)


class PIDStandingController:
    """
    PID Standing Controller with integral term for improved steady-state tracking.
    
    Control law: τ = Kp * error + Ki * ∫error dt + Kd * d_error/dt
    
    The integral term helps eliminate steady-state errors and drift,
    which is particularly useful for long-term standing stability.
    """
    
    def __init__(self, model, data, ki_factor=0.1):
        """
        Args:
            model: MuJoCo model
            data: MuJoCo data
            ki_factor: Scaling factor for integral gains (default: 0.1)
        """
        self.model = model
        self.data = data
        self.nu = model.nu
        
        # Use same standing pose as PD controller
        self.base_controller = PDStandingController(model, data)
        self.q_target = self.base_controller.q_target
        self.kp = self.base_controller.kp
        self.kd = self.base_controller.kd
        
        # Integral gains (typically much smaller than Kp)
        # Start with conservative values to avoid integral windup
        self.ki = self.kp * ki_factor
        
        # Integral error accumulator
        self.error_integral = np.zeros(self.nu)
        
        # Integral windup limits (prevent excessive accumulation)
        self.integral_limit = np.array([
            # Legs (prevent excessive buildup)
            5.0, 5.0, 5.0, 5.0, 5.0,  # Left leg
            5.0, 5.0, 5.0, 5.0, 5.0,  # Right leg
            # Torso
            5.0,
            # Arms (less critical)
            3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0,
        ])
        
        self.torque_limit = self.base_controller.torque_limit
        
        print("[PIDStanding] PID standing controller initialized")
        print(f"[PIDStanding] Ki factor: {ki_factor}")
        print(f"[PIDStanding] Integral term enabled for steady-state accuracy")
    
    def compute_control(self, dt=0.002):
        """
        Compute PID control with ankle strategy for balance.
        
        Args:
            dt: Time step for integral computation
            
        Returns:
            tau: Joint torques (nu,)
        """
        # Get current joint states
        q = self.data.qpos[7:7+self.nu]
        dq = self.data.qvel[6:6+self.nu]
        
        # Get base position for CoM estimation
        base_pos = self.data.qpos[0:3]
        base_vel = self.data.qvel[0:3]
        
        # Ankle strategy with adaptive gains
        ankle_base_gain = 130.0
        ankle_adaptive_gain = ankle_base_gain * (1.0 + 5.0 * abs(base_pos[0]))
        ankle_compensation = -ankle_adaptive_gain * base_pos[0] - 25.0 * base_vel[0]
        
        # Hip strategy
        hip_base_gain = 55.0
        hip_adaptive_gain = hip_base_gain * (1.0 + 3.0 * abs(base_pos[0]))
        hip_compensation = -hip_adaptive_gain * base_pos[0] - 12.0 * base_vel[0]
        
        # Torso strategy
        torso_base_gain = 250.0
        torso_adaptive_gain = torso_base_gain * (1.0 + 4.0 * abs(base_pos[0]))
        torso_compensation = -torso_adaptive_gain * base_pos[0] - 35.0 * base_vel[0]
        
        # Create modified target with balance compensation
        q_target_adjusted = self.q_target.copy()
        q_target_adjusted[2] += hip_compensation
        q_target_adjusted[7] += hip_compensation
        q_target_adjusted[4] += ankle_compensation
        q_target_adjusted[9] += ankle_compensation
        q_target_adjusted[10] += torso_compensation
        
        # Calculate tracking error
        error = q_target_adjusted - q
        
        # Update integral error with anti-windup
        self.error_integral += error * dt
        self.error_integral = np.clip(self.error_integral, -self.integral_limit, self.integral_limit)
        
        # PID control law
        tau_p = self.kp * error
        tau_i = self.ki * self.error_integral
        tau_d = -self.kd * dq
        
        tau = tau_p + tau_i + tau_d
        
        # Apply torque limits
        tau = np.clip(tau, -self.torque_limit, self.torque_limit)
        
        return tau
    
    def update_target_pose(self, q_new):
        """Update target pose"""
        self.q_target = q_new
        # Reset integral when target changes to prevent windup
        self.error_integral = np.zeros(self.nu)
    
    def reset_integral(self):
        """Reset integral error accumulator"""
        self.error_integral = np.zeros(self.nu)
