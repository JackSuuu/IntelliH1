"""
Quasi-Static Walking Controller for Unitree H1
Uses weight shifting and step-by-step balance for maximum stability
"""

import numpy as np
import mujoco
from control.pd_standing import PDStandingController


class QuasiStaticWalkingController:
    """
    Ultra-conservative walking using quasi-static approach:
    1. Shift weight to one leg
    2. Lift other leg slightly
    3. Move lifted leg forward
    4. Plant foot
    5. Shift weight to new foot
    6. Repeat
    
    This is SLOW but STABLE - like how a toddler learns to walk
    """
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.nu = model.nu
        
        # Base PD controller
        self.pd_controller = PDStandingController(model, data)
        
        # Increase gains for walking
        self.pd_controller.kp = (self.pd_controller.kp * 2.0).astype(self.pd_controller.kp.dtype)
        self.pd_controller.kd = (self.pd_controller.kd * 1.5).astype(self.pd_controller.kd.dtype)
        
        # Walking state machine
        self.state = "standing"  # standing, shift_left, step_right, shift_right, step_left
        self.state_time = 0.0
        self.state_duration = {
            "standing": 1.0,      # Stand still initially
            "shift_left": 0.5,    # Shift weight to left leg
            "step_right": 0.6,    # Swing right leg forward
            "shift_right": 0.5,   # Shift weight to right leg  
            "step_left": 0.6,     # Swing left leg forward
        }
        
        # Walking parameters
        self.step_length = 0.05  # Very small steps (5cm)
        self.hip_sway = 0.05     # Lateral weight shift (5cm)
        self.step_height = 0.02  # Minimal lift (2cm)
        
        # Target velocity
        self.is_walking = False
        
        print("[QuasiStaticWalking] Ultra-stable quasi-static walking initialized")
        print(f"  Step length: {self.step_length} m")
        print(f"  Step height: {self.step_height} m")
        print(f"  Hip sway: {self.hip_sway} m")
    
    def set_target_velocity(self, vx, vy=0.0):
        """Start/stop walking"""
        was_walking = self.is_walking
        self.is_walking = (abs(vx) > 0.01)
        
        if self.is_walking and not was_walking:
            self.state = "standing"
            self.state_time = 0.0
            print("[QuasiStaticWalking] Starting quasi-static walk cycle...")
    
    def _update_state_machine(self, dt):
        """Update walking state machine"""
        if not self.is_walking:
            self.state = "standing"
            return
        
        self.state_time += dt
        
        # Transition to next state
        if self.state_time >= self.state_duration[self.state]:
            self.state_time = 0.0
            
            if self.state == "standing":
                self.state = "shift_left"
            elif self.state == "shift_left":
                self.state = "step_right"
            elif self.state == "step_right":
                self.state = "shift_right"
            elif self.state == "shift_right":
                self.state = "step_left"
            elif self.state == "step_left":
                self.state = "shift_left"  # Loop
    
    def _generate_pose(self):
        """Generate target pose based on current state"""
        q = self.pd_controller.q_target.copy()
        
        if self.state == "standing":
            # Just standing pose
            return q
        
        # Progress through current state (0 to 1)
        progress = self.state_time / self.state_duration[self.state]
        progress = np.clip(progress, 0.0, 1.0)
        
        # Smooth interpolation (ease-in-out)
        smooth_progress = 3*progress**2 - 2*progress**3
        
        if self.state == "shift_left":
            # Shift weight to left leg (lean left)
            # Left leg: slight hip roll outward
            q[1] = self.hip_sway * smooth_progress  # Left hip roll
            # Right leg: slight hip roll inward  
            q[6] = self.hip_sway * smooth_progress  # Right hip roll
            
        elif self.state == "step_right":
            # Swing right leg forward while balanced on left
            # Maintain weight shift
            q[1] = self.hip_sway  # Left hip roll (keep shifted)
            q[6] = self.hip_sway  # Right hip roll
            
            # Right leg forward swing
            q[7] = 0.1 * smooth_progress  # Right hip pitch forward
            q[8] = 0.35 + 0.1 * np.sin(smooth_progress * np.pi)  # Right knee (lift foot)
            q[9] = -0.15  # Right ankle
            
        elif self.state == "shift_right":
            # Shift weight to right leg (lean right)
            q[1] = self.hip_sway * (1 - smooth_progress)  # Left hip roll (reduce)
            q[6] = self.hip_sway * (1 - smooth_progress)  # Right hip roll
            
            # Both legs grounded
            q[7] = 0.1 * (1 - smooth_progress)  # Right hip returning to neutral
            
        elif self.state == "step_left":
            # Swing left leg forward while balanced on right
            q[1] = -self.hip_sway  # Left hip roll (weight on right)
            q[6] = -self.hip_sway  # Right hip roll
            
            # Left leg forward swing
            q[2] = 0.1 * smooth_progress  # Left hip pitch forward
            q[3] = 0.35 + 0.1 * np.sin(smooth_progress * np.pi)  # Left knee (lift foot)
            q[4] = -0.15  # Left ankle
        
        return q
    
    def compute_control(self, dt=0.002):
        """Compute control torques"""
        # Update state machine
        self._update_state_machine(dt)
        
        # Generate target pose for current state
        q_target = self._generate_pose()
        
        # Update PD controller target
        self.pd_controller.q_target = q_target
        
        # Compute PD control
        tau = self.pd_controller.compute_control(dt)
        
        return tau
    
    def update_target_pose(self, q_new):
        """Update target pose"""
        self.pd_controller.q_target = q_new
