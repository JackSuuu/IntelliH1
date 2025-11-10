"""
Unitree H1 Humanoid Robot Controller
Simplified version using only PD Standing control based on official Unitree examples
Now with optional RL policy assistance and walking capability!
"""

import mujoco
import numpy as np

# Import official Unitree RL controller
from control.unitree_rl_controller import UnitreeRLWalkingController


class UnitreeH1Controller:
    """
    Unitree H1 controller using official Unitree RL walking controller.
    Supports walking and navigation in MuJoCo simulation.
    """
    
    def __init__(self, model, data, mode="walking", robot_type="h1", **kwargs):
        """
        Initialize H1 controller with official Unitree RL controller.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            mode: Control mode - "walking" or "navigation" (default: "walking")
            robot_type: Robot type ("h1", "h1_2", "g1")
            **kwargs: Ignored legacy arguments for backward compatibility
        """
        self.model = model
        self.data = data
        self.current_time = 0.0
        self.mode = mode
        
        # Initialize official Unitree RL walking controller
        self.controller = UnitreeRLWalkingController(model, data, robot_type=robot_type)
        print(f"[UnitreeH1] ⭐ Official Unitree RL {mode} controller initialized")
    
    def set_velocity_command(self, vx, vy, omega):
        """Set velocity command for walking controller"""
        self.controller.set_target_velocity(vx, vy, omega)
    
    def set_target(self, x, y):
        """Set navigation target for walking controller"""
        self.controller.set_target(x, y)

    def apply_control(self, dt=0.002):
        """
        Compute and apply control torques to maintain standing balance.
        
        Args:
            dt: Control timestep (default 2ms, same as official examples)
        """
        try:
            # Compute PD control torques with ankle/hip/torso strategies
            tau_optimal = self.controller.compute_control(dt)
        except Exception as e:
            if not hasattr(self, '_control_error_printed'):
                print(f"[UnitreeH1] Control error: {e}")
                self._control_error_printed = True
            tau_optimal = np.zeros(self.model.nu)
        
        # Apply torques to MuJoCo actuators
        self.data.ctrl[:] = tau_optimal

    def get_position(self):
        """Returns the current 3D position of the robot's pelvis."""
        return self.data.body('pelvis').xpos.copy()

    def get_heading_angle(self):
        """Returns the current heading angle (yaw) of the robot's torso."""
        torso_id = self.model.body('torso_link').id
        torso_mat = self.data.xmat[torso_id].reshape(3, 3)
        # Yaw is the rotation around the Z-axis
        yaw = np.arctan2(torso_mat[1, 0], torso_mat[0, 0])
        return yaw

    def update(self, time, dt=0.002):
        """
        Update controller at each timestep.
        
        Args:
            time: Current simulation time
            dt: Control timestep
        """
        self.current_time = time
        self.apply_control(dt)


    
    def set_walking_velocity(self, vx, vy=0.0):
        """
        Set walking velocity (only for walking mode)
        
        Args:
            vx: Forward velocity (m/s)
            vy: Lateral velocity (m/s)
        """
        if hasattr(self.controller, 'set_target_velocity'):
            self.controller.set_target_velocity(vx, vy)
        else:
            print("[UnitreeH1] ⚠️ Walking control not available in current mode")
    
    def set_navigation_target(self, target_x, target_y):
        """
        Set navigation target (only for navigation mode)
        
        Args:
            target_x: Target x position
            target_y: Target y position
        """
        if hasattr(self.controller, 'set_target'):
            self.controller.set_target(target_x, target_y)
        else:
            print("[UnitreeH1] ⚠️ Navigation not available in current mode")
