"""
Unitree H1 Humanoid Robot Controller
Simplified version using only PD Standing control based on official Unitree examples
Now with optional RL policy assistance and walking capability!
"""

import mujoco
import numpy as np
from control.pd_standing import PDStandingController, ImprovedPDController, RLAssistedPDController
from control.walking_controller import SimpleWalkingController, NavigationController
from control.quasi_static_walking import QuasiStaticWalkingController


class UnitreeH1Controller:
    """
    Simplified Unitree H1 controller using only PD standing control.
    Based on official Unitree SDK examples adapted for MuJoCo simulation.
    """
    
    def __init__(self, model, data, use_gravity_compensation=True, use_rl_assist=False, 
                 rl_policy_path="policies/h1_demo_policy.pt", mode="standing"):
        """
        Initialize H1 controller with PD standing control.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            use_gravity_compensation: Use improved PD with gravity compensation (recommended)
            use_rl_assist: Use RL policy to assist PD controller (EXPERIMENTAL)
            rl_policy_path: Path to RL policy file
            mode: Control mode - "standing", "walking", or "navigation"
        """
        self.model = model
        self.data = data
        self.current_time = 0.0
        self.mode = mode
        
        # Initialize controller based on mode
        if mode == "navigation":
            # 🗺️ Navigation mode: Walk to target positions
            self.controller = NavigationController(model, data)
            print("[UnitreeH1] 🗺️ Navigation controller initialized")
        elif mode == "walking":
            # 🚶 Walking mode: Quasi-static walking (slow but stable)
            self.controller = QuasiStaticWalkingController(model, data)
            print("[UnitreeH1] � Quasi-static walking controller initialized")
        elif use_rl_assist:
            # 🤖 RL-Assisted mode: Smart balance with learned behaviors
            self.controller = RLAssistedPDController(model, data, policy_path=rl_policy_path, rl_weight=0.3)
            print("[UnitreeH1] 🤖 RL-Assisted PD controller initialized")
        elif use_gravity_compensation:
            # Standard improved PD mode
            self.controller = ImprovedPDController(model, data)
            print("[UnitreeH1] ✅ PD controller with gravity compensation initialized")
        else:
            # Simple PD mode
            self.controller = PDStandingController(model, data)
            print("[UnitreeH1] ✅ Simple PD controller initialized")

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

    def update_target_pose(self, q_new):
        """
        Update target standing pose (for future walking implementation).
        
        Args:
            q_new: New target joint positions
        """
        if hasattr(self.controller, 'update_target_pose'):
            self.controller.update_target_pose(q_new)
    
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
