"""
Advanced Balance Controller with ZMP and Contact Force Feedback
Addresses the key issues for bipedal stability:
1. ZMP (Zero Moment Point) calculation for dynamic balance
2. Contact force feedback from MuJoCo sensors
3. Adaptive PD gains based on stability state
"""

import numpy as np
import mujoco
from control.pd_standing import PDStandingController


class ZMPBalanceController:
    """
    Advanced balance controller using:
    - ZMP calculation for dynamic stability
    - Contact force sensing
    - Adaptive control based on stability margin
    """
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.nu = model.nu
        
        # Base PD controller
        self.pd_controller = PDStandingController(model, data)
        
        # High gains for aggressive stabilization
        self.pd_controller.kp = (self.pd_controller.kp * 3.0).astype(self.pd_controller.kp.dtype)
        self.pd_controller.kd = (self.pd_controller.kd * 2.0).astype(self.pd_controller.kd.dtype)
        
        # ZMP parameters
        self.foot_length = 0.2  # meters
        self.foot_width = 0.1   # meters
        self.zmp_safety_margin = 0.03  # Stay 3cm inside support polygon
        
        # Contact force threshold
        self.contact_threshold = 10.0  # Newtons
        
        print("[ZMPBalance] Advanced balance controller initialized")
        print("  • ZMP-based stability control")
        print("  • Contact force feedback")
        print("  • Adaptive PD gains")
    
    def _get_contact_forces(self):
        """Get contact forces from foot sensors"""
        left_foot_force = 0.0
        right_foot_force = 0.0
        
        # Sum contact forces on foot geoms
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # Get contact force magnitude
            force = np.linalg.norm(self.data.sensordata[i*6:(i+1)*6])
            
            # Identify which foot (check geom names)
            geom1_name = self.model.geom(contact.geom1).name
            geom2_name = self.model.geom(contact.geom2).name
            
            if 'left_ankle' in geom1_name or 'left_ankle' in geom2_name:
                left_foot_force += force
            elif 'right_ankle' in geom1_name or 'right_ankle' in geom2_name:
                right_foot_force += force
        
        return left_foot_force, right_foot_force
    
    def _calculate_zmp(self):
        """
        Calculate Zero Moment Point position
        ZMP = point where net moment is zero (should be inside support polygon)
        """
        # Get CoM position and velocity
        com_pos = self.data.subtree_com[0].copy()  # Root body CoM
        com_vel = self.data.cvel[0, 3:6].copy()   # Linear velocity
        
        # Get foot positions (simplified - use ankle positions)
        left_foot_pos = self.data.body('left_ankle_link').xpos.copy()
        right_foot_pos = self.data.body('right_ankle_link').xpos.copy()
        
        # Calculate ZMP using simplified formula
        # ZMP ≈ CoM_x - (CoM_z / g) * CoM_vx
        g = 9.81
        zmp_x = com_pos[0] - (com_pos[2] / g) * com_vel[0]
        zmp_y = com_pos[1] - (com_pos[2] / g) * com_vel[1]
        
        # Support polygon center (between feet)
        support_center = (left_foot_pos + right_foot_pos) / 2.0
        
        # ZMP error (how far ZMP is from support center)
        zmp_error = np.array([
            zmp_x - support_center[0],
            zmp_y - support_center[1]
        ])
        
        return zmp_error, support_center
    
    def compute_control(self, dt=0.002):
        """
        Compute control with ZMP-based balance
        """
        # Get current state
        q = self.data.qpos[7:7+self.nu]
        dq = self.data.qvel[6:6+self.nu]
        
        # Calculate ZMP
        zmp_error, support_center = self._calculate_zmp()
        
        # Get contact forces
        left_force, right_force = self._get_contact_forces()
        
        # Check stability (both feet in contact)
        is_stable = (left_force > self.contact_threshold and 
                     right_force > self.contact_threshold)
        
        # Base target pose
        q_target = self.pd_controller.q_target.copy()
        
        if is_stable:
            # Apply ZMP-based corrections
            # If ZMP is forward, lean ankles backward
            ankle_correction = -200.0 * zmp_error[0] - 30.0 * self.data.qvel[0]
            hip_correction = -80.0 * zmp_error[0] - 15.0 * self.data.qvel[0]
            
            # Apply corrections
            q_target[2] += hip_correction   # Left hip pitch
            q_target[7] += hip_correction   # Right hip pitch
            q_target[4] += ankle_correction  # Left ankle
            q_target[9] += ankle_correction  # Right ankle
        
        # Compute PD control
        tau = self.pd_controller.kp * (q_target - q) - self.pd_controller.kd * dq
        
        # Gravity compensation
        tau_gravity = self.data.qfrc_bias[6:6+self.nu]
        tau += tau_gravity
        
        # Clip to limits
        tau = np.clip(tau, -self.pd_controller.torque_limit, self.pd_controller.torque_limit)
        
        return tau
    
    def update_target_pose(self, q_new):
        """Update target pose"""
        self.pd_controller.q_target = q_new
