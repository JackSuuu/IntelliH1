"""
Manipulator Controller for UR5e Robotic Arm
Handles inverse kinematics, motion planning, and gripper control
"""

import numpy as np
import mujoco
from typing import Tuple, Optional, List


class ManipulatorController:
    """
    Controller for UR5e 5-DOF robotic arm with gripper
    
    Joints:
    - shoulder_pan: Base rotation (yaw)
    - shoulder_lift: Shoulder pitch
    - elbow: Elbow flexion
    - wrist_rotate: Wrist roll
    - wrist_tilt: Wrist pitch
    - left/right_finger: Gripper fingers
    """
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        
        # Joint indices
        self.joint_names = [
            'shoulder_pan',
            'shoulder_lift', 
            'elbow',
            'wrist_rotate',
            'wrist_tilt'
        ]
        self.joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) 
                         for name in self.joint_names]
        
        # Actuator indices
        self.actuator_names = [
            'shoulder_pan_actuator',
            'shoulder_lift_actuator',
            'elbow_actuator', 
            'wrist_rotate_actuator',
            'wrist_tilt_actuator',
            'left_finger_actuator',
            'right_finger_actuator'
        ]
        self.actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                            for name in self.actuator_names]
        
        # End-effector site
        self.ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'ee_site')
        
        # Joint limits (from MJCF)
        self.joint_limits = {
            'shoulder_pan': (-3.14, 3.14),
            'shoulder_lift': (-1.57, 1.57),
            'elbow': (-2.0, 2.0),
            'wrist_rotate': (-3.14, 3.14),
            'wrist_tilt': (-1.57, 1.57)
        }
        
        # Home position (all zeros is safe)
        self.home_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Current target
        self.target_joint_pos = self.home_position.copy()
        self.gripper_target = 0.0  # 0=open, 0.025=closed
        
        # IK solver parameters - 优化后的参数
        self.ik_damping = 0.05  # 增加阻尼提高稳定性
        self.ik_max_iterations = 100  # 增加迭代次数
        self.ik_tolerance = 0.02  # 放宽容差到 2cm
        
        # Workspace limits (UR5e arm reach)
        self.max_reach = 0.35  # 20cm + 15cm arm segments
        self.min_reach = 0.10
        self.max_height = 0.40
        self.min_height = -0.05
        
        print(f"[Manipulator] Initialized with {len(self.joint_ids)} joints")
        print(f"[Manipulator] End-effector site ID: {self.ee_site_id}")
    
    def get_joint_positions(self) -> np.ndarray:
        """Get current joint angles"""
        positions = np.zeros(5)
        for i, joint_id in enumerate(self.joint_ids):
            positions[i] = self.data.qpos[joint_id]
        return positions
    
    def get_joint_velocities(self) -> np.ndarray:
        """Get current joint velocities"""
        velocities = np.zeros(5)
        for i, joint_id in enumerate(self.joint_ids):
            velocities[i] = self.data.qvel[joint_id]
        return velocities
    
    def get_end_effector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get end-effector position and orientation
        Returns: (position [x,y,z], quaternion [w,x,y,z])
        """
        # Get site position
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        
        # Get site orientation matrix
        ee_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        
        # Convert rotation matrix to quaternion
        ee_quat = self._mat_to_quat(ee_mat)
        
        return ee_pos, ee_quat
    
    def get_gripper_width(self) -> float:
        """Get current gripper width (0=closed, 0.05=fully open)"""
        left_pos = self.data.qpos[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, 'left_finger_joint')]
        right_pos = self.data.qpos[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, 'right_finger_joint')]
        return left_pos + right_pos
    
    def move_to_home(self):
        """Move arm to home position"""
        self.target_joint_pos = self.home_position.copy()
        print("[Manipulator] Moving to home position")
    
    def set_joint_targets(self, joint_angles: np.ndarray):
        """
        Set target joint angles directly
        
        Args:
            joint_angles: Array of 5 joint angles [shoulder_pan, shoulder_lift, 
                         elbow, wrist_rotate, wrist_tilt]
        """
        # Clamp to joint limits
        for i, name in enumerate(self.joint_names):
            lower, upper = self.joint_limits[name]
            joint_angles[i] = np.clip(joint_angles[i], lower, upper)
        
        self.target_joint_pos = joint_angles.copy()
    
    def is_in_workspace(self, target_pos: np.ndarray) -> bool:
        """
        Check if target position is within reachable workspace
        
        Args:
            target_pos: Target position [x, y, z]
        
        Returns:
            True if reachable, False otherwise
        """
        # Check XY plane distance
        distance_xy = np.linalg.norm(target_pos[:2])
        
        # Check height
        height = target_pos[2]
        
        in_reach = self.min_reach <= distance_xy <= self.max_reach
        in_height = self.min_height <= height <= self.max_height
        
        if not in_reach:
            print(f"[Manipulator] Target out of reach: {distance_xy:.3f}m (range: {self.min_reach}-{self.max_reach}m)")
        if not in_height:
            print(f"[Manipulator] Target height out of range: {height:.3f}m (range: {self.min_height}-{self.max_height}m)")
        
        return in_reach and in_height
    
    def move_to_pose(self, target_pos: np.ndarray, target_quat: Optional[np.ndarray] = None) -> bool:
        """
        Move end-effector to target pose using IK
        
        Args:
            target_pos: Target position [x, y, z]
            target_quat: Target orientation quaternion [w,x,y,z] (optional)
        
        Returns:
            True if IK solution found, False otherwise
        """
        # Check workspace first
        if not self.is_in_workspace(target_pos):
            print(f"[Manipulator] Target unreachable: {target_pos}")
            return False
        
        joint_solution = self.solve_ik(target_pos, target_quat)
        
        if joint_solution is not None:
            self.set_joint_targets(joint_solution)
            return True
        else:
            print(f"[Manipulator] IK failed for target: {target_pos}")
            return False
    
    def solve_ik(self, target_pos: np.ndarray, target_quat: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Solve inverse kinematics using Jacobian pseudo-inverse method
        
        Args:
            target_pos: Desired end-effector position [x, y, z]
            target_quat: Desired orientation quaternion [w,x,y,z] (optional)
        
        Returns:
            Joint angles solution or None if failed
        """
        # Start from current position
        q = self.get_joint_positions().copy()
        
        for iteration in range(self.ik_max_iterations):
            # Set current joint positions in data
            for i, joint_id in enumerate(self.joint_ids):
                self.data.qpos[joint_id] = q[i]
            
            # Forward kinematics
            mujoco.mj_kinematics(self.model, self.data)
            
            # Current end-effector pose
            current_pos, current_quat = self.get_end_effector_pose()
            
            # Position error
            pos_error = target_pos - current_pos
            error_norm = np.linalg.norm(pos_error)
            
            # Check convergence
            if error_norm < self.ik_tolerance:
                print(f"[Manipulator] IK converged in {iteration} iterations (error: {error_norm:.4f}m)")
                return q
            
            # Get Jacobian (only translational part for now)
            jacp = np.zeros((3, self.model.nv))  # Position Jacobian
            jacr = np.zeros((3, self.model.nv))  # Rotation Jacobian
            
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
            
            # Extract Jacobian for arm joints only
            J = jacp[:, :5]  # First 5 DOFs are arm joints
            
            # Damped least squares (pseudo-inverse)
            JtJ = J.T @ J + self.ik_damping * np.eye(5)
            J_pinv = np.linalg.solve(JtJ, J.T)
            
            # Joint velocity to minimize position error
            dq = J_pinv @ pos_error
            
            # Update joint angles with adaptive step size
            # Larger steps when far, smaller when close
            if error_norm > 0.1:
                step_size = 0.5
            elif error_norm > 0.05:
                step_size = 0.3
            else:
                step_size = 0.1
            
            q += step_size * dq
            
            # Clamp to joint limits
            for i, name in enumerate(self.joint_names):
                lower, upper = self.joint_limits[name]
                q[i] = np.clip(q[i], lower, upper)
        
        print(f"[Manipulator] IK failed to converge after {self.ik_max_iterations} iterations (error: {error_norm:.4f}m)")
        return None
    
    def set_gripper(self, width: float):
        """
        Set gripper width
        
        Args:
            width: 0.0 (closed) to 0.05 (fully open)
        """
        width = np.clip(width, 0.0, 0.025)
        self.gripper_target = width
    
    def open_gripper(self):
        """Fully open gripper"""
        self.set_gripper(0.025)
    
    def close_gripper(self):
        """Fully close gripper"""
        self.set_gripper(0.0)
    
    def grasp_object(self) -> bool:
        """
        Attempt to grasp object at current position
        Returns True if successful (force feedback detected)
        """
        self.close_gripper()
        # TODO: Add force sensing to detect successful grasp
        current_width = self.get_gripper_width()
        return current_width > 0.005  # Object present if gripper can't fully close
    
    def update(self):
        """
        Update controller - send commands to actuators
        Should be called every simulation step
        """
        # Set arm joint targets (position control)
        for i in range(5):
            self.data.ctrl[self.actuator_ids[i]] = self.target_joint_pos[i]
        
        # Set gripper targets (symmetric)
        self.data.ctrl[self.actuator_ids[5]] = self.gripper_target  # Left finger
        self.data.ctrl[self.actuator_ids[6]] = self.gripper_target  # Right finger
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get full state vector for neural network input
        Returns: [joint_pos (5), joint_vel (5), ee_pos (3), ee_quat (4), gripper_width (1)]
        Total: 18 dimensions
        """
        joint_pos = self.get_joint_positions()
        joint_vel = self.get_joint_velocities()
        ee_pos, ee_quat = self.get_end_effector_pose()
        gripper_width = self.get_gripper_width()
        
        state = np.concatenate([
            joint_pos,      # 5
            joint_vel,      # 5
            ee_pos,         # 3
            ee_quat,        # 4
            [gripper_width] # 1
        ])
        
        return state
    
    def execute_trajectory(self, waypoints: List[np.ndarray], duration: float = 1.0):
        """
        Execute a trajectory through multiple waypoints
        
        Args:
            waypoints: List of target positions [[x,y,z], ...]
            duration: Time to reach each waypoint (seconds)
        """
        print(f"[Manipulator] Executing trajectory with {len(waypoints)} waypoints")
        # TODO: Implement smooth trajectory interpolation
        for i, waypoint in enumerate(waypoints):
            print(f"[Manipulator] Moving to waypoint {i+1}/{len(waypoints)}: {waypoint}")
            success = self.move_to_pose(waypoint)
            if not success:
                print(f"[Manipulator] Failed to reach waypoint {i+1}")
                return False
        return True
    
    @staticmethod
    def _mat_to_quat(mat: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to quaternion [w,x,y,z]"""
        trace = np.trace(mat)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (mat[2, 1] - mat[1, 2]) * s
            y = (mat[0, 2] - mat[2, 0]) * s
            z = (mat[1, 0] - mat[0, 1]) * s
        elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
            s = 2.0 * np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2])
            w = (mat[2, 1] - mat[1, 2]) / s
            x = 0.25 * s
            y = (mat[0, 1] + mat[1, 0]) / s
            z = (mat[0, 2] + mat[2, 0]) / s
        elif mat[1, 1] > mat[2, 2]:
            s = 2.0 * np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2])
            w = (mat[0, 2] - mat[2, 0]) / s
            x = (mat[0, 1] + mat[1, 0]) / s
            y = 0.25 * s
            z = (mat[1, 2] + mat[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1])
            w = (mat[1, 0] - mat[0, 1]) / s
            x = (mat[0, 2] + mat[2, 0]) / s
            y = (mat[1, 2] + mat[2, 1]) / s
            z = 0.25 * s
        
        return np.array([w, x, y, z])


class TaskExecutor:
    """
    High-level task execution for manipulation tasks
    Coordinates navigation + manipulation
    """
    
    def __init__(self, manipulator: ManipulatorController):
        self.manipulator = manipulator
        self.current_task = None
    
    def pick_and_place(self, pick_pos: np.ndarray, place_pos: np.ndarray) -> bool:
        """
        Execute pick-and-place task
        
        Args:
            pick_pos: Position to pick object [x, y, z]
            place_pos: Position to place object [x, y, z]
        
        Returns:
            True if successful
        """
        print(f"[TaskExecutor] Pick-and-place: {pick_pos} -> {place_pos}")
        
        # 1. Move to pre-grasp position (above object)
        pre_grasp = pick_pos + np.array([0, 0, 0.1])
        if not self.manipulator.move_to_pose(pre_grasp):
            return False
        
        # 2. Open gripper
        self.manipulator.open_gripper()
        
        # 3. Move to grasp position
        if not self.manipulator.move_to_pose(pick_pos):
            return False
        
        # 4. Close gripper
        success = self.manipulator.grasp_object()
        if not success:
            print("[TaskExecutor] Failed to grasp object")
            return False
        
        # 5. Lift object
        lift_pos = pick_pos + np.array([0, 0, 0.15])
        if not self.manipulator.move_to_pose(lift_pos):
            return False
        
        # 6. Move to pre-place position
        pre_place = place_pos + np.array([0, 0, 0.15])
        if not self.manipulator.move_to_pose(pre_place):
            return False
        
        # 7. Lower to place position
        if not self.manipulator.move_to_pose(place_pos):
            return False
        
        # 8. Open gripper to release
        self.manipulator.open_gripper()
        
        # 9. Retract
        retract = place_pos + np.array([0, 0, 0.1])
        self.manipulator.move_to_pose(retract)
        
        print("[TaskExecutor] Pick-and-place completed successfully")
        return True
    
    def push_object(self, object_pos: np.ndarray, push_direction: np.ndarray, distance: float = 0.2):
        """Push object in specified direction"""
        print(f"[TaskExecutor] Pushing object at {object_pos} in direction {push_direction}")
        
        # Approach from behind
        approach = object_pos - push_direction * 0.1
        self.manipulator.move_to_pose(approach)
        
        # Push
        push_target = object_pos + push_direction * distance
        self.manipulator.move_to_pose(push_target)
        
        # Retract
        self.manipulator.move_to_home()
        print("[TaskExecutor] Push completed")
