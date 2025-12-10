#!/usr/bin/env python3
"""
RL Controller Node for ROS2
Wraps the existing unitree_rl_controller.py into a ROS2 node
Subscribes to /cmd_vel and publishes motor commands
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np

# ROS2 messages
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, Header

# IntelliH1 controller
import mujoco
from .utils import setup_intellih1_path

# Setup IntelliH1 paths
intellih1_root = setup_intellih1_path()

from control.unitree_rl_controller import UnitreeRLController


class RLControllerNode(Node):
    """
    ROS2 node for RL-based locomotion control
    - Subscribes to /joint_states for robot state
    - Subscribes to /cmd_vel for velocity commands
    - Publishes /motor_cmd for actuator torques
    """
    
    def __init__(self):
        super().__init__('rl_controller_node')
        
        # Parameters
        self.declare_parameter('robot_type', 'h1')
        self.declare_parameter('max_speed', 1.0)
        self.declare_parameter('model_path', 'extern/unitree_rl_gym/resources/robots/h1/scene_enhanced.xml')
        self.declare_parameter('policy_path', '')
        self.declare_parameter('update_rate', 50.0)  # 50Hz for RL inference
        
        robot_type = self.get_parameter('robot_type').value
        max_speed = self.get_parameter('max_speed').value
        model_path = self.get_parameter('model_path').value
        policy_path = self.get_parameter('policy_path').value
        update_rate = self.get_parameter('update_rate').value
        
        # Load MuJoCo model (needed for controller initialization)
        full_model_path = os.path.join(intellih1_root, model_path)
        if not os.path.exists(full_model_path):
            self.get_logger().error(f'Model file not found: {full_model_path}')
            raise FileNotFoundError(f'Model file not found: {full_model_path}')
        
        self.model = mujoco.MjModel.from_xml_path(full_model_path)
        self.data = mujoco.MjData(self.model)
        
        # Initialize RL controller
        self.get_logger().info(f'Initializing RL controller: {robot_type}')
        policy_path_arg = policy_path if policy_path else None
        self.controller = UnitreeRLController(
            self.model, 
            self.data, 
            policy_path=policy_path_arg,
            robot_type=robot_type,
            max_speed=max_speed
        )
        
        # Target velocity command
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_omega = 0.0
        
        # QoS profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            qos_profile
        )
        
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10  # Standard QoS for cmd_vel
        )
        
        # Publisher for motor commands
        self.motor_cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/motor_cmd',
            qos_profile
        )
        
        # Timer for control loop
        timer_period = 1.0 / update_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info(f'RL controller node initialized at {update_rate}Hz')
    
    def joint_state_callback(self, msg):
        """Update robot state from joint states"""
        # Update MuJoCo data structure with received joint states
        if len(msg.position) > 0:
            # Update joint positions (after floating base)
            num_joints = min(len(msg.position), len(self.data.qpos) - 7)
            self.data.qpos[7:7+num_joints] = msg.position[:num_joints]
        
        if len(msg.velocity) > 0:
            # Update joint velocities (after floating base)
            num_joints = min(len(msg.velocity), len(self.data.qvel) - 6)
            self.data.qvel[6:6+num_joints] = msg.velocity[:num_joints]
    
    def cmd_vel_callback(self, msg):
        """Receive velocity commands from brain node"""
        self.target_vx = msg.linear.x
        self.target_vy = msg.linear.y
        self.target_omega = msg.angular.z
        
        # Update controller target
        self.controller.set_target_velocity(
            self.target_vx, 
            self.target_vy, 
            self.target_omega
        )
    
    def timer_callback(self):
        """Control loop - compute and publish motor commands"""
        try:
            # Compute control action using RL policy
            action = self.controller.compute_action()
            
            # Publish motor commands
            msg = Float64MultiArray()
            msg.data = action.tolist()
            self.motor_cmd_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f'Error in control loop: {e}')


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = RLControllerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error in rl_node: {e}')
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
