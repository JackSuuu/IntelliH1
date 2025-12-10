#!/usr/bin/env python3
"""
MuJoCo Simulation Node for ROS2
Wraps the existing simulation/environment.py into a ROS2 node
Publishes sensor data and receives motor commands
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
from scipy.spatial.transform import Rotation

# ROS2 messages
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Header
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

# MuJoCo and IntelliH1
import sys
import os
# Add IntelliH1 src to path
intellih1_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..'))
sys.path.insert(0, os.path.join(intellih1_root, 'src'))

import mujoco


class MuJoCoSimNode(Node):
    """
    ROS2 node that runs MuJoCo physics simulation at 500Hz
    - Publishes /joint_states for robot visualization
    - Publishes /tf for robot pose in world frame
    - Subscribes to /motor_cmd for actuator torques
    """
    
    def __init__(self):
        super().__init__('mujoco_sim_node')
        
        # Parameters
        self.declare_parameter('model_path', 'extern/unitree_rl_gym/resources/robots/h1/scene_enhanced.xml')
        self.declare_parameter('update_rate', 500.0)  # 500Hz for physics
        
        model_path = self.get_parameter('model_path').value
        update_rate = self.get_parameter('update_rate').value
        
        # Load MuJoCo model
        self.get_logger().info(f'Loading MuJoCo model: {model_path}')
        full_model_path = os.path.join(intellih1_root, model_path)
        
        if not os.path.exists(full_model_path):
            self.get_logger().error(f'Model file not found: {full_model_path}')
            raise FileNotFoundError(f'Model file not found: {full_model_path}')
        
        self.model = mujoco.MjModel.from_xml_path(full_model_path)
        self.data = mujoco.MjData(self.model)
        
        # Motor commands (torques)
        self.motor_commands = np.zeros(self.model.nu)
        
        # QoS profile for real-time data
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Publishers
        self.joint_state_pub = self.create_publisher(
            JointState, 
            '/joint_states', 
            qos_profile
        )
        
        # TF broadcaster for robot pose
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Subscriber for motor commands
        self.motor_cmd_sub = self.create_subscription(
            Float64MultiArray,
            '/motor_cmd',
            self.motor_cmd_callback,
            qos_profile
        )
        
        # Timer for physics update (500Hz)
        timer_period = 1.0 / update_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # Extract joint names from MuJoCo model
        self.joint_names = []
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name is not None:
                self.joint_names.append(joint_name)
        
        self.get_logger().info(f'Simulation node initialized at {update_rate}Hz')
        self.get_logger().info(f'Number of joints: {len(self.joint_names)}')
        self.get_logger().info(f'Number of actuators: {self.model.nu}')
    
    def motor_cmd_callback(self, msg):
        """Receive motor commands (torques) from RL controller"""
        if len(msg.data) == self.model.nu:
            self.motor_commands = np.array(msg.data)
        else:
            self.get_logger().warn(
                f'Received motor command size {len(msg.data)} does not match '
                f'expected size {self.model.nu}'
            )
    
    def timer_callback(self):
        """Main simulation loop - runs at 500Hz"""
        # Apply motor commands (torques)
        self.data.ctrl[:] = self.motor_commands
        
        # Step physics
        mujoco.mj_step(self.model, self.data)
        
        # Publish robot state
        self.publish_joint_states()
        self.publish_tf()
    
    def publish_joint_states(self):
        """Publish joint states for RViz visualization"""
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        
        # Get joint positions and velocities
        # Note: qpos includes the floating base (7 DOF: position + quaternion)
        # We need to extract only the joint angles
        if self.data.qpos.size > 7:
            joint_positions = self.data.qpos[7:].tolist()
            joint_velocities = self.data.qvel[6:].tolist()  # qvel has 6 DOF for floating base
            
            # Match number of joints
            num_joints = min(len(self.joint_names), len(joint_positions))
            
            msg.name = self.joint_names[:num_joints]
            msg.position = joint_positions[:num_joints]
            msg.velocity = joint_velocities[:num_joints]
            
            self.joint_state_pub.publish(msg)
    
    def publish_tf(self):
        """Publish TF transform for robot base_link in world frame"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'
        
        # Get base position (first 3 elements of qpos)
        t.transform.translation.x = float(self.data.qpos[0])
        t.transform.translation.y = float(self.data.qpos[1])
        t.transform.translation.z = float(self.data.qpos[2])
        
        # Get base orientation (quaternion: qpos[3:7])
        # MuJoCo uses [w, x, y, z] format
        # ROS uses [x, y, z, w] format
        t.transform.rotation.x = float(self.data.qpos[4])
        t.transform.rotation.y = float(self.data.qpos[5])
        t.transform.rotation.z = float(self.data.qpos[6])
        t.transform.rotation.w = float(self.data.qpos[3])
        
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = MuJoCoSimNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error in sim_node: {e}')
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
