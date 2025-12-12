#!/usr/bin/env python3
"""
MuJoCo Simulation Node for ROS2 with VIEWER support (macOS compatible)
This version includes the RL controller directly for standalone operation.

Run this with: mjpython sim_node_viewer.py
Or use the provided shell script: ros2_ws/scripts/run_sim_viewer.sh
"""

import os
import sys
import threading
import time
import numpy as np

# ROS2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.executors import SingleThreadedExecutor

# ROS2 messages
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Header
from geometry_msgs.msg import Twist, TransformStamped
from tf2_ros import TransformBroadcaster

# MuJoCo
import mujoco
import mujoco.viewer


def setup_intellih1_path():
    """Setup path to IntelliH1 root directory"""
    current = os.path.dirname(os.path.abspath(__file__))
    
    for _ in range(10):
        if os.path.exists(os.path.join(current, 'extern', 'unitree_rl_gym')):
            if current not in sys.path:
                sys.path.insert(0, os.path.join(current, 'src'))
            return current
        current = os.path.dirname(current)
    
    home = os.path.expanduser('~')
    possible_paths = [
        os.path.join(home, 'Desktop', 'PROJECTS', '2025', 'IntelliH1'),
        os.path.join(home, 'IntelliH1'),
        '/Users/jacksu/Desktop/PROJECTS/2025/IntelliH1',
    ]
    
    for path in possible_paths:
        if os.path.exists(os.path.join(path, 'extern', 'unitree_rl_gym')):
            if os.path.join(path, 'src') not in sys.path:
                sys.path.insert(0, os.path.join(path, 'src'))
            return path
    
    raise RuntimeError("Could not find IntelliH1 root directory")


# Setup paths before importing IntelliH1 modules
intellih1_root = setup_intellih1_path()

# Now import the RL controller
from control.unitree_rl_controller import UnitreeRLController


def mujoco_to_ros_quaternion(mujoco_quat):
    """Convert MuJoCo quaternion [w,x,y,z] to ROS quaternion [x,y,z,w]"""
    return [mujoco_quat[1], mujoco_quat[2], mujoco_quat[3], mujoco_quat[0]]


class MuJoCoSimViewerNode(Node):
    """
    ROS2 node that runs MuJoCo simulation with integrated RL controller and viewer
    """
    
    def __init__(self, model, data, controller):
        super().__init__('mujoco_sim_node')
        
        self.model = model
        self.data = data
        self.controller = controller
        
        # Velocity commands
        self.target_vx = 0.8  # Default forward velocity
        self.target_vy = 0.0
        self.target_omega = 0.0
        self.lock = threading.Lock()
        
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
        
        # Subscriber for velocity commands
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        # Extract joint names from MuJoCo model
        self.joint_names = []
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name is not None:
                self.joint_names.append(joint_name)
        
        self.get_logger().info(f'Simulation node initialized with RL controller and viewer')
        self.get_logger().info(f'Number of joints: {len(self.joint_names)}')
        self.get_logger().info(f'Number of actuators: {self.model.nu}')
        self.get_logger().info(f'Default velocity: vx={self.target_vx} m/s')
    
    def cmd_vel_callback(self, msg):
        """Receive velocity commands"""
        with self.lock:
            self.target_vx = msg.linear.x
            self.target_vy = msg.linear.y
            self.target_omega = msg.angular.z
            self.get_logger().info(f'Received cmd_vel: vx={self.target_vx:.2f}, vy={self.target_vy:.2f}, omega={self.target_omega:.2f}')
    
    def get_velocity_command(self):
        """Get velocity command thread-safely"""
        with self.lock:
            return self.target_vx, self.target_vy, self.target_omega
    
    def step_simulation(self):
        """Run one step of simulation with RL controller"""
        # Get velocity command
        vx, vy, omega = self.get_velocity_command()
        
        # Update controller command
        self.controller.set_velocity_command(vx, vy, omega)
        
        # Get action from RL policy (compute_control returns torques)
        action = self.controller.compute_control()
        
        # Apply action to actuators
        self.data.ctrl[:] = action
        
        # Step physics
        mujoco.mj_step(self.model, self.data)
    
    def publish_state(self):
        """Publish robot state to ROS2 topics"""
        self.publish_joint_states()
        self.publish_tf()
    
    def publish_joint_states(self):
        """Publish joint states"""
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        
        if self.data.qpos.size > 7:
            joint_positions = self.data.qpos[7:].tolist()
            joint_velocities = self.data.qvel[6:].tolist()
            
            num_joints = min(len(self.joint_names), len(joint_positions))
            
            msg.name = self.joint_names[:num_joints]
            msg.position = joint_positions[:num_joints]
            msg.velocity = joint_velocities[:num_joints]
            
            self.joint_state_pub.publish(msg)
    
    def publish_tf(self):
        """Publish TF transform"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'base_link'
        
        t.transform.translation.x = float(self.data.qpos[0])
        t.transform.translation.y = float(self.data.qpos[1])
        t.transform.translation.z = float(self.data.qpos[2])
        
        mujoco_quat = self.data.qpos[3:7]
        ros_quat = mujoco_to_ros_quaternion(mujoco_quat)
        
        t.transform.rotation.x = ros_quat[0]
        t.transform.rotation.y = ros_quat[1]
        t.transform.rotation.z = ros_quat[2]
        t.transform.rotation.w = ros_quat[3]
        
        self.tf_broadcaster.sendTransform(t)


def main():
    """Main function - runs MuJoCo viewer with integrated RL controller"""
    
    # Initialize ROS2
    rclpy.init()
    
    # Load model
    model_path = os.path.join(
        intellih1_root, 
        'extern/unitree_rl_gym/resources/robots/h1/scene_enhanced.xml'
    )
    
    print(f"Loading MuJoCo model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)
    
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # Initialize RL controller
    print("Initializing RL controller...")
    controller = UnitreeRLController(
        model, 
        data, 
        robot_type='h1',
        max_speed=1.0
    )
    print("RL controller initialized!")
    
    # Create ROS2 node
    node = MuJoCoSimViewerNode(model, data, controller)
    
    # Create executor for spinning in a separate thread
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    
    # Spin ROS2 in background thread
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()
    
    print("=" * 60)
    print("MuJoCo Simulation with ROS2 + RL Controller")
    print("=" * 60)
    print("ROS2 Topics:")
    print("  - Publishing: /joint_states, /tf")
    print("  - Subscribing: /cmd_vel")
    print("")
    print("Control the robot with:")
    print("  ros2 topic pub /cmd_vel geometry_msgs/Twist \\")
    print("    \"{linear: {x: 1.0}, angular: {z: 0.0}}\" --rate 10")
    print("")
    print("Robot will walk forward by default (vx=0.8 m/s)")
    print("=" * 60)
    
    # Simulation timing
    sim_dt = model.opt.timestep  # MuJoCo timestep
    control_dt = 0.02  # 50Hz control rate
    steps_per_control = int(control_dt / sim_dt)
    publish_every = 10  # Publish ROS messages every N control steps
    
    step_count = 0
    
    # Launch MuJoCo viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("MuJoCo viewer launched! Close the window to exit.")
        
        start_time = time.time()
        sim_time = 0.0
        
        while viewer.is_running():
            step_start = time.time()
            
            # Run RL controller and physics
            for _ in range(steps_per_control):
                node.step_simulation()
                sim_time += sim_dt
            
            # Publish state at lower rate
            step_count += 1
            if step_count >= publish_every:
                node.publish_state()
                step_count = 0
            
            # Sync viewer
            viewer.sync()
            
            # Maintain real-time (approximately)
            elapsed = time.time() - step_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    print("Viewer closed, shutting down...")
    
    # Cleanup
    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
