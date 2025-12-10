#!/usr/bin/env python3
"""
Brain Node for ROS2
Wraps the existing cognitive_controller.py for high-level planning
LLM + A* path planning + navigation logic
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import asyncio
from typing import Optional, List, Tuple

# ROS2 messages
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

# IntelliH1 modules
from .utils import setup_intellih1_path, DEFAULT_LOCATIONS

# Setup IntelliH1 paths
intellih1_root = setup_intellih1_path()

from llm.navigation_planner import NavigationPlanner
from perception.path_planner import AStarPlanner


class BrainNode(Node):
    """
    ROS2 node for high-level cognitive control
    - Receives natural language commands
    - Plans paths using A* 
    - Publishes velocity commands to RL controller
    - Visualizes planned path in RViz
    """
    
    def __init__(self):
        super().__init__('brain_node')
        
        # Parameters
        self.declare_parameter('planning_rate', 1.0)  # 1Hz for LLM planning
        self.declare_parameter('control_rate', 10.0)  # 10Hz for navigation control
        self.declare_parameter('goal_tolerance', 1.2)  # meters
        self.declare_parameter('max_speed', 1.0)  # m/s
        
        planning_rate = self.get_parameter('planning_rate').value
        control_rate = self.get_parameter('control_rate').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.max_speed = self.get_parameter('max_speed').value
        
        # Initialize planners
        self.get_logger().info('Initializing navigation planner...')
        try:
            self.nav_planner = NavigationPlanner()
            self.get_logger().info('✅ Navigation planner initialized')
        except Exception as e:
            self.get_logger().warn(f'⚠️ Navigation planner initialization failed: {e}')
            self.nav_planner = None
        
        self.path_planner = AStarPlanner(grid_resolution=0.3, map_size=30)
        self.get_logger().info('✅ A* path planner initialized')
        
        # TF listener for robot pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Navigation state
        self.current_goal: Optional[Tuple[float, float]] = None
        self.waypoints: List[Tuple[float, float]] = []
        self.current_waypoint_idx = 0
        self.is_navigating = False
        
        # Robot pose
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        
        # Known locations (can be overridden via ROS parameters in future)
        self.locations = DEFAULT_LOCATIONS.copy()
        
        # TODO: Add ROS parameter for custom locations
        # self.declare_parameter('locations', {})
        # custom_locations = self.get_parameter('locations').value
        # self.locations.update(custom_locations)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/path', 10)
        self.status_pub = self.create_publisher(String, '/brain_status', 10)
        
        # Subscribers
        self.command_sub = self.create_subscription(
            String,
            '/navigation_command',
            self.command_callback,
            10
        )
        
        # Timers
        # Planning timer (low frequency)
        self.planning_timer = self.create_timer(
            1.0 / planning_rate, 
            self.planning_callback
        )
        
        # Control timer (higher frequency)
        self.control_timer = self.create_timer(
            1.0 / control_rate,
            self.control_callback
        )
        
        self.get_logger().info('Brain node initialized')
        self.get_logger().info(f'Planning rate: {planning_rate}Hz, Control rate: {control_rate}Hz')
    
    def command_callback(self, msg):
        """Receive navigation commands"""
        command = msg.data.lower().strip()
        self.get_logger().info(f'📝 Received command: "{command}"')
        
        # Parse command
        if self.nav_planner:
            # Use LLM to parse command
            try:
                task = asyncio.run(self.nav_planner.parse_navigation_command(command))
                if task and 'target_location' in task:
                    target = task['target_location']
                    self.set_goal(target[0], target[1])
                    return
            except Exception as e:
                self.get_logger().warn(f'LLM parsing failed: {e}')
        
        # Fallback: Simple keyword matching
        for location_name, coords in self.locations.items():
            if location_name in command:
                self.get_logger().info(f'🎯 Going to {location_name}: {coords}')
                self.set_goal(coords[0], coords[1])
                return
        
        self.get_logger().warn(f'❌ Could not parse command: "{command}"')
    
    def set_goal(self, x: float, y: float):
        """Set navigation goal and plan path"""
        self.current_goal = (x, y)
        self.is_navigating = True
        self.current_waypoint_idx = 0
        
        # Get current robot position
        self.update_robot_pose()
        
        # Plan path using A*
        start = (self.robot_x, self.robot_y)
        goal = (x, y)
        
        self.get_logger().info(f'🗺️ Planning path from {start} to {goal}')
        
        # TODO: Update occupancy grid from sensor data
        # For now, use static obstacles or empty grid
        
        path = self.path_planner.plan(start, goal)
        
        if path and len(path) > 0:
            self.waypoints = path
            self.current_waypoint_idx = 0
            self.get_logger().info(f'✅ Path planned with {len(path)} waypoints')
            
            # Publish path for visualization
            self.publish_path()
        else:
            self.get_logger().error('❌ Path planning failed!')
            self.is_navigating = False
            self.waypoints = []
    
    def update_robot_pose(self):
        """Get robot pose from TF"""
        try:
            # Get transform from map to base_link
            trans = self.tf_buffer.lookup_transform(
                'map', 
                'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            self.robot_x = trans.transform.translation.x
            self.robot_y = trans.transform.translation.y
            
            # Extract yaw from quaternion
            qx = trans.transform.rotation.x
            qy = trans.transform.rotation.y
            qz = trans.transform.rotation.z
            qw = trans.transform.rotation.w
            
            # Convert quaternion to yaw
            siny_cosp = 2 * (qw * qz + qx * qy)
            cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
            self.robot_yaw = np.arctan2(siny_cosp, cosy_cosp)
            
        except Exception as e:
            # TF not available yet, use default
            pass
    
    def planning_callback(self):
        """Low-frequency planning updates"""
        if self.is_navigating and self.current_goal:
            # Update robot pose
            self.update_robot_pose()
            
            # Check if goal reached
            goal_x, goal_y = self.current_goal
            distance_to_goal = np.sqrt(
                (self.robot_x - goal_x)**2 + (self.robot_y - goal_y)**2
            )
            
            if distance_to_goal < self.goal_tolerance:
                self.get_logger().info('🎉 Goal reached!')
                self.is_navigating = False
                self.stop_robot()
                
                # Publish status
                status_msg = String()
                status_msg.data = 'Goal reached'
                self.status_pub.publish(status_msg)
    
    def control_callback(self):
        """High-frequency navigation control"""
        if not self.is_navigating or len(self.waypoints) == 0:
            return
        
        # Update robot pose
        self.update_robot_pose()
        
        # Get current waypoint
        if self.current_waypoint_idx >= len(self.waypoints):
            self.get_logger().info('All waypoints reached, stopping')
            self.stop_robot()
            self.is_navigating = False
            return
        
        target_x, target_y = self.waypoints[self.current_waypoint_idx]
        
        # Compute distance and heading to waypoint
        dx = target_x - self.robot_x
        dy = target_y - self.robot_y
        distance = np.sqrt(dx**2 + dy**2)
        target_heading = np.arctan2(dy, dx)
        
        # Heading error
        heading_error = target_heading - self.robot_yaw
        # Normalize to [-pi, pi]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # If waypoint reached, move to next
        if distance < 0.5:  # 0.5m threshold
            self.current_waypoint_idx += 1
            self.get_logger().info(f'Waypoint {self.current_waypoint_idx}/{len(self.waypoints)} reached')
            return
        
        # Compute velocity commands
        # Adaptive speed based on heading error
        if abs(heading_error) > np.deg2rad(60):
            # Large error: slow down significantly
            speed_factor = 0.1
        elif abs(heading_error) > np.deg2rad(30):
            # Medium error: moderate speed
            speed_factor = 0.3
        else:
            # Small error: full speed
            speed_factor = 1.0
        
        vx = self.max_speed * speed_factor
        omega = 0.8 * heading_error  # Proportional control, max 0.8 rad/s
        
        # Publish velocity command
        cmd = Twist()
        cmd.linear.x = vx
        cmd.linear.y = 0.0
        cmd.angular.z = omega
        self.cmd_vel_pub.publish(cmd)
    
    def stop_robot(self):
        """Send zero velocity command"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)
    
    def publish_path(self):
        """Publish planned path for RViz visualization"""
        if not self.waypoints:
            return
        
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        
        for wp in self.waypoints:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = wp[0]
            pose.pose.position.y = wp[1]
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = BrainNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error in brain_node: {e}')
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
