#!/usr/bin/env python3
"""
Launch file for IntelliH1 ROS2 demo
Starts all nodes: simulation, RL controller, brain, and RViz
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    """Generate launch description with all IntelliH1 nodes"""
    
    # Declare launch arguments
    robot_type_arg = DeclareLaunchArgument(
        'robot_type',
        default_value='h1',
        description='Robot type: h1, h1_2, or g1'
    )
    
    max_speed_arg = DeclareLaunchArgument(
        'max_speed',
        default_value='1.0',
        description='Maximum walking speed in m/s'
    )
    
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz for visualization'
    )
    
    # Get launch configurations
    robot_type = LaunchConfiguration('robot_type')
    max_speed = LaunchConfiguration('max_speed')
    use_rviz = LaunchConfiguration('use_rviz')
    
    # Path to RViz config
    rviz_config = PathJoinSubstitution([
        FindPackageShare('intelli_h1_ros'),
        'config',
        'intelli_h1.rviz'
    ])
    
    # 1. MuJoCo Simulation Node
    sim_node = Node(
        package='intelli_h1_ros',
        executable='sim_node',
        name='mujoco_sim',
        output='screen',
        parameters=[{
            'model_path': 'extern/unitree_rl_gym/resources/robots/h1/scene_enhanced.xml',
            'update_rate': 500.0,
        }]
    )
    
    # 2. RL Controller Node
    rl_node = Node(
        package='intelli_h1_ros',
        executable='rl_node',
        name='rl_controller',
        output='screen',
        parameters=[{
            'robot_type': robot_type,
            'max_speed': max_speed,
            'update_rate': 50.0,
        }]
    )
    
    # 3. Brain/Planning Node
    brain_node = Node(
        package='intelli_h1_ros',
        executable='brain_node',
        name='brain',
        output='screen',
        parameters=[{
            'planning_rate': 1.0,
            'control_rate': 10.0,
            'goal_tolerance': 1.2,
            'max_speed': max_speed,
        }]
    )
    
    # 4. Robot State Publisher (publishes URDF transforms)
    # Note: This requires a proper URDF file
    robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': '',  # TODO: Load from URDF file
            'publish_frequency': 50.0,
        }],
        condition=lambda context: False  # Disabled until URDF is ready
    )
    
    # 5. RViz2 for visualization
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        condition=lambda context: context.launch_configurations['use_rviz'] == 'true'
    )
    
    return LaunchDescription([
        # Arguments
        robot_type_arg,
        max_speed_arg,
        use_rviz_arg,
        
        # Nodes
        sim_node,
        rl_node,
        brain_node,
        # robot_state_pub,  # Disabled until URDF ready
        # rviz_node,  # Disabled until config ready
    ])
