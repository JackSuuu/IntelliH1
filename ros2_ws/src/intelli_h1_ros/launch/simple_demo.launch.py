#!/usr/bin/env python3
"""
Simplified launch file for quick testing
Launches only sim_node and rl_node without brain for basic walking
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Generate simplified launch for basic locomotion testing"""
    
    # Arguments
    max_speed_arg = DeclareLaunchArgument(
        'max_speed',
        default_value='1.0',
        description='Maximum walking speed in m/s'
    )
    
    robot_type_arg = DeclareLaunchArgument(
        'robot_type',
        default_value='h1',
        description='Robot type: h1, h1_2, or g1'
    )
    
    max_speed = LaunchConfiguration('max_speed')
    robot_type = LaunchConfiguration('robot_type')
    
    # Simulation node
    sim_node = Node(
        package='intelli_h1_ros',
        executable='sim_node',
        name='mujoco_sim',
        output='screen',
        parameters=[{
            'model_path': 'extern/unitree_rl_gym/resources/robots/h1/scene_enhanced.xml',
            'update_rate': 500.0,
        }],
        emulate_tty=True,
    )
    
    # RL controller node
    rl_node = Node(
        package='intelli_h1_ros',
        executable='rl_node',
        name='rl_controller',
        output='screen',
        parameters=[{
            'robot_type': robot_type,
            'max_speed': max_speed,
            'update_rate': 50.0,
        }],
        emulate_tty=True,
    )
    
    return LaunchDescription([
        max_speed_arg,
        robot_type_arg,
        sim_node,
        rl_node,
    ])
