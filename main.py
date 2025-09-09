import mujoco
import mujoco.viewer
import numpy as np
import requests
import json
import time
import os
from dotenv import load_dotenv

# DeepSeek API setup (replace with your key)
# Load environment variables from .env file
load_dotenv()

# DeepSeek API setup from environment variables
API_KEY = os.getenv("DEEPSEEK_API_KEY")
API_URL = os.getenv("DEEPSEEK_BASE_URL")
 
if not API_KEY or not API_URL:
    raise ValueError("Please set DEEPSEEK_API_KEY and DEEPSEEK_BASE_URL in your .env file")

# Enhanced MJCF model: Better graphics and visual elements
mjcf_xml = """
<mujoco model="enhanced_car">
  <compiler angle="degree"/>
  <option gravity="0 0 -9.81" timestep="0.002" integrator="Euler"/>
  
  <default>
    <joint damping="1" frictionloss="0.05"/>
    <geom condim="3" friction="0.8 0.1 0.1"/>
  </default>
  
  <asset>
    <!-- Enhanced textures and materials -->
    <texture type="skybox" builtin="gradient" rgb1=".2 .3 .4" rgb2="0 0 0" width="512" height="512"/>
    <texture builtin="checker" height="512" name="texplane" rgb1="0.2 0.3 0.2" rgb2="0.1 0.2 0.1" type="2d" width="512"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="gradient" height="100" name="texgrad" rgb1="0 0 0" rgb2="1 1 1" type="2d" width="100"/>
    
    <material name="MatPlane" reflectance="0.3" shininess="1" specular="0.4" texrepeat="20 20" texture="texplane"/>
    <material name="car_body" reflectance="0.7" shininess="0.3" specular="0.5" texture="texgeom"/>
    <material name="wheel_mat" reflectance="0.1" shininess="0.1" specular="0.1" rgba="0.1 0.1 0.1 1"/>
    <material name="goal_mat" reflectance="0.8" shininess="0.6" specular="0.7" rgba="0.2 0.8 0.2 0.9"/>
    <material name="obstacle_mat" reflectance="0.8" shininess="0.6" specular="0.7" rgba="0.8 0.2 0.2 0.9"/>
    <material name="lidar_mat" reflectance="0" shininess="0" specular="0" rgba="1 0 0 0.3"/>
  </asset>
  
  <worldbody>
    <!-- Enhanced lighting -->
    <light cutoff="100" diffuse="1 1 1" dir="0 0 -1" directional="true" exponent="1" pos="0 0 3" specular="0.3 0.3 0.3"/>
    <light cutoff="60" diffuse="0.5 0.5 0.8" dir="-1 -1 -0.5" directional="true" exponent="1" pos="2 2 2" specular="0.1 0.1 0.1"/>
    
    <!-- Enhanced floor -->
    <geom conaffinity="0" condim="3" material="MatPlane" name="floor" pos="0 0 0" size="40 40 40" type="plane"/>
    
    <!-- Car with enhanced graphics -->
    <body name="car" pos="0 0 0.1">
      <joint name="car_x" type="slide" axis="1 0 0" damping="0.5"/>
      <joint name="car_y" type="slide" axis="0 1 0" damping="0.5"/>
      <joint name="car_theta" type="hinge" axis="0 0 1" damping="0.5"/>
      <inertial pos="0 0 0" mass="5" diaginertia="0.5 0.5 0.5"/>
      
      <!-- Main body -->
      <geom type="box" size="0.15 0.08 0.04" material="car_body"/>
      
      <!-- Car details -->
      <geom name="windshield" type="box" size="0.12 0.06 0.01" pos="0.05 0 0.04" rgba="0.3 0.3 0.8 0.6"/>
      <geom name="hood" type="box" size="0.08 0.07 0.02" pos="0.1 0 0.02" rgba="0.9 0.7 0.5 1"/>
      
      <!-- LiDAR sensor visualization -->
      <geom name="lidar_base" type="cylinder" size="0.02 0.01" pos="0 0 0.06" rgba="0.2 0.2 0.2 1"/>
      <geom name="lidar_top" type="cylinder" size="0.015 0.005" pos="0 0 0.08" rgba="0.8 0.2 0.2 1"/>
      
      <!-- Wheels with better materials -->
      <body name="rear_left_wheel" pos="-0.1 0.08 -0.04">
        <joint name="rear_left_wheel_joint" type="hinge" axis="0 1 0" damping="0.1"/>
        <inertial pos="0 0 0" mass="0.2" diaginertia="0.005 0.005 0.005"/>
        <geom type="cylinder" size="0.035 0.015" material="wheel_mat" euler="0 90 0"/>
        <geom type="cylinder" size="0.025 0.016" pos="0 0 0" rgba="0.3 0.3 0.3 1" euler="0 90 0"/>
      </body>
      
      <body name="rear_right_wheel" pos="-0.1 -0.08 -0.04">
        <joint name="rear_right_wheel_joint" type="hinge" axis="0 1 0" damping="0.1"/>
        <inertial pos="0 0 0" mass="0.2" diaginertia="0.005 0.005 0.005"/>
        <geom type="cylinder" size="0.035 0.015" material="wheel_mat" euler="0 90 0"/>
        <geom type="cylinder" size="0.025 0.016" pos="0 0 0" rgba="0.3 0.3 0.3 1" euler="0 90 0"/>
      </body>
      
      <body name="front_left_wheel" pos="0.1 0.08 -0.04">
        <joint name="front_left_wheel_joint" type="hinge" axis="0 1 0" damping="0.1"/>
        <inertial pos="0 0 0" mass="0.2" diaginertia="0.005 0.005 0.005"/>
        <geom type="cylinder" size="0.035 0.015" material="wheel_mat" euler="0 90 0"/>
        <geom type="cylinder" size="0.025 0.016" pos="0 0 0" rgba="0.3 0.3 0.3 1" euler="0 90 0"/>
      </body>
      
      <body name="front_right_wheel" pos="0.1 -0.08 -0.04">
        <joint name="front_right_wheel_joint" type="hinge" axis="0 1 0" damping="0.1"/>
        <inertial pos="0 0 0" mass="0.2" diaginertia="0.005 0.005 0.005"/>
        <geom type="cylinder" size="0.035 0.015" material="wheel_mat" euler="0 90 0"/>
        <geom type="cylinder" size="0.025 0.016" pos="0 0 0" rgba="0.3 0.3 0.3 1" euler="0 90 0"/>
      </body>
    </body>
    
    <!-- Enhanced Kitchen (Goal) -->
    <body name="kitchen" pos="5 5 0.5">
      <geom type="box" size="1 1 1" material="goal_mat"/>
      <geom name="kitchen_sign" type="box" size="1.1 1.1 0.05" pos="0 0 1.05" rgba="1 1 1 0.9"/>
      <geom name="kitchen_roof" type="box" size="1.2 1.2 0.1" pos="0 0 1.2" rgba="0.8 0.4 0.2 1"/>
    </body>
    
    <!-- Enhanced Red Obstacle -->
    <body name="red_ob" pos="2 2 0.25">
      <geom type="box" size="0.5 0.5 0.5" material="obstacle_mat"/>
      <geom name="obstacle_top" type="cylinder" size="0.3 0.05" pos="0 0 0.55" rgba="1 0.5 0 1"/>
    </body>
    
    <!-- Additional environmental elements -->
    <body name="tree1" pos="7 1 0.5">
      <geom type="cylinder" size="0.1 0.5" rgba="0.4 0.2 0.1 1"/>
      <geom type="sphere" size="0.3" pos="0 0 0.7" rgba="0.2 0.6 0.2 1"/>
    </body>
    
    <body name="tree2" pos="1 7 0.5">
      <geom type="cylinder" size="0.1 0.5" rgba="0.4 0.2 0.1 1"/>
      <geom type="sphere" size="0.3" pos="0 0 0.7" rgba="0.2 0.6 0.2 1"/>
    </body>
    
    <!-- Path markers to show car's intended route -->
    <body name="waypoint1" pos="1 1 0.02">
      <geom type="cylinder" size="0.05 0.01" rgba="1 1 0 0.6"/>
    </body>
    <body name="waypoint2" pos="2 2 0.02">
      <geom type="cylinder" size="0.05 0.01" rgba="1 1 0 0.6"/>
    </body>
    <body name="waypoint3" pos="3 3 0.02">
      <geom type="cylinder" size="0.05 0.01" rgba="1 1 0 0.6"/>
    </body>
    <body name="waypoint4" pos="4 4 0.02">
      <geom type="cylinder" size="0.05 0.01" rgba="1 1 0 0.6"/>
    </body>
    
    <!-- Simple LiDAR visualization elements -->
    <body name="lidar_indicator_front" pos="0 0 0.25">
      <geom type="sphere" size="0.02" rgba="0 1 0 0.8"/>
    </body>
    <body name="lidar_indicator_back" pos="0 0 0.25">
      <geom type="sphere" size="0.02" rgba="1 0 0 0.8"/>
    </body>
    <body name="lidar_indicator_left" pos="0 0 0.25">
      <geom type="sphere" size="0.02" rgba="1 1 0 0.8"/>
    </body>
    <body name="lidar_indicator_right" pos="0 0 0.25">
      <geom type="sphere" size="0.02" rgba="0 0 1 0.8"/>
    </body>
  </worldbody>
  
  <actuator>
    <motor joint="car_x" gear="20"/>
    <motor joint="car_theta" gear="10"/>
  </actuator>
</mujoco>
"""

# Load the model
model = mujoco.MjModel.from_xml_string(mjcf_xml)
data = mujoco.MjData(model)

# Get IDs for bodies
car_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "car")
kitchen_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "kitchen")
red_ob_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_ob")

# Get LiDAR indicator body IDs for easier access
lidar_indicators = {
    "front": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lidar_indicator_front"),
    "left": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lidar_indicator_left"),  
    "back": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lidar_indicator_back"),
    "right": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lidar_indicator_right")
}

# Function to update visual LiDAR indicators in the scene
def update_visual_lidar(data, car_pos, car_quat, ranges):
    """Update the visual LiDAR indicators around the car"""
    # Compute car yaw
    yaw = np.arctan2(2 * (car_quat[0] * car_quat[3] + car_quat[1] * car_quat[2]), 
                     1 - 2 * (car_quat[2]**2 + car_quat[3]**2))
    
    lidar_height = 0.15
    
    # Get average ranges for 4 main directions
    front_range = min(ranges[0], ranges[1], ranges[11])  # Front sector
    left_range = min(ranges[2], ranges[3], ranges[4])    # Left sector
    back_range = min(ranges[5], ranges[6], ranges[7])    # Back sector  
    right_range = min(ranges[8], ranges[9], ranges[10])  # Right sector
    
    directions = [
        ("lidar_indicator_front", front_range, 0.3),      # Front
        ("lidar_indicator_left", left_range, np.pi/2),    # Left  
        ("lidar_indicator_back", back_range, np.pi),      # Back
        ("lidar_indicator_right", right_range, -np.pi/2)  # Right
    ]
    
    for name, range_dist, relative_angle in directions:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            # Position indicator based on car position and detected range
            angle = yaw + relative_angle
            indicator_dist = min(range_dist * 0.8, 1.0)  # Scale down for visibility
            
            data.body(body_id).xpos[0] = car_pos[0] + indicator_dist * np.cos(angle)
            data.body(body_id).xpos[1] = car_pos[1] + indicator_dist * np.sin(angle)  
            data.body(body_id).xpos[2] = car_pos[2] + lidar_height

# Function to print LiDAR visualization in terminal
def print_lidar_visualization(ranges, car_pos, step_count):
    """Print a simple ASCII LiDAR visualization"""
    if step_count % 60 == 0:  # Print every 60 steps (1 second at 60 FPS)
        print("\n" + "="*50)
        print(f"🚗 CAR POSITION: ({car_pos[0]:.1f}, {car_pos[1]:.1f})")
        print("📡 LIDAR SCAN:")
        
        # Create a simple ASCII representation
        directions = ['F', 'FL', 'L', 'RL', 'R', 'RR', 'R', 'FR', 'F', 'FL', 'L', 'RL']
        for i, (direction, distance) in enumerate(zip(directions, ranges)):
            bars = int(distance * 3)  # Scale for display
            bar_str = "█" * bars + "░" * (9 - bars)
            status = "🔴" if distance < 1.0 else "🟡" if distance < 2.0 else "🟢"
            print(f"  {direction:2s}: {status} {bar_str} {distance:.1f}m")
        print("="*50)

# Simplified LiDAR function using distance calculations
def lidar_fake():
    car_pos = data.body(car_id).xpos
    car_quat = data.body(car_id).xquat
    
    # Compute yaw from quaternion (w, x, y, z format)
    yaw = np.arctan2(2 * (car_quat[0] * car_quat[3] + car_quat[1] * car_quat[2]), 
                     1 - 2 * (car_quat[2]**2 + car_quat[3]**2))
    
    ranges = []
    max_range = 3.0
    
    # Get obstacle positions
    kitchen_pos = data.body(kitchen_id).xpos
    red_ob_pos = data.body(red_ob_id).xpos
    
    for i in range(12):
        angle = yaw + i * np.pi / 6  # 30-degree increments
        ray_dir = np.array([np.cos(angle), np.sin(angle)])
        
        min_dist = max_range
        
        # Check distance to obstacles
        for obs_pos, obs_size in [(kitchen_pos[:2], 1.0), (red_ob_pos[:2], 0.5)]:
            # Vector from car to obstacle
            to_obs = obs_pos - car_pos[:2]
            
            # Project onto ray direction
            proj_dist = np.dot(to_obs, ray_dir)
            
            if proj_dist > 0:  # Obstacle is in front of ray
                # Perpendicular distance to ray
                perp_dist = np.linalg.norm(to_obs - proj_dist * ray_dir)
                
                if perp_dist < obs_size:  # Ray intersects obstacle
                    # Approximate intersection distance
                    intersect_dist = proj_dist - np.sqrt(obs_size**2 - perp_dist**2)
                    if intersect_dist > 0:
                        min_dist = min(min_dist, intersect_dist)
        
        ranges.append(min_dist)
    
    return ranges

# Convert ranges to text
def ranges2text(ranges):
    dirs = ['front', 'front-left', 'left', 'rear-left', 'rear', 'rear-right',
            'right', 'front-right', 'front', 'front-left', 'left', 'rear-left']  # 12 directions
    text = ", ".join([f"{d} {r:.1f}m" for d, r in zip(dirs, ranges)])
    return "Current LiDAR: " + text

# LLM drive function using requests
def llm_drive(prompt: str):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system",
             "content": (
                "You are a robot car navigating to a goal. "
                "Reply ONLY JSON: {\"v\":float, \"w\":float} "
                "v=forward/backward speed [-1,1] m/s, w=rotation speed [-1,1] rad/s. "
                "GOAL: Navigate to kitchen at (5,5). Use the goal direction angle to orient yourself. "
                "OBSTACLE AVOIDANCE: Avoid red obstacle at (2,2) using LiDAR readings. "
                "STRATEGY: If obstacles detected (distance < 2m), turn away. Otherwise, turn towards goal direction."
             )},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "stream": False
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"API request failed with status {response.status_code}: {response.text}")
    return json.loads(response.json()["choices"][0]["message"]["content"])

# Main simulation loop with enhanced GUI
with mujoco.viewer.launch_passive(model, data) as viewer:
    step_count = 0
    cmd = {'v': 0.0, 'w': 0.0}  # Initial command
    
    # Set better camera view
    viewer.cam.azimuth = 45
    viewer.cam.elevation = -30
    viewer.cam.distance = 8
    viewer.cam.lookat = np.array([2.5, 2.5, 0])
    
    while viewer.is_running():
        # Get LiDAR data
        try:
            ranges = lidar_fake()
        except Exception as e:
            print(f"LiDAR error: {e}")
            ranges = [3.0] * 12  # Default max range

        # Update visual LiDAR rays and terminal display
        car_pos_3d = data.body(car_id).xpos
        car_quat = data.body(car_id).xquat
        update_visual_lidar(data, car_pos_3d, car_quat, ranges)
        print_lidar_visualization(ranges, car_pos_3d[:2], step_count)

        text = ranges2text(ranges)
        car_pos = car_pos_3d[:2]
        # Calculate direction to goal
        goal_vec = np.array([5.0, 5.0]) - car_pos
        goal_angle = np.arctan2(goal_vec[1], goal_vec[0]) * 180 / np.pi
        goal_distance = np.linalg.norm(goal_vec)
        
        prompt = (text + f"\nCurrent position: ({car_pos[0]:.1f}, {car_pos[1]:.1f}). "
                 f"Goal position (kitchen): (5.0, 5.0). "
                 f"Goal direction: {goal_angle:.0f} degrees, distance: {goal_distance:.1f}m. "
                 f"You need to turn towards the goal and avoid the red obstacle at (2,2).")
        
        # Only call LLM every 30 steps to avoid API spam and improve stability
        if step_count % 30 == 0:
            try:
                cmd = llm_drive(prompt)
                print(f"LLM output: {cmd}")
            except Exception as e:
                print(f"LLM error: {e}")
                cmd = {'v': 0.0, 'w': 0.0}

        # Apply linear and angular velocity directly to car body
        linear_vel = cmd['v']
        angular_vel = cmd['w']
        
        # Apply to actuators (car body motion)
        data.ctrl[0] = linear_vel * 10.0  # Forward/backward force
        data.ctrl[1] = angular_vel * 5.0   # Rotation torque

        # Step simulation
        mujoco.mj_step(model, data)

        # Check for obstacle warnings
        min_range = min(ranges)
        if min_range < 1.5 and step_count % 30 == 0:
            print("⚠️  OBSTACLE DETECTED! Distance: {:.1f}m".format(min_range))
        
        # Check distance to kitchen
        kitchen_pos = data.body(kitchen_id).xpos[:2]
        dist = np.linalg.norm(car_pos - kitchen_pos)
        
        if step_count % 30 == 0:
            print(f"🚗 Pos: ({car_pos[0]:.1f}, {car_pos[1]:.1f}) | 🎯 Goal: {dist:.1f}m | 📡 Min LiDAR: {min_range:.1f}m | ⚡ v={linear_vel:.2f}, w={angular_vel:.2f}")
        if dist < 1.0:
            print("\n" + "🎉" * 20)
            print("🎉 MISSION COMPLETE! 🎉")
            print("🚗 Car successfully reached the kitchen! 🍳")
            print("📊 Final Stats:")
            print(f"   • Final position: ({car_pos[0]:.1f}, {car_pos[1]:.1f})")
            print(f"   • Total steps: {step_count}")
            print(f"   • Distance to goal: {dist:.2f}m")
            print("🎉" * 20 + "\n")
            data.ctrl[:] = 0
            break

        # Sync viewer and sleep for real-time
        viewer.sync()
        time.sleep(1.0 / 60)  # 60 FPS for smooth visualization
        step_count += 1