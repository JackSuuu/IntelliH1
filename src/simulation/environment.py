import mujoco
import numpy as np
import os
from config import LIDAR_SAMPLES, LIDAR_FOV, LIDAR_RANGE

def create_car_model():
    """
    Creates a proper differential drive robot model with correct wheel orientation.
    Fixed version with wheels properly aligned for forward motion.
    """
    mjcf_xml = f"""
<mujoco model="text2wheel_car">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option gravity="0 0 -9.81" timestep="0.005" integrator="RK4"/>
  
  <default>
    <joint damping="0.5" armature="0.01"/>
    <geom condim="3" friction="1.5 0.1 0.1" solimp="0.9 0.95 0.001" solref="0.02 1"/>
  </default>
  
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".2 .3 .4" rgb2="0 0 0" width="512" height="512"/>
    <texture builtin="checker" height="512" name="texplane" rgb1="0.2 0.3 0.2" rgb2="0.1 0.2 0.1" type="2d" width="512"/>
    <material name="MatPlane" reflectance="0.3" shininess="1" specular="0.4" texrepeat="20 20" texture="texplane"/>
    <material name="car_body" reflectance="0.7" shininess="0.3" specular="0.5" rgba="0.2 0.5 0.9 1"/>
    <material name="wheel_mat" reflectance="0.1" shininess="0.1" specular="0.1" rgba="0.1 0.1 0.1 1"/>
    <material name="obstacle_mat" reflectance="0.8" shininess="0.6" specular="0.7" rgba="0.8 0.2 0.2 0.9"/>
  </asset>
  
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="0 0 -1" directional="true" exponent="1" pos="0 0 5" specular="0.3 0.3 0.3"/>
    <geom conaffinity="0" condim="3" material="MatPlane" name="floor" pos="0 0 0" size="40 40 0.1" type="plane"/>
    
    <!-- Differential Drive Robot Car - Low center of gravity, stable design -->
    <body name="car" pos="0 0 0.06">
      <freejoint name="car_joint"/>
      <!-- Lower and heavier body for stability -->
      <inertial pos="0 0 -0.01" mass="8" diaginertia="0.15 0.15 0.08"/>
      
      <!-- Main body - lower profile -->
      <geom type="box" size="0.2 0.12 0.03" pos="0 0 0" material="car_body"/>
      
      <!-- Front bumper for direction indication -->
      <geom type="box" size="0.05 0.1 0.015" pos="0.22 0 -0.005" rgba="1 0.3 0.3 1"/>
      
      <!-- Wider wheelbase for stability -->
      <!-- Left wheel (rear) -->
      <body name="rear_left_wheel" pos="-0.1 0.15 -0.01">
        <joint name="rear_left_wheel_joint" type="hinge" axis="0 1 0" damping="0.5"/>
        <inertial pos="0 0 0" mass="0.6" diaginertia="0.003 0.003 0.003"/>
        <geom type="cylinder" size="0.045 0.025" pos="0 0 0" material="wheel_mat" friction="2.5 0.01 0.001"/>
      </body>
      
      <!-- Right wheel (rear) -->
      <body name="rear_right_wheel" pos="-0.1 -0.15 -0.01">
        <joint name="rear_right_wheel_joint" type="hinge" axis="0 1 0" damping="0.5"/>
        <inertial pos="0 0 0" mass="0.6" diaginertia="0.003 0.003 0.003"/>
        <geom type="cylinder" size="0.045 0.025" pos="0 0 0" material="wheel_mat" friction="2.5 0.01 0.001"/>
      </body>
      
      <!-- Front left wheel (powered) -->
      <body name="front_left_wheel" pos="0.1 0.15 -0.01">
        <joint name="front_left_wheel_joint" type="hinge" axis="0 1 0" damping="0.5"/>
        <inertial pos="0 0 0" mass="0.6" diaginertia="0.003 0.003 0.003"/>
        <geom type="cylinder" size="0.045 0.025" pos="0 0 0" material="wheel_mat" friction="2.5 0.01 0.001"/>
      </body>
      
      <!-- Front right wheel (powered) -->
      <body name="front_right_wheel" pos="0.1 -0.15 -0.01">
        <joint name="front_right_wheel_joint" type="hinge" axis="0 1 0" damping="0.5"/>
        <inertial pos="0 0 0" mass="0.6" diaginertia="0.003 0.003 0.003"/>
        <geom type="cylinder" size="0.045 0.025" pos="0 0 0" material="wheel_mat" friction="2.5 0.01 0.001"/>
      </body>
      
      <!-- Multiple stabilizer wheels to prevent tipping -->
      <geom type="sphere" size="0.02" pos="0.15 0 -0.025" rgba="0.3 0.3 0.3 1" friction="0.5 0.01 0.001"/>
      <geom type="sphere" size="0.02" pos="-0.15 0 -0.025" rgba="0.3 0.3 0.3 1" friction="0.5 0.01 0.001"/>
    </body>

    <!-- Obstacles -->
    <body name="obstacle1" pos="5 2 0.25">
      <geom type="box" size="0.5 0.5 0.25" material="obstacle_mat"/>
      <inertial pos="0 0 0" mass="100" diaginertia="1 1 1"/>
    </body>
    <body name="obstacle2" pos="3 -3 0.3">
      <geom type="cylinder" size="0.7 0.3" material="obstacle_mat"/>
      <inertial pos="0 0 0" mass="100" diaginertia="1 1 1"/>
    </body>
    <body name="obstacle3" pos="-4 5 0.25">
      <geom type="box" size="1 2 0.25" material="obstacle_mat"/>
      <inertial pos="0 0 0" mass="100" diaginertia="1 1 1"/>
    </body>
    
    <!-- Goal markers for different rooms -->
    <body name="kitchen" pos="8 8 0.3">
      <geom type="sphere" size="0.4" rgba="1.0 0.5 0.0 0.8" contype="0" conaffinity="0"/>
      <site name="kitchen_site" pos="0 0 0.6" size="0.1" rgba="1.0 0.5 0.0 1.0"/>
    </body>
    
    <body name="bedroom" pos="-6 6 0.3">
      <geom type="sphere" size="0.4" rgba="0.5 0.5 1.0 0.8" contype="0" conaffinity="0"/>
      <site name="bedroom_site" pos="0 0 0.6" size="0.1" rgba="0.5 0.5 1.0 1.0"/>
    </body>
    
    <body name="living_room" pos="6 -6 0.3">
      <geom type="sphere" size="0.4" rgba="0.0 0.8 0.5 0.8" contype="0" conaffinity="0"/>
      <site name="living_site" pos="0 0 0.6" size="0.1" rgba="0.0 0.8 0.5 1.0"/>
    </body>
  </worldbody>

  <actuator>
    <!-- Differential drive: Only REAR wheels are powered -->
    <motor name="rear_left_motor" joint="rear_left_wheel_joint" gear="200" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="rear_right_motor" joint="rear_right_wheel_joint" gear="200" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
    """
    return mjcf_xml

class Simulation:
    """
    Manages the MuJoCo simulation environment.
    """
    def __init__(self, model_path=None):
        """
        Initialize simulation.
        
        Args:
            model_path: Path to MJCF XML file (optional). If None, uses Unitree H1 model.
        """
        if model_path is None:
            # Use default Unitree H1 model
            model_path = "models/unitree_h1/scene.xml"
        
        # Load model from file
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        
        # Define locations within the simulation environment
        self.locations = {
            "Kitchen": (5.0, 3.0),
            "Bedroom": (-3.0, 6.0),
            "Living Room": (0.0, -4.0),
        }

        print(f"[Simulation] Loaded model: {model_path}")

    def launch_viewer(self):
        """Launches the passive viewer."""
        import mujoco.viewer
        # Use launch instead of launch_passive for macOS compatibility
        self.viewer = mujoco.viewer.launch(self.model, self.data)
        return self.viewer

    def step(self):
        """Advances the simulation by one step."""
        mujoco.mj_step(self.model, self.data)

    def sync_viewer(self):
        """Synchronizes the viewer with the simulation data."""
        if self.viewer and self.viewer.is_running():
            self.viewer.sync()

    def is_running(self):
        """Checks if the viewer is still running."""
        return self.viewer and self.viewer.is_running()

    def close(self):
        """Closes the viewer."""
        if self.viewer:
            self.viewer.close()
