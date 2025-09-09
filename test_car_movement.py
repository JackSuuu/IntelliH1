import mujoco
import mujoco.viewer
import numpy as np
import time

# Simple test MJCF for a moving car
mjcf_xml = """
<mujoco model="test_car">
  <compiler angle="degree"/>
  <option gravity="0 0 -9.81" timestep="0.002" integrator="Euler"/>
  
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="0" condim="3" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
    
    <!-- Simple car with slide joints -->
    <body name="car" pos="0 0 0.1">
      <joint name="car_x" type="slide" axis="1 0 0" damping="0.1"/>
      <joint name="car_y" type="slide" axis="0 1 0" damping="0.1"/>
      <joint name="car_theta" type="hinge" axis="0 0 1" damping="0.1"/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
      <geom type="box" size="0.15 0.08 0.04" rgba="0.8 0.6 0.4 1"/>
    </body>
    
    <!-- Goal marker -->
    <body name="goal" pos="3 3 0.5">
      <geom type="box" size="0.5 0.5 0.5" rgba="0 1 0 1"/>
    </body>
  </worldbody>
  
  <!-- Direct force actuators on car body -->
  <actuator>
    <motor joint="car_x" gear="10"/>
    <motor joint="car_theta" gear="5"/>
  </actuator>
</mujoco>
"""

# Load model
model = mujoco.MjModel.from_xml_string(mjcf_xml)
data = mujoco.MjData(model)

# Get car body ID
car_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "car")

print(f"Number of actuators: {model.nu}")
print(f"Actuator names: {[model.actuator(i).name for i in range(model.nu)]}")

# Test simulation
with mujoco.viewer.launch_passive(model, data) as viewer:
    step_count = 0
    while viewer.is_running() and step_count < 1000:
        # Apply constant forward force and slight rotation
        data.ctrl[0] = 1.0  # Forward force
        data.ctrl[1] = 0.1  # Slight rotation
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Print position every 100 steps
        if step_count % 100 == 0:
            car_pos = data.body(car_id).xpos
            print(f"Step {step_count}: Car position = ({car_pos[0]:.3f}, {car_pos[1]:.3f}, {car_pos[2]:.3f})")
        
        viewer.sync()
        time.sleep(1.0 / 60)
        step_count += 1

print("Test complete!")
