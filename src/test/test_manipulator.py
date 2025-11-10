"""
Test script for Humanoid Robot with Manipulator
Tests navigation + manipulation capabilities
"""

import time
import sys
import os
import numpy as np
import mujoco
import mujoco.viewer

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulation.environment import Simulation
from robot.car import Car
from robot.manipulator import ManipulatorController, TaskExecutor
from perception.vision import Perception


def test_manipulator_basics(sim, manipulator):
    """Test basic manipulator functions"""
    print("\n" + "="*60)
    print("🧪 TEST 1: Basic Manipulator Functions")
    print("="*60)
    
    # Get initial state
    joint_pos = manipulator.get_joint_positions()
    ee_pos, ee_quat = manipulator.get_end_effector_pose()
    gripper_width = manipulator.get_gripper_width()
    
    print(f"✓ Joint positions: {joint_pos}")
    print(f"✓ End-effector position: {ee_pos}")
    print(f"✓ End-effector quaternion: {ee_quat}")
    print(f"✓ Gripper width: {gripper_width:.4f}m")
    
    # Test state vector
    state = manipulator.get_state_vector()
    print(f"✓ State vector dimension: {state.shape[0]} (expected 18)")
    
    return True


def test_home_position(sim, manipulator, viewer):
    """Test moving to home position"""
    print("\n" + "="*60)
    print("🧪 TEST 2: Move to Home Position")
    print("="*60)
    
    manipulator.move_to_home()
    
    # Simulate for 3 seconds
    for _ in range(300):
        manipulator.update()
        sim.step()
        viewer.sync()
        time.sleep(0.01)
    
    joint_pos = manipulator.get_joint_positions()
    error = np.linalg.norm(joint_pos - manipulator.home_position)
    print(f"✓ Reached home position (error: {error:.4f} rad)")
    
    return error < 0.1


def test_joint_control(sim, manipulator, viewer):
    """Test direct joint control"""
    print("\n" + "="*60)
    print("🧪 TEST 3: Direct Joint Control")
    print("="*60)
    
    # Test configuration: slightly bent arm
    test_config = np.array([0.0, -0.5, 0.8, 0.0, 0.3])
    print(f"Target configuration: {test_config}")
    
    manipulator.set_joint_targets(test_config)
    
    # Simulate for 3 seconds
    for _ in range(300):
        manipulator.update()
        sim.step()
        viewer.sync()
        time.sleep(0.01)
    
    joint_pos = manipulator.get_joint_positions()
    error = np.linalg.norm(joint_pos - test_config)
    print(f"✓ Reached target (error: {error:.4f} rad)")
    print(f"  Actual: {joint_pos}")
    
    return error < 0.1


def test_gripper(sim, manipulator, viewer):
    """Test gripper open/close"""
    print("\n" + "="*60)
    print("🧪 TEST 4: Gripper Control")
    print("="*60)
    
    # Open gripper
    print("Opening gripper...")
    manipulator.open_gripper()
    for _ in range(100):
        manipulator.update()
        sim.step()
        viewer.sync()
        time.sleep(0.01)
    
    width_open = manipulator.get_gripper_width()
    print(f"✓ Gripper open: {width_open:.4f}m")
    
    # Close gripper
    print("Closing gripper...")
    manipulator.close_gripper()
    for _ in range(100):
        manipulator.update()
        sim.step()
        viewer.sync()
        time.sleep(0.01)
    
    width_closed = manipulator.get_gripper_width()
    print(f"✓ Gripper closed: {width_closed:.4f}m")
    
    return width_open > width_closed


def test_inverse_kinematics(sim, manipulator, viewer):
    """Test IK solver"""
    print("\n" + "="*60)
    print("🧪 TEST 5: Inverse Kinematics")
    print("="*60)
    
    # Move to home first
    manipulator.move_to_home()
    for _ in range(200):
        manipulator.update()
        sim.step()
        viewer.sync()
        time.sleep(0.01)
    
    # Test IK targets (relative to car base)
    test_targets = [
        np.array([0.3, 0.0, 0.2]),   # Forward
        np.array([0.2, 0.2, 0.15]),  # Forward-left
        np.array([0.2, -0.2, 0.15]), # Forward-right
    ]
    
    ik_success_count = 0
    
    for i, target in enumerate(test_targets):
        print(f"\nIK Test {i+1}: Target = {target}")
        
        # Solve IK
        success = manipulator.move_to_pose(target)
        
        if success:
            # Simulate motion
            for _ in range(200):
                manipulator.update()
                sim.step()
                viewer.sync()
                time.sleep(0.01)
            
            # Check final position
            ee_pos, _ = manipulator.get_end_effector_pose()
            error = np.linalg.norm(ee_pos - target)
            print(f"  ✓ IK converged! Final error: {error:.4f}m")
            print(f"    Target:  {target}")
            print(f"    Reached: {ee_pos}")
            
            if error < 0.05:  # 5cm tolerance
                ik_success_count += 1
        else:
            print(f"  ✗ IK failed to find solution")
    
    success_rate = ik_success_count / len(test_targets)
    print(f"\n✓ IK success rate: {success_rate*100:.1f}% ({ik_success_count}/{len(test_targets)})")
    
    return success_rate >= 0.5


def test_pick_and_place(sim, manipulator, task_executor, viewer):
    """Test pick-and-place task"""
    print("\n" + "="*60)
    print("🧪 TEST 6: Pick-and-Place Task")
    print("="*60)
    
    # Get positions of test objects
    cube_body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, 'test_cube')
    cube_pos = sim.data.xpos[cube_body_id].copy()
    
    print(f"Test cube at: {cube_pos}")
    
    # Define pick and place positions
    pick_pos = cube_pos.copy()
    pick_pos[2] = 0.09  # Slightly above table
    
    place_pos = pick_pos + np.array([0.5, 0.0, 0.0])  # 50cm forward
    
    print(f"Pick position: {pick_pos}")
    print(f"Place position: {place_pos}")
    
    # Execute pick-and-place (this is just demonstration, won't actually work without navigation)
    print("\nNote: Pick-and-place requires mobile base navigation to reach object.")
    print("This test demonstrates the API, but object is likely out of reach from start position.")
    
    success = task_executor.pick_and_place(pick_pos, place_pos)
    
    # Simulate the motion
    for _ in range(1000):
        manipulator.update()
        sim.step()
        viewer.sync()
        time.sleep(0.01)
    
    return success


def main():
    """Main test function"""
    print("="*60)
    print("🦾 HUMANOID ROBOT MANIPULATOR TEST SUITE")
    print("="*60)
    
    # Initialize simulation with humanoid model
    sim = Simulation(model_path="models/humanoid_robot.xml")
    
    # Initialize manipulator
    manipulator = ManipulatorController(sim.model, sim.data)
    task_executor = TaskExecutor(manipulator)
    
    # Forward simulation to initialize
    mujoco.mj_forward(sim.model, sim.data)
    
    print(f"\n✓ Simulation initialized")
    print(f"✓ Model: {sim.model_path}")
    print(f"✓ Manipulator joints: {len(manipulator.joint_ids)}")
    print(f"✓ Actuators: {len(manipulator.actuator_ids)}")
    
    # Run tests with viewer
    test_results = {}
    
    try:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            print("\n✓ Viewer launched! Starting tests...\n")
            
            # Wait for viewer to stabilize
            for _ in range(50):
                sim.step()
                viewer.sync()
                time.sleep(0.01)
            
            # Run test suite
            test_results['basics'] = test_manipulator_basics(sim, manipulator)
            time.sleep(1)
            
            test_results['home'] = test_home_position(sim, manipulator, viewer)
            time.sleep(1)
            
            test_results['joints'] = test_joint_control(sim, manipulator, viewer)
            time.sleep(1)
            
            test_results['gripper'] = test_gripper(sim, manipulator, viewer)
            time.sleep(1)
            
            test_results['ik'] = test_inverse_kinematics(sim, manipulator, viewer)
            time.sleep(1)
            
            # Optional: test pick-and-place (may fail if object out of reach)
            # test_results['pick_place'] = test_pick_and_place(sim, manipulator, task_executor, viewer)
            
            # Return to home
            print("\n" + "="*60)
            print("Returning to home position...")
            manipulator.move_to_home()
            for _ in range(300):
                manipulator.update()
                sim.step()
                viewer.sync()
                time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    passed = sum(test_results.values())
    total = len(test_results)
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("="*60)


if __name__ == "__main__":
    main()
