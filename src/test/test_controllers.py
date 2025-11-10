#!/usr/bin/env python3
"""
Unit tests for standing controllers

Tests basic functionality without full simulation runs.
"""

import sys
import os
import numpy as np
import mujoco

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from control.pd_standing import PDStandingController, ImprovedPDController, PIDStandingController
from control.zmp_balance import ZMPBalanceController
from simulation.environment import Simulation


def test_controller_initialization():
    """Test that all controllers can be initialized"""
    print("Testing controller initialization...")
    
    # Create simulation
    sim = Simulation(model_path="models/unitree_h1/scene_enhanced.xml")
    
    # Test PD controller
    pd = PDStandingController(sim.model, sim.data)
    assert pd.nu == sim.model.nu
    assert pd.kp.shape[0] == sim.model.nu
    assert pd.kd.shape[0] == sim.model.nu
    print("  ✅ PDStandingController initialized")
    
    # Test Improved PD controller
    improved_pd = ImprovedPDController(sim.model, sim.data)
    assert improved_pd.nu == sim.model.nu
    print("  ✅ ImprovedPDController initialized")
    
    # Test PID controller
    pid = PIDStandingController(sim.model, sim.data, ki_factor=0.1)
    assert pid.nu == sim.model.nu
    assert pid.ki.shape[0] == sim.model.nu
    assert pid.error_integral.shape[0] == sim.model.nu
    print("  ✅ PIDStandingController initialized")
    
    # Test ZMP controller
    zmp = ZMPBalanceController(sim.model, sim.data)
    assert zmp.nu == sim.model.nu
    print("  ✅ ZMPBalanceController initialized")
    
    print("✅ All controllers initialized successfully\n")


def test_controller_compute_control():
    """Test that controllers can compute control torques"""
    print("Testing control computation...")
    
    # Create simulation
    sim = Simulation(model_path="models/unitree_h1/scene_enhanced.xml")
    mujoco.mj_forward(sim.model, sim.data)
    
    # Test each controller
    controllers = [
        (PDStandingController(sim.model, sim.data), "PD"),
        (ImprovedPDController(sim.model, sim.data), "ImprovedPD"),
        (PIDStandingController(sim.model, sim.data), "PID"),
        (ZMPBalanceController(sim.model, sim.data), "ZMP"),
    ]
    
    for controller, name in controllers:
        tau = controller.compute_control(dt=0.002)
        
        # Check output shape
        assert tau.shape[0] == sim.model.nu, f"{name}: Wrong output shape"
        
        # Check values are finite
        assert np.all(np.isfinite(tau)), f"{name}: Non-finite values in control output"
        
        # Check torques are within reasonable bounds (not all zero, not excessive)
        assert np.any(np.abs(tau) > 0.01), f"{name}: Control output is all near-zero"
        assert np.all(np.abs(tau) < 1000), f"{name}: Control output has excessive values"
        
        print(f"  ✅ {name}: compute_control() produces valid output")
    
    print("✅ All controllers compute control successfully\n")


def test_pid_integral_reset():
    """Test PID controller integral reset"""
    print("Testing PID integral reset...")
    
    sim = Simulation(model_path="models/unitree_h1/scene_enhanced.xml")
    mujoco.mj_forward(sim.model, sim.data)
    
    pid = PIDStandingController(sim.model, sim.data)
    
    # Compute control a few times to accumulate integral
    for _ in range(10):
        pid.compute_control(dt=0.002)
    
    # Check integral is non-zero
    assert np.any(np.abs(pid.error_integral) > 0.001), "Integral should accumulate"
    
    # Reset
    pid.reset_integral()
    
    # Check integral is zero
    assert np.allclose(pid.error_integral, 0), "Integral should be zero after reset"
    
    print("  ✅ PID integral resets correctly")
    print("✅ PID integral functionality works\n")


def test_zmp_calculation():
    """Test ZMP calculation returns proper format"""
    print("Testing ZMP calculation...")
    
    sim = Simulation(model_path="models/unitree_h1/scene_enhanced.xml")
    mujoco.mj_forward(sim.model, sim.data)
    
    zmp = ZMPBalanceController(sim.model, sim.data)
    
    # Calculate ZMP
    zmp_error, support_center, zmp_pos = zmp._calculate_zmp()
    
    # Check shapes
    assert zmp_error.shape == (2,), "ZMP error should be 2D"
    assert support_center.shape == (3,), "Support center should be 3D"
    assert zmp_pos.shape == (2,), "ZMP position should be 2D"
    
    # Check values are finite
    assert np.all(np.isfinite(zmp_error)), "ZMP error should be finite"
    assert np.all(np.isfinite(support_center)), "Support center should be finite"
    assert np.all(np.isfinite(zmp_pos)), "ZMP position should be finite"
    
    print("  ✅ ZMP calculation returns proper format")
    print("✅ ZMP calculation works\n")


def test_zmp_support_polygon():
    """Test support polygon checking"""
    print("Testing support polygon checking...")
    
    sim = Simulation(model_path="models/unitree_h1/scene_enhanced.xml")
    mujoco.mj_forward(sim.model, sim.data)
    
    zmp = ZMPBalanceController(sim.model, sim.data)
    
    # Get foot positions
    left_foot_pos = sim.data.body('left_ankle_link').xpos.copy()
    right_foot_pos = sim.data.body('right_ankle_link').xpos.copy()
    
    # Test with ZMP at center (should be safe)
    center_pos = np.array([
        (left_foot_pos[0] + right_foot_pos[0]) / 2,
        (left_foot_pos[1] + right_foot_pos[1]) / 2
    ])
    is_safe, penetration = zmp._is_zmp_in_support_polygon(center_pos, left_foot_pos, right_foot_pos)
    assert is_safe, "ZMP at center should be safe"
    assert np.allclose(penetration, 0), "Penetration should be zero when safe"
    print("  ✅ ZMP at center is correctly identified as safe")
    
    # Test with ZMP far forward (should be unsafe)
    far_forward = np.array([
        center_pos[0] + 1.0,  # 1m forward
        center_pos[1]
    ])
    is_safe, penetration = zmp._is_zmp_in_support_polygon(far_forward, left_foot_pos, right_foot_pos)
    assert not is_safe, "ZMP far forward should be unsafe"
    assert penetration[0] > 0, "Should have positive X penetration"
    print("  ✅ ZMP outside polygon is correctly identified as unsafe")
    
    print("✅ Support polygon checking works\n")


def test_pid_anti_windup():
    """Test that PID integral has anti-windup"""
    print("Testing PID anti-windup...")
    
    sim = Simulation(model_path="models/unitree_h1/scene_enhanced.xml")
    mujoco.mj_forward(sim.model, sim.data)
    
    pid = PIDStandingController(sim.model, sim.data)
    
    # Force large error by setting extreme target
    pid.q_target = pid.q_target + 10.0  # Unrealistic target
    
    # Run many iterations to try to saturate integral
    for _ in range(1000):
        pid.compute_control(dt=0.002)
    
    # Check integral is clamped
    assert np.all(np.abs(pid.error_integral) <= pid.integral_limit), \
        "Integral should be clamped by limits"
    
    print("  ✅ PID integral is properly clamped")
    print("✅ PID anti-windup works\n")


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("RUNNING CONTROLLER UNIT TESTS")
    print("="*60)
    print()
    
    tests = [
        test_controller_initialization,
        test_controller_compute_control,
        test_pid_integral_reset,
        test_zmp_calculation,
        test_zmp_support_polygon,
        test_pid_anti_windup,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ Test failed: {test.__name__}")
            print(f"   Error: {e}")
            print()
            failed += 1
    
    print("="*60)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
