#!/usr/bin/env python3
"""
Stability Test Script for Unitree H1 Standing Controller

This script runs automated stability tests to verify that the robot can
stand for 60+ seconds without falling, meeting the acceptance criteria.

Usage:
    python test_stability.py --controller pd --duration 60
    python test_stability.py --controller pid --duration 120
    python test_stability.py --all  # Test all controllers
"""

import numpy as np
import mujoco
import argparse
import sys
import os
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulation.environment import Simulation
from control.pd_standing import PDStandingController, ImprovedPDController, PIDStandingController
from control.zmp_balance import ZMPBalanceController


class StabilityTest:
    """Automated stability testing for standing controllers"""
    
    def __init__(self, duration=60.0, dt=0.002):
        """
        Args:
            duration: Required standing duration in seconds
            dt: Simulation timestep
        """
        self.duration = duration
        self.dt = dt
        self.max_steps = int(duration / dt)
        
    def run_test(self, controller, controller_name):
        """
        Run stability test for a given controller.
        
        Args:
            controller: Controller instance
            controller_name: Name for display
            
        Returns:
            dict: Test results
        """
        print(f"\n{'='*60}")
        print(f"Testing: {controller_name}")
        print(f"{'='*60}")
        
        # Create simulation
        sim = Simulation(model_path="models/unitree_h1/scene_enhanced.xml")
        
        # Recreate controller with sim's model and data
        if isinstance(controller, PDStandingController):
            controller = PDStandingController(sim.model, sim.data)
        elif isinstance(controller, ImprovedPDController):
            controller = ImprovedPDController(sim.model, sim.data)
        elif isinstance(controller, PIDStandingController):
            controller = PIDStandingController(sim.model, sim.data)
        elif isinstance(controller, ZMPBalanceController):
            controller = ZMPBalanceController(sim.model, sim.data)
        
        # Initialize
        mujoco.mj_forward(sim.model, sim.data)
        
        # Track metrics
        com_positions = []
        heights = []
        fell = False
        fall_time = None
        
        start_real_time = time.time()
        last_print = start_real_time
        
        for step in range(self.max_steps):
            sim_time = step * self.dt
            
            # Compute control
            tau = controller.compute_control(dt=self.dt)
            sim.data.ctrl[:] = tau
            
            # Step simulation
            sim.step()
            
            # Record data
            base_pos = sim.data.qpos[0:3].copy()
            com_positions.append(base_pos)
            heights.append(base_pos[2])
            
            # Check for fall
            if base_pos[2] < 0.5:  # Height threshold
                fell = True
                fall_time = sim_time
                break
            
            # Progress indicator (every 2 seconds real time)
            current_real_time = time.time()
            if current_real_time - last_print > 2.0:
                print(f"  Progress: {sim_time:.1f}s / {self.duration:.1f}s "
                      f"| Height: {base_pos[2]:.3f}m | Status: ✅ Standing")
                last_print = current_real_time
        
        elapsed_real_time = time.time() - start_real_time
        
        # Calculate metrics
        com_positions = np.array(com_positions)
        heights = np.array(heights)
        
        com_drift_x = abs(com_positions[-1, 0] - com_positions[0, 0])
        com_drift_y = abs(com_positions[-1, 1] - com_positions[0, 1])
        com_drift_total = np.sqrt(com_drift_x**2 + com_drift_y**2)
        com_var_x = np.std(com_positions[:, 0])
        com_var_y = np.std(com_positions[:, 1])
        height_mean = np.mean(heights)
        height_std = np.std(heights)
        
        # Determine test result
        passed = not fell and com_var_x < 0.05 and com_var_y < 0.05
        
        results = {
            'controller': controller_name,
            'passed': passed,
            'fell': fell,
            'fall_time': fall_time,
            'duration': self.duration if not fell else fall_time,
            'com_drift_x': com_drift_x,
            'com_drift_y': com_drift_y,
            'com_drift_total': com_drift_total,
            'com_var_x': com_var_x,
            'com_var_y': com_var_y,
            'height_mean': height_mean,
            'height_std': height_std,
            'elapsed_real_time': elapsed_real_time
        }
        
        # Print results
        print(f"\n{'='*60}")
        print(f"RESULTS: {controller_name}")
        print(f"{'='*60}")
        
        if fell:
            print(f"Status: ❌ FAILED (fell at {fall_time:.2f}s)")
        else:
            print(f"Status: ✅ PASSED (stood for {self.duration:.1f}s)")
        
        print(f"\nMetrics:")
        print(f"  CoM drift X: {com_drift_x:.4f}m")
        print(f"  CoM drift Y: {com_drift_y:.4f}m")
        print(f"  CoM drift total: {com_drift_total:.4f}m")
        print(f"  CoM variation X: {com_var_x:.4f}m {'✅' if com_var_x < 0.05 else '❌'}")
        print(f"  CoM variation Y: {com_var_y:.4f}m {'✅' if com_var_y < 0.05 else '❌'}")
        print(f"  Height mean: {height_mean:.3f}m")
        print(f"  Height std: {height_std:.4f}m")
        print(f"  Real time: {elapsed_real_time:.1f}s")
        
        print(f"\nAcceptance Criteria:")
        print(f"  Duration ≥ {self.duration}s: {'✅ PASS' if not fell else '❌ FAIL'}")
        print(f"  CoM variation < 0.05m: {'✅ PASS' if com_var_x < 0.05 and com_var_y < 0.05 else '❌ FAIL'}")
        print(f"\nOverall: {'✅✅✅ TEST PASSED ✅✅✅' if passed else '❌❌❌ TEST FAILED ❌❌❌'}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Test standing controller stability')
    parser.add_argument('--controller', type=str, default='improved_pd',
                       choices=['pd', 'improved_pd', 'pid', 'zmp'],
                       help='Controller to test (default: improved_pd)')
    parser.add_argument('--duration', type=float, default=60.0,
                       help='Required standing duration in seconds (default: 60)')
    parser.add_argument('--all', action='store_true',
                       help='Test all controllers')
    parser.add_argument('--ki', type=float, default=0.1,
                       help='Ki factor for PID controller (default: 0.1)')
    args = parser.parse_args()
    
    tester = StabilityTest(duration=args.duration)
    
    if args.all:
        # Test all controllers
        print("="*60)
        print("TESTING ALL CONTROLLERS")
        print("="*60)
        
        controllers = [
            (lambda m, d: PDStandingController(m, d), "PD Controller"),
            (lambda m, d: ImprovedPDController(m, d), "Improved PD (with gravity comp)"),
            (lambda m, d: PIDStandingController(m, d, ki_factor=args.ki), f"PID Controller (Ki={args.ki})"),
            (lambda m, d: ZMPBalanceController(m, d), "ZMP Balance Controller"),
        ]
        
        all_results = []
        
        # Create dummy model/data for initialization
        sim = Simulation(model_path="models/unitree_h1/scene_enhanced.xml")
        
        for controller_factory, name in controllers:
            controller = controller_factory(sim.model, sim.data)
            result = tester.run_test(controller, name)
            all_results.append(result)
            time.sleep(1)  # Brief pause between tests
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY OF ALL TESTS")
        print("="*60)
        
        for result in all_results:
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            duration = f"{result['duration']:.1f}s" if not result['fell'] else f"fell@{result['fall_time']:.1f}s"
            print(f"{status} | {result['controller']:30s} | {duration:12s} | "
                  f"Var: {result['com_var_x']:.4f}m, {result['com_var_y']:.4f}m")
        
        # Overall result
        passed_count = sum(1 for r in all_results if r['passed'])
        total_count = len(all_results)
        print(f"\nOverall: {passed_count}/{total_count} tests passed")
        
    else:
        # Test single controller
        sim = Simulation(model_path="models/unitree_h1/scene_enhanced.xml")
        
        if args.controller == 'pd':
            controller = PDStandingController(sim.model, sim.data)
            name = "PD Controller"
        elif args.controller == 'improved_pd':
            controller = ImprovedPDController(sim.model, sim.data)
            name = "Improved PD (with gravity comp)"
        elif args.controller == 'pid':
            controller = PIDStandingController(sim.model, sim.data, ki_factor=args.ki)
            name = f"PID Controller (Ki={args.ki})"
        elif args.controller == 'zmp':
            controller = ZMPBalanceController(sim.model, sim.data)
            name = "ZMP Balance Controller"
        
        result = tester.run_test(controller, name)
        
        # Exit with appropriate code
        sys.exit(0 if result['passed'] else 1)


if __name__ == "__main__":
    main()
