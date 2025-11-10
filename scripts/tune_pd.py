#!/usr/bin/env python3
"""
PID Parameter Tuning Script for Unitree H1 Standing Controller

This script performs automated grid search to find optimal Kp, Kd, and Ki gains
for the standing controller. It runs simulations with different parameter sets
and evaluates stability based on multiple metrics.

Usage:
    python tune_pd.py --mode grid  # Grid search
    python tune_pd.py --mode test --kp 100 --kd 10 --ki 0.1  # Test specific values
"""

import numpy as np
import mujoco
import argparse
import sys
import os
from pathlib import Path
import time
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulation.environment import Simulation
from control.pd_standing import PDStandingController, PIDStandingController


class ControllerTuner:
    """Automated tuning for PD/PID controller gains"""
    
    def __init__(self, model_path="models/unitree_h1/scene_enhanced.xml", 
                 sim_duration=60.0, dt=0.002):
        """
        Args:
            model_path: Path to MuJoCo scene XML
            sim_duration: Simulation duration in seconds
            dt: Simulation timestep
        """
        self.model_path = model_path
        self.sim_duration = sim_duration
        self.dt = dt
        self.max_steps = int(sim_duration / dt)
        
    def evaluate_parameters(self, kp_scale=1.0, kd_scale=1.0, ki_factor=0.0, 
                          use_pid=False, verbose=False):
        """
        Evaluate controller performance with given parameters.
        
        Args:
            kp_scale: Scaling factor for Kp gains
            kd_scale: Scaling factor for Kd gains
            ki_factor: Integral gain factor (for PID)
            use_pid: Use PID controller instead of PD
            verbose: Print detailed progress
            
        Returns:
            dict: Performance metrics
        """
        # Create simulation
        sim = Simulation(model_path=self.model_path)
        
        # Create controller with scaled gains
        if use_pid:
            controller = PIDStandingController(sim.model, sim.data, ki_factor=ki_factor)
        else:
            controller = PDStandingController(sim.model, sim.data)
        
        # Scale gains
        controller.kp = controller.kp * kp_scale
        controller.kd = controller.kd * kd_scale
        
        # Initialize MuJoCo
        mujoco.mj_forward(sim.model, sim.data)
        
        # Tracking variables
        com_positions = []
        com_velocities = []
        heights = []
        fell = False
        fall_time = self.sim_duration
        
        # Run simulation
        for step in range(self.max_steps):
            current_time = step * self.dt
            
            # Compute control
            tau = controller.compute_control(dt=self.dt)
            sim.data.ctrl[:] = tau
            
            # Step simulation
            sim.step()
            
            # Record data
            base_pos = sim.data.qpos[0:3].copy()
            base_vel = sim.data.qvel[0:3].copy()
            
            com_positions.append(base_pos.copy())
            com_velocities.append(base_vel.copy())
            heights.append(base_pos[2])
            
            # Check if robot fell (height threshold)
            if base_pos[2] < 0.5:
                fell = True
                fall_time = current_time
                if verbose:
                    print(f"  Robot fell at t={current_time:.2f}s")
                break
            
            # Progress indicator
            if verbose and step % 5000 == 0:
                print(f"  Progress: {current_time:.1f}s / {self.sim_duration:.1f}s")
        
        # Calculate metrics
        com_positions = np.array(com_positions)
        com_velocities = np.array(com_velocities)
        heights = np.array(heights)
        
        # CoM drift (total displacement from start)
        com_drift_x = abs(com_positions[-1, 0] - com_positions[0, 0])
        com_drift_y = abs(com_positions[-1, 1] - com_positions[0, 1])
        com_drift_total = np.sqrt(com_drift_x**2 + com_drift_y**2)
        
        # CoM variation (standard deviation)
        com_var_x = np.std(com_positions[:, 0])
        com_var_y = np.std(com_positions[:, 1])
        
        # Height stability
        height_mean = np.mean(heights)
        height_std = np.std(heights)
        
        # Velocity magnitude (lower is more stable)
        velocity_magnitude = np.mean(np.linalg.norm(com_velocities, axis=1))
        
        # Overall score (lower is better)
        # Heavily penalize falling
        if fell:
            stability_score = 1000.0 + (self.sim_duration - fall_time) * 10
        else:
            # Weighted combination of metrics
            stability_score = (
                com_drift_total * 10.0 +  # Penalize drift
                (com_var_x + com_var_y) * 20.0 +  # Penalize variation
                height_std * 5.0 +  # Penalize height variation
                velocity_magnitude * 2.0  # Penalize excess movement
            )
        
        metrics = {
            'kp_scale': kp_scale,
            'kd_scale': kd_scale,
            'ki_factor': ki_factor,
            'use_pid': use_pid,
            'fell': fell,
            'fall_time': fall_time,
            'com_drift_x': com_drift_x,
            'com_drift_y': com_drift_y,
            'com_drift_total': com_drift_total,
            'com_var_x': com_var_x,
            'com_var_y': com_var_y,
            'height_mean': height_mean,
            'height_std': height_std,
            'velocity_magnitude': velocity_magnitude,
            'stability_score': stability_score
        }
        
        return metrics
    
    def grid_search(self, kp_range=(0.5, 1.5, 5), kd_range=(0.5, 1.5, 5),
                   ki_range=(0.0, 0.2, 3), save_results=True):
        """
        Perform grid search over parameter ranges.
        
        Args:
            kp_range: (min, max, num_points) for Kp scaling
            kd_range: (min, max, num_points) for Kd scaling
            ki_range: (min, max, num_points) for Ki factor
            save_results: Save results to JSON file
            
        Returns:
            list: All evaluation results
            dict: Best parameters
        """
        print("="*60)
        print("PD/PID PARAMETER GRID SEARCH")
        print("="*60)
        
        # Generate parameter grid
        kp_values = np.linspace(kp_range[0], kp_range[1], kp_range[2])
        kd_values = np.linspace(kd_range[0], kd_range[1], kd_range[2])
        ki_values = np.linspace(ki_range[0], ki_range[1], ki_range[2])
        
        total_tests = len(kp_values) * len(kd_values) * len(ki_values)
        print(f"\nTotal parameter combinations: {total_tests}")
        print(f"Simulation duration per test: {self.sim_duration}s")
        print(f"Estimated total time: {total_tests * self.sim_duration / 60:.1f} minutes")
        print()
        
        results = []
        best_result = None
        best_score = float('inf')
        
        test_num = 0
        for kp in kp_values:
            for kd in kd_values:
                for ki in ki_values:
                    test_num += 1
                    use_pid = ki > 0.001
                    
                    print(f"Test {test_num}/{total_tests}: Kp={kp:.2f}, Kd={kd:.2f}, Ki={ki:.3f}")
                    
                    try:
                        metrics = self.evaluate_parameters(
                            kp_scale=kp, 
                            kd_scale=kd, 
                            ki_factor=ki,
                            use_pid=use_pid,
                            verbose=False
                        )
                        
                        results.append(metrics)
                        
                        # Print summary
                        status = "❌ FELL" if metrics['fell'] else "✅ STABLE"
                        print(f"  {status} | Score: {metrics['stability_score']:.2f} | "
                              f"Drift: {metrics['com_drift_total']:.4f}m | "
                              f"Vel: {metrics['velocity_magnitude']:.4f}m/s")
                        
                        # Track best result
                        if metrics['stability_score'] < best_score:
                            best_score = metrics['stability_score']
                            best_result = metrics
                            print(f"  🏆 NEW BEST! Score: {best_score:.2f}")
                        
                        print()
                        
                    except Exception as e:
                        print(f"  ⚠️ Error: {e}")
                        print()
        
        # Print best results
        print("="*60)
        print("BEST PARAMETERS FOUND")
        print("="*60)
        if best_result:
            print(f"Kp scale: {best_result['kp_scale']:.3f}")
            print(f"Kd scale: {best_result['kd_scale']:.3f}")
            print(f"Ki factor: {best_result['ki_factor']:.3f}")
            print(f"Controller: {'PID' if best_result['use_pid'] else 'PD'}")
            print(f"Stability score: {best_result['stability_score']:.2f}")
            print(f"CoM drift: {best_result['com_drift_total']:.4f}m")
            print(f"CoM variance X: {best_result['com_var_x']:.4f}m")
            print(f"CoM variance Y: {best_result['com_var_y']:.4f}m")
            print(f"Status: {'Fell' if best_result['fell'] else 'Stable'}")
        
        # Save results
        if save_results:
            output_file = "logs/tuning_results.json"
            os.makedirs("logs", exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump({
                    'results': results,
                    'best': best_result
                }, f, indent=2)
            print(f"\n✅ Results saved to {output_file}")
        
        return results, best_result


def main():
    parser = argparse.ArgumentParser(description='Tune PD/PID controller parameters')
    parser.add_argument('--mode', type=str, default='grid', 
                       choices=['grid', 'test'],
                       help='Tuning mode: grid search or single test')
    parser.add_argument('--duration', type=float, default=60.0,
                       help='Simulation duration in seconds (default: 60)')
    parser.add_argument('--kp', type=float, default=1.0,
                       help='Kp scale factor for test mode (default: 1.0)')
    parser.add_argument('--kd', type=float, default=1.0,
                       help='Kd scale factor for test mode (default: 1.0)')
    parser.add_argument('--ki', type=float, default=0.1,
                       help='Ki factor for test mode (default: 0.1)')
    parser.add_argument('--use-pid', action='store_true',
                       help='Use PID controller in test mode')
    args = parser.parse_args()
    
    # Create tuner
    tuner = ControllerTuner(
        model_path="models/unitree_h1/scene_enhanced.xml",
        sim_duration=args.duration
    )
    
    if args.mode == 'grid':
        # Grid search
        # Start with conservative ranges around default values
        results, best = tuner.grid_search(
            kp_range=(0.5, 1.5, 5),  # 50% to 150% of default
            kd_range=(0.5, 1.5, 5),
            ki_range=(0.0, 0.2, 3),  # Test PD vs PID
            save_results=True
        )
    else:
        # Single test
        print("="*60)
        print("TESTING SPECIFIC PARAMETERS")
        print("="*60)
        print(f"Kp scale: {args.kp}")
        print(f"Kd scale: {args.kd}")
        print(f"Ki factor: {args.ki}")
        print(f"Controller: {'PID' if args.use_pid else 'PD'}")
        print()
        
        metrics = tuner.evaluate_parameters(
            kp_scale=args.kp,
            kd_scale=args.kd,
            ki_factor=args.ki,
            use_pid=args.use_pid,
            verbose=True
        )
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Status: {'❌ Fell at {:.2f}s'.format(metrics['fall_time']) if metrics['fell'] else '✅ Stable'}")
        print(f"CoM drift X: {metrics['com_drift_x']:.4f}m")
        print(f"CoM drift Y: {metrics['com_drift_y']:.4f}m")
        print(f"CoM drift total: {metrics['com_drift_total']:.4f}m")
        print(f"CoM variance X: {metrics['com_var_x']:.4f}m")
        print(f"CoM variance Y: {metrics['com_var_y']:.4f}m")
        print(f"Height mean: {metrics['height_mean']:.3f}m")
        print(f"Height std: {metrics['height_std']:.4f}m")
        print(f"Velocity magnitude: {metrics['velocity_magnitude']:.4f}m/s")
        print(f"Stability score: {metrics['stability_score']:.2f}")


if __name__ == "__main__":
    main()
