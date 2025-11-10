#!/usr/bin/env python3
"""
Logging and Visualization Script for Unitree H1 Standing Controller

This script runs simulations with data logging and generates visualizations
to analyze CoM and ZMP behavior, helping identify causes of drift.

Usage:
    python log_and_visualize.py --duration 60 --controller pd
    python log_and_visualize.py --duration 120 --controller pid --ki 0.1
    python log_and_visualize.py --duration 60 --controller zmp
"""

import numpy as np
import matplotlib.pyplot as plt
import mujoco
import argparse
import sys
import os
from pathlib import Path
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulation.environment import Simulation
from control.pd_standing import PDStandingController, ImprovedPDController, PIDStandingController
from control.zmp_balance import ZMPBalanceController


class DataLogger:
    """Logger for CoM and ZMP data during simulation"""
    
    def __init__(self):
        self.time = []
        self.com_pos = []
        self.com_vel = []
        self.zmp_pos = []
        self.zmp_error = []
        self.joint_pos = []
        self.joint_vel = []
        self.control_torques = []
        
    def log(self, t, data, zmp_pos=None, zmp_error=None, tau=None):
        """Log data at current timestep"""
        self.time.append(t)
        self.com_pos.append(data.qpos[0:3].copy())
        self.com_vel.append(data.qvel[0:3].copy())
        
        if zmp_pos is not None:
            self.zmp_pos.append(zmp_pos.copy())
        
        if zmp_error is not None:
            self.zmp_error.append(zmp_error.copy())
            
        if tau is not None:
            self.control_torques.append(tau.copy())
        
        # Log joint positions (skip floating base)
        self.joint_pos.append(data.qpos[7:].copy())
        self.joint_vel.append(data.qvel[6:].copy())
    
    def to_arrays(self):
        """Convert lists to numpy arrays"""
        self.time = np.array(self.time)
        self.com_pos = np.array(self.com_pos)
        self.com_vel = np.array(self.com_vel)
        self.joint_pos = np.array(self.joint_pos)
        self.joint_vel = np.array(self.joint_vel)
        
        if self.zmp_pos:
            self.zmp_pos = np.array(self.zmp_pos)
        if self.zmp_error:
            self.zmp_error = np.array(self.zmp_error)
        if self.control_torques:
            self.control_torques = np.array(self.control_torques)
    
    def save_csv(self, output_dir="logs"):
        """Save logged data to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save CoM data
        com_data = np.column_stack([self.time, self.com_pos, self.com_vel])
        np.savetxt(
            f"{output_dir}/com_data.csv", 
            com_data,
            delimiter=',',
            header='time,com_x,com_y,com_z,vel_x,vel_y,vel_z',
            comments=''
        )
        print(f"✅ CoM data saved to {output_dir}/com_data.csv")
        
        # Save ZMP data if available
        if len(self.zmp_pos) > 0:
            zmp_data = np.column_stack([self.time, self.zmp_pos])
            np.savetxt(
                f"{output_dir}/zmp_data.csv",
                zmp_data,
                delimiter=',',
                header='time,zmp_x,zmp_y',
                comments=''
            )
            print(f"✅ ZMP data saved to {output_dir}/zmp_data.csv")
        
        if len(self.zmp_error) > 0:
            zmp_error_data = np.column_stack([self.time, self.zmp_error])
            np.savetxt(
                f"{output_dir}/zmp_error.csv",
                zmp_error_data,
                delimiter=',',
                header='time,zmp_error_x,zmp_error_y',
                comments=''
            )
            print(f"✅ ZMP error data saved to {output_dir}/zmp_error.csv")


def visualize_data(logger, controller_name, output_dir="logs"):
    """Generate visualization plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. CoM Position over time
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(logger.time, logger.com_pos[:, 0], label='X', linewidth=1.5)
    ax1.plot(logger.time, logger.com_pos[:, 1], label='Y', linewidth=1.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax1.set_title('CoM Position (Horizontal)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. CoM Height
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(logger.time, logger.com_pos[:, 2], linewidth=1.5, color='green')
    ax2.axhline(y=0.9, color='r', linestyle='--', label='Fall threshold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Height (m)')
    ax2.set_title('CoM Height')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. CoM Velocity
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(logger.time, logger.com_vel[:, 0], label='Vx', linewidth=1.5)
    ax3.plot(logger.time, logger.com_vel[:, 1], label='Vy', linewidth=1.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('CoM Velocity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. CoM Trajectory (top view)
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(logger.com_pos[:, 0], logger.com_pos[:, 1], linewidth=1.5)
    ax4.scatter(logger.com_pos[0, 0], logger.com_pos[0, 1], 
                color='green', s=100, marker='o', label='Start', zorder=5)
    ax4.scatter(logger.com_pos[-1, 0], logger.com_pos[-1, 1], 
                color='red', s=100, marker='x', label='End', zorder=5)
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_title('CoM Trajectory (Top View)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')
    
    # 5. ZMP Error (if available)
    if len(logger.zmp_error) > 0:
        ax5 = plt.subplot(3, 2, 5)
        ax5.plot(logger.time, logger.zmp_error[:, 0], label='ZMP Error X', linewidth=1.5)
        ax5.plot(logger.time, logger.zmp_error[:, 1], label='ZMP Error Y', linewidth=1.5)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Error (m)')
        ax5.set_title('ZMP Error from Support Center')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    else:
        ax5 = plt.subplot(3, 2, 5)
        ax5.text(0.5, 0.5, 'ZMP data not available\n(use ZMP controller)', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('ZMP Error')
    
    # 6. Statistics Summary
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    
    # Calculate statistics
    com_drift_x = abs(logger.com_pos[-1, 0] - logger.com_pos[0, 0])
    com_drift_y = abs(logger.com_pos[-1, 1] - logger.com_pos[0, 1])
    com_drift_total = np.sqrt(com_drift_x**2 + com_drift_y**2)
    com_var_x = np.std(logger.com_pos[:, 0])
    com_var_y = np.std(logger.com_pos[:, 1])
    height_mean = np.mean(logger.com_pos[:, 2])
    height_std = np.std(logger.com_pos[:, 2])
    vel_mean = np.mean(np.linalg.norm(logger.com_vel, axis=1))
    
    # Stability assessment
    is_stable = logger.com_pos[-1, 2] > 0.9
    status = "✅ STABLE" if is_stable else "❌ FELL"
    
    stats_text = f"""
PERFORMANCE SUMMARY

Controller: {controller_name}
Duration: {logger.time[-1]:.1f}s
Status: {status}

CoM Drift:
  X: {com_drift_x:.4f} m
  Y: {com_drift_y:.4f} m
  Total: {com_drift_total:.4f} m

CoM Variation:
  σ(X): {com_var_x:.4f} m
  σ(Y): {com_var_y:.4f} m

Height:
  Mean: {height_mean:.3f} m
  Std: {height_std:.4f} m

Velocity:
  Mean: {vel_mean:.4f} m/s

Acceptance: {'✅ PASS' if com_var_x < 0.05 and com_var_y < 0.05 else '❌ FAIL'}
(Criteria: CoM variation < 0.05m)
    """
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes,
            fontfamily='monospace', fontsize=10, verticalalignment='top')
    
    plt.suptitle(f'Standing Controller Analysis - {controller_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_file = f"{output_dir}/stability_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to {output_file}")
    
    # Show plot (if display available)
    try:
        plt.show()
    except:
        print("⚠️ Display not available, plot saved to file only")


def run_simulation_with_logging(controller_type='pd', duration=60.0, ki_factor=0.1):
    """
    Run simulation with specified controller and log data.
    
    Args:
        controller_type: 'pd', 'improved_pd', 'pid', or 'zmp'
        duration: Simulation duration in seconds
        ki_factor: Ki factor for PID controller
    """
    print("="*60)
    print("SIMULATION WITH DATA LOGGING")
    print("="*60)
    print(f"Controller: {controller_type.upper()}")
    print(f"Duration: {duration}s")
    if controller_type == 'pid':
        print(f"Ki factor: {ki_factor}")
    print()
    
    # Create simulation
    sim = Simulation(model_path="models/unitree_h1/scene_enhanced.xml")
    
    # Create controller
    if controller_type == 'pd':
        controller = PDStandingController(sim.model, sim.data)
        controller_name = "PD Controller"
    elif controller_type == 'improved_pd':
        controller = ImprovedPDController(sim.model, sim.data)
        controller_name = "Improved PD (with gravity comp)"
    elif controller_type == 'pid':
        controller = PIDStandingController(sim.model, sim.data, ki_factor=ki_factor)
        controller_name = f"PID Controller (Ki={ki_factor})"
    elif controller_type == 'zmp':
        controller = ZMPBalanceController(sim.model, sim.data)
        controller_name = "ZMP Balance Controller"
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")
    
    # Initialize
    mujoco.mj_forward(sim.model, sim.data)
    
    # Create logger
    logger = DataLogger()
    
    # Run simulation
    dt = 0.002
    max_steps = int(duration / dt)
    
    print("Running simulation...")
    start_time = time.time()
    fell = False
    
    for step in range(max_steps):
        current_time = step * dt
        
        # Compute control
        tau = controller.compute_control(dt=dt)
        sim.data.ctrl[:] = tau
        
        # Get ZMP data if available
        zmp_pos = None
        zmp_error = None
        if hasattr(controller, '_calculate_zmp'):
            try:
                zmp_error_val, support_center, zmp_pos_val = controller._calculate_zmp()
                zmp_pos = zmp_pos_val
                zmp_error = zmp_error_val
            except:
                pass
        
        # Log data
        logger.log(current_time, sim.data, zmp_pos=zmp_pos, zmp_error=zmp_error, tau=tau)
        
        # Step simulation
        sim.step()
        
        # Check for fall
        if sim.data.qpos[2] < 0.5:
            fell = True
            print(f"\n❌ Robot fell at t={current_time:.2f}s")
            break
        
        # Progress indicator
        if step % 5000 == 0:
            elapsed = time.time() - start_time
            print(f"  Progress: {current_time:.1f}s / {duration:.1f}s (elapsed: {elapsed:.1f}s)")
    
    elapsed_time = time.time() - start_time
    
    if not fell:
        print(f"\n✅ Simulation completed successfully in {elapsed_time:.1f}s")
    
    # Convert to arrays
    logger.to_arrays()
    
    # Save data
    print("\nSaving data...")
    logger.save_csv()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_data(logger, controller_name)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    com_drift_x = abs(logger.com_pos[-1, 0] - logger.com_pos[0, 0])
    com_drift_y = abs(logger.com_pos[-1, 1] - logger.com_pos[0, 1])
    com_var_x = np.std(logger.com_pos[:, 0])
    com_var_y = np.std(logger.com_pos[:, 1])
    
    print(f"Status: {'❌ Fell' if fell else '✅ Stable'}")
    print(f"CoM drift X: {com_drift_x:.4f}m")
    print(f"CoM drift Y: {com_drift_y:.4f}m")
    print(f"CoM variation X: {com_var_x:.4f}m")
    print(f"CoM variation Y: {com_var_y:.4f}m")
    print(f"Acceptance criteria: CoM variation < 0.05m")
    print(f"Result: {'✅ PASS' if com_var_x < 0.05 and com_var_y < 0.05 else '❌ FAIL'}")


def main():
    parser = argparse.ArgumentParser(description='Log and visualize controller performance')
    parser.add_argument('--controller', type=str, default='pd',
                       choices=['pd', 'improved_pd', 'pid', 'zmp'],
                       help='Controller type (default: pd)')
    parser.add_argument('--duration', type=float, default=60.0,
                       help='Simulation duration in seconds (default: 60)')
    parser.add_argument('--ki', type=float, default=0.1,
                       help='Ki factor for PID controller (default: 0.1)')
    args = parser.parse_args()
    
    run_simulation_with_logging(
        controller_type=args.controller,
        duration=args.duration,
        ki_factor=args.ki
    )


if __name__ == "__main__":
    main()
