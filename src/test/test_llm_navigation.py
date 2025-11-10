"""
LLM-Driven Navigation Test for Unitree H1
Integrates natural language understanding with locomotion control
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import sys
import os
import asyncio

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from simulation.environment import Simulation
from control.cognitive_controller import CognitiveHumanoidController
import logging

# Configure logging to show INFO level messages
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Simple format, just the message
)


async def execute_llm_navigation(command: str, cognitive_controller, sim, viewer):
    """Execute a natural language navigation command"""
    
    print(f"\n{'='*60}")
    print(f"🤖 Executing command: \"{command}\"")
    print(f"{'='*60}\n")
    
    # Parse and execute command through cognitive pipeline
    task = await cognitive_controller.execute_command(command)
    
    if task is None or task.get('task_type') == 'stop':
        print("Task cancelled or stop command received.")
        return
    
    # Monitor execution
    start_time = time.time()
    last_print_time = start_time
    
    while viewer.is_running():
        current_time = time.time() - start_time
        
        # Update cognitive controller (includes motion control)
        status = cognitive_controller.update(current_time, dt=0.002)
        
        # Update simulation
        sim.step()
        viewer.sync()
        
        # Check status
        current_pos = sim.data.qpos[:3]
        
        # Print status every 2 seconds
        if time.time() - last_print_time > 2.0:
            is_stable = current_pos[2] > 0.9
            stability_status = "✅ STABLE" if is_stable else "❌ UNSTABLE"
            
            if status['status'] == 'executing':
                distance = status.get('distance_to_goal', 0.0)
                print(f"[{current_time:.1f}s] Pos: ({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}) | "
                      f"Height: {current_pos[2]:.2f}m | Distance: {distance:.2f}m | {stability_status}")
            elif status['status'] == 'completed':
                print(f"\n{'='*60}")
                print(f"🎯 TASK COMPLETED!")
                print(f"{'='*60}\n")
                break
            else:
                print(f"[{current_time:.1f}s] Pos: ({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}) | "
                      f"Height: {current_pos[2]:.2f}m | Status: {status['status']} | {stability_status}")
            
            last_print_time = time.time()
        
        # Stop if robot falls
        if current_pos[2] < 0.5:
            print("\n⚠️  Robot fell! Stopping execution.")
            break
        
        time.sleep(0.01)
    
    total_time = time.time() - start_time
    final_pos = sim.data.qpos[:3]
    distance_traveled = np.linalg.norm(final_pos[:2] - current_pos[:2])
    
    print(f"\n{'='*60}")
    print(f"📊 EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"  • Duration: {total_time:.1f}s")
    print(f"  • Distance traveled: {distance_traveled:.2f}m")
    print(f"  • Final position: ({final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.2f})")
    print(f"  • Final height: {final_pos[2]:.2f}m")
    print(f"{'='*60}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='LLM-Driven Humanoid Navigation')
    parser.add_argument('--command', type=str, default='walk to the kitchen',
                        help='Natural language navigation command')
    parser.add_argument('--walk-speed', type=float, default=0.8,
                        help='Default walking speed in m/s (default: 0.8)')
    parser.add_argument('--robot-type', type=str, default='h1', choices=['h1', 'h1_2', 'g1'],
                        help='Robot type (default: h1)')
    args = parser.parse_args()
    
    print("="*60)
    print("🤖 LLM-DRIVEN HUMANOID NAVIGATION")
    print("="*60)
    print(f"Robot: Unitree {args.robot_type.upper()}")
    print(f"Default speed: {args.walk_speed} m/s")
    print(f"Command: \"{args.command}\"")
    print("="*60)
    print()
    
    # Load enhanced scene
    print(f"[Setup] Loading enhanced scene...")
    sim = Simulation(model_path="extern/unitree_rl_gym/resources/robots/h1/scene_enhanced.xml")
    
    # Create cognitive controller (integrates LLM, perception, planning, and motion)
    cognitive_controller = CognitiveHumanoidController(
        sim.model,
        sim.data,
        robot_type=args.robot_type,
        max_speed=args.walk_speed  # Pass command-line speed to controller
    )
    
    print(f"✓ Scene loaded successfully!")
    print(f"  • Model: {sim.model_path}")
    print(f"  • Bodies: {sim.model.nbody}")
    print(f"  • Robot start: ({sim.data.qpos[0]:.2f}, {sim.data.qpos[1]:.2f}, {sim.data.qpos[2]:.2f})")
    print()
    
    print("🗺️  Known locations:")
    print("  • Kitchen: (5.0, 3.0)")
    print("  • Bedroom: (-3.0, 6.0)")
    print("  • Living Room: (0.0, -4.0)")
    print("  • Center: (0.0, 0.0)")
    print()
    
    print("📹 Camera controls:")
    print("  • Mouse drag: Rotate view")
    print("  • Scroll: Zoom")
    print("  • ESC: Exit")
    print()
    
    try:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            # Run LLM navigation with cognitive controller
            asyncio.run(execute_llm_navigation(args.command, cognitive_controller, sim, viewer))
    
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
