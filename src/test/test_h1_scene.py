"""
Test Unitree H1 with Enhanced Scene
Simplified version using only PD standing control
Based on official Unitree SDK examples
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from robot.humanoid import UnitreeH1Controller
from simulation.environment import Simulation


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Unitree H1 Control Test')
    parser.add_argument('--simple', action='store_true', 
                        help='Use simple PD (default: improved PD with gravity compensation)')
    parser.add_argument('--rl', action='store_true',
                        help='🤖 Use RL-Assisted PD (EXPERIMENTAL - uses h1_demo_policy.pt)')
    parser.add_argument('--rl-weight', type=float, default=0.3,
                        help='RL blend weight (0=pure PD, 1=pure RL, default=0.3)')
    parser.add_argument('--rl-policy', type=str, default='policies/h1_demo_policy.pt',
                        help='Path to RL policy file')
    parser.add_argument('--walk', action='store_true',
                        help='🚶 Enable walking mode (simple gait)')
    parser.add_argument('--navigate', action='store_true',
                        help='🗺️ Enable navigation mode (walk to target location)')
    parser.add_argument('--target', type=str, default='kitchen',
                        help='Navigation target location (kitchen/bedroom/livingroom or x,y coordinates)')
    parser.add_argument('--walk-speed', type=float, default=0.3,
                        help='Walking speed in m/s (default: 0.3)')
    args = parser.parse_args()
    
    # Determine control mode
    if args.navigate:
        mode = "navigation"
        print("="*60)
        print("🤖 UNITREE H1 - NAVIGATION TEST")
        print(f"   Target: {args.target}")
        print("="*60)
    elif args.walk:
        mode = "walking"
        print("="*60)
        print("🤖 UNITREE H1 - WALKING TEST")
        print(f"   Speed: {args.walk_speed} m/s")
        print("="*60)
    else:
        mode = "standing"
        print("="*60)
        print("🤖 UNITREE H1 - STANDING BALANCE TEST")
        if args.rl:
            print("   Using RL-Assisted PD Control (EXPERIMENTAL)")
        else:
            print("   Using PD Control (Based on Official Unitree Examples)")
        print("="*60)
    
    sim = Simulation(model_path="models/unitree_h1/scene_enhanced.xml")
    
    use_gravity_comp = not args.simple
    robot = UnitreeH1Controller(
        sim.model, 
        sim.data, 
        use_gravity_compensation=use_gravity_comp,
        use_rl_assist=args.rl,
        rl_policy_path=args.rl_policy,
        mode=mode
    )
    
    mujoco.mj_forward(sim.model, sim.data)
    
    print(f"\n✓ Scene loaded successfully!")
    print(f"✓ Model: {sim.model_path}")
    print(f"✓ Bodies: {sim.model.nbody}")
    print(f"✓ Geoms: {sim.model.ngeom}")
    print(f"✓ Joints: {sim.model.njnt}")
    print()

    print("🗺 Available locations:")
    if hasattr(sim, 'locations'):
        for loc_name, loc_pos in sim.locations.items():
            print(f"   • {loc_name}: {loc_pos}")
    print()
    
    print(f"🤖 Robot starting position: ({sim.data.qpos[0]:.2f}, {sim.data.qpos[1]:.2f}, {sim.data.qpos[2]:.2f})")
    print()
    
    if args.rl:
        print("🤖 RL-ASSISTED PD CONTROLLER (EXPERIMENTAL)")
        print(f"   • RL blend weight: {args.rl_weight:.1%}")
        print(f"   • Policy: {args.rl_policy}")
        print("   • Safety baseline: Improved PD with gravity compensation")
    elif use_gravity_comp:
        print("⚡ IMPROVED PD CONTROLLER (with gravity compensation)")
        print("   • Active balance strategies: Ankle + Hip + Torso")
        print("   • Gravity compensation: Enabled")
    else:
        print("📊 SIMPLE PD CONTROLLER")
        print("   • Active balance strategies: Ankle + Hip + Torso")
        print("   • Gravity compensation: Disabled")
    
    print()
    
    # Setup mode-specific configuration
    if mode == "navigation":
        # Parse target location
        locations = {
            'kitchen': (5.0, 3.0),
            'bedroom': (-3.0, 6.0),
            'livingroom': (0.0, -4.0),
        }
        
        target_str = args.target.lower()
        if target_str in locations:
            target_x, target_y = locations[target_str]
            print(f"🗺️  NAVIGATION MODE: Walking to {target_str.upper()}")
        else:
            # Try parsing as coordinates
            try:
                coords = [float(x.strip()) for x in args.target.split(',')]
                if len(coords) == 2:
                    target_x, target_y = coords
                    print(f"🗺️  NAVIGATION MODE: Walking to ({target_x:.2f}, {target_y:.2f})")
                else:
                    raise ValueError()
            except:
                print(f"⚠️  Invalid target: {args.target}")
                print("   Using default: Kitchen (5.0, 3.0)")
                target_x, target_y = 5.0, 3.0
        
        robot.set_navigation_target(target_x, target_y)
        print(f"   Target: ({target_x:.2f}, {target_y:.2f})")
        print(f"   Speed: {args.walk_speed} m/s")
        
    elif mode == "walking":
        print(f"🚶 WALKING MODE: Continuous forward walking")
        print(f"   Speed: {args.walk_speed} m/s")
        robot.set_walking_velocity(args.walk_speed, 0.0)
        
    else:
        print("⚠️  STANDING MODE: Robot will maintain standing balance")
        print("   Target: Maintain height ≈ 1.0m")
        print("   Based on official Unitree SDK control parameters")
    
    print(f"\n▶ Starting simulation...")
    print(f"📹 Camera controls:")
    print(f"   - Mouse drag: Rotate view")
    print(f"   - Scroll: Zoom")
    print(f"   - Tab: Switch camera")
    print(f"   - ESC: Exit")
    print("="*60 + "\n")
    
    step_count = 0
    start_time = time.time()
    last_print_time = start_time
    target_reached = False
    
    try:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            while viewer.is_running():
                current_time = time.time() - start_time
                
                robot.update(current_time, dt=0.002)
                
                sim.step()
                viewer.sync()
                
                step_count += 1
                
                # Check if navigation target reached
                if mode == "navigation" and not target_reached:
                    if hasattr(robot.controller, 'target_position'):
                        if robot.controller.target_position is None:
                            target_reached = True
                
                if time.time() - last_print_time > 2.0:
                    pos = robot.get_position()
                    
                    is_stable = pos[2] > 0.9
                    status = "✅ STABLE" if is_stable else "❌ FALLING"
                    
                    # Different output for navigation mode
                    if mode == "navigation":
                        if target_reached:
                            status = "🎯 TARGET REACHED"
                        else:
                            # Calculate distance to target
                            dist = np.linalg.norm([pos[0] - target_x, pos[1] - target_y])
                            status = f"🗺️ Distance: {dist:.2f}m"
                        
                        print(f"[{current_time:.1f}s] Pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) | "
                              f"Height: {pos[2]:.2f}m | {status}")
                    else:
                        print(f"[{current_time:.1f}s] Pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) | "
                              f"Height: {pos[2]:.2f}m | {status}")
                    
                    last_print_time = time.time()
                
                time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
    finally:
        print(f"\nSimulation ended.")
        print(f"Total steps: {step_count}")
        print(f"Total time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
