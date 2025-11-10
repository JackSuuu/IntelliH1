"""
Cognitive Humanoid Framework - Main Controller
Integrates LLM planning, perception, path planning, and locomotion control
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import asyncio
import time
import sys
import os

import sys
import os

# Add project root to path for absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.llm.navigation_planner import NavigationPlanner
from src.perception.path_planner import AStarPlanner
from .unitree_rl_controller import UnitreeRLWalkingController

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Import C++ perception module
CPP_PERCEPTION_AVAILABLE = False
try:
    # Add cpp directory to Python path for C++ module
    cpp_dir = os.path.join(os.path.dirname(__file__), '..', 'perception', 'cpp')
    if cpp_dir not in sys.path:
        sys.path.insert(0, cpp_dir)
    import perception_cpp
    CPP_PERCEPTION_AVAILABLE = True
    logger.info("✅ C++ perception module loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️  C++ perception module not available: {e}")


class CognitiveHumanoidController:
    """
    Main cognitive controller that integrates:
    - LLM Planning Layer (high-level command understanding)
    - Perception Layer (LIDAR, vision, occupancy mapping)
    - Path Planning Layer (A* global path planning)
    - Motion Control Layer (Unitree RL locomotion)
    """
    
    def __init__(self, model, data, robot_type="h1", max_speed=1.0):
        """
        Initialize cognitive controller
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            robot_type: Robot type ("h1", "h1_2", "g1")
            max_speed: Maximum forward velocity in m/s (default: 1.0, recommended: 0.8-1.5)
        """
        self.model = model
        self.data = data
        self.robot_type = robot_type
        self.max_speed = max_speed
        
        # Initialize all layers
        logger.info("="*60)
        logger.info("🧠 COGNITIVE HUMANOID FRAMEWORK - INITIALIZING")
        logger.info("="*60)
        
        # Layer 1: LLM Planning (Brain)
        self.llm_planner = NavigationPlanner()
        logger.info("✓ Layer 1: LLM Planning Layer initialized")
        
        # Layer 2: Perception (C++ point cloud processor + LIDAR)
        if CPP_PERCEPTION_AVAILABLE:
            self.point_cloud_processor = perception_cpp.PointCloudProcessor(10.0)  # 10m range
            logger.info("✓ Layer 2: C++ Point Cloud Processor initialized")
        else:
            self.point_cloud_processor = None
            logger.warning("⚠️  Layer 2: Using simulated perception (C++ module not available)")
        
        self.lidar_samples = 360  # 360 degree coverage
        self.lidar_range = 10.0   # 10m range
        self.current_lidar = None
        
        # Layer 3: Path Planning (A* for obstacle avoidance)
        self.path_planner = AStarPlanner(grid_resolution=0.3, map_size=30)
        self.current_path = None
        self.path_index = 0
        logger.info("✓ Layer 3: Path Planning Layer initialized (A*)")
        
        # Layer 4: Motion Control (Unitree RL)
        self.motion_controller = UnitreeRLWalkingController(model, data, robot_type=robot_type, max_speed=max_speed)
        logger.info(f"✓ Layer 4: Motion Control Layer initialized (Unitree RL, max_speed={max_speed} m/s)")
        
        # State management
        self.current_task = None
        self.execution_state = "idle"  # idle, planning, executing, completed
        self.last_perception_time = 0
        self.perception_interval = 0.5  # Update perception every 0.5s
        
        # Performance tracking
        self.total_distance_traveled = 0.0
        self.last_position = None
        
        logger.info("="*60)
        logger.info("🧠 COGNITIVE FRAMEWORK READY")
        logger.info("="*60)
    
    async def execute_command(self, command: str) -> Dict[str, Any]:
        """
        Execute natural language command through full cognitive pipeline
        
        Pipeline:
        1. LLM parses command → target location
        2. Perception scans environment → occupancy map
        3. A* plans path → waypoints
        4. Motion controller follows path → locomotion
        
        Args:
            command: Natural language command (e.g., "walk to the kitchen")
            
        Returns:
            Task information dictionary
        """
        current_pos = self.data.qpos[:3]  # [x, y, z]
        
        logger.info("\n" + "="*60)
        logger.info(f"🤖 EXECUTING COMMAND: \"{command}\"")
        logger.info("="*60)
        
        # Step 1: LLM Planning - Parse command to target
        logger.info("\n[Step 1/4] 🧠 LLM Planning Layer...")
        task = await self.llm_planner.parse_natural_language_command(command, current_pos)
        self.current_task = task
        self.execution_state = "planning"
        
        if task["task_type"] == "stop":
            logger.info("  → Stop command received")
            self.motion_controller.set_target_velocity(0.0, 0.0, 0.0)
            self.execution_state = "idle"
            return task
        
        if task["task_type"] != "navigate" or task.get("target_position") is None:
            logger.warning("  → Invalid navigation task")
            return task
        
        target = np.array(task["target_position"])
        logger.info(f"  ✓ Target parsed: {task.get('target_name', 'unknown')} → ({target[0]:.2f}, {target[1]:.2f})")
        logger.info(f"  ✓ Speed: {task['speed']:.2f} m/s")
        logger.info(f"  ✓ Reasoning: {task['reasoning']}")
        
        # Step 2: Perception - Scan environment and build occupancy map
        logger.info("\n[Step 2/4] 👁️  Perception Layer...")
        self._update_perception()
        self._update_occupancy_map()
        logger.info(f"  ✓ LIDAR scan completed: {self.lidar_samples} samples")
        logger.info(f"  ✓ Occupancy map updated")
        
        # Step 3: Path Planning - A* from current to target
        logger.info("\n[Step 3/4] 🗺️  Path Planning Layer...")
        start = current_pos[:2]
        path = self.path_planner.plan_path((start[0], start[1]), (target[0], target[1]))
        
        if path is None or len(path) == 0:
            logger.error("  ✗ Path planning failed! No valid path found.")
            logger.error("  → Possible reasons: target unreachable or in obstacle")
            self.execution_state = "idle"
            return task
        
        self.current_path = path
        self.path_index = 0
        logger.info(f"  ✓ Path found: {len(path)} waypoints")
        logger.info(f"  ✓ Path length: {self._calculate_path_length(path):.2f}m")
        logger.info(f"  ✓ Start: ({start[0]:.2f}, {start[1]:.2f}) → Goal: ({path[-1][0]:.2f}, {path[-1][1]:.2f})")
        
        # Step 4: Motion Control - Set initial target
        logger.info("\n[Step 4/4] 🚶 Motion Control Layer...")
        # Set first waypoint as target
        waypoint = self.current_path[self.path_index]
        self.motion_controller.set_target(waypoint[0], waypoint[1])
        logger.info(f"  ✓ Motion controller activated")
        logger.info(f"  ✓ Following waypoint {self.path_index + 1}/{len(self.current_path)}")
        
        self.execution_state = "executing"
        self.last_position = current_pos[:2].copy()
        
        logger.info("\n" + "="*60)
        logger.info("🎯 NAVIGATION STARTED")
        logger.info("="*60 + "\n")
        
        return task
    
    def update(self, current_time: float, dt: float = 0.002) -> Dict[str, Any]:
        """
        Update cognitive controller (call every simulation step)
        
        Args:
            current_time: Current simulation time
            dt: Time step
            
        Returns:
            Status dictionary with current state
        """
        # Update motion controller
        tau = self.motion_controller.compute_control(dt)
        self.data.ctrl[:] = tau
        
        # Track distance traveled
        current_pos = self.data.qpos[:2]
        if self.last_position is not None:
            self.total_distance_traveled += np.linalg.norm(current_pos - self.last_position)
        self.last_position = current_pos.copy()
        
        # Update perception periodically
        if current_time - self.last_perception_time > self.perception_interval:
            self._update_perception()
            self._update_occupancy_map()
            self.last_perception_time = current_time
        
        # Navigation state machine
        if self.execution_state == "executing" and self.current_path is not None and self.current_task is not None:
            # Update navigation (move to next waypoint if current reached)
            self.motion_controller.update_navigation()
            
            # Get final goal from task (original target, not last waypoint)
            final_goal = np.array(self.current_task["target_position"][:2])
            distance_to_goal = np.linalg.norm(current_pos - final_goal)
            
            # STOPPING CONDITION 1: Distance-based goal detection
            # Stop if within goal_tolerance of the FINAL GOAL (not just waypoint)
            goal_tolerance = 1.2  # 1.2m radius around goal (increased for reliability)
            
            # Log goal distance every 50 updates for debugging
            if not hasattr(self, '_goal_check_counter'):
                self._goal_check_counter = 0
            self._goal_check_counter += 1
            
            if self._goal_check_counter % 50 == 0:
                logger.info(f"[GOAL CHECK] Distance to final goal: {distance_to_goal:.2f}m (threshold: {goal_tolerance}m)")
            
            if distance_to_goal < goal_tolerance:
                logger.info("\n" + "="*60)
                logger.info("🎯 GOAL REACHED!")
                logger.info("="*60)
                logger.info(f"  Target: {self.current_task.get('target_name', 'unknown')}")
                logger.info(f"  Distance to goal: {distance_to_goal:.2f}m (< {goal_tolerance}m)")
                logger.info(f"  Total distance traveled: {self.total_distance_traveled:.2f}m")
                logger.info(f"  Final position: ({current_pos[0]:.2f}, {current_pos[1]:.2f})")
                logger.info("="*60 + "\n")
                
                # FORCE STOP THE ROBOT - clear both target and velocity
                self.motion_controller.target_position = None
                self.motion_controller.set_target_velocity(0.0, 0.0, 0.0)
                logger.info("🛑 Robot stopped - target cleared, velocity set to zero")
                
                self.execution_state = "completed"
                self.current_path = None
                self.current_task = None
                return {
                    "status": "completed",
                    "distance_traveled": self.total_distance_traveled,
                    "final_position": current_pos.tolist(),
                    "distance_to_goal": float(distance_to_goal)
                }
            
            # Check if current waypoint reached
            waypoint = self.current_path[self.path_index]
            distance_to_waypoint = np.linalg.norm(current_pos - np.array(waypoint))
            
            # Use tighter tolerance for final waypoint, looser for intermediate
            is_final_waypoint = (self.path_index == len(self.current_path) - 1)
            waypoint_tolerance = 0.4 if is_final_waypoint else 0.6
            
            if distance_to_waypoint < waypoint_tolerance:
                # Move to next waypoint
                self.path_index += 1
                
                if self.path_index >= len(self.current_path):
                    # Path completed! Check if we're actually at goal
                    if distance_to_goal > goal_tolerance:
                        logger.warning(f"⚠️  Path finished but still {distance_to_goal:.2f}m from goal")
                        logger.warning(f"   Continuing to goal: ({final_goal[0]:.2f}, {final_goal[1]:.2f})")
                        # Set final goal directly as target
                        self.motion_controller.set_target(final_goal[0], final_goal[1])
                    else:
                        # Actually at goal, stop the robot
                        logger.info("\n" + "="*60)
                        logger.info("🎯 NAVIGATION COMPLETED!")
                        logger.info("="*60)
                        logger.info(f"  Total distance: {self.total_distance_traveled:.2f}m")
                        logger.info(f"  Final position: ({current_pos[0]:.2f}, {current_pos[1]:.2f})")
                        logger.info("="*60 + "\n")
                        
                        # FORCE STOP THE ROBOT - clear both target and velocity
                        self.motion_controller.target_position = None
                        self.motion_controller.set_target_velocity(0.0, 0.0, 0.0)
                        logger.info("🛑 Robot stopped - target cleared, velocity set to zero")
                        
                        self.execution_state = "completed"
                        self.current_path = None
                        self.current_task = None
                        return {
                            "status": "completed",
                            "distance_traveled": self.total_distance_traveled,
                            "final_position": current_pos.tolist()
                        }
                else:
                    # Set next waypoint
                    waypoint = self.current_path[self.path_index]
                    self.motion_controller.set_target(waypoint[0], waypoint[1])
                    logger.info(f"📍 Waypoint {self.path_index}/{len(self.current_path)} reached")
                    logger.info(f"   Next: ({waypoint[0]:.2f}, {waypoint[1]:.2f}) | {distance_to_goal:.2f}m to goal")
            
            # Return status with distance to ACTUAL GOAL
            
            return {
                "status": "executing",
                "waypoint": f"{self.path_index + 1}/{len(self.current_path)}",
                "distance_to_goal": float(distance_to_goal),
                "distance_traveled": self.total_distance_traveled
            }
        
        return {"status": self.execution_state}
    
    def _update_perception(self):
        """Update perception using C++ point cloud processor or simulated LIDAR"""
        # Get robot position and heading
        robot_pos = self.data.qpos[:2]
        robot_quat = self.data.qpos[3:7]
        robot_heading = self._quat_to_yaw(robot_quat)

        if CPP_PERCEPTION_AVAILABLE and self.point_cloud_processor is not None:
            # Use C++ radar perception
            logger.debug("🔍 Using C++ radar perception")

            # Simulate LIDAR data (in real implementation, this would come from hardware)
            angles = np.linspace(0, 2*np.pi, self.lidar_samples, endpoint=False)
            distances = []

            for angle in angles:
                # Simulate distance measurement with some obstacles
                ray_angle = robot_heading + angle
                ray_dir = np.array([np.cos(ray_angle), np.sin(ray_angle)])

                # Add some simulated obstacles for testing
                obstacle_distance = self.lidar_range

                # Simulate a wall at 3m in front
                if abs(ray_angle - robot_heading) < 0.3:  # within 17 degrees of forward
                    obstacle_distance = min(obstacle_distance, 3.0)

                # Simulate obstacles on the sides
                if abs(abs(ray_angle - robot_heading) - np.pi/2) < 0.2:  # within 11 degrees of sides
                    obstacle_distance = min(obstacle_distance, 2.0)

                # Add some noise to make it realistic
                noise = np.random.normal(0, 0.05)
                obstacle_distance = max(0.1, min(self.lidar_range, obstacle_distance + noise))

                distances.append(obstacle_distance)

            # Convert to point cloud using C++
            try:
                # lidar_to_point_cloud expects (distances, fov_rad)
                # fov_rad should be 2π for 360-degree LIDAR
                point_cloud = self.point_cloud_processor.lidar_to_point_cloud(distances, 2 * np.pi)
                logger.debug(f"📊 Generated {len(point_cloud)} points from LIDAR")

                # Remove noise using C++ - use positional arguments, not keyword
                # min_dist_sq is minimum distance squared (0.1^2 = 0.01)
                filtered_points = self.point_cloud_processor.remove_noise(point_cloud, 0.01)
                logger.debug(f"🧹 Filtered to {len(filtered_points)} points after noise removal")

                # Store filtered point cloud for path planning
                # Convert points back to distances for LIDAR format
                self.current_lidar = np.array([np.sqrt(p.x*p.x + p.y*p.y) for p in filtered_points], dtype=np.float32)

            except Exception as e:
                logger.warning(f"⚠️  C++ perception failed, falling back to simulation: {e}")
                self._simulate_lidar_perception(robot_pos, robot_heading)

        else:
            # Fallback to simulated perception
            logger.debug("🔍 Using simulated LIDAR perception")
            self._simulate_lidar_perception(robot_pos, robot_heading)
    
    def _simulate_lidar_perception(self, robot_pos, robot_heading):
        """Simulate LIDAR perception for testing/fallback"""
        self.current_lidar = np.full(self.lidar_samples, self.lidar_range, dtype=np.float32)

        # Simulate LIDAR by checking distance to scene objects
        angles = np.linspace(0, 2*np.pi, self.lidar_samples, endpoint=False)

        for i, angle in enumerate(angles):
            # Ray direction in world frame
            ray_angle = robot_heading + angle
            ray_dir = np.array([np.cos(ray_angle), np.sin(ray_angle)])

            # Check intersections with scene objects (simplified)
            min_dist = self.lidar_range

            # Check known obstacle positions from scene
            obstacles = [
                (5.0, 2.5, 0.5),   # Kitchen counter
                (5.5, 3.5, 0.5),   # Kitchen cabinet
                (-3.5, 6.0, 0.5),  # Bed
                (-2.0, 6.0, 0.3),  # Nightstand
                (0.0, -4.5, 0.5),  # Couch
                (0.0, -3.0, 0.3),  # Coffee table
            ]

            for obs_x, obs_y, obs_radius in obstacles:
                obs_pos = np.array([obs_x, obs_y])
                to_obs = obs_pos - robot_pos

                # Project onto ray direction
                proj = np.dot(to_obs, ray_dir)
                if proj > 0:  # Obstacle in front
                    # Distance to obstacle center
                    dist = np.linalg.norm(to_obs)
                    # Approximate distance to surface
                    dist_to_surface = max(0.1, dist - obs_radius)
                    min_dist = min(min_dist, dist_to_surface)

            self.current_lidar[i] = min_dist
    
    def _update_occupancy_map(self):
        """Update occupancy grid from LIDAR data"""
        if self.current_lidar is None:
            return
        
        # Get robot position and heading
        robot_pos = self.data.qpos[:2]
        robot_quat = self.data.qpos[3:7]
        robot_heading = self._quat_to_yaw(robot_quat)
        
        # Collect all obstacle positions
        obstacles = []
        
        # Convert LIDAR to obstacle positions
        angles = np.linspace(0, 2*np.pi, self.lidar_samples, endpoint=False)
        
        for i, (angle, distance) in enumerate(zip(angles, self.current_lidar)):
            if distance < self.lidar_range - 0.1:  # Hit detected
                # Calculate obstacle position in world frame
                ray_angle = robot_heading + angle
                obs_x = robot_pos[0] + distance * np.cos(ray_angle)
                obs_y = robot_pos[1] + distance * np.sin(ray_angle)
                obstacles.append((obs_x, obs_y))
        
        # Also mark known static obstacles from scene
        # Use approximate positions near furniture, not exact landmark positions
        static_obstacles = [
            # Kitchen area obstacles (around 5.0, 3.0)
            (5.0, 2.3),   # Kitchen counter (south side)
            (5.5, 2.8),   # Kitchen counter (corner)
            (5.8, 3.5),   # Kitchen cabinet (east side)
            
            # Bedroom obstacles (around -3.0, 6.0)
            (-3.8, 6.0),  # Bed (west side)
            (-3.2, 6.5),  # Bed (north side)
            (-1.8, 6.2),  # Nightstand (east side)
            
            # Living room obstacles (around 0.0, -4.0)
            (0.0, -4.8),  # Couch (south side)
            (0.5, -4.5),  # Couch (east end)
            (-0.5, -4.5), # Couch (west end)
            (0.0, -2.7),  # Coffee table (north side)
        ]
        
        obstacles.extend(static_obstacles)
        
        # Update occupancy grid
        self.path_planner.update_obstacles(obstacles)
    
    def _calculate_path_length(self, path: List[Tuple[float, float]]) -> float:
        """Calculate total length of path"""
        if len(path) < 2:
            return 0.0
        
        length = 0.0
        for i in range(len(path) - 1):
            p1 = np.array(path[i])
            p2 = np.array(path[i + 1])
            length += float(np.linalg.norm(p2 - p1))
        
        return length
    
    def _quat_to_yaw(self, quat):
        """Convert quaternion to yaw angle"""
        qw, qx, qy, qz = quat
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        return yaw
    
    def get_status_summary(self) -> str:
        """Get human-readable status summary"""
        if self.execution_state == "idle":
            return "🟢 IDLE - Ready for commands"
        elif self.execution_state == "planning":
            return "🟡 PLANNING - Processing command..."
        elif self.execution_state == "executing":
            if self.current_path:
                return f"🔵 EXECUTING - Waypoint {self.path_index + 1}/{len(self.current_path)}"
            return "🔵 EXECUTING"
        elif self.execution_state == "completed":
            return "✅ COMPLETED - Task finished"
        else:
            return f"❓ UNKNOWN STATE: {self.execution_state}"
