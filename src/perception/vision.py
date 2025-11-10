import numpy as np
import mujoco
import time
from config import LIDAR_RANGE, LIDAR_SAMPLES, LIDAR_FOV
from llm.planner import LLMPlanner
from perception.path_planner import AStarPlanner
import logging

logger = logging.getLogger(__name__)

# Import the C++ perception module for fast point cloud processing
try:
    import perception_cpp
    CPP_AVAILABLE = True
    print("C++ perception module loaded successfully!")
except ImportError:
    CPP_AVAILABLE = False
    print("Warning: C++ perception module not available. Using Python fallback.")

class Perception:
    """
    Handles sensor data processing and navigation computation.
    Uses A* for global path planning and C++ for local reactive control.
    Enhanced with LLM-based high-level planning for complex scenarios.
    """
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.lidar_range = LIDAR_RANGE
        self.lidar_samples = LIDAR_SAMPLES
        self.lidar_fov = np.deg2rad(LIDAR_FOV)
        self.lidar_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'lidar_ray_{i}') for i in range(LIDAR_SAMPLES)]
        
        # Initialize C++ processor if available
        if CPP_AVAILABLE:
            self.processor = perception_cpp.PointCloudProcessor(LIDAR_RANGE)
        
        # Cache car body ID
        self.car_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'car')
        
        # A* path planner for global navigation
        self.path_planner = AStarPlanner(grid_resolution=0.4, map_size=30)  # Coarser grid for speed
        self.current_path = None
        self.current_waypoint_idx = 0
        self.waypoint_reach_threshold = 0.8  # meters - increased for smoother transitions
        self.replan_counter = 0
        self.replan_interval = 200  # Replan every 200 steps (~20 seconds) - less frequent
        self.last_goal = None  # Track if goal changed

    def get_lidar_readings(self) -> np.ndarray:
        """
        Simulates LiDAR sensor readings using ray casting.

        Returns:
            np.ndarray: An array of distances for each LiDAR ray.
        """
        # Get car position and orientation
        car_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'car')
        car_pos = self.data.xpos[car_body_id].copy()
        car_quat = self.data.xquat[car_body_id].copy()
        
        # Convert quaternion to rotation matrix
        car_mat = np.zeros(9)
        mujoco.mju_quat2Mat(car_mat, car_quat)
        car_mat = car_mat.reshape(3, 3)
        
        # Initialize readings
        readings = np.full(self.lidar_samples, self.lidar_range)
        
        # Cast rays in a 360-degree arc
        angles = np.linspace(-self.lidar_fov / 2, self.lidar_fov / 2, self.lidar_samples)
        
        for i, angle in enumerate(angles):
            # Ray direction in car's local frame
            local_dir = np.array([np.cos(angle), np.sin(angle), 0])
            
            # Transform to world frame
            world_dir = car_mat @ local_dir
            
            # Ray origin (slightly above ground to avoid floor detection)
            ray_start = car_pos + np.array([0, 0, 0.05])
            ray_end = ray_start + world_dir * self.lidar_range
            
            # MuJoCo ray casting
            geom_id = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(
                self.model, self.data,
                ray_start, world_dir,
                None,  # geomgroup
                1,     # flg_static (include static geoms)
                -1,    # bodyexclude (car body to exclude)
                geom_id
            )
            
            if distance >= 0 and distance < self.lidar_range:
                readings[i] = distance
                
        return readings

    def get_obstacle_avoidance_vector(self, readings: np.ndarray) -> np.ndarray:
        """
        Calculates an avoidance vector based on LiDAR readings.
        The vector points away from the closest obstacles.
        
        Uses C++ implementation for faster processing if available.

        Args:
            readings (np.ndarray): LiDAR distance readings.

        Returns:
            np.ndarray: A 2D vector suggesting a direction to move to avoid obstacles.
        """
        if CPP_AVAILABLE:
            # Convert LiDAR readings to point cloud using C++
            point_cloud = self.processor.lidar_to_point_cloud(readings.tolist(), self.lidar_fov)
            
            # Apply noise filtering
            filtered_cloud = self.processor.remove_noise(point_cloud, 0.25)  # min_dist_sq = 0.5^2
            
            # Calculate avoidance vector
            avoidance = self.processor.get_avoidance_vector(filtered_cloud)
            return np.array([avoidance.x, avoidance.y])
        else:
            # Fallback to Python implementation
            return self._python_avoidance_vector(readings)
    
    def _python_avoidance_vector(self, readings: np.ndarray) -> np.ndarray:
        """
        Python fallback for avoidance vector calculation.
        """
        avoidance_vector = np.zeros(2)
        angles = np.linspace(-self.lidar_fov / 2, self.lidar_fov / 2, self.lidar_samples)

        for i, dist in enumerate(readings):
            if dist < self.lidar_range:
                # Obstacle detected, calculate repulsive force
                repulsive_force = (1.0 / dist) * (self.lidar_range - dist)
                
                # Vector pointing away from the obstacle
                angle = angles[i]
                avoidance_vector[0] -= repulsive_force * np.cos(angle)
                avoidance_vector[1] -= repulsive_force * np.sin(angle)
        
        if np.linalg.norm(avoidance_vector) > 1.0:
            avoidance_vector /= np.linalg.norm(avoidance_vector)

        return avoidance_vector
    
    def compute_navigation(self, car_pos, car_heading, goal_pos):
        """
        Main navigation computation with A* global planning and C++ local control.
        - A* generates optimal path avoiding obstacles
        - C++ follows waypoints with reactive obstacle avoidance
        - LLM provides high-level strategy when stuck
        
        Args:
            car_pos: Current position [x, y, z]
            car_heading: Current heading angle (radians)
            goal_pos: Goal position [x, y, z]
            
        Returns:
            dict: {'linear_velocity': float, 'angular_velocity': float}
        """
        if not CPP_AVAILABLE:
            # Fallback to Python (simplified)
            return self._python_navigation(car_pos, car_heading, goal_pos)
        
        # Get LiDAR readings
        lidar_readings = self.get_lidar_readings()
        
        # Convert to point cloud
        point_cloud = self.processor.lidar_to_point_cloud(lidar_readings.tolist(), self.lidar_fov)
        
        # Filter noise
        filtered_cloud = self.processor.remove_noise(point_cloud, 0.25)
        
        # === A* GLOBAL PATH PLANNING ===
        # Update occupancy grid with current obstacles
        obstacles_2d = [(p.x, p.y) for p in filtered_cloud]
        
        # Replan only when needed (not every frame!)
        need_replan = False
        
        # Check if goal changed
        goal_key = (goal_pos[0], goal_pos[1])
        if self.last_goal != goal_key:
            need_replan = True
            self.last_goal = goal_key
            logger.info(f"🎯 New goal detected, replanning...")
        elif self.current_path is None:
            need_replan = True
        elif self.current_waypoint_idx >= len(self.current_path):
            # Reached end of path
            need_replan = True
        else:
            # Replan periodically (every 20 seconds) - MUCH less frequent
            self.replan_counter += 1
            if self.replan_counter >= self.replan_interval:
                need_replan = True
        
        if need_replan:
            # Update grid before planning
            self.path_planner.update_obstacles(obstacles_2d)
            
            # Plan new path
            self.current_path = self.path_planner.plan_path(
                (car_pos[0], car_pos[1]),
                (goal_pos[0], goal_pos[1])
            )
            self.current_waypoint_idx = 0
            self.replan_counter = 0
            
            if self.current_path is None:
                logger.warning("❌ A* failed to find path, using direct navigation")
                # Fallback: navigate directly to goal
                target_point = goal_pos
            else:
                logger.info(f"🗺️  A* path: {len(self.current_path)} waypoints")
                target_point = self.current_path[0]
        else:
            # Follow existing path - get current waypoint
            if self.current_path and self.current_waypoint_idx < len(self.current_path):
                current_waypoint = self.current_path[self.current_waypoint_idx]
                
                # Check if reached current waypoint
                dist_to_waypoint = np.sqrt(
                    (car_pos[0] - current_waypoint[0])**2 + 
                    (car_pos[1] - current_waypoint[1])**2
                )
                
                if dist_to_waypoint < self.waypoint_reach_threshold:
                    # Move to next waypoint
                    self.current_waypoint_idx += 1
                    logger.debug(f"✓ Waypoint {self.current_waypoint_idx}/{len(self.current_path)} reached")
                    
                    if self.current_waypoint_idx < len(self.current_path):
                        target_point = self.current_path[self.current_waypoint_idx]
                    else:
                        target_point = goal_pos  # Final goal
                else:
                    target_point = current_waypoint
            else:
                target_point = goal_pos
        
        # === C++ LOCAL CONTROL ===
        # Use C++ for reactive control towards current waypoint
        current_point = perception_cpp.Point(car_pos[0], car_pos[1])
        
        # Navigate to target waypoint (not final goal)
        if isinstance(target_point, tuple):
            target_cpp = perception_cpp.Point(target_point[0], target_point[1])
        else:
            target_cpp = perception_cpp.Point(target_point[0], target_point[1])
        
        nav_cmd = self.processor.compute_navigation_command(
            current_point,
            car_heading,
            target_cpp,
            filtered_cloud
        )
        
        base_command = {
            'linear_velocity': nav_cmd.linear_velocity,
            'angular_velocity': nav_cmd.angular_velocity
        }
        
        # === NO LLM - PURE A* + C++ CONTROL ===
        # LLM was causing 1-second blocking calls every 4 seconds
        # Result: jerky motion and poor performance
        # A* + C++ reactive control is sufficient and MUCH faster
        
        # Simply return the C++ command
        return base_command
    
    def _update_stuck_detection(self, car_pos):
        """Track if car is stuck in one place"""
        if self.last_position is None:
            self.last_position = car_pos[:2].copy()
            return
        
        # Check if moved significantly
        distance_moved = np.linalg.norm(car_pos[:2] - self.last_position)
        
        if distance_moved < 0.05:  # Moved less than 5cm
            self.stuck_counter += 1
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)
        
        self.last_position = car_pos[:2].copy()
        
        # Keep recent position history
        self.position_history.append(car_pos[:2].copy())
        if len(self.position_history) > 20:
            self.position_history.pop(0)
    
    def _describe_situation(self, env_analysis: dict, stuck_counter: int) -> str:
        """Create natural language description of current situation"""
        descriptions = []
        
        if stuck_counter > 10:
            descriptions.append("Car is STUCK - not moving for extended period")
        elif stuck_counter > 5:
            descriptions.append("Car seems stuck or making very slow progress")
        
        # Check for surrounded situation
        close_sides = [name for name, info in env_analysis['obstacles'].items() 
                      if info.get('very_close', False)]
        if len(close_sides) >= 2:
            descriptions.append(f"Surrounded by obstacles on: {', '.join(close_sides)}")
        
        # Check for large heading error with blocked front
        if (abs(env_analysis['heading_error_deg']) > 120 and 
            env_analysis['obstacles']['front']['has_obstacle']):
            descriptions.append(f"Need to turn {env_analysis['heading_error_deg']:.0f}° but front is blocked")
        
        if not descriptions:
            descriptions.append("Complex navigation situation requiring strategic decision")
        
        return "; ".join(descriptions)
    
    def _python_navigation(self, car_pos, car_heading, goal_pos):
        """Python fallback for navigation computation."""
        to_goal = goal_pos[:2] - car_pos[:2]
        distance = np.linalg.norm(to_goal)
        
        if distance < 0.5:
            return {'linear_velocity': 0.0, 'angular_velocity': 0.0}
        
        desired_heading = np.arctan2(to_goal[1], to_goal[0])
        heading_error = desired_heading - car_heading
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        # Simple proportional control
        linear_vel = 0.5 if abs(heading_error) < np.pi/4 else 0.2
        angular_vel = np.clip(1.0 * heading_error, -0.8, 0.8)
        
        return {'linear_velocity': linear_vel, 'angular_velocity': angular_vel}
