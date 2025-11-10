"""
A* Path Planner for obstacle avoidance
Generates global path from start to goal considering obstacles
"""

import numpy as np
import heapq
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AStarPlanner:
    """A* path planner with occupancy grid"""
    
    def __init__(self, grid_resolution=0.3, map_size=30):
        """
        Args:
            grid_resolution: Size of each grid cell in meters (0.3m = 30cm)
            map_size: Total map size in meters (30m x 30m centered at origin)
        """
        self.resolution = grid_resolution
        self.map_size = map_size
        
        # Grid dimensions
        self.grid_width = int(map_size / grid_resolution)
        self.grid_height = int(map_size / grid_resolution)
        
        # Offset to convert world coordinates to grid
        self.offset_x = map_size / 2.0
        self.offset_y = map_size / 2.0
        
        # Occupancy grid (0=free, 1=occupied)
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        
        # Robot safety margin (inflate obstacles)
        self.robot_radius = 0.3  # meters (humanoid is relatively narrow)
        self.inflation_cells = int(np.ceil(self.robot_radius / grid_resolution))
        
        logger.info(f"A* Planner initialized: {self.grid_width}x{self.grid_height} grid, "
                   f"{grid_resolution}m resolution, {self.inflation_cells} cell inflation")
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices"""
        grid_x = int((x + self.offset_x) / self.resolution)
        grid_y = int((y + self.offset_y) / self.resolution)
        
        # Clamp to grid bounds
        grid_x = max(0, min(self.grid_width - 1, grid_x))
        grid_y = max(0, min(self.grid_height - 1, grid_y))
        
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates (cell center)"""
        x = grid_x * self.resolution - self.offset_x + self.resolution / 2
        y = grid_y * self.resolution - self.offset_y + self.resolution / 2
        return x, y
    
    def update_obstacles(self, obstacles: List[Tuple[float, float]]):
        """
        Update occupancy grid with current obstacles
        
        Args:
            obstacles: List of (x, y) obstacle positions in world coordinates
        """
        # Clear grid
        self.grid.fill(0)
        
        # Mark obstacle cells
        for obs_x, obs_y in obstacles:
            grid_x, grid_y = self.world_to_grid(obs_x, obs_y)
            
            # Inflate obstacle for safety
            for dx in range(-self.inflation_cells, self.inflation_cells + 1):
                for dy in range(-self.inflation_cells, self.inflation_cells + 1):
                    gx = grid_x + dx
                    gy = grid_y + dy
                    
                    if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                        # Distance-based inflation (gradient)
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist <= self.inflation_cells:
                            self.grid[gy, gx] = 1
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance heuristic"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int, float]]:
        """
        Get valid neighboring cells
        
        Returns:
            List of (neighbor_x, neighbor_y, cost) tuples
        """
        x, y = node
        neighbors = []
        
        # 8-connected grid (diagonal movement allowed)
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nx, ny = x + dx, y + dy
            
            # Check bounds
            if not (0 <= nx < self.grid_width and 0 <= ny < self.grid_height):
                continue
            
            # Check if occupied
            if self.grid[ny, nx] == 1:
                continue
            
            # Cost: 1.0 for cardinal, 1.414 for diagonal
            cost = 1.414 if (dx != 0 and dy != 0) else 1.0
            
            neighbors.append((nx, ny, cost))
        
        return neighbors
    
    def plan_path(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """
        Plan path from start to goal using A*
        
        Args:
            start: (x, y) start position in world coordinates
            goal: (x, y) goal position in world coordinates
            
        Returns:
            List of (x, y) waypoints in world coordinates, or None if no path found
        """
        # Convert to grid coordinates
        start_grid = self.world_to_grid(start[0], start[1])
        goal_grid = self.world_to_grid(goal[0], goal[1])
        
        # Check if start or goal is in obstacle
        if self.grid[start_grid[1], start_grid[0]] == 1:
            logger.warning(f"Start position {start} is in obstacle!")
            # Try to find nearest free cell
            start_grid = self._find_nearest_free_cell(start_grid)
            if start_grid is None:
                return None
        
        if self.grid[goal_grid[1], goal_grid[0]] == 1:
            logger.warning(f"Goal position {goal} is in obstacle!")
            # Try to find nearest free cell
            goal_grid = self._find_nearest_free_cell(goal_grid)
            if goal_grid is None:
                return None
        
        # A* algorithm
        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}
        
        visited = set()
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current in visited:
                continue
            visited.add(current)
            
            # Goal reached
            if current == goal_grid:
                path_grid = self._reconstruct_path(came_from, current)
                path_world = [self.grid_to_world(gx, gy) for gx, gy in path_grid]
                
                # Simplify path (remove redundant waypoints)
                path_world = self._simplify_path(path_world)
                
                logger.info(f"✅ A* path found: {len(path_world)} waypoints")
                return path_world
            
            # Explore neighbors
            for neighbor_x, neighbor_y, move_cost in self.get_neighbors(current):
                neighbor = (neighbor_x, neighbor_y)
                
                if neighbor in visited:
                    continue
                
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        logger.warning(f"❌ A* failed to find path from {start} to {goal}")
        return None
    
    def _find_nearest_free_cell(self, start: Tuple[int, int], max_search=20) -> Optional[Tuple[int, int]]:
        """Find nearest free cell using BFS"""
        queue = [start]
        visited = {start}
        
        while queue:
            current = queue.pop(0)
            
            if self.grid[current[1], current[0]] == 0:
                return current
            
            # Check neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    nx, ny = current[0] + dx, current[1] + dy
                    
                    if not (0 <= nx < self.grid_width and 0 <= ny < self.grid_height):
                        continue
                    
                    if (nx, ny) in visited:
                        continue
                    
                    visited.add((nx, ny))
                    queue.append((nx, ny))
                    
                    if len(visited) > max_search:
                        return None
        
        return None
    
    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from map"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def _simplify_path(self, path: List[Tuple[float, float]], tolerance=0.5) -> List[Tuple[float, float]]:
        """
        Simplify path using Ramer-Douglas-Peucker algorithm
        Keep start, goal, and critical turning points
        """
        if len(path) <= 2:
            return path
        
        # Always keep start and end
        simplified = [path[0]]
        
        # Keep waypoints that are turning points (change direction significantly)
        for i in range(1, len(path) - 1):
            prev = path[i - 1]
            curr = path[i]
            next_pt = path[i + 1]
            
            # Calculate angle change
            vec1 = (curr[0] - prev[0], curr[1] - prev[1])
            vec2 = (next_pt[0] - curr[0], next_pt[1] - curr[1])
            
            # Normalize
            norm1 = np.sqrt(vec1[0]**2 + vec1[1]**2)
            norm2 = np.sqrt(vec2[0]**2 + vec2[1]**2)
            
            if norm1 > 0 and norm2 > 0:
                vec1_norm = (vec1[0] / norm1, vec1[1] / norm1)
                vec2_norm = (vec2[0] / norm2, vec2[1] / norm2)
                
                # Dot product (cosine of angle)
                dot = vec1_norm[0] * vec2_norm[0] + vec1_norm[1] * vec2_norm[1]
                
                # If angle change > 30 degrees, keep this waypoint
                if dot < 0.866:  # cos(30°)
                    simplified.append(curr)
        
        simplified.append(path[-1])
        
        logger.debug(f"Path simplified: {len(path)} → {len(simplified)} waypoints")
        return simplified
