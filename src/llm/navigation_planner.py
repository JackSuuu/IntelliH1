"""
LLM-Driven Navigation Planner for Humanoid Robot
Integrates natural language understanding with Unitree RL locomotion controller
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
import asyncio
from pathlib import Path

from groq import Groq
from config import API_KEY

logger = logging.getLogger(__name__)


class NavigationPlanner:
    """
    LLM-based navigation planner that converts natural language commands
    to robot navigation targets and velocity commands
    """
    
    def __init__(self):
        self.client = Groq(api_key=API_KEY)
        self.current_plan = None
        self.execution_state = "idle"  # idle, planning, executing, completed
        
        # Environment landmarks (from scene_enhanced.xml)
        self.landmarks = {
            "kitchen": {"position": (5.0, 3.0), "description": "kitchen area with counter and cabinet"},
            "bedroom": {"position": (-3.0, 6.0), "description": "bedroom with bed and nightstand"},
            "living_room": {"position": (0.0, -4.0), "description": "living room with couch and coffee table"},
            "center": {"position": (0.0, 0.0), "description": "center of the room"},
            "origin": {"position": (0.0, 0.0), "description": "starting position"},
        }
        
        # Object database (from scene)
        self.known_objects = {
            "counter": (5.0, 2.5),
            "cabinet": (5.5, 3.5),
            "bed": (-3.5, 6.0),
            "nightstand": (-2.0, 6.0),
            "couch": (0.0, -4.5),
            "coffee_table": (0.0, -3.0),
            "red_cube": None,  # Movable objects - positions change
            "green_cube": None,
            "blue_sphere": None,
        }
        
        logger.info("[NavigationPlanner] Initialized with LLM-based planning")
        logger.info(f"  Known landmarks: {list(self.landmarks.keys())}")
        logger.info(f"  Known objects: {list(self.known_objects.keys())}")
    
    async def parse_natural_language_command(self, command: str, current_position: np.ndarray) -> Dict[str, Any]:
        """
        Parse natural language command into structured navigation task
        
        Args:
            command: Natural language command (e.g., "walk to the kitchen")
            current_position: Current robot position [x, y, z]
            
        Returns:
            Dictionary with parsed task:
            {
                "task_type": "navigate" | "explore" | "follow" | "stop",
                "target": (x, y) or None,
                "speed": float (0.0 to 1.5),
                "description": str,
                "reasoning": str
            }
        """
        # Build environment context
        env_context = self._build_environment_context(current_position)
        
        # Create LLM prompt
        prompt = f"""You are a navigation planner for a humanoid robot (Unitree H1) in an indoor environment.

Current robot position: ({current_position[0]:.2f}, {current_position[1]:.2f})

Known landmarks and objects:
{json.dumps(env_context, indent=2)}

User command: "{command}"

Parse this command into a structured navigation task. Respond in JSON format:
{{
    "task_type": "navigate" | "explore" | "stop",
    "target_name": "kitchen" | "bedroom" | "living_room" | object name | null,
    "target_position": [x, y] or null,
    "speed": 0.0 to 1.5 (in m/s, recommend 0.8-1.0 for efficient navigation),
    "description": "brief description of the task",
    "reasoning": "brief explanation of your interpretation"
}}

Examples:
- "go to the kitchen" → navigate to kitchen at (5.0, 3.0)
- "walk to the bedroom" → navigate to bedroom at (-3.0, 6.0)
- "move forward slowly" → navigate 2m ahead at slow speed
- "stop" → stop task

Important:
- Use exact landmark positions from the context above
- Choose efficient speeds (0.8-1.0 m/s for normal walking)
- If direction only (forward/back/left/right), calculate relative position
"""

        try:
            # Call LLM
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a robot navigation planner. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Parse response
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            task = json.loads(content)
            
            # Validate and fill in defaults
            task.setdefault("task_type", "navigate")
            task.setdefault("speed", 0.9)  # Increased default speed
            task.setdefault("description", "Navigate to target")
            task.setdefault("reasoning", "Parsed from user command")
            
            # If target_name is provided, lookup position
            if task.get("target_name") and task["target_name"] in self.landmarks:
                task["target_position"] = list(self.landmarks[task["target_name"]]["position"])
            
            logger.info(f"[NavigationPlanner] Parsed command: {command}")
            logger.info(f"  → Task: {task['task_type']} to {task.get('target_name', 'unknown')}")
            logger.info(f"  → Target: {task.get('target_position')}, Speed: {task['speed']} m/s")
            logger.info(f"  → Reasoning: {task['reasoning']}")
            
            return task
            
        except Exception as e:
            logger.error(f"Failed to parse command with LLM: {e}")
            # Fallback: simple keyword matching
            return self._fallback_parse(command, current_position)
    
    def _fallback_parse(self, command: str, current_position: np.ndarray) -> Dict[str, Any]:
        """Simple keyword-based fallback parser"""
        command_lower = command.lower()
        
        # Check for stop command
        if "stop" in command_lower or "halt" in command_lower:
            return {
                "task_type": "stop",
                "target_position": None,
                "speed": 0.0,
                "description": "Stop movement",
                "reasoning": "Fallback parser: stop keyword detected"
            }
        
        # Check for landmark keywords
        for landmark_name, landmark_info in self.landmarks.items():
            if landmark_name in command_lower:
                return {
                    "task_type": "navigate",
                    "target_name": landmark_name,
                    "target_position": list(landmark_info["position"]),
                    "speed": 0.9,  # Faster default
                    "description": f"Navigate to {landmark_name}",
                    "reasoning": f"Fallback parser: detected '{landmark_name}' keyword"
                }
        
        # Default: explore forward
        target = current_position[:2] + np.array([2.0, 0.0])
        return {
            "task_type": "navigate",
            "target_position": list(target),
            "speed": 0.5,
            "description": "Move forward",
            "reasoning": "Fallback parser: no specific target, moving forward"
        }
    
    def _build_environment_context(self, current_position: np.ndarray) -> Dict[str, Any]:
        """Build environment context for LLM"""
        context = {
            "landmarks": {},
            "current_location": {
                "position": [float(current_position[0]), float(current_position[1])],
                "nearest_landmark": self._find_nearest_landmark(current_position)
            }
        }
        
        # Add landmark info with distances
        for name, info in self.landmarks.items():
            pos = np.array(info["position"])
            distance = np.linalg.norm(pos - current_position[:2])
            context["landmarks"][name] = {
                "position": list(info["position"]),
                "description": info["description"],
                "distance": float(distance)
            }
        
        return context
    
    def _find_nearest_landmark(self, position: np.ndarray) -> str:
        """Find nearest landmark to current position"""
        min_dist = float('inf')
        nearest = "unknown"
        
        for name, info in self.landmarks.items():
            pos = np.array(info["position"])
            dist = np.linalg.norm(pos - position[:2])
            if dist < min_dist:
                min_dist = dist
                nearest = name
        
        return nearest
    
    async def generate_velocity_command(self, 
                                       task: Dict[str, Any],
                                       current_position: np.ndarray,
                                       current_velocity: np.ndarray,
                                       obstacles: Optional[List[Dict]] = None) -> Dict[str, float]:
        """
        Generate velocity command based on task and current state
        
        Args:
            task: Parsed task from parse_natural_language_command
            current_position: Current position [x, y, z]
            current_velocity: Current velocity [vx, vy, vz]
            obstacles: Optional list of detected obstacles
            
        Returns:
            Dictionary with velocity commands:
            {
                "vx": forward velocity (m/s),
                "vy": lateral velocity (m/s),
                "omega": angular velocity (rad/s)
            }
        """
        if task["task_type"] == "stop":
            return {"vx": 0.0, "vy": 0.0, "omega": 0.0}
        
        if task["task_type"] != "navigate" or task.get("target_position") is None:
            return {"vx": 0.0, "vy": 0.0, "omega": 0.0}
        
        # Calculate direction to target
        target = np.array(task["target_position"])
        current_pos_2d = current_position[:2]
        to_target = target - current_pos_2d
        distance = np.linalg.norm(to_target)
        
        # Stop if close enough
        if distance < 0.5:
            logger.info(f"[NavigationPlanner] Reached target (distance: {distance:.2f}m)")
            return {"vx": 0.0, "vy": 0.0, "omega": 0.0}
        
        # Normalize direction
        direction = to_target / (distance + 1e-6)
        
        # Target speed (with distance-based scaling)
        target_speed = task.get("speed", 0.9)  # Default to faster speed
        if distance < 2.0:
            # Slow down when approaching target
            target_speed *= (distance / 2.0)
        
        # Simple proportional control
        # For now, only use forward velocity (vx) and angular velocity (omega)
        # The RL controller handles the actual locomotion
        vx = direction[0] * target_speed
        vy = direction[1] * target_speed
        
        # Calculate desired heading
        desired_heading = np.arctan2(direction[1], direction[0])
        
        # For simplicity, convert to forward + angular velocity
        # (Most humanoid locomotion uses forward walking + turning)
        speed_magnitude = np.sqrt(vx**2 + vy**2)
        omega = 0.0  # Angular velocity (would need current heading to calculate)
        
        logger.debug(f"[NavigationPlanner] Velocity command: vx={vx:.2f}, vy={vy:.2f}, distance={distance:.2f}m")
        
        return {
            "vx": float(vx),
            "vy": float(vy), 
            "omega": float(omega),
            "distance_to_target": float(distance)
        }


class LLMRobotController:
    """
    High-level controller that combines LLM planning with Unitree RL locomotion
    """
    
    def __init__(self, robot_controller, navigation_planner: NavigationPlanner):
        """
        Args:
            robot_controller: UnitreeRLWalkingController instance
            navigation_planner: NavigationPlanner instance
        """
        self.robot = robot_controller
        self.planner = navigation_planner
        self.current_task = None
        self.task_start_time = None
        
        logger.info("[LLMRobotController] Initialized")
    
    async def execute_command(self, command: str, current_position: np.ndarray):
        """
        Execute natural language command
        
        Args:
            command: Natural language command (e.g., "walk to the kitchen")
            current_position: Current robot position [x, y, z]
        """
        # Parse command with LLM
        task = await self.planner.parse_natural_language_command(command, current_position)
        self.current_task = task
        
        # If navigate task, set target on robot controller
        if task["task_type"] == "navigate" and task.get("target_position"):
            target = task["target_position"]
            self.robot.set_target(target[0], target[1])
            logger.info(f"[LLMRobotController] Executing: {task['description']}")
            logger.info(f"  Target: ({target[0]:.2f}, {target[1]:.2f})")
            logger.info(f"  Speed: {task['speed']:.2f} m/s")
        elif task["task_type"] == "stop":
            self.robot.set_velocity_command(0.0, 0.0, 0.0)
            logger.info("[LLMRobotController] Stopping robot")
        
        return task
    
    def update(self, current_position: np.ndarray) -> Dict[str, Any]:
        """
        Update controller state
        
        Returns:
            Status dictionary with current state
        """
        if self.current_task is None:
            return {"status": "idle"}
        
        # Check if navigation task is complete
        if self.current_task["task_type"] == "navigate":
            target = self.current_task.get("target_position")
            if target:
                distance = np.linalg.norm(np.array(target) - current_position[:2])
                if distance < 0.5:
                    logger.info("[LLMRobotController] Task completed!")
                    self.current_task = None
                    return {"status": "completed", "message": "Reached target"}
                
                return {
                    "status": "executing",
                    "task": self.current_task["description"],
                    "distance_remaining": float(distance)
                }
        
        return {"status": "executing"}
