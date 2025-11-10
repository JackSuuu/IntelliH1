"""
LLM-assisted Path Planner
Uses LLM to analyze lidar data and provide high-level navigation strategies
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import json
from groq import Groq
from config import API_KEY

logger = logging.getLogger(__name__)


class LLMPlanner:
    """LLM-based path planner for complex navigation scenarios"""
    
    def __init__(self):
        self.last_strategy = None
        self.strategy_update_interval = 5.0  # Update strategy every 5 seconds
        self.last_strategy_time = 0
        self.client = Groq(api_key=API_KEY)
        
    def analyze_environment(self, 
                           lidar_data: np.ndarray,
                           car_pos: np.ndarray,
                           car_heading: float,
                           goal_pos: np.ndarray) -> Dict:
        """
        Convert lidar data to natural language environment description
        
        Args:
            lidar_data: Array of lidar distance readings
            car_pos: Current car position [x, y]
            car_heading: Current heading angle (radians)
            goal_pos: Goal position [x, y]
            
        Returns:
            Dictionary with environment analysis
        """
        # Calculate direction to goal
        to_goal = goal_pos[:2] - car_pos[:2]
        distance_to_goal = np.linalg.norm(to_goal)
        desired_heading = np.arctan2(to_goal[1], to_goal[0])
        
        # Heading error
        heading_error = desired_heading - car_heading
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        heading_error_deg = np.degrees(heading_error)
        
        # Analyze obstacles in different sectors
        num_readings = len(lidar_data)
        sectors = {
            'front': (num_readings // 2 - num_readings // 8, num_readings // 2 + num_readings // 8),
            'left': (num_readings * 3 // 4 - num_readings // 8, num_readings * 3 // 4 + num_readings // 8),
            'right': (num_readings // 4 - num_readings // 8, num_readings // 4 + num_readings // 8),
            'rear': (0, num_readings // 8),  # Also check the very back
        }
        
        obstacle_info = {}
        for sector_name, (start, end) in sectors.items():
            sector_data = lidar_data[start:end]
            if len(sector_data) > 0:
                min_dist = np.min(sector_data)
                obstacle_info[sector_name] = {
                    'min_distance': float(min_dist),
                    'has_obstacle': min_dist < 3.0,
                    'very_close': min_dist < 1.0
                }
        
        # Determine which direction has most clearance
        clearances = {
            'front': obstacle_info['front']['min_distance'],
            'left': obstacle_info['left']['min_distance'],
            'right': obstacle_info['right']['min_distance']
        }
        best_direction = max(clearances, key=clearances.get)
        
        return {
            'distance_to_goal': float(distance_to_goal),
            'heading_error_deg': float(heading_error_deg),
            'obstacles': obstacle_info,
            'best_clearance_direction': best_direction,
            'max_clearance': clearances[best_direction]
        }
    
    def request_llm_strategy(self,
                            env_analysis: Dict,
                            current_situation: str) -> Optional[Dict]:
        """
        Ask LLM for navigation strategy based on environment
        
        Args:
            env_analysis: Environment analysis from analyze_environment()
            current_situation: Description of current problem
            
        Returns:
            Strategy dictionary with recommendations
        """
        # Build prompt for LLM
        obstacles_desc = []
        for sector, info in env_analysis['obstacles'].items():
            if info['very_close']:
                obstacles_desc.append(f"- {sector.upper()}: VERY CLOSE ({info['min_distance']:.1f}m)")
            elif info['has_obstacle']:
                obstacles_desc.append(f"- {sector.upper()}: obstacle at {info['min_distance']:.1f}m")
            else:
                obstacles_desc.append(f"- {sector.upper()}: clear (>{info['min_distance']:.1f}m)")
        
        obstacles_text = "\n".join(obstacles_desc)
        
        prompt = f"""You are an expert autonomous navigation system. Analyze this situation and provide a strategy.

CURRENT SITUATION:
{current_situation}

SENSOR DATA:
- Distance to goal: {env_analysis['distance_to_goal']:.1f} meters
- Heading error: {env_analysis['heading_error_deg']:.0f} degrees (positive = need to turn left, negative = turn right)
- Best clearance: {env_analysis['best_clearance_direction']} ({env_analysis['max_clearance']:.1f}m)

OBSTACLE MAP:
{obstacles_text}

PROVIDE A NAVIGATION STRATEGY:
1. Should we turn in place first, or navigate while moving?
2. Which direction should we favor (left/right/straight)?
3. What speed should we use (slow/medium/fast)?
4. Any special maneuvers needed?

Respond in JSON format:
{{
    "strategy": "brief strategy description",
    "turn_in_place": true/false,
    "preferred_direction": "left/right/straight",
    "speed_mode": "slow/medium/fast",
    "special_notes": "any important considerations"
}}"""

        try:
            completion = self.client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": "You are a navigation assistant for an autonomous robot."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            # Parse JSON response
            text = completion.choices[0].message.content or "{}"
            # Find JSON in the text
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                json_text = text[start:end]
                strategy = json.loads(json_text)
                
                logger.info(f"🤖 LLM Strategy: {strategy.get('strategy', 'N/A')}")
                logger.info(f"   Direction: {strategy.get('preferred_direction', 'N/A')}, "
                           f"Speed: {strategy.get('speed_mode', 'N/A')}")
                
                return strategy
            else:
                logger.warning("Could not parse LLM response as JSON")
                return None
                
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return None
    
    def should_request_llm_help(self,
                                env_analysis: Dict,
                                current_time: float,
                                stuck_counter: int = 0) -> bool:
        """
        Decide if we should ask LLM for help
        
        Returns True if:
        - Haven't asked recently AND
        - (Stuck in one place for long time OR completely surrounded)
        """
        # Don't ask too frequently - give C++ more time to work
        if current_time - self.last_strategy_time < self.strategy_update_interval:
            return False
        
        # Only ask if REALLY stuck (increased threshold)
        if stuck_counter > 10:  # Increased from 5 to 10
            return True
        
        # Only ask if COMPLETELY surrounded (all 4 directions)
        close_obstacles = sum(1 for info in env_analysis['obstacles'].values() 
                            if info['very_close'])
        if close_obstacles >= 4:  # Changed from 2 to 4 - must be completely trapped
            return True
        
        # Ask if heading error is large and front is blocked
        if (abs(env_analysis['heading_error_deg']) > 120 and 
            env_analysis['obstacles']['front']['has_obstacle']):
            return True
        
        return False
    
    def apply_llm_strategy(self, strategy: Dict, nav_command: Dict) -> Dict:
        """
        Modify C++ navigation command based on LLM strategy
        Applies GENTLE modifications to avoid disrupting C++ control
        
        Args:
            strategy: Strategy from LLM
            nav_command: Original command from C++
            
        Returns:
            Modified navigation command
        """
        if strategy is None:
            return nav_command
        
        # Apply GENTLE speed adjustment - don't drastically change speeds
        speed_mode = strategy.get('speed_mode', 'medium')
        if speed_mode == 'slow':
            nav_command['linear_velocity'] *= 0.8  # Reduced from 0.5 - gentler
        elif speed_mode == 'fast':
            nav_command['linear_velocity'] *= 1.1  # Reduced from 1.3 - gentler
        
        # Apply GENTLE directional bias - only nudge, don't force
        preferred_dir = strategy.get('preferred_direction', 'straight')
        if preferred_dir == 'left' and nav_command['angular_velocity'] > 0:
            nav_command['angular_velocity'] *= 1.1  # Reduced from 1.2 - gentler
        elif preferred_dir == 'right' and nav_command['angular_velocity'] < 0:
            nav_command['angular_velocity'] *= 1.1  # Reduced from 1.2 - gentler
        
        # Turn in place ONLY if C++ already suggested it (very low linear velocity)
        if strategy.get('turn_in_place', False):
            if nav_command['linear_velocity'] < 0.08:  # C++ wants slow/stop
                nav_command['linear_velocity'] = 0.0  # Stop moving forward
                # Keep C++ angular velocity, just ensure it's strong enough
                if abs(nav_command['angular_velocity']) < 0.3:
                    nav_command['angular_velocity'] = np.sign(nav_command['angular_velocity']) * 0.35
        
        return nav_command
