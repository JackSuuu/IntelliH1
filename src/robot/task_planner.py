"""
Integrated Navigation + Manipulation Task System
Coordinates mobile base navigation and arm manipulation for complete tasks
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class Task:
    """Task definition"""
    name: str
    task_type: str  # "navigate", "pick", "place", "wait"
    target_location: Optional[str] = None
    target_position: Optional[np.ndarray] = None
    object_name: Optional[str] = None
    duration: float = 0.0


@dataclass
class TaskResult:
    """Task execution result"""
    success: bool
    message: str
    duration: float


class IntegratedTaskPlanner:
    """
    高层任务规划器
    协调导航和抓取任务
    
    示例任务序列:
    1. 导航到Kitchen → 抓取Cube → 导航到Bedroom → 放置Cube → 导航到Living Room
    """
    
    def __init__(self, robot, manipulator, task_executor, perception):
        """
        Args:
            robot: Car instance (mobile base)
            manipulator: ManipulatorController instance
            task_executor: TaskExecutor instance
            perception: Perception instance
        """
        self.robot = robot
        self.manipulator = manipulator
        self.task_executor = task_executor
        self.perception = perception
        
        # Task state
        self.current_task_list: List[Task] = []
        self.current_task_idx = 0
        self.task_start_time = None
        self.is_executing = False
        
        # Known object positions (in world frame)
        self.objects = {
            'cube': np.array([2.0, 0.0, 0.09]),      # Red cube
            'cylinder': np.array([2.5, 0.5, 0.11]),  # Green cylinder
        }
        
        # Task history
        self.task_history: List[Tuple[Task, TaskResult]] = []
        
        print("[TaskPlanner] Initialized!")
        print(f"  Available objects: {list(self.objects.keys())}")
    
    def create_demo_task_sequence(self) -> List[Task]:
        """
        创建演示任务序列:
        1. 导航到Kitchen
        2. 抓取Cube
        3. 导航到Bedroom
        4. 放置Cube
        5. 导航到Living Room
        """
        tasks = [
            Task(
                name="Navigate to Kitchen",
                task_type="navigate",
                target_location="kitchen"
            ),
            Task(
                name="Pick up Cube",
                task_type="pick",
                object_name="cube",
                target_position=self.objects['cube']
            ),
            Task(
                name="Navigate to Bedroom",
                task_type="navigate",
                target_location="bedroom"
            ),
            Task(
                name="Place Cube",
                task_type="place",
                target_position=np.array([0.3, 0.0, 0.12])  # Relative to car
            ),
            Task(
                name="Navigate to Living Room",
                task_type="navigate",
                target_location="living room"
            )
        ]
        
        print("\n[TaskPlanner] Created demo task sequence:")
        for i, task in enumerate(tasks, 1):
            print(f"  {i}. {task.name} ({task.task_type})")
        
        return tasks
    
    def start_task_sequence(self, tasks: Optional[List[Task]] = None):
        """
        开始执行任务序列
        
        Args:
            tasks: Task list. If None, uses demo sequence
        """
        if tasks is None:
            tasks = self.create_demo_task_sequence()
        
        self.current_task_list = tasks
        self.current_task_idx = 0
        self.is_executing = True
        self.task_history = []
        
        print(f"\n[TaskPlanner] ▶ Starting task sequence ({len(tasks)} tasks)")
        print("="*60)
    
    def get_current_task(self) -> Optional[Task]:
        """Get current task"""
        if not self.is_executing or self.current_task_idx >= len(self.current_task_list):
            return None
        return self.current_task_list[self.current_task_idx]
    
    def execute_navigation_task(self, task: Task) -> TaskResult:
        """
        执行导航任务
        
        Args:
            task: Navigation task
        
        Returns:
            TaskResult
        """
        print(f"\n[TaskPlanner] 🚗 Executing: {task.name}")
        print(f"  Target: {task.target_location}")
        
        start_time = time.time()
        
        # Set goal
        success = self.robot.set_goal(task.target_location)
        
        if not success:
            return TaskResult(
                success=False,
                message=f"Failed to set goal: {task.target_location}",
                duration=time.time() - start_time
            )
        
        # Wait for navigation to complete (handled in main loop)
        # This is a non-blocking call - just marks the goal
        
        return TaskResult(
            success=True,
            message=f"Navigation goal set: {task.target_location}",
            duration=time.time() - start_time
        )
    
    def execute_pick_task(self, task: Task) -> TaskResult:
        """
        执行抓取任务
        
        Args:
            task: Pick task
        
        Returns:
            TaskResult
        """
        print(f"\n[TaskPlanner] 🦾 Executing: {task.name}")
        print(f"  Object: {task.object_name}")
        print(f"  Position: {task.target_position}")
        
        start_time = time.time()
        
        # Convert world position to car-relative position
        car_pos = self.robot.get_position()
        car_heading = self.robot.get_heading_angle()
        
        # Transform from world to car frame
        relative_pos = task.target_position - car_pos
        
        # Rotate to car frame
        cos_h = np.cos(-car_heading)
        sin_h = np.sin(-car_heading)
        car_frame_pos = np.array([
            relative_pos[0] * cos_h - relative_pos[1] * sin_h,
            relative_pos[0] * sin_h + relative_pos[1] * cos_h,
            relative_pos[2]
        ])
        
        print(f"  Car-relative position: {car_frame_pos}")
        
        # Check if object is reachable
        distance = np.linalg.norm(car_frame_pos[:2])
        if distance > 0.40:  # Max reach
            return TaskResult(
                success=False,
                message=f"Object too far: {distance:.2f}m > 0.40m",
                duration=time.time() - start_time
            )
        
        # Execute pick
        # Pre-grasp position (10cm above)
        pre_grasp = car_frame_pos.copy()
        pre_grasp[2] += 0.10
        
        print(f"  Moving to pre-grasp position...")
        if not self.manipulator.move_to_pose(pre_grasp):
            return TaskResult(
                success=False,
                message="Failed to reach pre-grasp position",
                duration=time.time() - start_time
            )
        
        # Open gripper
        print(f"  Opening gripper...")
        self.manipulator.open_gripper()
        
        # Move to grasp position
        print(f"  Moving to grasp position...")
        if not self.manipulator.move_to_pose(car_frame_pos):
            return TaskResult(
                success=False,
                message="Failed to reach grasp position",
                duration=time.time() - start_time
            )
        
        # Close gripper
        print(f"  Closing gripper...")
        success = self.manipulator.grasp_object()
        
        if not success:
            return TaskResult(
                success=False,
                message="Failed to grasp object",
                duration=time.time() - start_time
            )
        
        # Lift object
        lift_pos = car_frame_pos.copy()
        lift_pos[2] += 0.15
        print(f"  Lifting object...")
        self.manipulator.move_to_pose(lift_pos)
        
        return TaskResult(
            success=True,
            message=f"Successfully picked {task.object_name}",
            duration=time.time() - start_time
        )
    
    def execute_place_task(self, task: Task) -> TaskResult:
        """
        执行放置任务
        
        Args:
            task: Place task
        
        Returns:
            TaskResult
        """
        print(f"\n[TaskPlanner] 🦾 Executing: {task.name}")
        print(f"  Position: {task.target_position}")
        
        start_time = time.time()
        
        # Pre-place position (above target)
        pre_place = task.target_position.copy()
        pre_place[2] += 0.15
        
        print(f"  Moving to pre-place position...")
        if not self.manipulator.move_to_pose(pre_place):
            return TaskResult(
                success=False,
                message="Failed to reach pre-place position",
                duration=time.time() - start_time
            )
        
        # Move to place position
        print(f"  Lowering to place position...")
        if not self.manipulator.move_to_pose(task.target_position):
            return TaskResult(
                success=False,
                message="Failed to reach place position",
                duration=time.time() - start_time
            )
        
        # Open gripper to release
        print(f"  Releasing object...")
        self.manipulator.open_gripper()
        
        # Retract
        retract = task.target_position.copy()
        retract[2] += 0.10
        print(f"  Retracting...")
        self.manipulator.move_to_pose(retract)
        
        # Return to home
        print(f"  Returning arm to home...")
        self.manipulator.move_to_home()
        
        return TaskResult(
            success=True,
            message="Successfully placed object",
            duration=time.time() - start_time
        )
    
    def update(self) -> bool:
        """
        Update task execution (call every frame)
        
        Returns:
            True if still executing, False if all tasks complete
        """
        if not self.is_executing:
            return False
        
        # Check if all tasks complete
        if self.current_task_idx >= len(self.current_task_list):
            self.is_executing = False
            self._print_summary()
            return False
        
        current_task = self.get_current_task()
        
        if current_task is None:
            return False
        
        # For navigation tasks, check if goal reached
        if current_task.task_type == "navigate":
            if self.robot.current_goal_name is None:
                # No active goal, task complete
                return True
            
            # Check if reached goal
            car_pos = self.robot.get_position()
            goal_pos = self.robot.get_current_goal_position()
            
            if goal_pos is not None:
                distance = np.linalg.norm(car_pos[:2] - goal_pos[:2])
                
                if distance < 1.5:  # Match C++ threshold
                    # Goal reached!
                    result = TaskResult(
                        success=True,
                        message=f"Reached {current_task.target_location}",
                        duration=time.time() - self.task_start_time if self.task_start_time else 0.0
                    )
                    self._complete_task(current_task, result)
                    return True
        
        return True
    
    def execute_next_task(self) -> bool:
        """
        Execute next task immediately (for manipulation tasks)
        
        Returns:
            True if task executed, False if no more tasks
        """
        if not self.is_executing or self.current_task_idx >= len(self.current_task_list):
            return False
        
        current_task = self.get_current_task()
        
        if current_task is None:
            return False
        
        # Start timing
        self.task_start_time = time.time()
        
        # Execute based on task type
        if current_task.task_type == "navigate":
            result = self.execute_navigation_task(current_task)
            if not result.success:
                self._complete_task(current_task, result)
        
        elif current_task.task_type == "pick":
            result = self.execute_pick_task(current_task)
            self._complete_task(current_task, result)
        
        elif current_task.task_type == "place":
            result = self.execute_place_task(current_task)
            self._complete_task(current_task, result)
        
        elif current_task.task_type == "wait":
            result = TaskResult(
                success=True,
                message="Wait complete",
                duration=current_task.duration
            )
            self._complete_task(current_task, result)
        
        return True
    
    def _complete_task(self, task: Task, result: TaskResult):
        """Mark task as complete and move to next"""
        status = "✅" if result.success else "❌"
        print(f"\n{status} Task {self.current_task_idx + 1}/{len(self.current_task_list)} Complete: {task.name}")
        print(f"   {result.message} ({result.duration:.1f}s)")
        
        self.task_history.append((task, result))
        self.current_task_idx += 1
        self.task_start_time = None
        
        # Execute next task if not navigation
        if self.current_task_idx < len(self.current_task_list):
            next_task = self.current_task_list[self.current_task_idx]
            if next_task.task_type != "navigate":
                # Manipulation tasks execute immediately
                print(f"\n→ Proceeding to next task...")
                time.sleep(0.5)  # Brief pause
                self.execute_next_task()
    
    def _print_summary(self):
        """Print task execution summary"""
        print("\n" + "="*60)
        print("🏁 ALL TASKS COMPLETE!")
        print("="*60)
        
        success_count = sum(1 for _, result in self.task_history if result.success)
        total_duration = sum(result.duration for _, result in self.task_history)
        
        print(f"\nSummary:")
        print(f"  Total tasks: {len(self.task_history)}")
        print(f"  Successful: {success_count}")
        print(f"  Failed: {len(self.task_history) - success_count}")
        print(f"  Total time: {total_duration:.1f}s")
        
        print(f"\nTask Details:")
        for i, (task, result) in enumerate(self.task_history, 1):
            status = "✅" if result.success else "❌"
            print(f"  {status} {i}. {task.name} ({result.duration:.1f}s)")
        
        print("="*60)
