"""
Multimodal Perception Module (Phase 2)
Integrates RGB cameras, semantic segmentation, object detection, and scene graphs
"""

import numpy as np
import mujoco
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import vision libraries (will gracefully handle missing deps)
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available. Install with: pip install opencv-python")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("YOLOv8 not available. Install with: pip install ultralytics")

try:
    import torch
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    SEGFORMER_AVAILABLE = True
except ImportError:
    SEGFORMER_AVAILABLE = False
    logger.warning("Transformers/SegFormer not available. Install with: pip install transformers torch")


class MultimodalPerception:
    """
    Multimodal Perception System for Embodied AI
    
    Features:
    - RGB camera capture from MuJoCo
    - Object detection (YOLOv8)
    - Semantic segmentation (SegFormer)
    - 3D position estimation (depth + detection)
    - Scene graph generation
    """
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        
        # Get camera IDs
        self.cameras = {
            'robot_head': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'robot_head'),
            'left_wrist_cam': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'left_wrist_cam'),
            'right_wrist_cam': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'right_wrist_cam'),
        }
        
        # Initialize vision models if available
        self.yolo_model = None
        self.segformer_processor = None
        self.segformer_model = None
        
        if YOLO_AVAILABLE:
            try:
                # Use YOLOv8n (nano) for fast inference
                self.yolo_model = YOLO('yolov8n.pt')
                logger.info("✓ YOLOv8 loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load YOLOv8: {e}")
        
        if SEGFORMER_AVAILABLE:
            try:
                # Use SegFormer for semantic segmentation
                self.segformer_processor = SegformerImageProcessor.from_pretrained(
                    "nvidia/segformer-b0-finetuned-ade-512-512"
                )
                self.segformer_model = SegformerForSemanticSegmentation.from_pretrained(
                    "nvidia/segformer-b0-finetuned-ade-512-512"
                )
                logger.info("✓ SegFormer loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load SegFormer: {e}")
        
        # Scene understanding state
        self.detected_objects = []
        self.scene_graph = {
            "objects": [],
            "relationships": []
        }
        
        # Renderer for capturing RGB images
        self.renderer = mujoco.Renderer(model, height=480, width=640)
        
        logger.info("[MultimodalPerception] Initialized")
        logger.info(f"  Cameras: {list(self.cameras.keys())}")
        logger.info(f"  YOLOv8: {'✓' if self.yolo_model else '✗'}")
        logger.info(f"  SegFormer: {'✓' if self.segformer_model else '✗'}")
    
    def capture_rgb_image(self, camera_name: str = 'robot_head') -> Optional[np.ndarray]:
        """
        Capture RGB image from specified camera
        
        Args:
            camera_name: Name of camera ('robot_head', 'left_wrist_cam', 'right_wrist_cam')
        
        Returns:
            RGB image as numpy array (H, W, 3) or None if failed
        """
        if camera_name not in self.cameras:
            logger.error(f"Unknown camera: {camera_name}")
            return None
        
        try:
            # Update renderer with current camera
            self.renderer.update_scene(self.data, camera=camera_name)
            
            # Render and get RGB pixels
            rgb = self.renderer.render()
            
            return rgb
        
        except Exception as e:
            logger.error(f"Failed to capture image from {camera_name}: {e}")
            return None
    
    def capture_depth_image(self, camera_name: str = 'robot_head') -> Optional[np.ndarray]:
        """
        Capture depth image from specified camera
        
        Args:
            camera_name: Name of camera
        
        Returns:
            Depth image as numpy array (H, W) or None if failed
        """
        if camera_name not in self.cameras:
            logger.error(f"Unknown camera: {camera_name}")
            return None
        
        try:
            # Enable depth rendering
            self.renderer.enable_depth_rendering()
            self.renderer.update_scene(self.data, camera=camera_name)
            
            # Render depth
            depth = self.renderer.render()
            
            self.renderer.disable_depth_rendering()
            
            return depth
        
        except Exception as e:
            logger.error(f"Failed to capture depth from {camera_name}: {e}")
            return None
    
    def detect_objects(self, rgb_image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Detect objects in RGB image using YOLOv8
        
        Args:
            rgb_image: RGB image (H, W, 3)
            confidence_threshold: Minimum confidence for detection
        
        Returns:
            List of detected objects with bounding boxes and labels
        """
        if not YOLO_AVAILABLE or self.yolo_model is None:
            logger.warning("YOLOv8 not available for object detection")
            return []
        
        try:
            # Run YOLOv8 inference
            results = self.yolo_model(rgb_image, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf >= confidence_threshold:
                        # Get box coordinates (xyxy format)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cls = int(box.cls[0])
                        label = self.yolo_model.names[cls]
                        
                        detections.append({
                            'label': label,
                            'confidence': conf,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                        })
            
            logger.debug(f"Detected {len(detections)} objects")
            return detections
        
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return []
    
    def semantic_segmentation(self, rgb_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Perform semantic segmentation using SegFormer
        
        Args:
            rgb_image: RGB image (H, W, 3)
        
        Returns:
            Segmentation map (H, W) with class IDs or None if failed
        """
        if not SEGFORMER_AVAILABLE or self.segformer_model is None:
            logger.warning("SegFormer not available for semantic segmentation")
            return None
        
        try:
            # Preprocess image
            inputs = self.segformer_processor(images=rgb_image, return_tensors="pt")
            
            # Run inference
            with torch.no_grad():
                outputs = self.segformer_model(**inputs)
            
            # Get segmentation map
            logits = outputs.logits
            seg_map = logits.argmax(dim=1)[0].cpu().numpy()
            
            return seg_map
        
        except Exception as e:
            logger.error(f"Semantic segmentation failed: {e}")
            return None
    
    def estimate_3d_position(self, detection: Dict, depth_image: np.ndarray, 
                            camera_name: str = 'robot_head') -> Optional[np.ndarray]:
        """
        Estimate 3D position of detected object using depth
        
        Args:
            detection: Object detection dictionary with 'center' key
            depth_image: Depth image (H, W)
            camera_name: Name of camera
        
        Returns:
            3D position [x, y, z] in world frame or None if failed
        """
        try:
            # Get pixel coordinates
            u, v = detection['center']
            u, v = int(u), int(v)
            
            # Get depth value
            depth = depth_image[v, u]
            
            # Get camera parameters
            cam_id = self.cameras[camera_name]
            cam_pos = self.data.cam_xpos[cam_id]
            cam_mat = self.data.cam_xmat[cam_id].reshape(3, 3)
            
            # Simple unprojection (approximate - proper implementation needs intrinsics)
            # This is a simplified version
            h, w = depth_image.shape
            
            # Normalized coordinates
            x_norm = (u - w/2) / (w/2)
            y_norm = (v - h/2) / (h/2)
            
            # 3D point in camera frame
            point_cam = np.array([x_norm * depth, y_norm * depth, depth])
            
            # Transform to world frame
            point_world = cam_pos + cam_mat @ point_cam
            
            return point_world
        
        except Exception as e:
            logger.error(f"3D position estimation failed: {e}")
            return None
    
    def build_scene_graph(self, camera_name: str = 'robot_head') -> Dict:
        """
        Build scene graph from current camera view
        
        Args:
            camera_name: Name of camera to use
        
        Returns:
            Scene graph dictionary with objects and relationships
        """
        # Capture images
        rgb = self.capture_rgb_image(camera_name)
        if rgb is None:
            return {"objects": [], "relationships": []}
        
        depth = self.capture_depth_image(camera_name)
        
        # Detect objects
        detections = self.detect_objects(rgb)
        
        # Build object list with 3D positions
        objects = []
        for i, det in enumerate(detections):
            obj = {
                "id": f"{det['label']}_{i}",
                "type": det['label'],
                "confidence": det['confidence'],
                "bbox_2d": det['bbox']
            }
            
            # Add 3D position if depth available
            if depth is not None:
                pos_3d = self.estimate_3d_position(det, depth, camera_name)
                if pos_3d is not None:
                    obj["pos_3d"] = pos_3d.tolist()
            
            objects.append(obj)
        
        # Infer spatial relationships (simple heuristics)
        relationships = []
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i >= j:
                    continue
                
                # Check if obj1 is on obj2 (based on vertical position)
                if 'pos_3d' in obj1 and 'pos_3d' in obj2:
                    z1, z2 = obj1['pos_3d'][2], obj2['pos_3d'][2]
                    xy_dist = np.linalg.norm(
                        np.array(obj1['pos_3d'][:2]) - np.array(obj2['pos_3d'][:2])
                    )
                    
                    # If obj1 is above obj2 and horizontally close
                    if z1 > z2 + 0.1 and xy_dist < 0.5:
                        relationships.append({
                            "type": "on",
                            "obj1": obj1['id'],
                            "obj2": obj2['id']
                        })
        
        scene_graph = {
            "objects": objects,
            "relationships": relationships,
            "timestamp": self.data.time
        }
        
        self.scene_graph = scene_graph
        return scene_graph
    
    def get_scene_description(self) -> str:
        """
        Generate natural language description of scene
        
        Returns:
            Human-readable scene description
        """
        if not self.scene_graph["objects"]:
            return "No objects detected in scene."
        
        desc = f"Scene contains {len(self.scene_graph['objects'])} objects:\n"
        
        for obj in self.scene_graph["objects"]:
            desc += f"  - {obj['type']}"
            if 'pos_3d' in obj:
                pos = obj['pos_3d']
                desc += f" at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
            desc += f" (confidence: {obj['confidence']:.2f})\n"
        
        if self.scene_graph["relationships"]:
            desc += "\nRelationships:\n"
            for rel in self.scene_graph["relationships"]:
                desc += f"  - {rel['obj1']} is {rel['type']} {rel['obj2']}\n"
        
        return desc
    
    def visualize_detections(self, rgb_image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes on RGB image
        
        Args:
            rgb_image: RGB image
            detections: List of detections
        
        Returns:
            Annotated image
        """
        if not CV2_AVAILABLE:
            return rgb_image
        
        annotated = rgb_image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['label']} {det['confidence']:.2f}"
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return annotated


# Example usage
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    from simulation.environment import Simulation
    
    # Load scene
    sim = Simulation(model_path="models/unitree_h1/scene_enhanced.xml")
    
    # Initialize perception
    perception = MultimodalPerception(sim.model, sim.data)
    
    # Update simulation
    mujoco.mj_forward(sim.model, sim.data)
    
    # Capture and analyze scene
    print("\n" + "="*60)
    print("Testing Multimodal Perception")
    print("="*60)
    
    rgb = perception.capture_rgb_image('robot_head')
    if rgb is not None:
        print(f"✓ Captured RGB image: {rgb.shape}")
    
    scene_graph = perception.build_scene_graph('robot_head')
    print(f"\n{perception.get_scene_description()}")
    
    print("\nScene Graph:")
    print(scene_graph)
