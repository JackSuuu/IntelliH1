"""
Physics Knowledge Base (Phase 3)
Stores physical properties of objects and manipulation knowledge
"""

import json
from typing import Dict, List, Any, Optional
import numpy as np


class PhysicsKnowledgeBase:
    """
    Knowledge base containing physical properties and manipulation strategies
    
    This forms the foundation for RAG-based physics-aware planning
    """
    
    def __init__(self):
        self.objects_db = self._create_objects_database()
        self.actions_db = self._create_actions_database()
        self.materials_db = self._create_materials_database()
        self.failure_modes_db = self._create_failure_modes_database()
    
    def _create_objects_database(self) -> List[Dict[str, Any]]:
        """Create database of common household objects with physics properties"""
        return [
            # Kitchen objects
            {
                "name": "coffee_mug",
                "type": "container",
                "shape": "cylinder",
                "typical_weight": {"min": 0.2, "max": 0.5, "unit": "kg"},
                "material": "ceramic",
                "fragile": True,
                "grasp_points": ["handle", "body"],
                "preferred_grasp": "handle",
                "friction_coefficient": 0.6,
                "center_of_mass": "bottom_third",
                "stability": {
                    "base_area": "small",
                    "height_to_width_ratio": 1.5,
                    "tip_angle": 30
                },
                "manipulation_notes": "Grasp by handle when full. Use gentle force (5-10N). Risk of tipping if grasped from side when full."
            },
            {
                "name": "apple",
                "type": "food",
                "shape": "sphere",
                "typical_weight": {"min": 0.1, "max": 0.2, "unit": "kg"},
                "material": "organic",
                "fragile": True,
                "grasp_points": ["top", "sides"],
                "preferred_grasp": "enveloping",
                "friction_coefficient": 0.7,
                "deformable": True,
                "manipulation_notes": "Apply even pressure (3-5N). Avoid point contact to prevent bruising."
            },
            {
                "name": "banana",
                "type": "food",
                "shape": "curved_cylinder",
                "typical_weight": {"min": 0.1, "max": 0.15, "unit": "kg"},
                "material": "organic",
                "fragile": True,
                "grasp_points": ["stem", "body"],
                "preferred_grasp": "stem",
                "friction_coefficient": 0.5,
                "deformable": True,
                "manipulation_notes": "Grasp at stem end. Body bruises easily with >2N pressure."
            },
            {
                "name": "water_bottle",
                "type": "container",
                "shape": "cylinder",
                "typical_weight": {"min": 0.05, "max": 0.6, "unit": "kg"},
                "material": "plastic",
                "fragile": False,
                "grasp_points": ["cap", "body"],
                "preferred_grasp": "body",
                "friction_coefficient": 0.4,
                "center_of_mass": "variable",
                "manipulation_notes": "Weight varies with contents. Center of mass shifts when tilted."
            },
            
            # Task objects
            {
                "name": "red_cube",
                "type": "geometric_primitive",
                "shape": "box",
                "typical_weight": {"min": 0.2, "max": 0.3, "unit": "kg"},
                "material": "plastic",
                "fragile": False,
                "grasp_points": ["edges", "faces"],
                "preferred_grasp": "parallel_jaw",
                "friction_coefficient": 0.8,
                "stability": {
                    "base_area": "large",
                    "tip_angle": 45
                },
                "manipulation_notes": "Stable when placed on any face. High friction - secure grasp."
            },
            {
                "name": "green_cylinder",
                "type": "geometric_primitive",
                "shape": "cylinder",
                "typical_weight": {"min": 0.25, "max": 0.35, "unit": "kg"},
                "material": "plastic",
                "fragile": False,
                "grasp_points": ["sides", "top"],
                "preferred_grasp": "wrap_around",
                "friction_coefficient": 0.8,
                "stability": {
                    "base_area": "medium",
                    "tip_angle": 25
                },
                "manipulation_notes": "Can roll when placed horizontally. Place upright for stability."
            },
            {
                "name": "yellow_ball",
                "type": "geometric_primitive",
                "shape": "sphere",
                "typical_weight": {"min": 0.15, "max": 0.25, "unit": "kg"},
                "material": "plastic",
                "fragile": False,
                "grasp_points": ["any"],
                "preferred_grasp": "enveloping",
                "friction_coefficient": 0.7,
                "stability": {
                    "base_area": "minimal",
                    "rolls": True
                },
                "manipulation_notes": "Will roll on any surface. Requires containment (bowl/box) or active stabilization."
            },
            
            # Furniture/surfaces
            {
                "name": "table",
                "type": "surface",
                "shape": "rectangular_prism",
                "material": "wood",
                "fragile": False,
                "surface_friction": 0.6,
                "load_capacity": {"max": 50, "unit": "kg"},
                "manipulation_notes": "Stable surface for object placement. Ensure objects are centered to prevent tipping."
            },
            {
                "name": "counter",
                "type": "surface",
                "shape": "rectangular_prism",
                "material": "wood",
                "fragile": False,
                "surface_friction": 0.5,
                "load_capacity": {"max": 100, "unit": "kg"},
                "manipulation_notes": "Fixed surface. Good for heavy object placement."
            }
        ]
    
    def _create_actions_database(self) -> List[Dict[str, Any]]:
        """Create database of manipulation actions and their requirements"""
        return [
            {
                "action": "grasp_cylinder",
                "object_shapes": ["cylinder"],
                "requirements": {
                    "gripper_size": "must exceed object diameter",
                    "approach_angle": ["top", "side"],
                    "force_range": {"min": 5, "max": 15, "unit": "N"},
                    "contact_points": 2
                },
                "steps": [
                    "Align gripper axis with cylinder axis",
                    "Approach from above or side",
                    "Close gripper with 5-15N force",
                    "Verify secure grasp before lifting"
                ],
                "failure_modes": ["slip_if_wet", "slip_if_insufficient_force", "damage_if_excessive_force"],
                "success_indicators": ["stable_contact", "no_slippage_on_lift"]
            },
            {
                "action": "grasp_sphere",
                "object_shapes": ["sphere"],
                "requirements": {
                    "gripper_size": "must exceed object diameter",
                    "approach_angle": ["any"],
                    "force_range": {"min": 3, "max": 10, "unit": "N"},
                    "contact_points": 3,
                    "gripper_type": "enveloping or suction"
                },
                "steps": [
                    "Center gripper above sphere",
                    "Envelop sphere with multi-finger grasp",
                    "Apply gentle inward pressure",
                    "Lift slowly"
                ],
                "failure_modes": ["slip_due_to_point_contact", "roll_away_on_approach"],
                "success_indicators": ["three_point_contact", "stable_in_grasp"]
            },
            {
                "action": "grasp_box",
                "object_shapes": ["box", "rectangular_prism"],
                "requirements": {
                    "gripper_size": "must fit around edge or face",
                    "approach_angle": ["side", "top"],
                    "force_range": {"min": 5, "max": 20, "unit": "N"},
                    "contact_points": 2
                },
                "steps": [
                    "Identify opposing faces or edges",
                    "Position gripper perpendicular to contact surfaces",
                    "Close with firm grip",
                    "Lift vertically"
                ],
                "failure_modes": ["slip_on_smooth_surfaces", "corner_stress_on_fragile_materials"],
                "success_indicators": ["parallel_contact", "no_rotation"]
            },
            {
                "action": "place_on_surface",
                "object_shapes": ["any"],
                "requirements": {
                    "surface_clearance": {"min": 0.05, "unit": "m"},
                    "descent_speed": {"max": 0.1, "unit": "m/s"},
                    "alignment": "vertical"
                },
                "steps": [
                    "Position object above target location",
                    "Align object base with surface normal",
                    "Lower slowly until contact",
                    "Release grasp gently",
                    "Retract gripper vertically"
                ],
                "failure_modes": ["tip_over_if_misaligned", "bounce_if_too_fast", "roll_away_if_unstable_shape"],
                "success_indicators": ["stable_rest", "no_rocking"]
            },
            {
                "action": "pour_liquid",
                "object_shapes": ["container"],
                "requirements": {
                    "grasp_stability": "high",
                    "tilt_angle_control": "precise",
                    "target_container": "present",
                    "force_range": {"min": 5, "max": 10, "unit": "N"}
                },
                "steps": [
                    "Grasp container securely (prefer handle)",
                    "Position spout above target",
                    "Tilt slowly (10-30 degrees)",
                    "Monitor flow rate",
                    "Return to upright when done"
                ],
                "failure_modes": ["spill_if_too_fast", "miss_target_if_misaligned", "drop_if_grasp_fails"],
                "success_indicators": ["controlled_flow", "no_spillage"]
            },
            {
                "action": "push_object",
                "object_shapes": ["any"],
                "requirements": {
                    "contact_point": "center_of_mass",
                    "force_direction": "parallel_to_surface",
                    "force_range": {"min": 1, "max": 10, "unit": "N"}
                },
                "steps": [
                    "Approach object from desired push direction",
                    "Make contact near center of mass",
                    "Apply horizontal force",
                    "Monitor object motion",
                    "Stop when goal reached"
                ],
                "failure_modes": ["tip_over_if_high_contact", "rotate_if_off_center", "slip_if_low_friction"],
                "success_indicators": ["straight_line_motion", "no_tipping"]
            }
        ]
    
    def _create_materials_database(self) -> Dict[str, Dict[str, Any]]:
        """Create database of material properties"""
        return {
            "ceramic": {
                "density": {"value": 2.4, "unit": "g/cm³"},
                "friction_coefficient": 0.6,
                "fragile": True,
                "break_force": {"value": 50, "unit": "N"},
                "thermal_properties": "poor_conductor",
                "notes": "High compressive strength but brittle. Avoid impacts and excessive point loads."
            },
            "plastic": {
                "density": {"value": 1.2, "unit": "g/cm³"},
                "friction_coefficient": 0.4,
                "fragile": False,
                "deformable": True,
                "thermal_properties": "insulator",
                "notes": "Flexible and durable. Can deform under pressure but recovers."
            },
            "wood": {
                "density": {"value": 0.7, "unit": "g/cm³"},
                "friction_coefficient": 0.5,
                "fragile": False,
                "grain_direction": "important",
                "thermal_properties": "poor_conductor",
                "notes": "Anisotropic - stronger along grain. Moisture affects properties."
            },
            "metal": {
                "density": {"value": 7.8, "unit": "g/cm³"},
                "friction_coefficient": 0.7,
                "fragile": False,
                "thermal_properties": "good_conductor",
                "notes": "Heavy and strong. Watch for sharp edges. Can be slippery when polished."
            },
            "organic": {
                "density": {"value": 1.0, "unit": "g/cm³"},
                "friction_coefficient": 0.7,
                "fragile": True,
                "deformable": True,
                "perishable": True,
                "notes": "Soft and easily damaged. Use gentle force. Properties vary with ripeness."
            }
        }
    
    def _create_failure_modes_database(self) -> List[Dict[str, Any]]:
        """Create database of common failure modes and solutions"""
        return [
            {
                "failure": "object_slips_during_grasp",
                "causes": ["insufficient_friction", "wet_surface", "low_grasp_force"],
                "solutions": [
                    "Increase gripper force by 20%",
                    "Change grasp point to higher friction area",
                    "Use enveloping grasp instead of pinch",
                    "Dry surface before grasping"
                ],
                "prevention": "Pre-check surface friction and adjust force"
            },
            {
                "failure": "object_tips_over_when_placed",
                "causes": ["high_center_of_mass", "narrow_base", "uneven_surface", "fast_release"],
                "solutions": [
                    "Reposition for wider base",
                    "Lower more slowly",
                    "Ensure surface is level",
                    "Provide temporary stabilization"
                ],
                "prevention": "Check stability ratio (height/base) < 2.0"
            },
            {
                "failure": "fragile_object_breaks",
                "causes": ["excessive_force", "point_contact", "impact", "squeeze"],
                "solutions": [
                    "Use force < material break_force / 5",
                    "Increase contact area",
                    "Lower descent speed to < 0.05 m/s",
                    "Use compliant gripper padding"
                ],
                "prevention": "Query material fragility before manipulation"
            },
            {
                "failure": "object_rolls_away",
                "causes": ["spherical_shape", "tilted_surface", "momentum"],
                "solutions": [
                    "Place in container or against wall",
                    "Use two-stage placement (lower, stabilize, release)",
                    "Create temporary barrier"
                ],
                "prevention": "Identify rolling hazard for spheres/cylinders"
            },
            {
                "failure": "grasp_fails_on_lift",
                "causes": ["underestimated_weight", "poor_grasp_geometry", "friction_too_low"],
                "solutions": [
                    "Query object weight before attempt",
                    "Use two-handed grasp for heavy objects",
                    "Improve contact geometry",
                    "Increase grasp force proportional to weight"
                ],
                "prevention": "Weight check + grasp force = 2 × (weight × safety_factor)"
            }
        ]
    
    def query_object(self, object_name: str) -> Optional[Dict[str, Any]]:
        """Query physics properties of an object"""
        for obj in self.objects_db:
            if obj["name"].lower() == object_name.lower():
                return obj
        return None
    
    def query_action(self, action_name: str) -> Optional[Dict[str, Any]]:
        """Query requirements for a manipulation action"""
        for action in self.actions_db:
            if action["action"].lower() == action_name.lower():
                return action
        return None
    
    def query_material(self, material_name: str) -> Optional[Dict[str, Any]]:
        """Query properties of a material"""
        return self.materials_db.get(material_name.lower())
    
    def get_grasp_strategy(self, object_name: str) -> str:
        """Get recommended grasp strategy for an object"""
        obj = self.query_object(object_name)
        if not obj:
            return f"Unknown object: {object_name}"
        
        shape = obj["shape"]
        preferred = obj.get("preferred_grasp", "standard")
        
        # Find matching action
        action_name = f"grasp_{shape.split('_')[0]}"  # e.g., grasp_cylinder
        action = self.query_action(action_name)
        
        strategy = f"Grasp strategy for {object_name}:\n"
        strategy += f"  Shape: {shape}\n"
        strategy += f"  Material: {obj['material']}\n"
        strategy += f"  Preferred grasp: {preferred}\n"
        
        if action:
            strategy += f"\nRequirements:\n"
            for key, value in action["requirements"].items():
                strategy += f"  - {key}: {value}\n"
            
            strategy += f"\nSteps:\n"
            for i, step in enumerate(action["steps"], 1):
                strategy += f"  {i}. {step}\n"
        
        strategy += f"\nNotes: {obj.get('manipulation_notes', 'No special notes')}\n"
        
        return strategy
    
    def predict_failure_risk(self, object_name: str, action_name: str, 
                            conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict risk of failure for a planned action
        
        Args:
            object_name: Name of object
            action_name: Action to perform
            conditions: Current conditions (force, speed, etc.)
        
        Returns:
            Risk assessment dictionary
        """
        obj = self.query_object(object_name)
        action = self.query_action(action_name)
        
        if not obj or not action:
            return {"risk": "unknown", "reason": "object or action not in database"}
        
        risks = []
        
        # Check fragility
        if obj.get("fragile") and conditions.get("force", 0) > 10:
            risks.append({
                "type": "breakage",
                "severity": "high",
                "reason": "Fragile object + high force"
            })
        
        # Check stability
        if "place" in action_name and obj.get("stability", {}).get("rolls"):
            risks.append({
                "type": "rolling",
                "severity": "medium",
                "reason": "Object shape prone to rolling"
            })
        
        # Check weight
        if "grasp" in action_name:
            weight = obj["typical_weight"]["max"]
            force = conditions.get("force", 0)
            if force < weight * 9.81 * 2:  # Need 2x weight for secure grasp
                risks.append({
                    "type": "grasp_failure",
                    "severity": "high",
                    "reason": f"Insufficient force: {force}N < {weight * 9.81 * 2}N"
                })
        
        overall_risk = "low"
        if any(r["severity"] == "high" for r in risks):
            overall_risk = "high"
        elif any(r["severity"] == "medium" for r in risks):
            overall_risk = "medium"
        
        return {
            "risk_level": overall_risk,
            "identified_risks": risks,
            "recommendations": self._get_risk_mitigation(risks)
        }
    
    def _get_risk_mitigation(self, risks: List[Dict]) -> List[str]:
        """Generate risk mitigation recommendations"""
        recommendations = []
        
        for risk in risks:
            if risk["type"] == "breakage":
                recommendations.append("Reduce force to 5N or less")
                recommendations.append("Increase contact area")
            elif risk["type"] == "rolling":
                recommendations.append("Place object in container or against wall")
                recommendations.append("Use two-phase placement with stabilization")
            elif risk["type"] == "grasp_failure":
                recommendations.append("Increase grasp force by 50%")
                recommendations.append("Use two-handed grasp if possible")
        
        return recommendations
    
    def export_to_text(self) -> str:
        """Export knowledge base to text format for LLM context"""
        text = "=== PHYSICS KNOWLEDGE BASE ===\n\n"
        
        text += "## OBJECTS ##\n"
        for obj in self.objects_db:
            text += f"\n{obj['name'].upper()}:\n"
            text += f"  Type: {obj['type']}\n"
            text += f"  Shape: {obj['shape']}\n"
            text += f"  Material: {obj['material']}\n"
            text += f"  Fragile: {obj['fragile']}\n"
            text += f"  Notes: {obj.get('manipulation_notes', 'N/A')}\n"
        
        text += "\n## ACTIONS ##\n"
        for action in self.actions_db:
            text += f"\n{action['action'].upper()}:\n"
            text += f"  Applies to: {action['object_shapes']}\n"
            text += f"  Steps: {', '.join(action['steps'])}\n"
            text += f"  Failure modes: {', '.join(action['failure_modes'])}\n"
        
        return text


# Example usage
if __name__ == "__main__":
    kb = PhysicsKnowledgeBase()
    
    print("=== Testing Physics Knowledge Base ===\n")
    
    # Query object
    print("1. Query coffee mug:")
    mug = kb.query_object("coffee_mug")
    print(f"   Fragile: {mug['fragile']}")
    print(f"   Preferred grasp: {mug['preferred_grasp']}")
    print(f"   Notes: {mug['manipulation_notes']}\n")
    
    # Get grasp strategy
    print("2. Grasp strategy for coffee mug:")
    strategy = kb.get_grasp_strategy("coffee_mug")
    print(strategy)
    
    # Predict failure risk
    print("\n3. Failure risk prediction:")
    risk = kb.predict_failure_risk(
        "coffee_mug",
        "grasp_cylinder",
        {"force": 20}  # Too much force!
    )
    print(f"   Risk level: {risk['risk_level']}")
    print(f"   Recommendations: {risk['recommendations']}")
    
    # Export for LLM
    print("\n4. Exporting to text format...")
    text_export = kb.export_to_text()
    print(f"   Exported {len(text_export)} characters")
