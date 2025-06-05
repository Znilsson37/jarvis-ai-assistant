"""3D Scene Understanding Module"""
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
import logging
from modules.vision import vision_system
from modules.ai_vision import ai_vision_system

logger = logging.getLogger(__name__)

class Scene3DAnalyzer:
    def __init__(self):
        self.depth_scale = 1.0  # Meters per unit
        self.camera_matrix = None
        self.dist_coeffs = None
        self.initialize_camera_params()

    def initialize_camera_params(self):
        """Initialize camera intrinsic parameters"""
        # Default parameters for 1920x1080 camera
        fx = 1000  # Focal length x
        fy = 1000  # Focal length y
        cx = 960   # Principal point x
        cy = 540   # Principal point y
        
        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Assuming no distortion
        self.dist_coeffs = np.zeros(5)

    def analyze_scene(self, frame: np.ndarray) -> Dict:
        """Perform comprehensive 3D scene analysis"""
        try:
            # Get depth map
            depth_map = vision_system.get_depth_map(frame)
            if depth_map.size == 0:
                return {"error": "Could not generate depth map"}

            # Get AI vision results
            ai_results = ai_vision_system.get_latest_results()
            if not ai_results:
                return {"error": "No AI vision results available"}

            # Get structural analysis
            wireframe = vision_system._generate_wireframe(frame)

            # Combine all data for 3D scene understanding
            scene_3d = self._reconstruct_3d_scene(
                frame, depth_map, wireframe, ai_results
            )

            return {
                "status": "success",
                "scene_3d": scene_3d,
                "depth_map": depth_map,
                "structural_analysis": wireframe
            }

        except Exception as e:
            logger.error(f"Error in scene analysis: {e}")
            return {"error": str(e)}

    def _reconstruct_3d_scene(
        self, 
        frame: np.ndarray, 
        depth_map: np.ndarray,
        wireframe: np.ndarray,
        ai_results: Dict
    ) -> Dict:
        """Reconstruct 3D scene from multiple data sources"""
        scene = {
            "objects": [],
            "structure": {
                "planes": [],
                "edges": []
            },
            "spatial_relations": []
        }

        # Process detected objects
        if "objects" in ai_results:
            for obj in ai_results["objects"]:
                obj_3d = self._get_object_3d_position(
                    obj["bbox"], 
                    depth_map
                )
                if obj_3d:
                    scene["objects"].append({
                        "class": obj["class"],
                        "confidence": obj["confidence"],
                        "position_3d": obj_3d,
                        "bbox": obj["bbox"]
                    })

        # Process structural elements
        edges = self._extract_structural_edges(wireframe)
        scene["structure"]["edges"] = edges

        # Find major planes using depth map
        planes = self._detect_planes(depth_map)
        scene["structure"]["planes"] = planes

        # Analyze spatial relationships
        scene["spatial_relations"] = self._analyze_spatial_relations(
            scene["objects"],
            scene["structure"]
        )

        return scene

    def _get_object_3d_position(
        self, 
        bbox: List[int], 
        depth_map: np.ndarray
    ) -> Optional[Dict]:
        """Calculate 3D position of object using depth map"""
        try:
            x1, y1, x2, y2 = bbox
            # Get depth in object region
            obj_depth = depth_map[y1:y2, x1:x2]
            if obj_depth.size == 0:
                return None

            # Use median depth for robustness
            depth = np.median(obj_depth) * self.depth_scale

            # Calculate 3D coordinates
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Convert to 3D coordinates
            X = (center_x - self.camera_matrix[0, 2]) * depth / self.camera_matrix[0, 0]
            Y = (center_y - self.camera_matrix[1, 2]) * depth / self.camera_matrix[1, 1]
            Z = depth

            return {
                "x": float(X),
                "y": float(Y),
                "z": float(Z),
                "confidence": float(np.mean(obj_depth > 0))
            }

        except Exception as e:
            logger.error(f"Error calculating 3D position: {e}")
            return None

    def _extract_structural_edges(self, wireframe: np.ndarray) -> List[Dict]:
        """Extract structural edges from wireframe"""
        edges = []
        try:
            # Convert to grayscale if needed
            if len(wireframe.shape) == 3:
                gray = cv2.cvtColor(wireframe, cv2.COLOR_BGR2GRAY)
            else:
                gray = wireframe

            # Find lines using probabilistic Hough transform
            lines = cv2.HoughLinesP(
                gray, 1, np.pi/180, 50, 
                minLineLength=100, maxLineGap=10
            )

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    edges.append({
                        "start": {"x": int(x1), "y": int(y1)},
                        "end": {"x": int(x2), "y": int(y2)},
                        "length": float(np.sqrt((x2-x1)**2 + (y2-y1)**2))
                    })

        except Exception as e:
            logger.error(f"Error extracting structural edges: {e}")

        return edges

    def _detect_planes(self, depth_map: np.ndarray) -> List[Dict]:
        """Detect major planes in the scene using depth map"""
        planes = []
        try:
            # Simple plane detection using depth discontinuities
            depth_grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
            depth_grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
            
            # Find regions with consistent depth
            grad_mag = np.sqrt(depth_grad_x**2 + depth_grad_y**2)
            plane_mask = grad_mag < np.mean(grad_mag)
            
            # Label connected components
            num_labels, labels = cv2.connectedComponents(plane_mask.astype(np.uint8))
            
            for label in range(1, num_labels):
                mask = labels == label
                if np.sum(mask) > 1000:  # Min plane size
                    mean_depth = np.mean(depth_map[mask])
                    planes.append({
                        "depth": float(mean_depth),
                        "area": float(np.sum(mask)),
                        "center": {
                            "x": float(np.mean(np.where(mask)[1])),
                            "y": float(np.mean(np.where(mask)[0]))
                        }
                    })

        except Exception as e:
            logger.error(f"Error detecting planes: {e}")

        return planes

    def _analyze_spatial_relations(
        self, 
        objects: List[Dict],
        structure: Dict
    ) -> List[Dict]:
        """Analyze spatial relationships between objects and structure"""
        relations = []
        try:
            # Analyze object-object relations
            for i, obj1 in enumerate(objects):
                for obj2 in objects[i+1:]:
                    relation = self._get_spatial_relation(obj1, obj2)
                    if relation:
                        relations.append(relation)

            # Analyze object-plane relations
            for obj in objects:
                for plane in structure["planes"]:
                    if abs(obj["position_3d"]["z"] - plane["depth"]) < 0.5:
                        relations.append({
                            "type": "on",
                            "object": obj["class"],
                            "surface": "plane",
                            "confidence": min(obj["confidence"], 
                                           obj["position_3d"]["confidence"])
                        })

        except Exception as e:
            logger.error(f"Error analyzing spatial relations: {e}")

        return relations

    def _get_spatial_relation(
        self, 
        obj1: Dict,
        obj2: Dict
    ) -> Optional[Dict]:
        """Determine spatial relation between two objects"""
        try:
            # Calculate relative position
            dx = obj1["position_3d"]["x"] - obj2["position_3d"]["x"]
            dy = obj1["position_3d"]["y"] - obj2["position_3d"]["y"]
            dz = obj1["position_3d"]["z"] - obj2["position_3d"]["z"]
            
            # Determine primary relation
            if abs(dz) > max(abs(dx), abs(dy)):
                relation = "in_front_of" if dz < 0 else "behind"
            elif abs(dy) > abs(dx):
                relation = "above" if dy < 0 else "below"
            else:
                relation = "left_of" if dx < 0 else "right_of"
            
            return {
                "type": relation,
                "object1": obj1["class"],
                "object2": obj2["class"],
                "confidence": min(obj1["confidence"], 
                                obj2["confidence"],
                                obj1["position_3d"]["confidence"],
                                obj2["position_3d"]["confidence"])
            }

        except Exception as e:
            logger.error(f"Error getting spatial relation: {e}")
            return None

# Create global instance
scene_3d_analyzer = Scene3DAnalyzer()
