import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch
from modules.scene_3d import scene_3d_analyzer
from modules.vision import vision_system
from modules.ai_vision import ai_vision_system

class TestScene3DAnalyzer:
    @classmethod
    def setup_class(cls):
        """Set up test environment"""
        # Create mock frame and depth data
        cls.mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cls.mock_depth_map = np.ones((480, 640), dtype=np.uint8) * 128
        
        # Add some depth variations for plane detection
        cls.mock_depth_map[100:200, 100:200] = 64  # Closer plane
        cls.mock_depth_map[300:400, 300:400] = 192  # Further plane
        
        # Create mock wireframe
        cls.mock_wireframe = np.zeros((480, 640), dtype=np.uint8)
        # Add some lines for edge detection
        cv2.line(cls.mock_wireframe, (100, 100), (200, 100), 255, 2)
        cv2.line(cls.mock_wireframe, (100, 100), (100, 200), 255, 2)
        
        # Mock AI vision results
        cls.mock_ai_results = {
            "objects": [
                {
                    "class": "person",
                    "confidence": 0.95,
                    "bbox": [50, 50, 150, 250]
                },
                {
                    "class": "chair",
                    "confidence": 0.85,
                    "bbox": [300, 200, 400, 350]
                }
            ]
        }
    
    def setup_method(self):
        """Set up each test method"""
        # Mock vision system methods
        vision_system.get_depth_map = MagicMock(return_value=self.mock_depth_map)
        vision_system._generate_wireframe = MagicMock(return_value=self.mock_wireframe)
        
        # Mock AI vision system methods
        ai_vision_system.get_latest_results = MagicMock(return_value=self.mock_ai_results)
    
    def test_scene_analysis(self):
        """Test complete scene analysis"""
        result = scene_3d_analyzer.analyze_scene(self.mock_frame)
        
        assert result["status"] == "success"
        assert "scene_3d" in result
        assert "depth_map" in result
        assert "structural_analysis" in result
        
        scene_3d = result["scene_3d"]
        assert "objects" in scene_3d
        assert "structure" in scene_3d
        assert "spatial_relations" in scene_3d
        
        # Verify objects were processed
        assert len(scene_3d["objects"]) == 2
        assert scene_3d["objects"][0]["class"] == "person"
        assert scene_3d["objects"][1]["class"] == "chair"
        
        # Verify structural analysis
        assert len(scene_3d["structure"]["edges"]) > 0
        assert len(scene_3d["structure"]["planes"]) > 0
    
    def test_object_3d_position(self):
        """Test 3D position calculation"""
        bbox = [100, 100, 200, 200]
        position = scene_3d_analyzer._get_object_3d_position(bbox, self.mock_depth_map)
        
        assert position is not None
        assert "x" in position
        assert "y" in position
        assert "z" in position
        assert "confidence" in position
        assert 0 <= position["confidence"] <= 1
    
    def test_structural_edge_detection(self):
        """Test structural edge detection"""
        edges = scene_3d_analyzer._extract_structural_edges(self.mock_wireframe)
        
        assert len(edges) > 0
        for edge in edges:
            assert "start" in edge
            assert "end" in edge
            assert "length" in edge
            assert edge["length"] > 0
    
    def test_plane_detection(self):
        """Test plane detection"""
        planes = scene_3d_analyzer._detect_planes(self.mock_depth_map)
        
        assert len(planes) > 0
        for plane in planes:
            assert "depth" in plane
            assert "area" in plane
            assert "center" in plane
            assert plane["area"] > 0
    
    def test_spatial_relations(self):
        """Test spatial relationship analysis"""
        # Create test objects with known positions
        objects = [
            {
                "class": "person",
                "confidence": 0.9,
                "position_3d": {"x": 0, "y": 0, "z": 1, "confidence": 0.8},
                "bbox": [0, 0, 100, 200]
            },
            {
                "class": "chair",
                "confidence": 0.8,
                "position_3d": {"x": 0, "y": 0, "z": 2, "confidence": 0.7},
                "bbox": [100, 100, 200, 300]
            }
        ]
        
        structure = {
            "planes": [
                {"depth": 1.0, "area": 1000, "center": {"x": 100, "y": 100}}
            ]
        }
        
        relations = scene_3d_analyzer._analyze_spatial_relations(objects, structure)
        
        assert len(relations) > 0
        for relation in relations:
            assert "type" in relation
            assert "confidence" in relation
            assert 0 <= relation["confidence"] <= 1
    
    def test_error_handling(self):
        """Test error handling"""
        # Test with invalid depth map
        vision_system.get_depth_map = MagicMock(return_value=np.array([]))
        result = scene_3d_analyzer.analyze_scene(self.mock_frame)
        assert "error" in result
        
        # Test with missing AI results
        ai_vision_system.get_latest_results = MagicMock(return_value=None)
        result = scene_3d_analyzer.analyze_scene(self.mock_frame)
        assert "error" in result
    
    def teardown_method(self):
        """Clean up after each test"""
        vision_system.cleanup()

if __name__ == "__main__":
    pytest.main([__file__])
