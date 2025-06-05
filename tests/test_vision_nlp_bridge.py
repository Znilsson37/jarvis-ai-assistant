import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from modules.vision_nlp_bridge import VisionNLPBridge
from modules.brain import BrainManager

@pytest.fixture
def bridge():
    brain = BrainManager(storage_path="test_storage")
    return VisionNLPBridge(brain)

def test_material_recognition_no_frame(bridge):
    with patch('modules.vision.vision_system.frame_buffer', []):
        intent_result = {"intent": "material_recognition"}
        response = bridge.handle_vision_intent(intent_result)
        assert response["status"] == "error"
        assert "No frames available" in response["message"]

def test_material_recognition_success(bridge):
    mock_frame = MagicMock()
    mock_result = {
        "material": "wood",
        "confidence": 0.85,
        "properties": {
            "reflectivity": 0.6,
            "texture_complexity": 5.0,
            "color_properties": {"hue": 20, "saturation": 0.5, "color_consistency": 0.9}
        }
    }
    with patch('modules.vision.vision_system.frame_buffer', [mock_frame]), \
         patch('modules.material_recognition.material_recognizer.analyze_material', return_value=mock_result):
        intent_result = {"intent": "material_recognition"}
        response = bridge.handle_vision_intent(intent_result)
        assert response["status"] == "success"
        assert "material_recognition" in response
        assert response["material_recognition"]["material"] == "wood"
        assert "Material recognized" in response["description"]

def test_material_recognition_error(bridge):
    mock_frame = MagicMock()
    mock_error = {"error": "Failed to analyze material"}
    with patch('modules.vision.vision_system.frame_buffer', [mock_frame]), \
         patch('modules.material_recognition.material_recognizer.analyze_material', return_value=mock_error):
        intent_result = {"intent": "material_recognition"}
        response = bridge.handle_vision_intent(intent_result)
        assert response["status"] == "error"
        assert "Failed to analyze material" in response["message"]

def test_object_detection(bridge):
    """Test object detection integration"""
    mock_frame = np.random.rand(100, 100, 3)
    mock_objects = [
        {"label": "person", "confidence": 0.95, "bbox": [10, 10, 50, 50]},
        {"label": "chair", "confidence": 0.85, "bbox": [60, 60, 90, 90]}
    ]
    
    with patch('modules.vision.vision_system.frame_buffer', [mock_frame]), \
         patch('modules.vision.VisionProcessor.detect_objects', return_value=mock_objects):
        intent_result = {"intent": "object_detection"}
        response = bridge.handle_vision_intent(intent_result)
        assert response["status"] == "success"
        assert "objects" in response
        assert len(response["objects"]) == 2
        assert response["objects"][0]["label"] == "person"

def test_scene_understanding(bridge):
    """Test scene understanding integration"""
    mock_frame = np.random.rand(100, 100, 3)
    mock_scene = {
        "scene_type": "indoor",
        "environment": "office",
        "lighting": "bright",
        "objects_present": ["desk", "computer", "chair"],
        "spatial_relations": [
            {"subject": "computer", "relation": "on", "object": "desk"}
        ]
    }
    
    with patch('modules.vision.vision_system.frame_buffer', [mock_frame]), \
         patch('modules.vision.VisionProcessor.analyze_scene', return_value=mock_scene):
        intent_result = {"intent": "scene_understanding"}
        response = bridge.handle_vision_intent(intent_result)
        assert response["status"] == "success"
        assert "scene_analysis" in response
        assert response["scene_analysis"]["scene_type"] == "indoor"
        assert "desk" in response["scene_analysis"]["objects_present"]

def test_text_recognition(bridge):
    """Test OCR/text recognition integration"""
    mock_frame = np.random.rand(100, 100, 3)
    mock_text = {
        "text": "Hello World",
        "confidence": 0.98,
        "bounding_boxes": [[10, 10, 50, 30]],
        "language": "en"
    }
    
    with patch('modules.vision.vision_system.frame_buffer', [mock_frame]), \
         patch('modules.vision.VisionProcessor.recognize_text', return_value=mock_text):
        intent_result = {"intent": "text_recognition"}
        response = bridge.handle_vision_intent(intent_result)
        assert response["status"] == "success"
        assert "text_recognition" in response
        assert response["text_recognition"]["text"] == "Hello World"

def test_face_analysis(bridge):
    """Test face detection and analysis integration"""
    mock_frame = np.random.rand(100, 100, 3)
    mock_faces = [{
        "bbox": [20, 20, 60, 60],
        "confidence": 0.92,
        "landmarks": {"eyes": [[30, 30], [50, 30]], "nose": [40, 40]},
        "emotions": {"happy": 0.8, "neutral": 0.2},
        "age_estimation": 25,
        "gender": "female"
    }]
    
    with patch('modules.vision.vision_system.frame_buffer', [mock_frame]), \
         patch('modules.vision.VisionProcessor.analyze_faces', return_value=mock_faces):
        intent_result = {"intent": "face_analysis"}
        response = bridge.handle_vision_intent(intent_result)
        assert response["status"] == "success"
        assert "faces" in response
        assert len(response["faces"]) == 1
        assert response["faces"][0]["emotions"]["happy"] == 0.8

def test_integration_with_brain(bridge):
    """Test full integration with brain module"""
    mock_frame = np.random.rand(100, 100, 3)
    mock_scene = {
        "scene_type": "indoor",
        "objects_present": ["desk", "computer"]
    }
    
    with patch('modules.vision.vision_system.frame_buffer', [mock_frame]), \
         patch('modules.vision.VisionProcessor.analyze_scene', return_value=mock_scene):
        # Process visual input
        response = bridge.process_visual_input("What objects are in the room?")
        
        # Verify brain integration
        assert response is not None
        assert "scene_type" in str(response).lower()
        assert "desk" in str(response).lower()
        
        # Verify memory storage
        results = bridge.brain.search_memory("scene_analysis")
        assert len(results) > 0
        assert any("indoor" in str(value) for _, value in results)

def test_error_handling(bridge):
    """Test error handling in vision-nlp bridge"""
    # Test with invalid intent
    response = bridge.handle_vision_intent({"intent": "invalid_intent"})
    assert response["status"] == "error"
    assert "Unsupported intent" in response["message"]
    
    # Test with missing frame
    with patch('modules.vision.vision_system.frame_buffer', []):
        response = bridge.process_visual_input("What's in the image?")
        assert "error" in response.lower()
        assert "no frame" in response.lower()
    
    # Test with processing error
    with patch('modules.vision.VisionProcessor.analyze_scene', side_effect=Exception("Processing error")):
        response = bridge.handle_vision_intent({"intent": "scene_understanding"})
        assert response["status"] == "error"
        assert "Processing error" in response["message"]

def test_concurrent_processing(bridge):
    """Test concurrent processing of multiple frames"""
    mock_frames = [np.random.rand(100, 100, 3) for _ in range(3)]
    mock_results = [
        {"scene_type": "indoor", "objects_present": ["desk"]},
        {"scene_type": "outdoor", "objects_present": ["tree"]},
        {"scene_type": "indoor", "objects_present": ["chair"]}
    ]
    
    with patch('modules.vision.vision_system.frame_buffer', mock_frames), \
         patch('modules.vision.VisionProcessor.analyze_scene', side_effect=mock_results):
        responses = []
        for _ in range(3):
            response = bridge.handle_vision_intent({"intent": "scene_understanding"})
            responses.append(response)
        
        assert all(r["status"] == "success" for r in responses)
        assert len(responses) == 3
        assert any("desk" in str(r) for r in responses)
        assert any("tree" in str(r) for r in responses)
        assert any("chair" in str(r) for r in responses)
