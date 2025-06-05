import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch
from modules.material_recognition import material_recognizer
from modules.vision import vision_system

class TestMaterialRecognition:
    @classmethod
    def setup_class(cls):
        """Set up test environment"""
        # Create test images for different materials
        cls.test_size = (200, 200, 3)
        
        # Wood texture (brownish with grain pattern)
        cls.wood_image = np.zeros(cls.test_size, dtype=np.uint8)
        cls.wood_image[:] = [102, 51, 0]  # Brown color
        # Add grain pattern
        for i in range(0, 200, 10):
            cv2.line(cls.wood_image, (0, i), (200, i), [120, 60, 0], 2)
        
        # Metal texture (gray with high reflectivity)
        cls.metal_image = np.ones(cls.test_size, dtype=np.uint8) * 192
        # Add reflective highlights
        cv2.circle(cls.metal_image, (100, 100), 50, [255, 255, 255], -1)
        
        # Fabric texture (regular pattern)
        cls.fabric_image = np.zeros(cls.test_size, dtype=np.uint8)
        for i in range(0, 200, 5):
            for j in range(0, 200, 5):
                cls.fabric_image[i:i+3, j:j+3] = [150, 150, 150]
        
        # Mock spectral analysis results
        cls.mock_spectral = {
            "magnitude_spectrum": np.random.rand(200, 200),
            "dominant_frequencies": [1.0, 0.8, 0.6, 0.4, 0.2]
        }
    
    def setup_method(self):
        """Set up each test method"""
        # Mock vision system's spectral analysis
        vision_system._spectral_analysis = MagicMock(return_value=self.mock_spectral)
    
    def test_feature_extraction(self):
        """Test feature extraction from different materials"""
        # Test wood features
        features = material_recognizer._extract_features(self.wood_image, self.mock_spectral)
        assert "frequency_response" in features
        assert "texture_entropy" in features
        assert "reflectivity" in features
        assert "color_properties" in features
        
        # Verify color properties for wood
        color_props = features["color_properties"]
        assert "hue" in color_props
        assert "saturation" in color_props
        assert "color_consistency" in color_props
        assert 0 <= color_props["color_consistency"] <= 1
        
        # Test metal features
        features = material_recognizer._extract_features(self.metal_image, self.mock_spectral)
        assert features["reflectivity"] > 0.5  # Metal should have high reflectivity
    
    def test_texture_entropy(self):
        """Test texture entropy calculation"""
        # Calculate entropy for different textures
        wood_entropy = material_recognizer._calculate_texture_entropy(self.wood_image)
        metal_entropy = material_recognizer._calculate_texture_entropy(self.metal_image)
        fabric_entropy = material_recognizer._calculate_texture_entropy(self.fabric_image)
        
        # Wood should have medium entropy due to grain pattern
        assert 3.0 < wood_entropy < 7.0
        
        # Metal should have low entropy (smooth surface)
        assert metal_entropy < wood_entropy
        
        # Fabric should have high entropy due to regular pattern
        assert fabric_entropy > metal_entropy
    
    def test_material_classification(self):
        """Test material classification"""
        # Test wood classification
        result = material_recognizer.analyze_material(self.wood_image)
        assert "material" in result
        assert "confidence" in result
        assert "properties" in result
        assert 0 <= result["confidence"] <= 1
        
        # Test metal classification
        result = material_recognizer.analyze_material(self.metal_image)
        assert "material" in result
        assert result["properties"]["reflectivity"] > 0.5
        
        # Test with invalid input
        result = material_recognizer.analyze_material(np.zeros((1, 1, 3)))
        assert "error" in result
    
    def test_frequency_comparison(self):
        """Test frequency response comparison"""
        freq1 = [0.8, 0.6, 0.4, 0.2]
        freq2 = [0.7, 0.5, 0.3, 0.1]
        
        similarity = material_recognizer._compare_frequency_response(freq1, freq2)
        assert 0 <= similarity <= 1
        
        # Same frequencies should have perfect similarity
        perfect_similarity = material_recognizer._compare_frequency_response(freq1, freq1)
        assert perfect_similarity > 0.99
        
        # Test with empty input
        zero_similarity = material_recognizer._compare_frequency_response([], [])
        assert zero_similarity == 0
    
    def test_color_comparison(self):
        """Test color properties comparison"""
        properties = {
            "hue": 20,
            "saturation": 0.5,
            "color_consistency": 0.8
        }
        
        # Test with matching range
        ranges = {
            "hue": (0, 30),
            "saturation": (0.3, 0.7)
        }
        match_score = material_recognizer._compare_color_properties(properties, ranges)
        assert match_score > 0.7
        
        # Test with non-matching range
        ranges = {
            "hue": (180, 200),
            "saturation": (0.8, 1.0)
        }
        no_match_score = material_recognizer._compare_color_properties(properties, ranges)
        assert no_match_score < 0.3
    
    def test_error_handling(self):
        """Test error handling"""
        # Test with None input
        result = material_recognizer.analyze_material(None)
        assert "error" in result
        
        # Test with invalid image shape
        invalid_image = np.zeros((10, 10))  # Missing color channels
        result = material_recognizer.analyze_material(invalid_image)
        assert "error" in result
        
        # Test with corrupted spectral data
        vision_system._spectral_analysis = MagicMock(return_value={"error": "Failed"})
        result = material_recognizer.analyze_material(self.wood_image)
        assert "error" in result
    
    def teardown_method(self):
        """Clean up after each test"""
        pass

if __name__ == "__main__":
    pytest.main([__file__])
