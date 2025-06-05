"""Material Recognition Module using Spectral Analysis"""
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
import logging
from scipy.stats import entropy
from modules.vision import vision_system

logger = logging.getLogger(__name__)

class MaterialRecognizer:
    def __init__(self):
        # Material spectral signatures database
        # Values are normalized frequency responses for different materials
        self.material_signatures = {
            "wood": {
                "frequency_response": [0.8, 0.6, 0.4, 0.3, 0.2],
                "texture_entropy": (4.5, 6.5),
                "color_range": {
                    "hue": (10, 30),
                    "saturation": (0.3, 0.8)
                }
            },
            "metal": {
                "frequency_response": [0.9, 0.8, 0.7, 0.6, 0.5],
                "texture_entropy": (2.0, 4.0),
                "color_range": {
                    "hue": (0, 360),  # Any hue
                    "saturation": (0.0, 0.3)
                }
            },
            "fabric": {
                "frequency_response": [0.5, 0.4, 0.3, 0.2, 0.1],
                "texture_entropy": (5.5, 7.5),
                "color_range": {
                    "hue": (0, 360),  # Any hue
                    "saturation": (0.2, 1.0)
                }
            },
            "plastic": {
                "frequency_response": [0.7, 0.5, 0.3, 0.2, 0.1],
                "texture_entropy": (3.0, 5.0),
                "color_range": {
                    "hue": (0, 360),  # Any hue
                    "saturation": (0.4, 1.0)
                }
            },
            "glass": {
                "frequency_response": [0.95, 0.9, 0.85, 0.8, 0.75],
                "texture_entropy": (1.0, 3.0),
                "color_range": {
                    "hue": (0, 360),  # Any hue
                    "saturation": (0.0, 0.2)
                }
            }
        }

    def analyze_material(self, region: np.ndarray) -> Dict:
        """Analyze material properties of a region"""
        try:
            # Input validation
            if region is None or not isinstance(region, np.ndarray):
                return {"error": "Invalid input: region must be a numpy array"}
            
            if len(region.shape) != 3:
                return {"error": "Invalid input: region must be a 3-channel image"}
                
            if region.shape[0] < 2 or region.shape[1] < 2:
                return {"error": "Invalid input: image too small (minimum 2x2)"}
            
            # Convert to uint8 if needed
            if region.dtype != np.uint8:
                if region.max() <= 1.0:  # Normalized values
                    region = (region * 255).astype(np.uint8)
                else:
                    region = region.astype(np.uint8)
            
            # Get spectral analysis
            spectral = vision_system._spectral_analysis(region)
            if "error" in spectral:
                return {"error": "Failed to perform spectral analysis"}

            # Extract features
            features = self._extract_features(region, spectral)
            if not features:
                return {"error": "Failed to extract features"}

            # Classify material
            material = self._classify_material(features)
            if material["class"] == "unknown":
                return {"error": "Could not classify material"}

            return {
                "material": material["class"],
                "confidence": material["confidence"],
                "properties": {
                    "reflectivity": features.get("reflectivity", 0.0),
                    "texture_complexity": features.get("texture_entropy", 0.0),
                    "color_properties": features.get("color_properties", {})
                }
            }

        except Exception as e:
            logger.error(f"Error in material analysis: {e}")
            return {"error": str(e)}

    def _extract_features(
        self, 
        region: np.ndarray, 
        spectral: Dict
    ) -> Dict:
        """Extract material features from region"""
        features = {}

        try:
            # Normalize frequency components
            freq_response = self._normalize_frequencies(
                spectral["dominant_frequencies"]
            )
            features["frequency_response"] = freq_response

            # Calculate texture entropy
            features["texture_entropy"] = self._calculate_texture_entropy(region)

            # Calculate reflectivity from magnitude spectrum
            features["reflectivity"] = self._calculate_reflectivity(
                spectral["magnitude_spectrum"]
            )

            # Extract color properties
            features["color_properties"] = self._extract_color_properties(region)

            return features

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {}

    def _normalize_frequencies(self, frequencies: List[float]) -> List[float]:
        """Normalize frequency components"""
        if not frequencies:
            return []
        freq_array = np.array(frequencies)
        return (freq_array / np.max(freq_array)).tolist()

    def _calculate_texture_entropy(self, region: np.ndarray) -> float:
        """Calculate texture entropy using GLCM"""
        try:
            # Ensure uint8 type
            if region.dtype != np.uint8:
                region = (region * 255).astype(np.uint8)
            
            # Convert to grayscale
            if len(region.shape) == 3:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray = region.copy()

            # Calculate GLCM
            glcm = self._get_glcm(gray)
            
            # Calculate entropy
            if glcm.size > 0:
                return float(entropy(glcm.flatten()))
            return 0.0

        except Exception as e:
            logger.error(f"Error calculating texture entropy: {e}")
            return 0.0

    def _get_glcm(self, gray: np.ndarray) -> np.ndarray:
        """Calculate Gray Level Co-occurrence Matrix with multiple directions"""
        try:
            # Ensure minimum size
            if gray.shape[0] < 2 or gray.shape[1] < 2:
                return np.zeros((32, 32))

            # Quantize to 32 levels for better discrimination
            levels = 32
            max_val = np.max(gray)
            if max_val == 0:
                max_val = 1
            gray_scaled = ((gray / max_val) * (levels - 1)).astype(np.uint8)
            
            # Initialize GLCM for multiple directions
            glcm = np.zeros((levels, levels))
            h, w = gray_scaled.shape
            
            # Define offsets for different directions and distances
            offsets = []
            distances = [1, 2]  # Consider both immediate and skip-one neighbors
            angles = [0, 45, 90, 135]  # degrees
            
            for d in distances:
                for angle in angles:
                    rad = np.radians(angle)
                    di = int(round(-d * np.sin(rad)))
                    dj = int(round(d * np.cos(rad)))
                    offsets.append((di, dj))
            
            # Calculate GLCM for each direction and distance
            for di, dj in offsets:
                temp_glcm = np.zeros((levels, levels))
                
                # Define valid range based on offset
                i_start = max(0, -di)
                i_end = min(h, h - di)
                j_start = max(0, -dj)
                j_end = min(w, w - dj)
                
                # Calculate co-occurrence with weights based on distance
                weight = 1.0 / np.sqrt(di*di + dj*dj) if di != 0 or dj != 0 else 1.0
                for i in range(i_start, i_end):
                    for j in range(j_start, j_end):
                        i2, j2 = i + di, j + dj
                        if 0 <= i2 < h and 0 <= j2 < w:
                            temp_glcm[gray_scaled[i,j], gray_scaled[i2,j2]] += weight
                            # Add symmetric counterpart
                            temp_glcm[gray_scaled[i2,j2], gray_scaled[i,j]] += weight
                
                # Normalize and add to total GLCM
                total = temp_glcm.sum()
                if total > 0:
                    glcm += temp_glcm / total
            
            # Average over all directions and distances
            glcm /= len(offsets)
            
            # Apply Gaussian smoothing with larger kernel
            glcm = cv2.GaussianBlur(glcm, (5, 5), 0)
            
            # Apply contrast enhancement
            glcm = cv2.normalize(glcm, None, 0, 1, cv2.NORM_MINMAX)
            
            # Add small epsilon to avoid log(0) in entropy calculation
            glcm += 1e-10
            
            return glcm

        except Exception as e:
            logger.error(f"Error calculating GLCM: {e}")
            return np.zeros((32, 32))

    def _calculate_reflectivity(self, magnitude_spectrum: np.ndarray) -> float:
        """Calculate material reflectivity from spectral magnitude"""
        try:
            h, w = magnitude_spectrum.shape
            
            # Create frequency mask emphasizing high frequencies
            y, x = np.ogrid[-h//2:h//2, -w//2:w//2]
            freq_radius = np.sqrt(x*x + y*y)
            
            # Normalize radius to [0, 1]
            max_radius = np.sqrt((h/2)**2 + (w/2)**2)
            freq_radius = freq_radius / max_radius
            
            # Create weight mask emphasizing high frequencies
            weight_mask = freq_radius ** 2  # Quadratic weighting
            
            # Apply mask to magnitude spectrum
            weighted_spectrum = magnitude_spectrum * weight_mask
            
            # Calculate reflectivity as weighted mean
            reflectivity = np.mean(weighted_spectrum) / np.mean(magnitude_spectrum)
            
            # Normalize to [0, 1] range with emphasis on high values
            reflectivity = np.tanh(2 * reflectivity)
            
            return float(reflectivity)

        except Exception as e:
            logger.error(f"Error calculating reflectivity: {e}")
            return 0.0

    def _extract_color_properties(self, region: np.ndarray) -> Dict:
        """Extract color properties in HSV space"""
        try:
            # Ensure uint8 type
            if region.dtype != np.uint8:
                region = (region * 255).astype(np.uint8)
                
            # Convert to HSV
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            
            # Calculate average hue and saturation
            hue_mean = float(np.mean(hsv[:,:,0]))
            sat_mean = float(np.mean(hsv[:,:,1])) / 255.0
            
            # Calculate color consistency
            hue_std = float(np.std(hsv[:,:,0]))
            sat_std = float(np.std(hsv[:,:,1])) / 255.0
            
            return {
                "hue": hue_mean,
                "saturation": sat_mean,
                "color_consistency": 1.0 - min(1.0, (hue_std + sat_std) / 100)
            }

        except Exception as e:
            logger.error(f"Error extracting color properties: {e}")
            return {}

    def _classify_material(self, features: Dict) -> Dict:
        """Classify material based on extracted features"""
        try:
            scores = {}
            
            for material, signature in self.material_signatures.items():
                # Compare frequency response
                freq_score = self._compare_frequency_response(
                    features["frequency_response"],
                    signature["frequency_response"]
                )
                
                # Compare texture entropy
                texture_score = self._compare_texture_entropy(
                    features["texture_entropy"],
                    signature["texture_entropy"]
                )
                
                # Compare color properties
                color_score = self._compare_color_properties(
                    features["color_properties"],
                    signature["color_range"]
                )
                
                # Calculate weighted average
                scores[material] = (
                    0.4 * freq_score +
                    0.3 * texture_score +
                    0.3 * color_score
                )
            
            # Get best match
            best_material = max(scores.items(), key=lambda x: x[1])
            
            return {
                "class": best_material[0],
                "confidence": best_material[1]
            }

        except Exception as e:
            logger.error(f"Error classifying material: {e}")
            return {"class": "unknown", "confidence": 0.0}

    def _compare_frequency_response(
        self,
        freq1: List[float],
        freq2: List[float]
    ) -> float:
        """Compare two frequency responses"""
        try:
            if not freq1 or not freq2:
                return 0.0
            
            # Ensure same length
            min_len = min(len(freq1), len(freq2))
            freq1 = freq1[:min_len]
            freq2 = freq2[:min_len]
            
            # Calculate similarity
            diff = np.array(freq1) - np.array(freq2)
            return float(1.0 - min(1.0, np.mean(np.abs(diff))))

        except Exception as e:
            logger.error(f"Error comparing frequency responses: {e}")
            return 0.0

    def _compare_texture_entropy(
        self,
        entropy: float,
        range_tuple: Tuple[float, float]
    ) -> float:
        """Compare texture entropy with expected range"""
        try:
            min_val, max_val = range_tuple
            if entropy < min_val:
                return max(0.0, 1.0 - (min_val - entropy) / min_val)
            elif entropy > max_val:
                return max(0.0, 1.0 - (entropy - max_val) / max_val)
            else:
                return 1.0

        except Exception as e:
            logger.error(f"Error comparing texture entropy: {e}")
            return 0.0

    def _compare_color_properties(
        self,
        properties: Dict,
        ranges: Dict
    ) -> float:
        """Compare color properties with expected ranges"""
        try:
            hue = properties["hue"]
            sat = properties["saturation"]
            
            # Check if color is within expected ranges
            hue_min, hue_max = ranges["hue"]
            sat_min, sat_max = ranges["saturation"]
            
            hue_match = (hue_min <= hue <= hue_max)
            sat_match = (sat_min <= sat <= sat_max)
            
            # Consider color consistency
            consistency = properties.get("color_consistency", 1.0)
            
            if hue_match and sat_match:
                return consistency
            elif hue_match or sat_match:
                return 0.5 * consistency
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Error comparing color properties: {e}")
            return 0.0

# Create global instance
material_recognizer = MaterialRecognizer()
