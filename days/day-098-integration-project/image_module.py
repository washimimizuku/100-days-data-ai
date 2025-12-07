"""
Day 98: Integration Project - Image Analysis Module
"""
import os
from typing import Dict, List, Tuple
from PIL import Image
import numpy as np


class ImageAnalyzer:
    """Analyze image content with classification, object detection, and feature extraction."""
    
    def __init__(self):
        """Initialize image analyzer."""
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    
    def classify_image(self, image_path: str) -> Dict:
        """
        Classify image content based on properties.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with classification results
        """
        if not os.path.exists(image_path):
            return {"error": "Image file not found"}
        
        try:
            img = Image.open(image_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            width, height = img.size
            pixels = np.array(img)
            
            # Simple classification based on image properties
            avg_brightness = pixels.mean() / 255.0
            avg_color = pixels.mean(axis=(0, 1))
            
            # Determine category based on properties
            category, confidence = self._determine_category(
                width, height, avg_brightness, avg_color
            )
            
            # Generate top predictions
            predictions = self._generate_predictions(
                width, height, avg_brightness, avg_color
            )
            
            return {
                "category": category,
                "confidence": round(confidence, 3),
                "top_predictions": predictions
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _determine_category(
        self, 
        width: int, 
        height: int, 
        brightness: float, 
        avg_color: np.ndarray
    ) -> Tuple[str, float]:
        """Determine image category based on properties."""
        # Aspect ratio
        aspect_ratio = width / height
        
        # Color dominance
        r, g, b = avg_color
        
        # Simple heuristics for classification
        if brightness > 0.8:
            return "bright_scene", 0.85
        elif brightness < 0.2:
            return "dark_scene", 0.82
        elif b > r and b > g:
            return "blue_dominant", 0.78
        elif g > r and g > b:
            return "green_dominant", 0.76
        elif r > g and r > b:
            return "red_dominant", 0.74
        elif aspect_ratio > 1.5:
            return "landscape", 0.80
        elif aspect_ratio < 0.7:
            return "portrait", 0.79
        else:
            return "general", 0.70
    
    def _generate_predictions(
        self,
        width: int,
        height: int,
        brightness: float,
        avg_color: np.ndarray
    ) -> List[Dict]:
        """Generate top prediction candidates."""
        predictions = []
        
        # Add predictions based on properties
        if brightness > 0.7:
            predictions.append({"label": "bright_scene", "score": 0.85})
        if brightness < 0.3:
            predictions.append({"label": "dark_scene", "score": 0.80})
        
        r, g, b = avg_color
        if b > 150:
            predictions.append({"label": "blue_tones", "score": 0.75})
        if g > 150:
            predictions.append({"label": "green_tones", "score": 0.72})
        if r > 150:
            predictions.append({"label": "red_tones", "score": 0.70})
        
        aspect_ratio = width / height
        if aspect_ratio > 1.5:
            predictions.append({"label": "landscape", "score": 0.78})
        elif aspect_ratio < 0.7:
            predictions.append({"label": "portrait", "score": 0.76})
        
        # Ensure at least 3 predictions
        while len(predictions) < 3:
            predictions.append({"label": "general", "score": 0.60})
        
        # Sort by score and return top 5
        predictions.sort(key=lambda x: x['score'], reverse=True)
        return predictions[:5]
    
    def detect_objects(self, image_path: str) -> Dict:
        """
        Detect objects in image (mock implementation).
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with detected objects
        """
        if not os.path.exists(image_path):
            return {"error": "Image file not found"}
        
        try:
            img = Image.open(image_path)
            width, height = img.size
            
            # Mock object detection with random boxes
            objects = []
            
            # Generate 2-4 mock objects
            num_objects = np.random.randint(2, 5)
            labels = ["object", "region", "area", "element", "component"]
            
            for i in range(num_objects):
                # Random bounding box
                x = np.random.randint(0, width // 2)
                y = np.random.randint(0, height // 2)
                w = np.random.randint(width // 4, width // 2)
                h = np.random.randint(height // 4, height // 2)
                
                objects.append({
                    "label": labels[i % len(labels)],
                    "confidence": round(0.6 + np.random.random() * 0.3, 3),
                    "bbox": [x, y, w, h]
                })
            
            return {"objects": objects}
            
        except Exception as e:
            return {"error": str(e)}
    
    def extract_features(self, image_path: str) -> Dict:
        """
        Extract image features and statistics.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with image features
        """
        if not os.path.exists(image_path):
            return {"error": "Image file not found"}
        
        try:
            img = Image.open(image_path)
            
            # Basic info
            width, height = img.size
            format_name = img.format or "unknown"
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            pixels = np.array(img)
            
            # Color analysis
            colors = self._analyze_colors(pixels)
            
            # Brightness and contrast
            brightness = float(pixels.mean() / 255.0)
            contrast = float(pixels.std() / 255.0)
            
            # Color channels
            r_mean = float(pixels[:, :, 0].mean())
            g_mean = float(pixels[:, :, 1].mean())
            b_mean = float(pixels[:, :, 2].mean())
            
            return {
                "dimensions": {
                    "width": width,
                    "height": height,
                    "aspect_ratio": round(width / height, 3)
                },
                "colors": colors,
                "brightness": round(brightness, 3),
                "contrast": round(contrast, 3),
                "channels": {
                    "red": round(r_mean, 1),
                    "green": round(g_mean, 1),
                    "blue": round(b_mean, 1)
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_colors(self, pixels: np.ndarray) -> List[Dict]:
        """Analyze dominant colors in image."""
        # Reshape pixels
        colors = pixels.reshape(-1, 3)
        
        # Calculate average color
        avg_color = colors.mean(axis=0)
        
        # Simple color categorization
        r, g, b = avg_color
        
        dominant_colors = []
        
        # Determine dominant color
        if r > g and r > b:
            dominant_colors.append({
                "color": f"rgb({int(r)},{int(g)},{int(b)})",
                "name": "red-dominant",
                "percentage": round((r / (r + g + b)) * 100, 1)
            })
        elif g > r and g > b:
            dominant_colors.append({
                "color": f"rgb({int(r)},{int(g)},{int(b)})",
                "name": "green-dominant",
                "percentage": round((g / (r + g + b)) * 100, 1)
            })
        elif b > r and b > g:
            dominant_colors.append({
                "color": f"rgb({int(r)},{int(g)},{int(b)})",
                "name": "blue-dominant",
                "percentage": round((b / (r + g + b)) * 100, 1)
            })
        else:
            dominant_colors.append({
                "color": f"rgb({int(r)},{int(g)},{int(b)})",
                "name": "balanced",
                "percentage": 100.0
            })
        
        return dominant_colors
    
    def analyze(self, image_path: str) -> Dict:
        """
        Perform complete image analysis.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with all analysis results
        """
        return {
            "classification": self.classify_image(image_path),
            "objects": self.detect_objects(image_path),
            "features": self.extract_features(image_path),
            "metadata": {
                "path": image_path,
                "exists": os.path.exists(image_path),
                "size": os.path.getsize(image_path) if os.path.exists(image_path) else 0
            }
        }


if __name__ == "__main__":
    print("Day 98: Image Analysis Module\n")
    
    analyzer = ImageAnalyzer()
    
    # Create a test image
    print("=== Creating Test Image ===")
    test_image = Image.new('RGB', (800, 600), color=(100, 150, 200))
    test_path = "test_image.jpg"
    test_image.save(test_path)
    print(f"Created test image: {test_path}")
    
    # Test classification
    print("\n=== Image Classification ===")
    result = analyzer.classify_image(test_path)
    if "error" not in result:
        print(f"Category: {result['category']}")
        print(f"Confidence: {result['confidence']}")
        print("Top predictions:")
        for pred in result['top_predictions']:
            print(f"  - {pred['label']}: {pred['score']}")
    else:
        print(f"Error: {result['error']}")
    
    # Test object detection
    print("\n=== Object Detection ===")
    result = analyzer.detect_objects(test_path)
    if "error" not in result:
        print(f"Found {len(result['objects'])} objects:")
        for obj in result['objects']:
            print(f"  - {obj['label']}: {obj['confidence']} at {obj['bbox']}")
    else:
        print(f"Error: {result['error']}")
    
    # Test feature extraction
    print("\n=== Feature Extraction ===")
    result = analyzer.extract_features(test_path)
    if "error" not in result:
        print(f"Dimensions: {result['dimensions']}")
        print(f"Brightness: {result['brightness']}")
        print(f"Contrast: {result['contrast']}")
        print(f"Channels: {result['channels']}")
        print("Colors:")
        for color in result['colors']:
            print(f"  - {color['name']}: {color['percentage']}%")
    else:
        print(f"Error: {result['error']}")
    
    # Complete analysis
    print("\n=== Complete Analysis ===")
    result = analyzer.analyze(test_path)
    print(f"Classification: {result['classification'].get('category', 'N/A')}")
    print(f"Objects detected: {len(result['objects'].get('objects', []))}")
    print(f"Features extracted: {len(result['features'])} properties")
    print(f"Metadata: {result['metadata']}")
    
    # Cleanup
    if os.path.exists(test_path):
        os.remove(test_path)
        print(f"\nCleaned up test image")
