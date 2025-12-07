"""Day 93: Computer Vision - Solutions

NOTE: Mock implementations for learning without large model downloads.
"""

import numpy as np
from typing import List, Tuple, Dict, Any


# Exercise 1: Image Preprocessing
class ImagePreprocessor:
    """Preprocess images for CV models."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
    
    def resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size."""
        # Simple resize simulation
        h, w = self.target_size
        return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image values."""
        # Scale to [0, 1]
        normalized = image.astype(np.float32) / 255.0
        
        # Apply ImageNet mean/std
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        
        return normalized
    
    def augment(self, image: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        augmented = image.copy()
        
        # Random horizontal flip
        if np.random.rand() > 0.5:
            augmented = np.fliplr(augmented)
        
        # Random brightness adjustment
        brightness_factor = np.random.uniform(0.8, 1.2)
        augmented = np.clip(augmented * brightness_factor, 0, 255).astype(np.uint8)
        
        return augmented


# Exercise 2: Image Classification
class ImageClassifier:
    """Classify images using mock model."""
    
    def __init__(self, num_classes: int = 1000):
        self.num_classes = num_classes
        self.classes = [f"class_{i}" for i in range(num_classes)]
    
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Predict image class."""
        # Mock prediction
        class_id = np.random.randint(0, self.num_classes)
        confidence = np.random.uniform(0.7, 0.99)
        
        return {
            'class_id': class_id,
            'class_name': self.classes[class_id],
            'confidence': confidence
        }
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Predict multiple images."""
        return [self.predict(img) for img in images]


# Exercise 3: Object Detection
class ObjectDetector:
    """Detect objects in images."""
    
    def __init__(self):
        self.classes = ["person", "car", "dog", "cat", "bird"]
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in image."""
        h, w = image.shape[:2]
        
        # Mock detections
        num_objects = np.random.randint(1, 4)
        detections = []
        
        for _ in range(num_objects):
            x1 = np.random.randint(0, w // 2)
            y1 = np.random.randint(0, h // 2)
            x2 = np.random.randint(x1 + 50, w)
            y2 = np.random.randint(y1 + 50, h)
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'class': np.random.choice(self.classes),
                'confidence': np.random.uniform(0.7, 0.99)
            })
        
        return detections
    
    def draw_boxes(self, image: np.ndarray, 
                   detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw bounding boxes on image."""
        result = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            # Simulate drawing (in real implementation, use cv2.rectangle)
            result[y1:y1+2, x1:x2] = [0, 255, 0]  # Top
            result[y2:y2+2, x1:x2] = [0, 255, 0]  # Bottom
            result[y1:y2, x1:x1+2] = [0, 255, 0]  # Left
            result[y1:y2, x2:x2+2] = [0, 255, 0]  # Right
        
        return result


# Exercise 4: Face Detection
class FaceDetector:
    """Detect faces in images."""
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image."""
        h, w = image.shape[:2]
        
        # Mock face detection
        num_faces = np.random.randint(0, 3)
        faces = []
        
        for _ in range(num_faces):
            x = np.random.randint(0, w - 100)
            y = np.random.randint(0, h - 100)
            w_face = np.random.randint(80, 150)
            h_face = np.random.randint(80, 150)
            faces.append((x, y, w_face, h_face))
        
        return faces
    
    def extract_faces(self, image: np.ndarray) -> List[np.ndarray]:
        """Extract face regions."""
        faces = self.detect_faces(image)
        face_images = []
        
        for (x, y, w, h) in faces:
            # Extract and resize face region
            face = image[y:y+h, x:x+w]
            # Resize to standard size (mock)
            face_resized = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            face_images.append(face_resized)
        
        return face_images


# Exercise 5: Feature Extraction
class FeatureExtractor:
    """Extract features from images."""
    
    def __init__(self, feature_dim: int = 512):
        self.feature_dim = feature_dim
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract feature vector from image."""
        # Mock feature extraction
        features = np.random.randn(self.feature_dim)
        # Normalize
        features = features / np.linalg.norm(features)
        return features
    
    def compute_similarity(self, features1: np.ndarray, 
                          features2: np.ndarray) -> float:
        """Compute similarity between feature vectors."""
        # Cosine similarity
        similarity = np.dot(features1, features2) / (
            np.linalg.norm(features1) * np.linalg.norm(features2)
        )
        # Convert to [0, 1] range
        return (similarity + 1) / 2
    
    def find_similar(self, query_features: np.ndarray,
                    database_features: List[np.ndarray],
                    top_k: int = 5) -> List[Tuple[int, float]]:
        """Find most similar images."""
        similarities = []
        
        for idx, db_features in enumerate(database_features):
            sim = self.compute_similarity(query_features, db_features)
            similarities.append((idx, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


# Bonus: Image Segmentation
class ImageSegmenter:
    """Segment images into regions."""
    
    def segment(self, image: np.ndarray) -> np.ndarray:
        """Segment image."""
        h, w = image.shape[:2]
        # Mock segmentation mask with class IDs
        mask = np.random.randint(0, 21, (h, w), dtype=np.uint8)
        return mask
    
    def visualize_mask(self, image: np.ndarray, 
                      mask: np.ndarray) -> np.ndarray:
        """Visualize segmentation mask."""
        # Create color map
        colors = np.random.randint(0, 255, (21, 3), dtype=np.uint8)
        
        # Color code mask
        colored_mask = colors[mask]
        
        # Blend with original image
        alpha = 0.5
        result = (alpha * image + (1 - alpha) * colored_mask).astype(np.uint8)
        
        return result


def demo_computer_vision():
    """Demonstrate computer vision tasks."""
    print("Day 93: Computer Vision - Solutions Demo\n" + "=" * 60)
    
    # Create mock image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("\n1. Image Preprocessing")
    preprocessor = ImagePreprocessor()
    resized = preprocessor.resize(image)
    normalized = preprocessor.normalize(resized)
    augmented = preprocessor.augment(image)
    print(f"   Original shape: {image.shape}")
    print(f"   Resized shape: {resized.shape}")
    print(f"   Normalized range: [{normalized.min():.2f}, {normalized.max():.2f}]")
    
    print("\n2. Image Classification")
    classifier = ImageClassifier(num_classes=10)
    prediction = classifier.predict(image)
    print(f"   Predicted: {prediction['class_name']}")
    print(f"   Confidence: {prediction['confidence']:.2%}")
    
    print("\n3. Object Detection")
    detector = ObjectDetector()
    detections = detector.detect(image)
    print(f"   Detected {len(detections)} objects:")
    for det in detections:
        print(f"     - {det['class']}: {det['confidence']:.2%}")
    
    print("\n4. Face Detection")
    face_detector = FaceDetector()
    faces = face_detector.detect_faces(image)
    print(f"   Detected {len(faces)} faces")
    face_images = face_detector.extract_faces(image)
    print(f"   Extracted {len(face_images)} face images")
    
    print("\n5. Feature Extraction")
    extractor = FeatureExtractor(feature_dim=128)
    features1 = extractor.extract(image)
    features2 = extractor.extract(image)
    similarity = extractor.compute_similarity(features1, features2)
    print(f"   Feature dimension: {len(features1)}")
    print(f"   Similarity score: {similarity:.3f}")
    
    # Find similar images
    database = [extractor.extract(image) for _ in range(10)]
    similar = extractor.find_similar(features1, database, top_k=3)
    print(f"   Top 3 similar images: {[idx for idx, _ in similar]}")
    
    print("\n6. Image Segmentation")
    segmenter = ImageSegmenter()
    mask = segmenter.segment(image)
    visualization = segmenter.visualize_mask(image, mask)
    print(f"   Segmentation mask shape: {mask.shape}")
    print(f"   Unique classes: {len(np.unique(mask))}")
    
    print("\n" + "=" * 60)
    print("All computer vision tasks demonstrated!")


if __name__ == "__main__":
    demo_computer_vision()
