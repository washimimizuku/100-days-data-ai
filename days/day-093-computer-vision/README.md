# Day 93: Computer Vision

## Learning Objectives

**Time**: 1 hour

- Understand computer vision fundamentals
- Learn image preprocessing and augmentation
- Implement object detection and classification
- Apply pre-trained models for CV tasks

## Theory (15 minutes)

### What is Computer Vision?

Computer vision enables machines to interpret and understand visual information from images and videos.

**Key Tasks**:
- Image classification
- Object detection
- Segmentation
- Face recognition
- Image generation

### Image Basics

**Image Representation**:
```python
import numpy as np
from PIL import Image

# Load image
img = Image.open('photo.jpg')
img_array = np.array(img)  # Shape: (height, width, channels)

# RGB channels
print(f"Shape: {img_array.shape}")  # (224, 224, 3)
```

**Color Spaces**:
- RGB: Red, Green, Blue
- Grayscale: Single channel
- HSV: Hue, Saturation, Value

### Image Preprocessing

**Resizing**:
```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

img_tensor = transform(img)
```

**Augmentation**:
```python
augment = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomCrop(200)
])
```

### Image Classification

**Using Pre-trained Models**:
```python
import torch
from torchvision import models

# Load pre-trained ResNet
model = models.resnet50(pretrained=True)
model.eval()

# Predict
with torch.no_grad():
    output = model(img_tensor.unsqueeze(0))
    _, predicted = torch.max(output, 1)
```

**Transfer Learning**:
```python
# Freeze layers
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Train only final layer
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
```

### Object Detection

**YOLO (You Only Look Once)**:
```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Detect objects
results = model('image.jpg')

# Process results
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        confidence = box.conf[0]
        class_id = box.cls[0]
```

**Bounding Boxes**:
```python
import cv2

# Draw bounding box
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, (0, 255, 0), 2)
```

### Image Segmentation

**Semantic Segmentation**:
```python
from torchvision.models.segmentation import deeplabv3_resnet50

model = deeplabv3_resnet50(pretrained=True)
model.eval()

with torch.no_grad():
    output = model(img_tensor.unsqueeze(0))['out']
    mask = output.argmax(1).squeeze().cpu().numpy()
```

**Instance Segmentation**:
```python
from torchvision.models.detection import maskrcnn_resnet50_fpn

model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

predictions = model(img_tensor.unsqueeze(0))
masks = predictions[0]['masks']
```

### Face Detection

**Using OpenCV**:
```python
import cv2

# Load cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Detect faces
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangles
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
```

### Feature Extraction

**CNN Features**:
```python
# Extract features from intermediate layer
class FeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = torch.nn.Sequential(*list(model.children())[:-1])
    
    def forward(self, x):
        return self.features(x)

extractor = FeatureExtractor(model)
features = extractor(img_tensor.unsqueeze(0))
```

### Image Similarity

**Cosine Similarity**:
```python
from sklearn.metrics.pairwise import cosine_similarity

# Extract features for two images
features1 = extractor(img1_tensor.unsqueeze(0)).flatten()
features2 = extractor(img2_tensor.unsqueeze(0)).flatten()

# Calculate similarity
similarity = cosine_similarity(
    features1.reshape(1, -1),
    features2.reshape(1, -1)
)[0][0]
```

### Common Architectures

**CNNs**:
- VGG: Deep networks with small filters
- ResNet: Skip connections for deep networks
- Inception: Multi-scale feature extraction
- EfficientNet: Optimized architecture

**Modern Models**:
- Vision Transformer (ViT): Attention-based
- CLIP: Vision-language model
- DINO: Self-supervised learning

### Data Augmentation

**Techniques**:
```python
augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                          saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5)
])
```

### Evaluation Metrics

**Classification**:
- Accuracy: Correct predictions / Total
- Precision: True positives / Predicted positives
- Recall: True positives / Actual positives
- F1-Score: Harmonic mean of precision and recall

**Detection**:
- IoU (Intersection over Union): Overlap measure
- mAP (mean Average Precision): Detection quality
- Precision-Recall curve

### Use Cases

**Medical Imaging**:
- Disease detection
- Tumor segmentation
- X-ray analysis

**Autonomous Vehicles**:
- Lane detection
- Object recognition
- Traffic sign classification

**Retail**:
- Product recognition
- Inventory management
- Visual search

**Security**:
- Face recognition
- Anomaly detection
- Surveillance

### Best Practices

1. **Data Quality**: Clean, diverse, labeled data
2. **Preprocessing**: Normalize, resize consistently
3. **Augmentation**: Increase dataset variety
4. **Transfer Learning**: Use pre-trained models
5. **Evaluation**: Test on unseen data
6. **Optimization**: Balance accuracy and speed

### Why This Matters

Computer vision powers many AI applications from autonomous vehicles to medical diagnosis. Understanding CV fundamentals enables building visual AI systems that can interpret and act on visual information.

## Exercise (40 minutes)

Complete the exercises in `exercise.py`:

1. **Image Preprocessing**: Load and preprocess images
2. **Classification**: Classify images with pre-trained model
3. **Object Detection**: Detect objects in images
4. **Face Detection**: Find faces in photos
5. **Feature Extraction**: Extract and compare image features

## Resources

- [PyTorch Vision](https://pytorch.org/vision/stable/index.html)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [Computer Vision Course](https://www.coursera.org/learn/convolutional-neural-networks)

## Next Steps

- Complete the exercises
- Review the solution
- Take the quiz
- Move to Day 94: NLP Tasks

Tomorrow you'll learn about natural language processing tasks including text classification, named entity recognition, and sentiment analysis.
