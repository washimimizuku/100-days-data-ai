# Image Classifier - Project Specification

## Overview

Build a production-ready image classification system using Vision Transformers. The system should handle data loading, model training, and inference with proper error handling and logging.

---

## Data Format

### Directory Structure
```
data/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── class2/
│   │   └── ...
│   └── class3/
│       └── ...
└── val/
    ├── class1/
    ├── class2/
    └── class3/
```

### Image Specifications
- Format: JPEG, PNG
- Size: Any (will be resized to 224x224)
- Channels: RGB (3 channels)
- Classes: 3-10 classes recommended

---

## Component Specifications

### 1. ImageDataset Class

```python
class ImageDataset(Dataset):
    """Custom dataset for image classification"""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Path to data directory
            transform: Optional transforms to apply
        """
        # Load image paths and labels
        # Store class names and mapping
    
    def __len__(self):
        # Return dataset size
    
    def __getitem__(self, idx):
        # Load image
        # Apply transforms
        # Return image tensor and label
```

### 2. ImageClassifier Class

```python
class ImageClassifier:
    """Vision Transformer based image classifier"""
    
    def __init__(self, num_classes, model_name="google/vit-base-patch16-224"):
        """
        Args:
            num_classes: Number of output classes
            model_name: Pre-trained model identifier
        """
        # Load pre-trained ViT
        # Replace classification head
        # Setup device (GPU/CPU)
    
    def train(self, train_loader, val_loader, epochs, lr):
        """Train the model"""
        # Training loop with validation
        # Save best model
        # Return training history
    
    def predict(self, image_path, top_k=3):
        """Predict single image"""
        # Load and preprocess image
        # Run inference
        # Return top-k predictions with confidence
    
    def predict_batch(self, image_paths):
        """Predict multiple images"""
        # Batch processing
        # Return predictions for all images
    
    def save(self, path):
        """Save model checkpoint"""
    
    def load(self, path):
        """Load model checkpoint"""
```

### 3. Training Configuration

```python
config = {
    'model_name': 'google/vit-base-patch16-224',
    'num_classes': 3,
    'epochs': 5,
    'batch_size': 16,
    'learning_rate': 5e-5,
    'weight_decay': 0.01,
    'image_size': 224,
    'num_workers': 2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
```

---

## Data Augmentation

### Training Transforms
```python
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

### Validation Transforms
```python
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

---

## Training Process

### Epoch Loop
```
For each epoch:
    1. Set model to training mode
    2. For each batch in train_loader:
        - Forward pass
        - Compute loss
        - Backward pass
        - Update weights
        - Track metrics
    
    3. Set model to evaluation mode
    4. For each batch in val_loader:
        - Forward pass (no gradients)
        - Compute loss and accuracy
        - Track metrics
    
    5. Print epoch summary
    6. Save checkpoint if best validation accuracy
    7. Early stopping check
```

### Metrics to Track
- Training loss
- Training accuracy
- Validation loss
- Validation accuracy
- Learning rate

---

## Inference Pipeline

### Single Image Prediction
```
1. Load image from path
2. Apply validation transforms
3. Add batch dimension
4. Move to device
5. Run model inference
6. Apply softmax for probabilities
7. Get top-k predictions
8. Return class names and confidence scores
```

### Batch Prediction
```
1. Load all images
2. Apply transforms
3. Create batch tensor
4. Run batch inference
5. Process all predictions
6. Return results for each image
```

---

## Output Format

### Training Output
```
Epoch 1/5
Train Loss: 0.8234, Train Acc: 65.32%
Val Loss: 0.6543, Val Acc: 72.15%
✓ Best model saved

Epoch 2/5
Train Loss: 0.5432, Train Acc: 78.45%
Val Loss: 0.5123, Val Acc: 80.23%
✓ Best model saved
...
```

### Prediction Output
```python
{
    'predictions': [
        {'class': 'cat', 'confidence': 0.9234},
        {'class': 'dog', 'confidence': 0.0543},
        {'class': 'bird', 'confidence': 0.0223}
    ],
    'top_class': 'cat',
    'top_confidence': 0.9234
}
```

---

## Error Handling

### Common Errors
1. **File Not Found**: Check image path exists
2. **Invalid Image**: Verify image format and integrity
3. **Out of Memory**: Reduce batch size or image resolution
4. **Model Load Error**: Verify checkpoint file exists
5. **CUDA Error**: Check GPU availability and memory

### Error Messages
```python
try:
    image = Image.open(image_path)
except FileNotFoundError:
    raise ValueError(f"Image not found: {image_path}")
except Exception as e:
    raise ValueError(f"Error loading image: {e}")
```

---

## Performance Benchmarks

### Training Performance
- **Small Dataset** (1000 images): ~2 min/epoch (CPU)
- **Medium Dataset** (5000 images): ~8 min/epoch (CPU)
- **Large Dataset** (10000 images): ~15 min/epoch (CPU)

### Inference Performance
- **Single Image**: 50-100ms (CPU), 10-20ms (GPU)
- **Batch (32 images)**: 500ms (CPU), 100ms (GPU)

### Memory Requirements
- **Model**: ~350 MB (ViT-base)
- **Training**: ~2 GB RAM (batch_size=16)
- **Inference**: ~500 MB RAM

---

## Testing Checklist

- [ ] Data generation creates proper directory structure
- [ ] Dataset loads images correctly
- [ ] Transforms apply without errors
- [ ] Model initializes with correct number of classes
- [ ] Training loop runs for specified epochs
- [ ] Validation metrics computed correctly
- [ ] Best model saves to disk
- [ ] Model loads from checkpoint
- [ ] Single image prediction works
- [ ] Batch prediction works
- [ ] Top-k predictions returned correctly
- [ ] Confidence scores sum to 1.0

---

## Example Usage

### Training
```python
from classifier import ImageClassifier
from torch.utils.data import DataLoader

# Create datasets
train_dataset = ImageDataset('data/train', transform=train_transform)
val_dataset = ImageDataset('data/val', transform=val_transform)

# Create loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Train model
classifier = ImageClassifier(num_classes=3)
history = classifier.train(train_loader, val_loader, epochs=5, lr=5e-5)
classifier.save('best_model.pth')
```

### Inference
```python
from classifier import ImageClassifier

# Load model
classifier = ImageClassifier(num_classes=3)
classifier.load('best_model.pth')

# Predict
result = classifier.predict('test_image.jpg', top_k=3)
print(f"Prediction: {result['top_class']} ({result['top_confidence']:.2%})")
```

---

## Success Metrics

### Minimum Requirements
- Training completes without errors
- Validation accuracy > 70%
- Model saves and loads correctly
- Predictions return valid probabilities

### Target Performance
- Validation accuracy > 80%
- Inference time < 100ms per image
- All tests pass
- Code is clean and documented

### Excellent Performance
- Validation accuracy > 90%
- Inference time < 50ms per image
- Implements early stopping
- Includes visualization
- Handles edge cases gracefully
