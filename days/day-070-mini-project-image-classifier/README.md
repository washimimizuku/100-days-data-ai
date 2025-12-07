# Day 70: Mini Project - Image Classifier

## 🎯 Project Overview

Build a complete image classification system using Vision Transformers (ViT) and PyTorch. This project integrates concepts from Days 67-69: PyTorch tensors, model building, and Hugging Face Transformers.

**Time**: 2 hours

---

## 🏗️ Architecture

```
Image Classifier System
├── Data Pipeline
│   ├── Image loading and preprocessing
│   ├── Data augmentation
│   └── Dataset creation
├── Model
│   ├── Pre-trained ViT from Hugging Face
│   ├── Custom classification head
│   └── Fine-tuning strategy
├── Training
│   ├── Training loop with validation
│   ├── Learning rate scheduling
│   └── Model checkpointing
└── Inference
    ├── Single image prediction
    ├── Batch prediction
    └── Confidence scores
```

---

## 📋 Requirements

### Functional Requirements

1. **Data Handling**
   - Load images from directory structure
   - Apply data augmentation (rotation, flip, crop)
   - Create train/validation splits
   - Batch processing with DataLoader

2. **Model Architecture**
   - Use pre-trained Vision Transformer
   - Add custom classification head
   - Support transfer learning
   - Save/load model checkpoints

3. **Training Pipeline**
   - Train with early stopping
   - Track metrics (accuracy, loss)
   - Validate after each epoch
   - Save best model

4. **Inference System**
   - Predict single images
   - Batch predictions
   - Return top-k predictions with confidence
   - Visualize predictions

### Technical Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers library
- Pillow for image processing
- All files < 400 lines

---

## 🚀 Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Project Structure

```
day-070-mini-project-image-classifier/
├── README.md              # This file
├── project.md             # Detailed specification
├── classifier.py          # Main classifier implementation
├── train.py              # Training script
├── predict.py            # Inference script
├── test_classifier.sh    # Automated testing
└── requirements.txt      # Dependencies
```

### Quick Start

```bash
# 1. Generate sample dataset
python classifier.py --generate-data

# 2. Train the model
python train.py --epochs 5 --batch-size 16

# 3. Make predictions
python predict.py --image sample.jpg --top-k 3

# 4. Run tests
./test_classifier.sh
```

---

## 💻 Implementation Guide

### Step 1: Data Pipeline (30 min)

Create dataset class with augmentation:
- Load images from folders (class = folder name)
- Apply transforms (resize, normalize, augment)
- Split into train/validation sets
- Create DataLoaders

### Step 2: Model Setup (20 min)

Load and customize ViT:
- Load pre-trained ViT from Hugging Face
- Replace classification head for custom classes
- Freeze/unfreeze layers for transfer learning
- Move model to GPU if available

### Step 3: Training Loop (40 min)

Implement training with validation:
- Forward pass and loss computation
- Backward pass and optimization
- Validation after each epoch
- Save best model checkpoint
- Early stopping

### Step 4: Inference (30 min)

Create prediction pipeline:
- Load trained model
- Preprocess input images
- Generate predictions with confidence
- Visualize results

---

## 🎓 Learning Objectives

By completing this project, you will:
- Build end-to-end image classification pipeline
- Use Vision Transformers for computer vision
- Implement transfer learning with pre-trained models
- Apply data augmentation techniques
- Create training loops with validation
- Save and load model checkpoints
- Build inference systems with confidence scores

---

## 📊 Expected Results

### Training Metrics
- Training accuracy: > 85%
- Validation accuracy: > 80%
- Training time: ~5-10 min (5 epochs, CPU)
- Model size: ~350 MB (ViT-base)

### Inference Performance
- Single image: < 100ms
- Batch (32 images): < 1s
- Top-1 accuracy: > 80%
- Top-3 accuracy: > 95%

---

## 🧪 Testing

The test script validates:
1. Data generation and loading
2. Model initialization
3. Training for 2 epochs
4. Model saving and loading
5. Inference on test images
6. Batch predictions

Run tests:
```bash
chmod +x test_classifier.sh
./test_classifier.sh
```

---

## 🔧 Configuration

### Model Options
- `vit-base`: 86M parameters, good accuracy
- `vit-small`: 22M parameters, faster training
- `vit-large`: 304M parameters, best accuracy

### Training Options
- Epochs: 5-10 for fine-tuning
- Batch size: 16-32 (depends on GPU memory)
- Learning rate: 1e-4 to 5e-5
- Optimizer: AdamW with weight decay

### Data Augmentation
- Random horizontal flip
- Random rotation (±15 degrees)
- Random crop and resize
- Color jitter

---

## 📈 Extensions

### Bonus Features (Optional)

1. **MLflow Integration**
   - Track experiments
   - Log metrics and parameters
   - Save model artifacts

2. **Gradio Interface**
   - Web UI for predictions
   - Upload images
   - Display results with confidence

3. **Model Optimization**
   - Quantization for faster inference
   - ONNX export
   - TorchScript compilation

4. **Advanced Techniques**
   - Mixup augmentation
   - Label smoothing
   - Test-time augmentation

---

## 🎯 Success Criteria

- [ ] Data pipeline loads and augments images
- [ ] Model trains without errors
- [ ] Validation accuracy > 80%
- [ ] Model saves and loads correctly
- [ ] Inference produces predictions with confidence
- [ ] Test script passes all checks
- [ ] All files < 400 lines

---

## 📚 Resources

- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [Hugging Face ViT Documentation](https://huggingface.co/docs/transformers/model_doc/vit)
- [PyTorch Image Classification Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Transfer Learning Guide](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Data Augmentation Techniques](https://pytorch.org/vision/stable/transforms.html)

---

## 🔍 Troubleshooting

**Out of Memory Error**:
- Reduce batch size
- Use smaller model (vit-small)
- Enable gradient checkpointing

**Low Accuracy**:
- Train for more epochs
- Adjust learning rate
- Add more data augmentation
- Unfreeze more layers

**Slow Training**:
- Use GPU if available
- Increase batch size
- Use mixed precision training
- Reduce image resolution

---

## Next Steps

After completing this project:
1. Review Week 10 concepts (Days 64-70)
2. Prepare for Week 11: GenAI Foundations
3. Explore advanced vision tasks (object detection, segmentation)
4. Build a portfolio project combining multiple weeks

**Tomorrow**: Day 71 - LLM Architecture
