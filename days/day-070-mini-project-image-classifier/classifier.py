"""
Image Classifier using Vision Transformer
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import os
from pathlib import Path
import numpy as np


class ImageDataset(Dataset):
    """Custom dataset for image classification"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all image paths and labels
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for img_path in class_dir.glob('*.jpg'):
                self.images.append(str(img_path))
                self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class ImageClassifier:
    """Vision Transformer based image classifier"""
    
    def __init__(self, num_classes, model_name="google/vit-base-patch16-224"):
        self.num_classes = num_classes
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained ViT
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        self.model.to(self.device)
        
        self.class_names = None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    def train(self, train_loader, val_loader, epochs=5, lr=5e-5):
        """Train the model"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images).logits
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = 100 * train_correct / train_total
            
            # Validation phase
            val_loss, val_acc = self._validate(val_loader, criterion)
            
            # Track history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save('best_model.pth')
                print("✓ Best model saved")
            print()
        
        return self.history
    
    def _validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images).logits
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        return val_loss, val_acc
    
    def predict(self, image_path, top_k=3):
        """Predict single image"""
        self.model.eval()
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor).logits
            probabilities = torch.softmax(outputs, dim=1)[0]
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, min(top_k, self.num_classes))
        
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            predictions.append({
                'class': self.class_names[idx] if self.class_names else f'class_{idx}',
                'confidence': prob.item()
            })
        
        return {
            'predictions': predictions,
            'top_class': predictions[0]['class'],
            'top_confidence': predictions[0]['confidence']
        }
    
    def predict_batch(self, image_paths):
        """Predict multiple images"""
        self.model.eval()
        results = []
        
        for image_path in image_paths:
            result = self.predict(image_path, top_k=1)
            results.append(result)
        
        return results
    
    def save(self, path):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'model_name': self.model_name,
            'class_names': self.class_names,
            'history': self.history
        }
        torch.save(checkpoint, path)
    
    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.class_names = checkpoint.get('class_names')
        self.history = checkpoint.get('history', self.history)


def get_transforms(train=True):
    """Get data transforms"""
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def generate_sample_data(output_dir='data', num_images_per_class=50):
    """Generate sample dataset for testing"""
    from PIL import Image, ImageDraw
    import random
    
    classes = ['cat', 'dog', 'bird']
    
    for split in ['train', 'val']:
        for class_name in classes:
            class_dir = Path(output_dir) / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            num_images = num_images_per_class if split == 'train' else num_images_per_class // 5
            
            for i in range(num_images):
                # Create random colored image
                img = Image.new('RGB', (224, 224), 
                              color=(random.randint(0, 255),
                                   random.randint(0, 255),
                                   random.randint(0, 255)))
                
                # Add some shapes
                draw = ImageDraw.Draw(img)
                for _ in range(5):
                    x1, y1 = random.randint(0, 200), random.randint(0, 200)
                    x2, y2 = x1 + random.randint(10, 50), y1 + random.randint(10, 50)
                    draw.rectangle([x1, y1, x2, y2], 
                                 fill=(random.randint(0, 255),
                                      random.randint(0, 255),
                                      random.randint(0, 255)))
                
                img.save(class_dir / f'{class_name}_{i:03d}.jpg')
    
    print(f"Generated sample dataset in {output_dir}/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-data', action='store_true', help='Generate sample dataset')
    args = parser.parse_args()
    
    if args.generate_data:
        generate_sample_data()
        print("Sample data generated successfully!")
