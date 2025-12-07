"""
Inference script for image classifier
"""

import torch
from classifier import ImageClassifier
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Predict with image classifier')
    parser.add_argument('--image', type=str, required=True, help='Image path')
    parser.add_argument('--model', type=str, default='best_model.pth', 
                       help='Model checkpoint path')
    parser.add_argument('--top-k', type=int, default=3, help='Number of top predictions')
    parser.add_argument('--num-classes', type=int, default=3, help='Number of classes')
    args = parser.parse_args()
    
    # Check if image exists
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        return
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        print("Please train a model first using train.py")
        return
    
    print("=" * 60)
    print("Image Classifier Inference")
    print("=" * 60)
    print(f"Image: {args.image}")
    print(f"Model: {args.model}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 60)
    print()
    
    # Load model
    print("Loading model...")
    classifier = ImageClassifier(num_classes=args.num_classes)
    classifier.load(args.model)
    print("Model loaded successfully!")
    print()
    
    # Predict
    print("Making prediction...")
    result = classifier.predict(args.image, top_k=args.top_k)
    
    # Display results
    print("=" * 60)
    print("Prediction Results:")
    print("=" * 60)
    print(f"Top Prediction: {result['top_class']} ({result['top_confidence']:.2%})")
    print()
    print(f"Top {args.top_k} Predictions:")
    for i, pred in enumerate(result['predictions'], 1):
        print(f"  {i}. {pred['class']:15s} {pred['confidence']:.2%}")
    print("=" * 60)


def predict_batch_demo():
    """Demo function for batch prediction"""
    import glob
    
    # Find all images in data/val
    image_paths = glob.glob('data/val/*/*.jpg')[:10]
    
    if not image_paths:
        print("No images found for batch prediction demo")
        return
    
    print("=" * 60)
    print("Batch Prediction Demo")
    print("=" * 60)
    print(f"Processing {len(image_paths)} images...")
    print()
    
    # Load model
    classifier = ImageClassifier(num_classes=3)
    classifier.load('best_model.pth')
    
    # Predict batch
    results = classifier.predict_batch(image_paths)
    
    # Display results
    print("Results:")
    for img_path, result in zip(image_paths, results):
        img_name = Path(img_path).name
        print(f"{img_name:30s} -> {result['top_class']:10s} ({result['top_confidence']:.2%})")
    print("=" * 60)


if __name__ == "__main__":
    main()
