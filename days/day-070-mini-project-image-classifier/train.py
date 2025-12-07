"""
Training script for image classifier
"""

import torch
from torch.utils.data import DataLoader
from classifier import ImageClassifier, ImageDataset, get_transforms
import argparse


def main():
    parser = argparse.ArgumentParser(description='Train image classifier')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--model', type=str, default='google/vit-base-patch16-224', 
                       help='Pre-trained model name')
    parser.add_argument('--output', type=str, default='best_model.pth', 
                       help='Output model path')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Image Classifier Training")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Model: {args.model}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 60)
    print()
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = ImageDataset(
        f'{args.data_dir}/train',
        transform=get_transforms(train=True)
    )
    val_dataset = ImageDataset(
        f'{args.data_dir}/val',
        transform=get_transforms(train=False)
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    print()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Create classifier
    print("Initializing model...")
    classifier = ImageClassifier(
        num_classes=len(train_dataset.classes),
        model_name=args.model
    )
    classifier.class_names = train_dataset.classes
    print(f"Model loaded: {args.model}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print()
    
    # Train
    print("Starting training...")
    print("=" * 60)
    history = classifier.train(
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr
    )
    
    # Save final model
    classifier.save(args.output)
    print("=" * 60)
    print(f"Training complete! Model saved to {args.output}")
    print()
    
    # Print summary
    print("Training Summary:")
    print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
