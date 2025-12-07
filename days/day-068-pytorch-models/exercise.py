"""
Day 68: PyTorch Models - Exercises

Practice building, training, and evaluating PyTorch models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def exercise_1_binary_classifier():
    """
    Exercise 1: Build a Binary Classifier
    
    Create a neural network for binary classification:
    - Input size: 20 features
    - Hidden layer 1: 64 neurons with ReLU
    - Hidden layer 2: 32 neurons with ReLU
    - Output: 1 neuron with Sigmoid
    
    TODO: Define the BinaryClassifier class
    TODO: Create an instance and print the model
    TODO: Test with random input (batch_size=10, features=20)
    """
    # TODO: Implement BinaryClassifier class
    pass


def exercise_2_training_loop():
    """
    Exercise 2: Implement Training Loop
    
    Create synthetic data and train a simple model:
    - Generate 1000 samples with 10 features
    - Binary classification task
    - Train for 50 epochs
    - Track and print loss every 10 epochs
    
    TODO: Generate synthetic data (X, y)
    TODO: Create DataLoader
    TODO: Define model, loss function, optimizer
    TODO: Implement training loop
    TODO: Print final loss
    """
    # TODO: Generate data
    # X = torch.randn(1000, 10)
    # y = (X.sum(dim=1) > 0).float().unsqueeze(1)
    
    # TODO: Create DataLoader
    
    # TODO: Define model
    
    # TODO: Training loop
    pass


def exercise_3_multiclass_classifier():
    """
    Exercise 3: Multi-Class Classifier
    
    Build and train a model for 5-class classification:
    - Input: 15 features
    - Hidden layers: 50, 30 neurons
    - Output: 5 classes
    - Use CrossEntropyLoss
    
    TODO: Define MultiClassNet
    TODO: Generate synthetic data (500 samples, 5 classes)
    TODO: Train for 30 epochs
    TODO: Compute and print final accuracy
    """
    # TODO: Implement MultiClassNet
    
    # TODO: Generate data
    # X = torch.randn(500, 15)
    # y = torch.randint(0, 5, (500,))
    
    # TODO: Train model
    
    # TODO: Compute accuracy
    pass


def exercise_4_model_checkpointing():
    """
    Exercise 4: Model Checkpointing
    
    Implement save/load functionality:
    - Train a simple model for 20 epochs
    - Save checkpoint every 5 epochs
    - Load the best checkpoint
    - Continue training from checkpoint
    
    TODO: Define model and training setup
    TODO: Implement checkpoint saving logic
    TODO: Save checkpoint with epoch, model state, optimizer state, loss
    TODO: Load checkpoint and verify
    """
    # TODO: Create model
    
    # TODO: Training with checkpointing
    
    # TODO: Load checkpoint
    
    # TODO: Verify loaded state
    pass


def exercise_5_model_evaluation():
    """
    Exercise 5: Model Evaluation
    
    Create comprehensive evaluation functions:
    - Compute accuracy
    - Compute precision and recall (binary classification)
    - Generate predictions with probabilities
    - Test on validation set
    
    TODO: Implement evaluate_accuracy()
    TODO: Implement evaluate_precision_recall()
    TODO: Implement predict_with_probabilities()
    TODO: Test all functions on a trained model
    """
    # TODO: Define evaluation functions
    
    # TODO: Train a simple model
    
    # TODO: Evaluate on test data
    
    # TODO: Print metrics
    pass


if __name__ == "__main__":
    print("Day 68: PyTorch Models - Exercises\n")
    
    print("=" * 60)
    print("Exercise 1: Binary Classifier")
    print("=" * 60)
    # exercise_1_binary_classifier()
    
    print("\n" + "=" * 60)
    print("Exercise 2: Training Loop")
    print("=" * 60)
    # exercise_2_training_loop()
    
    print("\n" + "=" * 60)
    print("Exercise 3: Multi-Class Classifier")
    print("=" * 60)
    # exercise_3_multiclass_classifier()
    
    print("\n" + "=" * 60)
    print("Exercise 4: Model Checkpointing")
    print("=" * 60)
    # exercise_4_model_checkpointing()
    
    print("\n" + "=" * 60)
    print("Exercise 5: Model Evaluation")
    print("=" * 60)
    # exercise_5_model_evaluation()
