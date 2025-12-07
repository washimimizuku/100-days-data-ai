"""
Day 67: PyTorch Tensors - Exercises
"""
import torch
import numpy as np


def exercise_1_tensor_creation():
    """
    Exercise 1: Tensor Creation
    
    Create tensors using different methods.
    """
    # TODO: Create tensor from list [1, 2, 3, 4, 5]
    # TODO: Create 3x3 tensor of zeros
    # TODO: Create 2x4 tensor of ones
    # TODO: Create 3x3 random tensor (uniform distribution)
    # TODO: Create tensor from 0 to 10 with step 2
    # TODO: Print all tensors and their shapes
    pass


def exercise_2_tensor_operations():
    """
    Exercise 2: Tensor Operations
    
    Perform mathematical operations on tensors.
    """
    # Create tensors
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    
    # TODO: Add x and y
    # TODO: Multiply x and y element-wise
    # TODO: Compute dot product
    # TODO: Create 2x3 and 3x4 matrices, multiply them
    # TODO: Print all results
    pass


def exercise_3_reshaping():
    """
    Exercise 3: Reshaping Tensors
    
    Practice tensor reshaping operations.
    """
    # Create tensor
    x = torch.arange(24)
    
    # TODO: Reshape to 4x6
    # TODO: Reshape to 2x3x4
    # TODO: Flatten back to 1D
    # TODO: Add dimension at position 0
    # TODO: Remove added dimension
    # TODO: Print shapes at each step
    pass


def exercise_4_autograd():
    """
    Exercise 4: Automatic Differentiation
    
    Use autograd to compute gradients.
    """
    # TODO: Create tensor x = [2.0] with requires_grad=True
    # TODO: Compute y = x^3 + 2*x^2 + x
    # TODO: Compute gradient dy/dx
    # TODO: Print x, y, and gradient
    # TODO: Verify gradient manually (3x^2 + 4x + 1 at x=2)
    pass


def exercise_5_linear_regression():
    """
    Exercise 5: Linear Regression with Tensors
    
    Implement simple linear regression.
    """
    # Generate data: y = 2x + 1 + noise
    torch.manual_seed(42)
    X = torch.randn(50, 1)
    y_true = 2 * X + 1 + 0.1 * torch.randn(50, 1)
    
    # TODO: Initialize w and b with requires_grad=True
    # TODO: Train for 100 epochs
    # TODO: In each epoch:
    #       - Compute predictions: y_pred = X * w + b
    #       - Compute MSE loss
    #       - Backward pass
    #       - Update w and b (learning_rate=0.01)
    #       - Zero gradients
    # TODO: Print final w and b
    # TODO: Compare with true values (w=2, b=1)
    pass


if __name__ == "__main__":
    print("Day 67: PyTorch Tensors - Exercises\n")
    
    print("=" * 60)
    print("Exercise 1: Tensor Creation")
    print("=" * 60)
    # exercise_1_tensor_creation()
    
    print("\n" + "=" * 60)
    print("Exercise 2: Tensor Operations")
    print("=" * 60)
    # exercise_2_tensor_operations()
    
    print("\n" + "=" * 60)
    print("Exercise 3: Reshaping")
    print("=" * 60)
    # exercise_3_reshaping()
    
    print("\n" + "=" * 60)
    print("Exercise 4: Autograd")
    print("=" * 60)
    # exercise_4_autograd()
    
    print("\n" + "=" * 60)
    print("Exercise 5: Linear Regression")
    print("=" * 60)
    # exercise_5_linear_regression()
