"""
Day 67: PyTorch Tensors - Solutions
"""
import torch
import numpy as np


def exercise_1_tensor_creation():
    """Exercise 1: Tensor Creation"""
    print("Creating tensors:\n")
    
    tensor_list = torch.tensor([1, 2, 3, 4, 5])
    print(f"From list: {tensor_list}")
    print(f"Shape: {tensor_list.shape}\n")
    
    zeros = torch.zeros(3, 3)
    print(f"Zeros (3x3):\n{zeros}")
    print(f"Shape: {zeros.shape}\n")
    
    ones = torch.ones(2, 4)
    print(f"Ones (2x4):\n{ones}")
    print(f"Shape: {ones.shape}\n")
    
    rand = torch.rand(3, 3)
    print(f"Random (3x3):\n{rand}")
    print(f"Shape: {rand.shape}\n")
    
    arange = torch.arange(0, 11, 2)
    print(f"Range (0 to 10, step 2): {arange}")
    print(f"Shape: {arange.shape}")


def exercise_2_tensor_operations():
    """Exercise 2: Tensor Operations"""
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    
    print(f"x: {x}")
    print(f"y: {y}\n")
    
    add = x + y
    print(f"x + y: {add}")
    
    mul = x * y
    print(f"x * y: {mul}")
    
    dot = torch.dot(x, y)
    print(f"dot(x, y): {dot}")
    
    A = torch.randn(2, 3)
    B = torch.randn(3, 4)
    C = torch.matmul(A, B)
    
    print(f"\nMatrix multiplication:")
    print(f"A shape: {A.shape}")
    print(f"B shape: {B.shape}")
    print(f"C = A @ B shape: {C.shape}")


def exercise_3_reshaping():
    """Exercise 3: Reshaping Tensors"""
    x = torch.arange(24)
    print(f"Original: shape {x.shape}")
    
    x_4x6 = x.view(4, 6)
    print(f"Reshaped to 4x6: shape {x_4x6.shape}")
    
    x_2x3x4 = x.view(2, 3, 4)
    print(f"Reshaped to 2x3x4: shape {x_2x3x4.shape}")
    
    x_flat = x_2x3x4.view(-1)
    print(f"Flattened: shape {x_flat.shape}")
    
    x_unsqueeze = x_flat.unsqueeze(0)
    print(f"Unsqueezed (dim=0): shape {x_unsqueeze.shape}")
    
    x_squeeze = x_unsqueeze.squeeze()
    print(f"Squeezed: shape {x_squeeze.shape}")


def exercise_4_autograd():
    """Exercise 4: Automatic Differentiation"""
    x = torch.tensor([2.0], requires_grad=True)
    
    y = x**3 + 2*x**2 + x
    
    y.backward()
    
    print(f"x: {x.item()}")
    print(f"y = x^3 + 2x^2 + x: {y.item()}")
    print(f"dy/dx: {x.grad.item()}")
    
    manual_grad = 3 * (2**2) + 4 * 2 + 1
    print(f"\nManual calculation: 3x^2 + 4x + 1 at x=2")
    print(f"= 3(4) + 4(2) + 1 = {manual_grad}")
    print(f"Match: {abs(x.grad.item() - manual_grad) < 0.001}")


def exercise_5_linear_regression():
    """Exercise 5: Linear Regression with Tensors"""
    torch.manual_seed(42)
    X = torch.randn(50, 1)
    y_true = 2 * X + 1 + 0.1 * torch.randn(50, 1)
    
    w = torch.randn(1, 1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    
    learning_rate = 0.01
    epochs = 100
    
    print("Training Linear Regression:")
    print(f"True parameters: w=2.00, b=1.00\n")
    
    for epoch in range(epochs):
        y_pred = X @ w + b
        
        loss = ((y_pred - y_true) ** 2).mean()
        
        loss.backward()
        
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad
            
            w.grad.zero_()
            b.grad.zero_()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}, Loss: {loss.item():.4f}, "
                  f"w: {w.item():.3f}, b: {b.item():.3f}")
    
    print(f"\nFinal learned parameters:")
    print(f"w: {w.item():.3f} (true: 2.00)")
    print(f"b: {b.item():.3f} (true: 1.00)")
    
    w_error = abs(w.item() - 2.0)
    b_error = abs(b.item() - 1.0)
    print(f"\nErrors: w={w_error:.3f}, b={b_error:.3f}")


if __name__ == "__main__":
    print("Day 67: PyTorch Tensors - Solutions\n")
    
    print("=" * 60)
    print("Exercise 1: Tensor Creation")
    print("=" * 60)
    exercise_1_tensor_creation()
    
    print("\n" + "=" * 60)
    print("Exercise 2: Tensor Operations")
    print("=" * 60)
    exercise_2_tensor_operations()
    
    print("\n" + "=" * 60)
    print("Exercise 3: Reshaping")
    print("=" * 60)
    exercise_3_reshaping()
    
    print("\n" + "=" * 60)
    print("Exercise 4: Autograd")
    print("=" * 60)
    exercise_4_autograd()
    
    print("\n" + "=" * 60)
    print("Exercise 5: Linear Regression")
    print("=" * 60)
    exercise_5_linear_regression()
