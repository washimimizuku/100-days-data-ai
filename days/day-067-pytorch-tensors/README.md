# Day 67: PyTorch Tensors

## 📖 Learning Objectives

By the end of this session, you will:
- Understand PyTorch tensors and their importance
- Create and manipulate tensors
- Perform tensor operations
- Use GPU acceleration
- Understand autograd for automatic differentiation
- Apply tensors to real problems

**Time**: 1 hour  
**Level**: Beginner

---

## What is PyTorch?

**PyTorch** is an open-source deep learning framework developed by Facebook AI Research.

**Why PyTorch?**
- Pythonic and intuitive
- Dynamic computation graphs
- Strong GPU acceleration
- Excellent for research and production
- Large community and ecosystem

---

## Tensors Basics

**Tensor**: Multi-dimensional array (like NumPy arrays but with GPU support)

### Creating Tensors

```python
import torch
import numpy as np

# From Python list or NumPy array
tensor = torch.tensor([1, 2, 3, 4])  # From list
tensor = torch.from_numpy(np.array([1, 2, 3, 4]))  # From NumPy

# Zeros, ones, random
zeros = torch.zeros(3, 4)  # 3x4 zeros
ones = torch.ones(2, 3)    # 2x3 ones
rand = torch.rand(2, 3)    # Uniform [0, 1)
randn = torch.randn(2, 3)  # Normal distribution

# Range
arange = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, 5)  # 5 values from 0 to 1

# Tensor attributes
tensor = torch.randn(3, 4)
print(tensor.shape, tensor.dtype, tensor.device, tensor.requires_grad)
# torch.Size([3, 4]), torch.float32, cpu, False
```

---

## Tensor Operations

### Basic Math & Matrix Operations

```python
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

# Element-wise operations
print(x + y, x - y, x * y, x / y)  # tensor([5, 7, 9]), tensor([-3, -3, -3]), ...
x.add_(y)  # In-place: x = x + y
x.mul_(2)  # In-place: x = x * 2

# Matrix multiplication
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = A @ B  # or torch.matmul(A, B)
print(C.shape)  # torch.Size([3, 5])

# Transpose and dot product
A_T = A.T  # or A.t()
dot = torch.dot(x, y)  # 1*4 + 2*5 + 3*6 = 32
```

### Reshaping

```python
x = torch.arange(12)
print(x.shape)  # torch.Size([12])

# Reshape
x = x.view(3, 4)  # or x.reshape(3, 4)
print(x.shape)  # torch.Size([3, 4])

# Flatten
x = x.view(-1)  # -1 infers dimension
print(x.shape)  # torch.Size([12])

# Add dimension
x = x.unsqueeze(0)  # Add dimension at position 0
print(x.shape)  # torch.Size([1, 12])

# Remove dimension
x = x.squeeze()
print(x.shape)  # torch.Size([12])
```

---

## Indexing and Slicing

```python
x = torch.arange(12).view(3, 4)
# tensor([[ 0,  1,  2,  3], [ 4,  5,  6,  7], [ 8,  9, 10, 11]])

# Indexing: x[0] (first row), x[:, 0] (first column), x[1, 2] (element at 1,2)
print(x[0:2, 1:3])  # Rows 0-1, columns 1-2: tensor([[1, 2], [5, 6]])

# Boolean indexing
print(x[x > 5])  # tensor([ 6,  7,  8,  9, 10, 11])
```

---

## GPU Acceleration

### Moving to GPU

```python
# Check if GPU available
print(torch.cuda.is_available())

# Create tensor on GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.randn(3, 4, device=device)
    
    # Or move existing tensor
    y = torch.randn(3, 4)
    y = y.to(device)
    
    # Operations on GPU
    z = x + y  # Computed on GPU
    
    # Move back to CPU
    z_cpu = z.to("cpu")
```

### Device Management

```python
# Best practice: device-agnostic code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.randn(3, 4).to(device)
y = torch.randn(3, 4).to(device)
z = x + y  # Works on GPU or CPU
```

---

## Autograd (Automatic Differentiation)

### Gradient Tracking

```python
# Enable gradient tracking
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2  # y = x²

# Compute gradients
y.backward()  # dy/dx = 2x

print(x.grad)  # tensor([4.0]) because 2*2 = 4
```

### Example: Linear Function

```python
# y = 3x + 2
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = 3 * x + 2

# Compute gradients
loss = y.sum()
loss.backward()

print(x.grad)  # tensor([3., 3., 3.]) because dy/dx = 3
```

### Gradient Accumulation

```python
x = torch.tensor([2.0], requires_grad=True)

# First computation
y = x ** 2
y.backward()
print(x.grad)  # tensor([4.0])

# Second computation (gradients accumulate!)
y = x ** 3
y.backward()
print(x.grad)  # tensor([16.0]) = 4 + 12

# Zero gradients before new computation
x.grad.zero_()
y = x ** 3
y.backward()
print(x.grad)  # tensor([12.0])
```

---

## Common Tensor Operations

### Aggregations & Concatenation

```python
x = torch.randn(3, 4)

# Aggregations: sum, mean, max, min (all support dim parameter)
print(x.sum(), x.sum(dim=0), x.sum(dim=1))  # All elements, column sums, row sums
print(x.mean(), x.max(), x.argmax())  # Mean, max value, index of max

# Concatenation
y = torch.randn(2, 3)
z = torch.randn(2, 3)
cat_v = torch.cat([y, z], dim=0)  # Vertical: torch.Size([4, 3])
cat_h = torch.cat([y, z], dim=1)  # Horizontal: torch.Size([2, 6])
stack = torch.stack([y, z], dim=0)  # New dimension: torch.Size([2, 2, 3])
```

---

## Practical Example: Linear Regression

```python
import torch
import matplotlib.pyplot as plt

# Generate data: y = 3x + 2 + noise
torch.manual_seed(42)
X = torch.randn(100, 1)
y_true = 3 * X + 2 + 0.1 * torch.randn(100, 1)

# Initialize parameters
w = torch.randn(1, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# Training
learning_rate = 0.01
epochs = 100

for epoch in range(epochs):
    # Forward pass
    y_pred = X @ w + b
    
    # Loss (MSE)
    loss = ((y_pred - y_true) ** 2).mean()
    
    # Backward pass
    loss.backward()
    
    # Update parameters
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        
        # Zero gradients
        w.grad.zero_()
        b.grad.zero_()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print(f"\nLearned: w={w.item():.2f}, b={b.item():.2f}")
print(f"True: w=3.00, b=2.00")
```

---

## NumPy Interoperability

```python
import numpy as np

# NumPy ↔ PyTorch (they share memory!)
np_array = np.array([1, 2, 3])
tensor = torch.from_numpy(np_array)  # NumPy to PyTorch
np_array = tensor.numpy()  # PyTorch to NumPy

# Shared memory example
np_array[0] = 10
print(tensor)  # tensor([10, 2, 3]) - changed!
```

---

## Best Practices

```python
# 1. Device-agnostic code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)
data = data.to(device)

# 2. Detach from computation graph when needed
y_detached = y.detach()  # No gradient tracking

# 3. Use torch.no_grad() for inference
with torch.no_grad():
    predictions = model(data)

# 4. Set random seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed(42)
```

---

## 💻 Exercises

Complete the exercises in `exercise.py`:

### Exercise 1: Tensor Creation
Create tensors using different methods.

### Exercise 2: Tensor Operations
Perform mathematical operations.

### Exercise 3: Reshaping
Practice tensor reshaping.

### Exercise 4: Autograd
Use automatic differentiation.

### Exercise 5: Linear Regression
Implement linear regression with tensors.

---

## ✅ Quiz

Test your understanding in `quiz.md`.

---

## 🎯 Key Takeaways

- Tensors are multi-dimensional arrays with GPU support
- PyTorch provides NumPy-like operations
- Use .to(device) for GPU acceleration
- Autograd enables automatic differentiation
- requires_grad=True tracks gradients
- Use torch.no_grad() for inference
- Tensors and NumPy arrays can share memory
- Always zero gradients before backward pass

---

## 📚 Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Tensor Operations](https://pytorch.org/docs/stable/torch.html)

---

## Tomorrow: Day 68 - PyTorch Models

Learn to build neural networks with PyTorch.
