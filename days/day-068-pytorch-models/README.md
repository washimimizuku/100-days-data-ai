# Day 68: PyTorch Models

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of this session, you will:
- Understand neural network architecture basics
- Build custom models using `nn.Module`
- Implement forward passes and training loops
- Use PyTorch's built-in layers and activation functions
- Save and load trained models
- Apply best practices for model development

---

## Neural Network Basics

### What is a Neural Network?

A neural network is a computational model inspired by biological neurons. It consists of:
- **Input Layer**: Receives raw data
- **Hidden Layers**: Transform data through learned weights
- **Output Layer**: Produces predictions

```python
import torch
import torch.nn as nn

# Simple 3-layer network
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create model
model = SimpleNet(10, 20, 2)
print(model)
```

### Key Components

**Layers**:
- `nn.Linear`: Fully connected layer (y = xW^T + b)
- `nn.Conv2d`: 2D convolution for images
- `nn.LSTM`: Recurrent layer for sequences
- `nn.Embedding`: Lookup table for discrete inputs

**Activation Functions**:
- `nn.ReLU()`: max(0, x) - most common
- `nn.Sigmoid()`: 1/(1+e^-x) - binary classification
- `nn.Tanh()`: Hyperbolic tangent
- `nn.Softmax()`: Probability distribution

**Loss Functions**:
- `nn.MSELoss()`: Mean squared error (regression)
- `nn.CrossEntropyLoss()`: Classification
- `nn.BCELoss()`: Binary cross-entropy

---

## Building Models with nn.Module

### The nn.Module Pattern

All PyTorch models inherit from `nn.Module`:

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers here
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 1)
    
    def forward(self, x):
        # Define forward pass
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x
```

### Sequential Models & Custom Layers

```python
# Simple architectures with nn.Sequential
model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 1))
output = model(torch.randn(5, 10))

# Custom reusable components
class ResidualBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(size, size), nn.Linear(size, size)
    
    def forward(self, x):
        residual = x
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.relu(x + residual)  # Skip connection
```

---

## Training Loop

### Basic Training Pattern

```python
# Setup
model = SimpleNet(10, 20, 2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward pass
    optimizer.zero_grad()  # Clear gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update weights
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
```

### Complete Training Example

```python
def train_model(model, train_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()  # Enable dropout, batch norm
        total_loss = 0
        
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()  # Clear gradients
            loss.backward()        # Compute gradients
            optimizer.step()       # Update weights
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
    return model

# Evaluation mode (disables dropout, batch norm)
model.eval()
with torch.no_grad():  # Disable gradient computation
    predictions = model(test_data)
```

---

## Optimizers

```python
# Common optimizers
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # SGD
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam (adaptive LR)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # Adam + weight decay

# Learning rate scheduling
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Usage with scheduler
for epoch in range(epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        optimizer.step()
    scheduler.step()  # Update learning rate
```

---

## Model Evaluation

```python
def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return 100 * correct / total

def predict(model, x):
    model.eval()
    with torch.no_grad():
        output = model(x)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
    return predicted_class, probabilities
```

---

## Saving and Loading Models

```python
# Save model weights (recommended)
torch.save(model.state_dict(), 'model_weights.pth')

# Save checkpoint (weights + optimizer + metadata)
checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}
torch.save(checkpoint, 'checkpoint.pth')

# Load weights
model = SimpleNet(10, 20, 2)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch, loss = checkpoint['epoch'], checkpoint['loss']
```

---

## Advanced Model Patterns

```python
# Dropout for regularization (only active in training mode)
class RegularizedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.dropout = nn.Dropout(0.5)  # Drop 50% during training
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))

# Batch normalization (normalizes activations)
class BatchNormNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.bn1 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.bn1(self.fc1(x))))

# Multi-input models
class MultiInputNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1, self.branch2 = nn.Linear(10, 20), nn.Linear(5, 20)
        self.combined = nn.Linear(40, 1)
    
    def forward(self, x1, x2):
        out1, out2 = torch.relu(self.branch1(x1)), torch.relu(self.branch2(x2))
        return self.combined(torch.cat([out1, out2], dim=1))
```

---

## Model Inspection

```python
# View model architecture and count parameters
print(model)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total: {total_params}, Trainable: {trainable_params}')

# Inspect layers
for name, param in model.named_parameters():
    print(f'{name}: {param.shape}')

# Hook for intermediate outputs
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

model.fc1.register_forward_hook(get_activation('fc1'))
output = model(x)
fc1_output = activations['fc1']  # Access intermediate activation
```

---

## 💻 Exercises (40 min)

Test your understanding with these hands-on exercises in `exercise.py`:

### Exercise 1: Build a Binary Classifier
Create a neural network for binary classification with 2 hidden layers.

### Exercise 2: Implement Training Loop
Write a complete training loop with loss tracking and validation.

### Exercise 3: Multi-Class Classifier
Build and train a model for multi-class classification (3+ classes).

### Exercise 4: Model Checkpointing
Implement save/load functionality with checkpoints during training.

### Exercise 5: Model Evaluation
Create evaluation functions that compute accuracy, precision, and recall.

---

## ✅ Quiz

Test your understanding of PyTorch models in `quiz.md`.

---

## 🎯 Key Takeaways

- **nn.Module** is the base class for all PyTorch models
- **forward()** method defines the computation graph
- Training loop: forward pass → compute loss → backward pass → update weights
- Use **model.train()** for training and **model.eval()** for inference
- **optimizer.zero_grad()** must be called before each backward pass
- Save model with **state_dict()** for portability
- Use **torch.no_grad()** during evaluation to save memory

---

## 📚 Resources

- [PyTorch nn.Module Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [PyTorch Tutorials - Neural Networks](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)
- [PyTorch Optimizers](https://pytorch.org/docs/stable/optim.html)
- [Model Saving Guide](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- [PyTorch Examples](https://github.com/pytorch/examples)

---

## Tomorrow: Day 69 - Hugging Face Transformers

Learn how to use pre-trained transformer models for NLP and vision tasks with the Hugging Face library.
