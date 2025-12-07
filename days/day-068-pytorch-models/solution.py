"""
Day 68: PyTorch Models - Solutions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def exercise_1_binary_classifier():
    """Exercise 1: Build a Binary Classifier"""
    
    class BinaryClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(20, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.sigmoid(self.fc3(x))
            return x
    
    model = BinaryClassifier()
    print(model)
    
    # Test with random input
    x = torch.randn(10, 20)
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")


def exercise_2_training_loop():
    """Exercise 2: Implement Training Loop"""
    
    # Generate synthetic data
    torch.manual_seed(42)
    X = torch.randn(1000, 10)
    y = (X.sum(dim=1) > 0).float().unsqueeze(1)
    
    # Create DataLoader
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Define model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1),
        nn.Sigmoid()
    )
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    print("Training binary classifier...")
    for epoch in range(50):
        total_loss = 0
        for batch_X, batch_y in loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}/50, Loss: {avg_loss:.4f}")
    
    print(f"Final loss: {avg_loss:.4f}")


def exercise_3_multiclass_classifier():
    """Exercise 3: Multi-Class Classifier"""
    
    class MultiClassNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(15, 50)
            self.fc2 = nn.Linear(50, 30)
            self.fc3 = nn.Linear(30, 5)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)  # No softmax (CrossEntropyLoss includes it)
            return x
    
    # Generate synthetic data
    torch.manual_seed(42)
    X = torch.randn(500, 15)
    y = torch.randint(0, 5, (500,))
    
    # Create DataLoader
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Setup
    model = MultiClassNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training
    print("Training multi-class classifier...")
    for epoch in range(30):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}/30, Loss: {avg_loss:.4f}")
    
    # Compute accuracy
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y).float().mean()
    
    print(f"Final accuracy: {accuracy:.2%}")


def exercise_4_model_checkpointing():
    """Exercise 4: Model Checkpointing"""
    
    # Create model and data
    torch.manual_seed(42)
    X = torch.randn(200, 5)
    y = (X.sum(dim=1) > 0).float().unsqueeze(1)
    
    model = nn.Sequential(
        nn.Linear(5, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
        nn.Sigmoid()
    )
    
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Training with checkpointing
    print("Training with checkpointing...")
    best_loss = float('inf')
    
    for epoch in range(20):
        outputs = model(X)
        loss = criterion(outputs, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pth')
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f} - Checkpoint saved")
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(checkpoint, 'best_checkpoint.pth')
    
    # Load best checkpoint
    print("\nLoading best checkpoint...")
    checkpoint = torch.load('best_checkpoint.pth')
    
    new_model = nn.Sequential(
        nn.Linear(5, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
        nn.Sigmoid()
    )
    new_model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")
    print(f"Checkpoint loss: {checkpoint['loss']:.4f}")
    
    # Verify
    new_model.eval()
    with torch.no_grad():
        outputs = new_model(X)
        loaded_loss = criterion(outputs, y)
    print(f"Verified loss: {loaded_loss.item():.4f}")


def exercise_5_model_evaluation():
    """Exercise 5: Model Evaluation"""
    
    def evaluate_accuracy(model, X, y):
        model.eval()
        with torch.no_grad():
            outputs = model(X)
            predicted = (outputs > 0.5).float()
            accuracy = (predicted == y).float().mean()
        return accuracy.item()
    
    def evaluate_precision_recall(model, X, y):
        model.eval()
        with torch.no_grad():
            outputs = model(X)
            predicted = (outputs > 0.5).float()
            
            # True positives, false positives, false negatives
            tp = ((predicted == 1) & (y == 1)).sum().float()
            fp = ((predicted == 1) & (y == 0)).sum().float()
            fn = ((predicted == 0) & (y == 1)).sum().float()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return precision.item(), recall.item()
    
    def predict_with_probabilities(model, X):
        model.eval()
        with torch.no_grad():
            probabilities = model(X)
            predicted = (probabilities > 0.5).float()
        return predicted, probabilities
    
    # Train a simple model
    torch.manual_seed(42)
    X_train = torch.randn(300, 8)
    y_train = (X_train.sum(dim=1) > 0).float().unsqueeze(1)
    
    X_test = torch.randn(100, 8)
    y_test = (X_test.sum(dim=1) > 0).float().unsqueeze(1)
    
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid()
    )
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Quick training
    print("Training model for evaluation...")
    for epoch in range(50):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluate
    print("\nEvaluation Results:")
    accuracy = evaluate_accuracy(model, X_test, y_test)
    print(f"Accuracy: {accuracy:.2%}")
    
    precision, recall = evaluate_precision_recall(model, X_test, y_test)
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f"F1 Score: {f1:.2%}")
    
    # Sample predictions
    predicted, probabilities = predict_with_probabilities(model, X_test[:5])
    print("\nSample Predictions:")
    for i in range(5):
        print(f"Sample {i+1}: Prob={probabilities[i].item():.4f}, "
              f"Predicted={int(predicted[i].item())}, "
              f"Actual={int(y_test[i].item())}")


if __name__ == "__main__":
    print("Day 68: PyTorch Models - Solutions\n")
    
    print("=" * 60)
    print("Exercise 1: Binary Classifier")
    print("=" * 60)
    exercise_1_binary_classifier()
    
    print("\n" + "=" * 60)
    print("Exercise 2: Training Loop")
    print("=" * 60)
    exercise_2_training_loop()
    
    print("\n" + "=" * 60)
    print("Exercise 3: Multi-Class Classifier")
    print("=" * 60)
    exercise_3_multiclass_classifier()
    
    print("\n" + "=" * 60)
    print("Exercise 4: Model Checkpointing")
    print("=" * 60)
    exercise_4_model_checkpointing()
    
    print("\n" + "=" * 60)
    print("Exercise 5: Model Evaluation")
    print("=" * 60)
    exercise_5_model_evaluation()
