# Day 63: Mini Project - ML with MLflow

## 🎯 Project Overview

Build a complete machine learning pipeline with experiment tracking using MLflow. Compare multiple algorithms, track experiments, and select the best model.

**Time**: 2 hours  
**Difficulty**: Intermediate

---

## What You'll Build

A complete ML system that:
- Loads and preprocesses data
- Trains multiple models
- Tracks experiments with MLflow
- Compares model performance
- Selects and registers best model
- Makes predictions with saved model

---

## Learning Objectives

- Build end-to-end ML pipeline
- Track experiments with MLflow
- Compare multiple models systematically
- Use MLflow Model Registry
- Deploy and use saved models
- Follow ML best practices

---

## Project Structure

```
day-063-mini-project-ml-mlflow/
├── README.md              # This file
├── project.md             # Detailed specification
├── ml_pipeline.py         # Main ML pipeline
├── train_models.py        # Model training with MLflow
├── evaluate_models.py     # Model evaluation
├── predict.py             # Make predictions
├── requirements.txt       # Dependencies
└── test_pipeline.sh       # Test script
```

---

## MLflow Overview

**MLflow** is an open-source platform for managing the ML lifecycle with 4 components:
1. **Tracking**: Log parameters, metrics, artifacts
2. **Projects**: Package code in reusable format
3. **Models**: Deploy models to various platforms
4. **Registry**: Centralized model store

Track experiments, compare models, reproduce results, and deploy consistently.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
python ml_pipeline.py
```

### 3. View MLflow UI

```bash
mlflow ui
```

Open http://localhost:5000 in browser

### 4. Run Tests

```bash
./test_pipeline.sh
```

---

## Implementation Guide

### Step 1: Data Preparation

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Step 2: MLflow Tracking

```python
import mlflow
import mlflow.sklearn

# Start MLflow run
with mlflow.start_run(run_name="logistic_regression"):
    # Log parameters
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("C", 1.0)
    
    # Train model
    model = LogisticRegression(C=1.0)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    accuracy = model.score(X_test_scaled, y_test)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

### Step 3: Compare Models

```python
models = {
    'LogisticRegression': LogisticRegression(),
    'RandomForest': RandomForestClassifier(),
    'SVM': SVC()
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        mlflow.log_param("model_type", name)
        model.fit(X_train_scaled, y_train)
        accuracy = model.score(X_test_scaled, y_test)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
```

### Step 4: Load Best Model

```python
# Get best run
runs = mlflow.search_runs(order_by=["metrics.accuracy DESC"])
best_run_id = runs.iloc[0]['run_id']

# Load model
model_uri = f"runs:/{best_run_id}/model"
loaded_model = mlflow.sklearn.load_model(model_uri)

# Predict
predictions = loaded_model.predict(X_new)
```

---

## Features to Implement

**Core Features**: Data pipeline (load, split, scale), model training (multiple algorithms, MLflow tracking), evaluation (metrics, comparison), prediction (load model, predict)

**MLflow Features**: Experiment tracking (log params/metrics/artifacts), model registry (register, version, stage transitions)

---

## Expected Output

### Console Output

```
=== ML Pipeline with MLflow ===

Loading data...
Dataset: Breast Cancer
Samples: 569, Features: 30

Training models...
[1/5] Training Logistic Regression...
  Accuracy: 0.965
  Run ID: abc123

[2/5] Training Random Forest...
  Accuracy: 0.974
  Run ID: def456

[3/5] Training SVM...
  Accuracy: 0.982
  Run ID: ghi789

Best Model: SVM (0.982)
Model saved to MLflow

View results: mlflow ui
```

### MLflow UI

- Experiment list with all runs
- Metrics comparison charts
- Parameter comparison
- Model artifacts
- Run details

---

## Testing

The test script validates:
- Data loading and preprocessing
- Model training
- MLflow tracking
- Model saving and loading
- Predictions

```bash
./test_pipeline.sh
```

---

## Bonus Challenges

### 1. Hyperparameter Tuning

Add Grid Search with MLflow tracking:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

with mlflow.start_run():
    grid_search = GridSearchCV(SVC(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_score", grid_search.best_score_)
```

### 2. Model Registry

Register and version models:

```python
# Register model
model_uri = f"runs:/{run_id}/model"
mlflow.register_model(model_uri, "breast_cancer_classifier")

# Transition to production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="breast_cancer_classifier",
    version=1,
    stage="Production"
)
```

### 3. Custom Metrics

Log custom visualizations:

```python
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()

# Save and log
plt.savefig("confusion_matrix.png")
mlflow.log_artifact("confusion_matrix.png")
```

---

## Key Concepts

**MLflow Tracking**: Run (single execution), Experiment (group of runs), Parameters (hyperparameters), Metrics (accuracy, loss), Artifacts (models, plots)

**Best Practices**: Organize experiments, log everything, use tags, version models, document runs

---

## Common Issues

```bash
# MLflow UI not starting - use different port
mlflow ui --port 5001

# Model not found - check run ID
mlflow runs list
mlflow artifacts list -r <run_id>

# Import errors - reinstall dependencies
pip install -r requirements.txt --upgrade
```

---

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Models](https://mlflow.org/docs/latest/models.html)
- [MLflow Registry](https://mlflow.org/docs/latest/model-registry.html)

---

## Success Criteria

✅ Load and preprocess data | ✅ Train 3+ models | ✅ Track with MLflow | ✅ Compare and select best | ✅ Save/load models | ✅ Make predictions | ✅ All tests pass

**Next Steps**: Explore Model Registry, try different datasets, add more models, implement hyperparameter tuning, deploy to production

---

## Detailed Specification

See `project.md` for detailed requirements and implementation guide.
