# Mini Project: ML with MLflow - Detailed Specification

## Project Goal

Build a complete machine learning pipeline with experiment tracking using MLflow to compare models and select the best performer.

---

## Requirements

### Functional Requirements

1. **Data Management**
   - Load breast cancer dataset
   - Split into train/test sets (80/20)
   - Scale features appropriately
   - Validate data quality

2. **Model Training**
   - Train at least 5 different models
   - Use consistent random seeds
   - Apply proper preprocessing
   - Track all experiments

3. **MLflow Integration**
   - Log all parameters
   - Log all metrics
   - Save model artifacts
   - Tag runs appropriately

4. **Model Evaluation**
   - Calculate multiple metrics
   - Compare all models
   - Select best performer
   - Generate comparison report

5. **Model Deployment**
   - Save best model
   - Load model from MLflow
   - Make predictions
   - Validate predictions

### Non-Functional Requirements

- All files < 400 lines
- Clear code structure
- Proper error handling
- Comprehensive logging
- Reproducible results

---

## Data Specification

### Dataset

**Source**: sklearn.datasets.load_breast_cancer()

**Features**:
- 30 numerical features
- 569 samples
- Binary classification (malignant/benign)

**Split**:
- Training: 80% (455 samples)
- Testing: 20% (114 samples)
- Random state: 42

**Preprocessing**:
- StandardScaler for feature scaling
- Fit on training data only

---

## Models to Implement

```python
# 1. Logistic Regression
LogisticRegression(C=1.0, max_iter=1000, random_state=42)

# 2. Decision Tree
DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)

# 3. Random Forest
RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# 4. Support Vector Machine
SVC(C=1.0, kernel='rbf', probability=True, random_state=42)

# 5. K-Nearest Neighbors
KNeighborsClassifier(n_neighbors=5, weights='uniform')
```

---

## MLflow Tracking Specification

### Parameters to Log

For each model:
- `model_type`: Algorithm name
- `random_state`: Random seed
- All hyperparameters specific to model

### Metrics to Log

For each model:
- `accuracy`: Overall accuracy
- `precision`: Weighted precision
- `recall`: Weighted recall
- `f1_score`: Weighted F1 score

### Artifacts to Log

For each model:
- Trained model (sklearn format)
- Feature names
- Scaler object

### Tags to Add

- `dataset`: "breast_cancer"
- `task`: "classification"
- `framework`: "sklearn"

---

## File Specifications

**ml_pipeline.py**: Main orchestrator - `load_data()`, `preprocess_data()`, `train_all_models()`, `evaluate_models()`, `select_best_model()`

**train_models.py**: Model training with MLflow tracking - `train_model()`, `get_model_params()`, `calculate_metrics()`

**evaluate_models.py**: Model comparison - `compare_models()`, `get_best_model()`, `print_comparison()`

**predict.py**: Make predictions - `load_model()`, `predict()`, `predict_proba()`

---

## Expected Metrics

Based on breast cancer dataset:

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~0.96 | ~0.96 | ~0.96 | ~0.96 |
| Decision Tree | ~0.93 | ~0.93 | ~0.93 | ~0.93 |
| Random Forest | ~0.97 | ~0.97 | ~0.97 | ~0.97 |
| SVM | ~0.98 | ~0.98 | ~0.98 | ~0.98 |
| KNN | ~0.96 | ~0.96 | ~0.96 | ~0.96 |

**Best Model**: SVM (typically)

---

## Test Cases

### test_pipeline.sh

```bash
#!/bin/bash

echo "Testing ML Pipeline with MLflow..."

# Test 1: Data loading
python -c "from ml_pipeline import load_data; load_data()"

# Test 2: Model training
python train_models.py

# Test 3: Model evaluation
python evaluate_models.py

# Test 4: Predictions
python predict.py

# Test 5: MLflow tracking
mlflow runs list

echo "All tests passed!"
```

---

## Implementation Steps

**Phase 1 - Setup (15 min)**: Create structure, install dependencies, test MLflow

**Phase 2 - Data Pipeline (20 min)**: Load dataset, split, scale, validate

**Phase 3 - Model Training (40 min)**: Implement training, add MLflow tracking, train 5 models

**Phase 4 - Evaluation (25 min)**: Compare models, calculate metrics, select best

**Phase 5 - Deployment (20 min)**: Save best model, load, predict, validate

---

## MLflow UI Guide

Start: `mlflow ui` → Open: http://localhost:5000 → View runs, compare metrics

**Sections**: Experiments (list), Runs (individual), Metrics (performance), Parameters (hyperparameters), Artifacts (models/files)

---

## Bonus Features

### 1. Hyperparameter Tuning

Add Grid Search with MLflow:

```python
def tune_hyperparameters(model, param_grid, X_train, y_train):
    with mlflow.start_run(run_name="grid_search"):
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)
        
        return grid_search.best_estimator_
```

### 2. Model Registry

Register best model:

```python
def register_best_model(run_id, model_name):
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, model_name)
    
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=1,
        stage="Production"
    )
```

### 3. Visualization

Add confusion matrix:

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def log_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()
```

---

## Validation Checklist

- [ ] Data loads correctly
- [ ] Train/test split is 80/20
- [ ] Features are scaled
- [ ] All 5 models train successfully
- [ ] MLflow tracks all runs
- [ ] Metrics are logged correctly
- [ ] Best model is identified
- [ ] Model can be loaded and used
- [ ] Predictions are accurate
- [ ] Test script passes

---

## Deliverables & Success Metrics

**Code**: ml_pipeline.py, train_models.py, evaluate_models.py, predict.py  
**Config**: requirements.txt, test_pipeline.sh  
**Docs**: README.md, project.md  
**MLflow**: Experiment with all runs, saved models, logged metrics

**Success**: All 5 models train, MLflow UI shows runs, best accuracy > 0.95, model loads/predicts, tests pass
