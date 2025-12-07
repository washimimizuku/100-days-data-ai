# Day 65: MLflow Tracking

## 📖 Learning Objectives

By the end of this session, you will:
- Master advanced MLflow tracking features
- Organize experiments effectively
- Log custom metrics and parameters
- Track artifacts and models
- Use MLflow UI for analysis
- Implement parent-child runs
- Compare experiments systematically

**Time**: 1 hour  
**Level**: Intermediate

---

## MLflow Tracking Overview

**MLflow Tracking** is a component for logging parameters, metrics, and artifacts during ML experiments.

### Core Concepts

**Run**: Single execution of ML code
**Experiment**: Group of related runs
**Parameters**: Input values (hyperparameters)
**Metrics**: Output values (accuracy, loss)
**Artifacts**: Output files (models, plots, data)
**Tags**: Metadata for organization

---

## Basic Tracking

```python
import mlflow

# Start run (use context manager)
with mlflow.start_run():
    # Log parameters (single or multiple)
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_params({"batch_size": 32, "epochs": 10, "optimizer": "adam"})
    
    # Log metrics (single or multiple)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metrics({"precision": 0.93, "recall": 0.92, "f1_score": 0.925})
    
    # Log metric at specific step (for training curves)
    for epoch in range(10):
        mlflow.log_metric("loss", train_epoch(), step=epoch)
```

---

## Experiment Organization

```python
# Set or create experiment
mlflow.set_experiment("customer_churn")
experiment_id = mlflow.create_experiment("customer_churn", tags={"team": "data-science"})

# Set by ID
mlflow.set_experiment(experiment_id=experiment_id)
```

### Run Names and Tags

```python
with mlflow.start_run(run_name="random_forest_v1"):
    # Set tags
    mlflow.set_tag("model_type", "RandomForest")
    mlflow.set_tag("version", "1.0")
    mlflow.set_tag("developer", "alice")
    
    # Set multiple tags
    mlflow.set_tags({
        "environment": "production",
        "dataset": "2024-Q1"
    })
```

---

## Artifact Logging

### Logging Files

```python
import matplotlib.pyplot as plt

with mlflow.start_run():
    # Create plot
    plt.plot([1, 2, 3], [4, 5, 6])
    plt.savefig("plot.png")
    
    # Log single file
    mlflow.log_artifact("plot.png")
    
    # Log file to specific path
    mlflow.log_artifact("plot.png", "plots")
    
    # Log entire directory
    mlflow.log_artifacts("output_dir", "results")
```

### Logging Models & Data

```python
import mlflow.sklearn
from mlflow.models.signature import infer_signature

with mlflow.start_run():
    model = RandomForestClassifier().fit(X_train, y_train)
    
    # Log model with signature
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(model, "model", signature=signature, registered_model_name="my_model")
    
    # Log data (CSV, JSON)
    df.to_csv("data.csv", index=False); mlflow.log_artifact("data.csv")
    with open("config.json", "w") as f: json.dump(config, f); mlflow.log_artifact("config.json")
```

---

## Parent-Child Runs

### Nested Runs

```python
# Parent run for overall experiment
with mlflow.start_run(run_name="hyperparameter_search"):
    mlflow.log_param("search_type", "grid")
    
    # Child runs for each configuration
    for lr in [0.001, 0.01, 0.1]:
        with mlflow.start_run(run_name=f"lr_{lr}", nested=True):
            mlflow.log_param("learning_rate", lr)
            
            model = train_model(lr)
            accuracy = evaluate(model)
            
            mlflow.log_metric("accuracy", accuracy)
```

### Use Case: Cross-Validation

```python
from sklearn.model_selection import KFold

with mlflow.start_run(run_name="cross_validation"):
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(KFold(n_splits=5).split(X)):
        with mlflow.start_run(run_name=f"fold_{fold}", nested=True):
            model = train_model(X[train_idx], y[train_idx])
            score = model.score(X[val_idx], y[val_idx])
            mlflow.log_metric("accuracy", score)
            fold_scores.append(score)
    
    # Log average in parent run
    mlflow.log_metrics({"avg_accuracy": np.mean(fold_scores), "std_accuracy": np.std(fold_scores)})
```

---

## Searching Runs

```python
from mlflow.tracking import MlflowClient
import mlflow

client = MlflowClient()

# Search runs with filter and ordering
runs = client.search_runs(experiment_ids=["1"], filter_string="metrics.accuracy > 0.9", order_by=["metrics.accuracy DESC"])

# Get runs as DataFrame for analysis
runs_df = mlflow.search_runs(experiment_ids=["1"], filter_string="params.model_type = 'RandomForest'")
best_run = runs_df.loc[runs_df["metrics.accuracy"].idxmax()]
print(f"Best run: {best_run['run_id']}, Accuracy: {best_run['metrics.accuracy']}")
```

---

## Custom Metrics

### Logging Custom Metrics

```python
from sklearn.metrics import confusion_matrix, classification_report

with mlflow.start_run():
    y_pred = model.predict(X_test)
    
    # Log confusion matrix values
    cm = confusion_matrix(y_test, y_pred)
    mlflow.log_metric("true_positives", cm[1, 1])
    mlflow.log_metric("false_positives", cm[0, 1])
    mlflow.log_metric("true_negatives", cm[0, 0])
    mlflow.log_metric("false_negatives", cm[1, 0])
    
    # Log classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{label}_{metric_name}", value)
```

### Time Series Metrics

```python
with mlflow.start_run():
    for epoch in range(100):
        train_loss, val_loss = train_epoch(), validate()
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        if val_loss < best_val_loss:
            best_val_loss = val_loss; mlflow.log_metric("best_val_loss", best_val_loss)
```

---

## Autologging

```python
# Enable autologging for automatic parameter, metric, and model logging
import mlflow.sklearn
mlflow.sklearn.autolog()

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
    score = model.score(X_test, y_test)  # Automatically logged

# Framework-specific: mlflow.tensorflow.autolog(), mlflow.pytorch.autolog(), mlflow.xgboost.autolog()
```

---

## MLflow UI

```bash
# Start UI (default: http://localhost:5000)
mlflow ui

# Custom port or backend
mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db
```

### UI Features

**Experiments View**:
- List all experiments
- Filter and search runs
- Compare runs side-by-side

**Run Details**:
- Parameters and metrics
- Artifacts and models
- Charts and visualizations

**Compare Runs**:
- Select multiple runs
- Compare metrics
- Parallel coordinates plot
- Scatter plots

---

## Best Practices

```python
# 1. Organize experiments with descriptive names and tags
mlflow.set_experiment("customer_churn_2024_q1")
mlflow.set_tags({"team": "data-science", "project": "retention", "priority": "high"})

# 2. Log everything - params, metrics, artifacts, models
with mlflow.start_run():
    mlflow.log_params(model.get_params())
    mlflow.log_metrics({"train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc})
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.sklearn.log_model(model, "model")

# 3. Use meaningful names for runs and parameters
with mlflow.start_run(run_name="rf_100trees_depth10"):
    mlflow.log_param("random_forest_n_estimators", 100)

# 4. Track metadata (git commit, data version, environment, developer)
mlflow.set_tags({"git_commit": "abc123", "data_version": "v2.1", "environment": "production"})
```

---

## Complete Example

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

# Setup
mlflow.set_experiment("breast_cancer_classification")
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Track experiment
with mlflow.start_run(run_name="random_forest_optimized"):
    mlflow.set_tags({"model_type": "RandomForest", "dataset": "breast_cancer"})
    
    # Train and log
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    mlflow.log_params(model.get_params())
    
    # Evaluate and log metrics
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": report['weighted avg']['precision'],
        "recall": report['weighted avg']['recall'],
        "f1_score": report['weighted avg']['f1-score']
    })
    
    # Log artifacts and model
    plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
    plt.savefig("feature_importance.png"); mlflow.log_artifact("feature_importance.png"); plt.close()
    mlflow.sklearn.log_model(model, "model")
```

---

## 💻 Exercises

Complete the exercises in `exercise.py`:

### Exercise 1: Advanced Logging
Log parameters, metrics, and artifacts.

### Exercise 2: Parent-Child Runs
Implement nested runs for cross-validation.

### Exercise 3: Search and Compare
Search runs and compare results.

### Exercise 4: Custom Metrics
Log custom evaluation metrics.

### Exercise 5: Autologging
Use MLflow autologging features.

---

## ✅ Quiz

Test your understanding in `quiz.md`.

---

## 🎯 Key Takeaways

- MLflow Tracking logs parameters, metrics, and artifacts
- Organize experiments with names and tags
- Use parent-child runs for complex workflows
- Search and filter runs programmatically
- Log custom metrics and artifacts
- Autologging simplifies tracking
- MLflow UI provides visualization and comparison
- Track everything for reproducibility

---

## 📚 Resources

- [MLflow Tracking Documentation](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Python API](https://mlflow.org/docs/latest/python_api/index.html)
- [MLflow Examples](https://github.com/mlflow/mlflow/tree/master/examples)

---

## Tomorrow: Day 66 - Model Monitoring

Learn to monitor ML models in production.
