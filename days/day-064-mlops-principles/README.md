# Day 64: MLOps Principles

## 📖 Learning Objectives

By the end of this session, you will:
- Understand MLOps and its importance
- Learn the ML lifecycle stages
- Know CI/CD practices for ML
- Understand model versioning and registry
- Learn monitoring and maintenance strategies
- Apply MLOps best practices

**Time**: 1 hour  
**Level**: Intermediate

---

## What is MLOps?

**MLOps** (Machine Learning Operations) is a set of practices that combines ML, DevOps, and Data Engineering to deploy and maintain ML systems in production reliably and efficiently.

### Why MLOps?

**Traditional ML Problems**:
- Models work in notebooks but fail in production
- No reproducibility (can't recreate results)
- Manual deployment processes
- No monitoring of model performance
- Difficult collaboration
- Model drift goes undetected

**MLOps Solutions**:
- Automated pipelines
- Version control for data, code, and models
- Continuous integration and deployment
- Monitoring and alerting
- Reproducible experiments
- Collaboration tools

---

## ML Lifecycle

**6 Stages**: Problem Definition → Data Collection & Preparation → Model Development (feature engineering, selection, tuning) → Model Evaluation (validate, compare baseline) → Model Deployment (package, deploy, A/B test) → Monitoring & Maintenance (track metrics, detect drift, retrain)

---

## MLOps Maturity Levels

| Level | Description | Key Features |
|-------|-------------|--------------|
| 0 | Manual Process | Notebooks, manual deployment, no monitoring |
| 1 | ML Pipeline Automation | Automated training, experiment tracking, model registry |
| 2 | CI/CD Automation | Automated testing, continuous training, deployment |
| 3 | Full Automation | Auto-retraining triggers, feature store, advanced monitoring |

---

## Key MLOps Components

### 1. Version Control

**Code Versioning**:
```bash
# Git for code
git add model.py
git commit -m "Update model architecture"
git push
```

**Data Versioning**:
```python
# DVC (Data Version Control)
dvc add data/train.csv
dvc push
```

**Model Versioning**:
```python
# MLflow Model Registry
mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="customer_churn_model"
)
```

### 2. Experiment Tracking

```python
import mlflow

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("batch_size", 32)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

### 3. Model Registry

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_uri = f"runs:/{run_id}/model"
mv = mlflow.register_model(model_uri, "my_model")

# Transition: None → Staging → Production
client.transition_model_version_stage(name="my_model", version=mv.version, stage="Production")
```

### 4. CI/CD for ML

**Continuous Integration**:
```yaml
# .github/workflows/ml-ci.yml
name: ML CI

on: [push]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: |
          pip install -r requirements.txt
          pytest tests/
      - name: Check data quality
        run: python validate_data.py
      - name: Train model
        run: python train.py
      - name: Evaluate model
        run: python evaluate.py
```

**Continuous Deployment**:
```yaml
# Deploy on merge to main
deploy:
  runs-on: ubuntu-latest
  steps:
    - name: Deploy model
      run: |
        mlflow models serve -m "models:/my_model/Production"
```

### 5. Model Monitoring

```python
import logging

def monitor_predictions(y_true, y_pred, y_proba):
    """Monitor model predictions"""
    
    # Accuracy
    accuracy = (y_true == y_pred).mean()
    logging.info(f"Current accuracy: {accuracy:.3f}")
    
    # Prediction distribution
    pred_dist = np.bincount(y_pred) / len(y_pred)
    logging.info(f"Prediction distribution: {pred_dist}")
    
    # Confidence scores
    avg_confidence = y_proba.max(axis=1).mean()
    logging.info(f"Average confidence: {avg_confidence:.3f}")
    
    # Alert if performance drops
    if accuracy < 0.85:
        logging.warning("Model accuracy below threshold!")
```

---

## Model Deployment Strategies

```python
# 1. Batch Prediction - process large datasets offline
def batch_predict(model, data_path, output_path):
    df = pd.read_csv(data_path)
    df['prediction'] = model.predict(df[feature_columns])
    df.to_csv(output_path, index=False)

# 2. Real-time API - serve predictions via REST API
from fastapi import FastAPI
app = FastAPI()
model = mlflow.sklearn.load_model("models:/my_model/Production")

@app.post("/predict")
def predict(features: dict):
    X = [[features[col] for col in feature_columns]]
    return {"prediction": int(model.predict(X)[0])}

# 3. Streaming - process events from Kafka
from kafka import KafkaConsumer, KafkaProducer
consumer = KafkaConsumer('input_topic')
producer = KafkaProducer('output_topic')
for message in consumer:
    data = json.loads(message.value)
    prediction = model.predict(preprocess(data))
    producer.send('output_topic', json.dumps({'prediction': prediction[0]}))
```

---

## Model Drift Detection

### Types of Drift

**1. Data Drift**: Input distribution changes
```python
from scipy.stats import ks_2samp

def detect_data_drift(reference_data, current_data):
    """Detect drift using Kolmogorov-Smirnov test"""
    statistic, p_value = ks_2samp(reference_data, current_data)
    
    if p_value < 0.05:
        print("Data drift detected!")
        return True
    return False
```

**2. Concept Drift**: Relationship between X and y changes
```python
def detect_concept_drift(model, X_test, y_test, threshold=0.05):
    """Detect concept drift by monitoring accuracy"""
    current_accuracy = model.score(X_test, y_test)
    
    if current_accuracy < baseline_accuracy - threshold:
        print("Concept drift detected!")
        return True
    return False
```

**3. Prediction Drift**: Output distribution changes
```python
def detect_prediction_drift(reference_preds, current_preds):
    """Detect drift in predictions"""
    ref_dist = np.bincount(reference_preds) / len(reference_preds)
    curr_dist = np.bincount(current_preds) / len(current_preds)
    
    # Chi-square test
    from scipy.stats import chisquare
    statistic, p_value = chisquare(curr_dist, ref_dist)
    
    if p_value < 0.05:
        print("Prediction drift detected!")
        return True
    return False
```

---

## Best Practices

```python
# 1. Reproducibility - set seeds, version everything (Git, DVC, MLflow, Docker)
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

# 2. Testing - test accuracy, shapes, data quality
def test_model_accuracy():
    assert load_model().score(X_test, y_test) > 0.85

def test_data_quality():
    df = load_data()
    assert df.isnull().sum().sum() == 0 and len(df) > 1000

# 3. Documentation - document model, features, performance, deployment
"""Model: Customer Churn Predictor v2.1.0
Features: tenure, monthly_charges, total_charges, contract_type
Performance: Accuracy 0.87, Precision 0.85, Recall 0.82, F1 0.83
Deployment: Production, /api/v1/predict, 99.9% uptime, <100ms latency"""

# 4. Monitoring - track predictions, latency, accuracy with Prometheus
import prometheus_client as prom
prediction_counter = prom.Counter('predictions_total', 'Total predictions')
prediction_latency = prom.Histogram('prediction_latency_seconds', 'Latency')
```

---

## MLOps Tools

| Category | Tools |
|----------|-------|
| Experiment Tracking | MLflow, Weights & Biases, Neptune.ai |
| Model Registry | MLflow Model Registry, Seldon Core, BentoML |
| CI/CD | GitHub Actions, GitLab CI, Jenkins |
| Monitoring | Prometheus + Grafana, Evidently AI, WhyLabs |
| Orchestration | Airflow, Kubeflow, Prefect |

---

## 💻 Exercises

Complete the exercises in `exercise.py`:

### Exercise 1: Model Versioning
Implement model versioning with MLflow.

### Exercise 2: CI/CD Pipeline
Create basic CI/CD workflow.

### Exercise 3: Drift Detection
Implement drift detection.

### Exercise 4: Model Monitoring
Build monitoring system.

### Exercise 5: Deployment Strategy
Design deployment approach.

---

## ✅ Quiz

Test your understanding in `quiz.md`.

---

## 🎯 Key Takeaways

- MLOps combines ML, DevOps, and Data Engineering
- Version control everything: code, data, models
- Automate training, testing, and deployment
- Monitor models in production continuously
- Detect and handle model drift
- Use model registry for version management
- Implement CI/CD for ML pipelines
- Test models before deployment

---

## 📚 Resources

- [MLOps Principles](https://ml-ops.org/)
- [Google MLOps Guide](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Evidently AI](https://www.evidentlyai.com/)

---

## Tomorrow: Day 65 - MLflow Tracking

Deep dive into MLflow tracking and experiment management.
