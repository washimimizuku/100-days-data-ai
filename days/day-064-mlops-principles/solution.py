"""
Day 64: MLOps Principles - Solutions
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import ks_2samp
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


def exercise_1_model_versioning():
    """Exercise 1: Model Versioning with MLflow"""
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    mlflow.set_experiment("mlops_demo")
    
    with mlflow.start_run(run_name="versioned_model"):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", accuracy)
        
        mlflow.sklearn.log_model(model, "model")
        
        run_id = mlflow.active_run().info.run_id
    
    print(f"Model trained with accuracy: {accuracy:.3f}")
    print(f"Run ID: {run_id}")
    
    model_uri = f"runs:/{run_id}/model"
    model_name = "breast_cancer_classifier"
    
    try:
        mv = mlflow.register_model(model_uri, model_name)
        print(f"\nModel registered: {model_name}")
        print(f"Version: {mv.version}")
        
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Staging"
        )
        print(f"Model transitioned to: Staging")
    except Exception as e:
        print(f"Note: Model registry requires MLflow server. Error: {e}")


def exercise_2_cicd_workflow():
    """Exercise 2: CI/CD Workflow Design"""
    workflow = {
        "continuous_integration": {
            "steps": [
                "1. Code checkout",
                "2. Install dependencies",
                "3. Run unit tests",
                "4. Validate data quality",
                "5. Train model",
                "6. Evaluate model",
                "7. Check performance threshold"
            ],
            "tests": [
                "Unit tests (pytest)",
                "Data validation tests",
                "Model performance tests",
                "Integration tests"
            ]
        },
        "continuous_deployment": {
            "stages": [
                "1. Deploy to Staging",
                "2. Run integration tests",
                "3. A/B testing (10% traffic)",
                "4. Monitor metrics",
                "5. Gradual rollout (50%, 100%)",
                "6. Deploy to Production"
            ],
            "rollback": "Automatic rollback if accuracy drops below 85%"
        }
    }
    
    print("CI/CD Workflow for ML Model:\n")
    
    print("Continuous Integration:")
    for step in workflow["continuous_integration"]["steps"]:
        print(f"  {step}")
    
    print("\nTests:")
    for test in workflow["continuous_integration"]["tests"]:
        print(f"  - {test}")
    
    print("\nContinuous Deployment:")
    for stage in workflow["continuous_deployment"]["stages"]:
        print(f"  {stage}")
    
    print(f"\nRollback Strategy:")
    print(f"  {workflow['continuous_deployment']['rollback']}")


def exercise_3_drift_detection():
    """Exercise 3: Drift Detection"""
    np.random.seed(42)
    reference_data = np.random.normal(0, 1, 1000)
    current_data_no_drift = np.random.normal(0, 1, 1000)
    current_data_with_drift = np.random.normal(0.5, 1.2, 1000)
    
    def detect_drift(reference, current, threshold=0.05):
        statistic, p_value = ks_2samp(reference, current)
        drift_detected = p_value < threshold
        return drift_detected, p_value
    
    print("Data Drift Detection:\n")
    
    drift1, p_value1 = detect_drift(reference_data, current_data_no_drift)
    print(f"Test 1: No Drift Expected")
    print(f"  P-value: {p_value1:.4f}")
    print(f"  Drift Detected: {drift1}")
    print(f"  Status: {'❌ ALERT' if drift1 else '✓ OK'}")
    
    print()
    
    drift2, p_value2 = detect_drift(reference_data, current_data_with_drift)
    print(f"Test 2: Drift Expected")
    print(f"  P-value: {p_value2:.4f}")
    print(f"  Drift Detected: {drift2}")
    print(f"  Status: {'❌ ALERT' if drift2 else '✓ OK'}")
    
    print("\nInterpretation:")
    print("  P-value < 0.05 indicates significant distribution change")
    print("  Drift detection triggers model retraining")


def exercise_4_model_monitoring():
    """Exercise 4: Model Monitoring"""
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    pred_dist = np.bincount(y_pred) / len(y_pred)
    avg_confidence = y_proba.max(axis=1).mean()
    
    print("Model Monitoring Report:\n")
    print(f"Performance Metrics:")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Threshold: 0.850")
    print(f"  Status: {'✓ OK' if accuracy >= 0.85 else '❌ BELOW THRESHOLD'}")
    
    print(f"\nPrediction Distribution:")
    print(f"  Class 0 (Malignant): {pred_dist[0]:.1%}")
    print(f"  Class 1 (Benign): {pred_dist[1]:.1%}")
    
    print(f"\nConfidence Metrics:")
    print(f"  Average Confidence: {avg_confidence:.3f}")
    print(f"  Status: {'✓ OK' if avg_confidence > 0.8 else '⚠ LOW CONFIDENCE'}")
    
    if accuracy < 0.85:
        print("\n❌ ALERT: Model accuracy below threshold!")
        print("   Action: Trigger model retraining")
    else:
        print("\n✓ All monitoring checks passed")


def exercise_5_deployment_strategy():
    """Exercise 5: Deployment Strategy"""
    scenarios = {
        "batch_prediction": {
            "description": "Daily predictions on large dataset",
            "strategy": "Scheduled batch processing",
            "tools": ["Apache Spark", "Airflow", "S3"],
            "frequency": "Daily at 2 AM",
            "details": "Process millions of records, store results in data warehouse"
        },
        "realtime_api": {
            "description": "Low-latency predictions via API",
            "strategy": "REST API with load balancing",
            "tools": ["FastAPI", "Docker", "Kubernetes", "Redis cache"],
            "sla": "<100ms latency, 99.9% uptime",
            "details": "Serve predictions via HTTP endpoint with caching"
        },
        "streaming": {
            "description": "Real-time predictions on stream",
            "strategy": "Stream processing pipeline",
            "tools": ["Kafka", "Spark Streaming", "MLflow"],
            "throughput": "10,000 predictions/second",
            "details": "Consume from Kafka, predict, publish results"
        }
    }
    
    print("Deployment Strategy Guide:\n")
    
    for name, config in scenarios.items():
        print(f"{name.replace('_', ' ').title()}:")
        print(f"  Description: {config['description']}")
        print(f"  Strategy: {config['strategy']}")
        print(f"  Tools: {', '.join(config['tools'])}")
        
        if 'frequency' in config:
            print(f"  Frequency: {config['frequency']}")
        if 'sla' in config:
            print(f"  SLA: {config['sla']}")
        if 'throughput' in config:
            print(f"  Throughput: {config['throughput']}")
        
        print(f"  Details: {config['details']}")
        print()


if __name__ == "__main__":
    print("Day 64: MLOps Principles - Solutions\n")
    
    print("=" * 60)
    print("Exercise 1: Model Versioning")
    print("=" * 60)
    exercise_1_model_versioning()
    
    print("\n" + "=" * 60)
    print("Exercise 2: CI/CD Workflow")
    print("=" * 60)
    exercise_2_cicd_workflow()
    
    print("\n" + "=" * 60)
    print("Exercise 3: Drift Detection")
    print("=" * 60)
    exercise_3_drift_detection()
    
    print("\n" + "=" * 60)
    print("Exercise 4: Model Monitoring")
    print("=" * 60)
    exercise_4_model_monitoring()
    
    print("\n" + "=" * 60)
    print("Exercise 5: Deployment Strategy")
    print("=" * 60)
    exercise_5_deployment_strategy()
