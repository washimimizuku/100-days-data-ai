"""
Day 64: MLOps Principles - Exercises
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


def exercise_1_model_versioning():
    """
    Exercise 1: Model Versioning with MLflow
    
    Implement model versioning and registry.
    """
    # Load data
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    # TODO: Set experiment name
    # TODO: Train model with MLflow tracking
    # TODO: Log parameters, metrics, and model
    # TODO: Register model in MLflow registry
    # TODO: Transition model to "Staging"
    # TODO: Print model version and stage
    pass


def exercise_2_cicd_workflow():
    """
    Exercise 2: CI/CD Workflow Design
    
    Design a CI/CD workflow for ML model.
    """
    workflow = {
        "continuous_integration": {
            "steps": [],  # TODO: Add CI steps
            "tests": []   # TODO: Add test types
        },
        "continuous_deployment": {
            "stages": [],  # TODO: Add deployment stages
            "rollback": None  # TODO: Add rollback strategy
        }
    }
    
    # TODO: Fill in CI steps (data validation, model training, testing)
    # TODO: Fill in CD stages (staging, production)
    # TODO: Define rollback strategy
    # TODO: Print workflow
    pass


def exercise_3_drift_detection():
    """
    Exercise 3: Drift Detection
    
    Implement data and concept drift detection.
    """
    # Generate reference and current data
    np.random.seed(42)
    reference_data = np.random.normal(0, 1, 1000)
    current_data_no_drift = np.random.normal(0, 1, 1000)
    current_data_with_drift = np.random.normal(0.5, 1.2, 1000)
    
    # TODO: Implement data drift detection using KS test
    # TODO: Test with no drift data
    # TODO: Test with drift data
    # TODO: Print results
    pass


def exercise_4_model_monitoring():
    """
    Exercise 4: Model Monitoring
    
    Build monitoring system for model predictions.
    """
    # Load data and train model
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # TODO: Calculate and log accuracy
    # TODO: Calculate prediction distribution
    # TODO: Calculate average confidence
    # TODO: Check if accuracy below threshold (0.85)
    # TODO: Print monitoring report
    pass


def exercise_5_deployment_strategy():
    """
    Exercise 5: Deployment Strategy
    
    Design deployment strategy for different scenarios.
    """
    scenarios = {
        "batch_prediction": {
            "description": "Daily predictions on large dataset",
            "strategy": None,  # TODO: Choose strategy
            "tools": [],       # TODO: List tools
            "frequency": None  # TODO: Define frequency
        },
        "realtime_api": {
            "description": "Low-latency predictions via API",
            "strategy": None,
            "tools": [],
            "sla": None  # TODO: Define SLA
        },
        "streaming": {
            "description": "Real-time predictions on stream",
            "strategy": None,
            "tools": [],
            "throughput": None  # TODO: Define throughput
        }
    }
    
    # TODO: Fill in deployment strategies
    # TODO: Choose appropriate tools for each
    # TODO: Define SLAs and requirements
    # TODO: Print deployment plan
    pass


if __name__ == "__main__":
    print("Day 64: MLOps Principles - Exercises\n")
    
    print("=" * 60)
    print("Exercise 1: Model Versioning")
    print("=" * 60)
    # exercise_1_model_versioning()
    
    print("\n" + "=" * 60)
    print("Exercise 2: CI/CD Workflow")
    print("=" * 60)
    # exercise_2_cicd_workflow()
    
    print("\n" + "=" * 60)
    print("Exercise 3: Drift Detection")
    print("=" * 60)
    # exercise_3_drift_detection()
    
    print("\n" + "=" * 60)
    print("Exercise 4: Model Monitoring")
    print("=" * 60)
    # exercise_4_model_monitoring()
    
    print("\n" + "=" * 60)
    print("Exercise 5: Deployment Strategy")
    print("=" * 60)
    # exercise_5_deployment_strategy()
