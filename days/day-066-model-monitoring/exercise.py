"""
Day 66: Model Monitoring - Exercises
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from scipy.stats import ks_2samp
import time


def exercise_1_performance_monitoring():
    """
    Exercise 1: Performance Monitoring
    
    Track model performance metrics over time.
    """
    # Load and split data
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # TODO: Make predictions
    # TODO: Calculate accuracy, precision, recall
    # TODO: Check if accuracy below threshold (0.85)
    # TODO: Log metrics
    # TODO: Print monitoring report
    pass


def exercise_2_drift_detection():
    """
    Exercise 2: Data Drift Detection
    
    Implement drift detection using KS test.
    """
    # Generate reference and current data
    np.random.seed(42)
    reference_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(5, 2, 1000)
    })
    
    # Current data with drift in feature1
    current_data = pd.DataFrame({
        'feature1': np.random.normal(0.5, 1.2, 1000),
        'feature2': np.random.normal(5, 2, 1000)
    })
    
    # TODO: For each feature, perform KS test
    # TODO: Check if p-value < 0.05 (drift detected)
    # TODO: Print drift detection results
    # TODO: Identify which features have drifted
    pass


def exercise_3_alerting_system():
    """
    Exercise 3: Alerting System
    
    Build alert system with thresholds.
    """
    # Simulate metrics
    metrics = {
        'accuracy': 0.82,
        'latency_ms': 150,
        'missing_pct': 8,
        'confidence': 0.65
    }
    
    thresholds = {
        'accuracy': 0.85,
        'latency_ms': 100,
        'missing_pct': 5,
        'confidence': 0.70
    }
    
    # TODO: Check each metric against threshold
    # TODO: Trigger alerts for violations
    # TODO: Assign severity (HIGH, MEDIUM, LOW)
    # TODO: Print alert summary
    pass


def exercise_4_dashboard_metrics():
    """
    Exercise 4: Dashboard Metrics
    
    Create monitoring dashboard data.
    """
    # Load data
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # TODO: Calculate current accuracy
    # TODO: Calculate prediction distribution
    # TODO: Calculate average confidence
    # TODO: Calculate low confidence percentage
    # TODO: Create dashboard dictionary
    # TODO: Print dashboard
    pass


def exercise_5_retraining_trigger():
    """
    Exercise 5: Retraining Trigger
    
    Implement automatic retraining logic.
    """
    # Simulate metrics over time
    metrics_history = [
        {'accuracy': 0.92, 'drift_pvalue': 0.15},
        {'accuracy': 0.90, 'drift_pvalue': 0.12},
        {'accuracy': 0.88, 'drift_pvalue': 0.08},
        {'accuracy': 0.85, 'drift_pvalue': 0.06},
        {'accuracy': 0.82, 'drift_pvalue': 0.03}
    ]
    
    accuracy_threshold = 0.85
    drift_threshold = 0.05
    
    # TODO: Check recent accuracy (last 3 metrics)
    # TODO: Check recent drift (last 3 metrics)
    # TODO: Determine if retraining needed
    # TODO: Print decision and reasoning
    pass


if __name__ == "__main__":
    print("Day 66: Model Monitoring - Exercises\n")
    
    print("=" * 60)
    print("Exercise 1: Performance Monitoring")
    print("=" * 60)
    # exercise_1_performance_monitoring()
    
    print("\n" + "=" * 60)
    print("Exercise 2: Drift Detection")
    print("=" * 60)
    # exercise_2_drift_detection()
    
    print("\n" + "=" * 60)
    print("Exercise 3: Alerting System")
    print("=" * 60)
    # exercise_3_alerting_system()
    
    print("\n" + "=" * 60)
    print("Exercise 4: Dashboard Metrics")
    print("=" * 60)
    # exercise_4_dashboard_metrics()
    
    print("\n" + "=" * 60)
    print("Exercise 5: Retraining Trigger")
    print("=" * 60)
    # exercise_5_retraining_trigger()
