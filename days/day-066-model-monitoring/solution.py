"""
Day 66: Model Monitoring - Solutions
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
    """Exercise 1: Performance Monitoring"""
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print("Performance Monitoring Report:")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    
    threshold = 0.85
    print(f"\nThreshold Check:")
    print(f"  Threshold: {threshold}")
    
    if accuracy < threshold:
        print(f"  ❌ ALERT: Accuracy {accuracy:.3f} below threshold!")
        print(f"  Action: Trigger retraining")
    else:
        print(f"  ✓ OK: Accuracy above threshold")


def exercise_2_drift_detection():
    """Exercise 2: Data Drift Detection"""
    np.random.seed(42)
    reference_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(5, 2, 1000)
    })
    
    current_data = pd.DataFrame({
        'feature1': np.random.normal(0.5, 1.2, 1000),
        'feature2': np.random.normal(5, 2, 1000)
    })
    
    print("Data Drift Detection:\n")
    
    drift_detected = {}
    threshold = 0.05
    
    for col in reference_data.columns:
        statistic, p_value = ks_2samp(
            reference_data[col],
            current_data[col]
        )
        
        drift_detected[col] = p_value < threshold
        
        print(f"{col}:")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Drift: {'Yes' if drift_detected[col] else 'No'}")
        print(f"  Status: {'❌ ALERT' if drift_detected[col] else '✓ OK'}")
        print()
    
    drifted_features = [col for col, drifted in drift_detected.items() if drifted]
    
    if drifted_features:
        print(f"Drifted Features: {', '.join(drifted_features)}")
        print("Action: Investigate data changes and consider retraining")
    else:
        print("No drift detected in any features")


def exercise_3_alerting_system():
    """Exercise 3: Alerting System"""
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
    
    print("Alerting System Report:\n")
    
    alerts = []
    
    if metrics['accuracy'] < thresholds['accuracy']:
        alerts.append({
            'severity': 'HIGH',
            'metric': 'accuracy',
            'message': f"Accuracy {metrics['accuracy']:.3f} below threshold {thresholds['accuracy']}"
        })
    
    if metrics['latency_ms'] > thresholds['latency_ms']:
        alerts.append({
            'severity': 'MEDIUM',
            'metric': 'latency_ms',
            'message': f"Latency {metrics['latency_ms']}ms above threshold {thresholds['latency_ms']}ms"
        })
    
    if metrics['missing_pct'] > thresholds['missing_pct']:
        alerts.append({
            'severity': 'MEDIUM',
            'metric': 'missing_pct',
            'message': f"Missing values {metrics['missing_pct']}% above threshold {thresholds['missing_pct']}%"
        })
    
    if metrics['confidence'] < thresholds['confidence']:
        alerts.append({
            'severity': 'LOW',
            'metric': 'confidence',
            'message': f"Confidence {metrics['confidence']:.3f} below threshold {thresholds['confidence']}"
        })
    
    if alerts:
        print(f"Total Alerts: {len(alerts)}\n")
        for alert in alerts:
            print(f"[{alert['severity']}] {alert['message']}")
    else:
        print("No alerts triggered - all metrics within thresholds")


def exercise_4_dashboard_metrics():
    """Exercise 4: Dashboard Metrics"""
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
    low_confidence_pct = (y_proba.max(axis=1) < 0.7).mean() * 100
    
    dashboard = {
        'current_accuracy': accuracy,
        'total_predictions': len(y_pred),
        'pred_class_0': pred_dist[0],
        'pred_class_1': pred_dist[1],
        'avg_confidence': avg_confidence,
        'low_confidence_pct': low_confidence_pct
    }
    
    print("Model Monitoring Dashboard:\n")
    print(f"Performance:")
    print(f"  Current Accuracy: {dashboard['current_accuracy']:.3f}")
    print(f"  Total Predictions: {dashboard['total_predictions']}")
    
    print(f"\nPrediction Distribution:")
    print(f"  Class 0: {dashboard['pred_class_0']:.1%}")
    print(f"  Class 1: {dashboard['pred_class_1']:.1%}")
    
    print(f"\nConfidence Metrics:")
    print(f"  Average Confidence: {dashboard['avg_confidence']:.3f}")
    print(f"  Low Confidence (<0.7): {dashboard['low_confidence_pct']:.1f}%")


def exercise_5_retraining_trigger():
    """Exercise 5: Retraining Trigger"""
    metrics_history = [
        {'accuracy': 0.92, 'drift_pvalue': 0.15},
        {'accuracy': 0.90, 'drift_pvalue': 0.12},
        {'accuracy': 0.88, 'drift_pvalue': 0.08},
        {'accuracy': 0.85, 'drift_pvalue': 0.06},
        {'accuracy': 0.82, 'drift_pvalue': 0.03}
    ]
    
    accuracy_threshold = 0.85
    drift_threshold = 0.05
    
    recent_metrics = metrics_history[-3:]
    
    recent_accuracy = np.mean([m['accuracy'] for m in recent_metrics])
    recent_drift = np.mean([m['drift_pvalue'] for m in recent_metrics])
    
    print("Retraining Trigger Analysis:\n")
    print(f"Recent Metrics (last 3):")
    for i, m in enumerate(recent_metrics, 1):
        print(f"  {i}. Accuracy: {m['accuracy']:.3f}, Drift P-value: {m['drift_pvalue']:.3f}")
    
    print(f"\nAverages:")
    print(f"  Recent Accuracy: {recent_accuracy:.3f}")
    print(f"  Recent Drift P-value: {recent_drift:.3f}")
    
    print(f"\nThresholds:")
    print(f"  Accuracy Threshold: {accuracy_threshold}")
    print(f"  Drift Threshold: {drift_threshold}")
    
    should_retrain = False
    reasons = []
    
    if recent_accuracy < accuracy_threshold:
        should_retrain = True
        reasons.append(f"Accuracy {recent_accuracy:.3f} < {accuracy_threshold}")
    
    if recent_drift < drift_threshold:
        should_retrain = True
        reasons.append(f"Drift detected (p={recent_drift:.3f} < {drift_threshold})")
    
    print(f"\nDecision: {'RETRAIN' if should_retrain else 'CONTINUE MONITORING'}")
    
    if should_retrain:
        print(f"Reasons:")
        for reason in reasons:
            print(f"  - {reason}")
    else:
        print("Model performance is acceptable")


if __name__ == "__main__":
    print("Day 66: Model Monitoring - Solutions\n")
    
    print("=" * 60)
    print("Exercise 1: Performance Monitoring")
    print("=" * 60)
    exercise_1_performance_monitoring()
    
    print("\n" + "=" * 60)
    print("Exercise 2: Drift Detection")
    print("=" * 60)
    exercise_2_drift_detection()
    
    print("\n" + "=" * 60)
    print("Exercise 3: Alerting System")
    print("=" * 60)
    exercise_3_alerting_system()
    
    print("\n" + "=" * 60)
    print("Exercise 4: Dashboard Metrics")
    print("=" * 60)
    exercise_4_dashboard_metrics()
    
    print("\n" + "=" * 60)
    print("Exercise 5: Retraining Trigger")
    print("=" * 60)
    exercise_5_retraining_trigger()
