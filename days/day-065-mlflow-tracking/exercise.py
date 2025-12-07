"""
Day 65: MLflow Tracking - Exercises
"""
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def exercise_1_advanced_logging():
    """
    Exercise 1: Advanced Logging
    
    Log parameters, metrics, and artifacts comprehensively.
    """
    # Load data
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    # TODO: Set experiment name
    # TODO: Start run with descriptive name
    # TODO: Set tags (model_type, dataset, developer)
    # TODO: Train RandomForestClassifier
    # TODO: Log all model parameters
    # TODO: Log accuracy, precision, recall, F1
    # TODO: Create and log feature importance plot
    # TODO: Log model
    # TODO: Print run ID
    pass


def exercise_2_parent_child_runs():
    """
    Exercise 2: Parent-Child Runs
    
    Implement nested runs for cross-validation.
    """
    # Load data
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # TODO: Start parent run for cross-validation
    # TODO: Log CV parameters (n_splits)
    # TODO: For each fold, create child run
    # TODO: Train model on fold
    # TODO: Log fold accuracy
    # TODO: In parent run, log average and std accuracy
    # TODO: Print summary
    pass


def exercise_3_search_compare():
    """
    Exercise 3: Search and Compare Runs
    
    Search runs and compare results.
    """
    # TODO: Train 3 different models with MLflow tracking
    #       - LogisticRegression
    #       - RandomForestClassifier (n_estimators=50)
    #       - RandomForestClassifier (n_estimators=100)
    # TODO: Search all runs in experiment
    # TODO: Filter runs with accuracy > 0.95
    # TODO: Find best run by accuracy
    # TODO: Print comparison table
    pass


def exercise_4_custom_metrics():
    """
    Exercise 4: Custom Metrics
    
    Log custom evaluation metrics.
    """
    # Load data
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    # TODO: Train model
    # TODO: Make predictions
    # TODO: Calculate confusion matrix
    # TODO: Log TP, FP, TN, FN as metrics
    # TODO: Calculate and log specificity
    # TODO: Calculate and log sensitivity
    # TODO: Print all metrics
    pass


def exercise_5_autologging():
    """
    Exercise 5: Autologging
    
    Use MLflow autologging features.
    """
    # Load data
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    # TODO: Enable sklearn autologging
    # TODO: Start run
    # TODO: Train RandomForestClassifier
    # TODO: Evaluate on test set
    # TODO: Check what was automatically logged
    # TODO: Print run details
    pass


if __name__ == "__main__":
    print("Day 65: MLflow Tracking - Exercises\n")
    
    print("=" * 60)
    print("Exercise 1: Advanced Logging")
    print("=" * 60)
    # exercise_1_advanced_logging()
    
    print("\n" + "=" * 60)
    print("Exercise 2: Parent-Child Runs")
    print("=" * 60)
    # exercise_2_parent_child_runs()
    
    print("\n" + "=" * 60)
    print("Exercise 3: Search and Compare")
    print("=" * 60)
    # exercise_3_search_compare()
    
    print("\n" + "=" * 60)
    print("Exercise 4: Custom Metrics")
    print("=" * 60)
    # exercise_4_custom_metrics()
    
    print("\n" + "=" * 60)
    print("Exercise 5: Autologging")
    print("=" * 60)
    # exercise_5_autologging()
