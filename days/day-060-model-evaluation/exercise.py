"""
Day 60: Model Evaluation Metrics - Exercises
"""
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
import numpy as np


def exercise_1_classification_metrics():
    """
    Exercise 1: Classification Metrics
    
    Calculate and interpret all classification metrics.
    """
    # Load breast cancer dataset
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    # TODO: Train LogisticRegression model
    # TODO: Make predictions
    # TODO: Calculate accuracy, precision, recall, F1-score
    # TODO: Print all metrics
    # TODO: Interpret results (which metric is most important?)
    pass


def exercise_2_confusion_matrix():
    """
    Exercise 2: Confusion Matrix Analysis
    
    Analyze confusion matrix to understand errors.
    """
    # Load breast cancer dataset
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    # TODO: Train model and make predictions
    # TODO: Calculate confusion matrix
    # TODO: Extract TP, TN, FP, FN
    # TODO: Calculate metrics manually from confusion matrix
    # TODO: Print confusion matrix and analysis
    pass


def exercise_3_regression_metrics():
    """
    Exercise 3: Regression Metrics
    
    Compare models using different regression metrics.
    """
    # Load diabetes dataset
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    # TODO: Train LinearRegression model
    # TODO: Make predictions
    # TODO: Calculate MAE, MSE, RMSE, R²
    # TODO: Print all metrics
    # TODO: Interpret R² score (is model good?)
    pass


def exercise_4_imbalanced_data():
    """
    Exercise 4: Handle Imbalanced Data
    
    Work with imbalanced dataset and appropriate metrics.
    """
    # Create imbalanced dataset (90% class 0, 10% class 1)
    np.random.seed(42)
    X = np.random.randn(1000, 5)
    y = np.array([0]*900 + [1]*100)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # TODO: Train LogisticRegression model
    # TODO: Make predictions
    # TODO: Calculate accuracy (will be high but misleading)
    # TODO: Calculate precision, recall, F1 for minority class
    # TODO: Print classification report
    # TODO: Explain why accuracy is misleading
    pass


def exercise_5_metric_selection():
    """
    Exercise 5: Choose Appropriate Metrics
    
    Select metrics for different scenarios.
    """
    scenarios = {
        "spam_detection": {
            "description": "Email spam filter",
            "cost": "False positives (good email marked spam) are very costly",
            "metric": None  # TODO: Choose metric
        },
        "cancer_screening": {
            "description": "Cancer detection",
            "cost": "False negatives (missing cancer) are very costly",
            "metric": None  # TODO: Choose metric
        },
        "house_price": {
            "description": "Predict house prices",
            "cost": "Need interpretable error in dollars",
            "metric": None  # TODO: Choose metric
        },
        "fraud_detection": {
            "description": "Credit card fraud (1% fraud rate)",
            "cost": "Imbalanced, need to catch fraud",
            "metric": None  # TODO: Choose metric
        },
        "sales_forecast": {
            "description": "Predict monthly sales",
            "cost": "Need to explain variance",
            "metric": None  # TODO: Choose metric
        }
    }
    
    # TODO: Fill in appropriate metrics for each scenario
    # TODO: Print scenarios with chosen metrics
    # TODO: Explain reasoning for each choice
    pass


if __name__ == "__main__":
    print("Day 60: Model Evaluation Metrics - Exercises\n")
    
    print("=" * 60)
    print("Exercise 1: Classification Metrics")
    print("=" * 60)
    # exercise_1_classification_metrics()
    
    print("\n" + "=" * 60)
    print("Exercise 2: Confusion Matrix")
    print("=" * 60)
    # exercise_2_confusion_matrix()
    
    print("\n" + "=" * 60)
    print("Exercise 3: Regression Metrics")
    print("=" * 60)
    # exercise_3_regression_metrics()
    
    print("\n" + "=" * 60)
    print("Exercise 4: Imbalanced Data")
    print("=" * 60)
    # exercise_4_imbalanced_data()
    
    print("\n" + "=" * 60)
    print("Exercise 5: Metric Selection")
    print("=" * 60)
    # exercise_5_metric_selection()
