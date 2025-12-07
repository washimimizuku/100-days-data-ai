"""
Day 60: Model Evaluation Metrics - Solutions
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
    """Exercise 1: Classification Metrics"""
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Dataset: Breast Cancer (Binary Classification)")
    print(f"Classes: {data.target_names}")
    print(f"Test samples: {len(y_test)}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.3f} - Overall correctness")
    print(f"  Precision: {precision:.3f} - Of predicted malignant, % correct")
    print(f"  Recall:    {recall:.3f} - Of actual malignant, % found")
    print(f"  F1-Score:  {f1:.3f} - Balance of precision and recall")
    
    print(f"\nInterpretation:")
    print(f"  For cancer detection, RECALL is most important")
    print(f"  We want to catch all cancer cases (minimize false negatives)")
    print(f"  Current recall of {recall:.1%} means we find {recall:.1%} of cancers")


def exercise_2_confusion_matrix():
    """Exercise 2: Confusion Matrix Analysis"""
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("Confusion Matrix:")
    print(f"                Predicted")
    print(f"                Neg    Pos")
    print(f"Actual  Neg     {tn:<6} {fp:<6}")
    print(f"        Pos     {fn:<6} {tp:<6}")
    
    print(f"\nBreakdown:")
    print(f"  True Negatives (TN):  {tn} - Correctly predicted benign")
    print(f"  False Positives (FP): {fp} - Benign predicted as malignant")
    print(f"  False Negatives (FN): {fn} - Malignant predicted as benign (BAD!)")
    print(f"  True Positives (TP):  {tp} - Correctly predicted malignant")
    
    print(f"\nManual Metric Calculation:")
    accuracy_manual = (tp + tn) / (tp + tn + fp + fn)
    precision_manual = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_manual = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_manual = 2 * (precision_manual * recall_manual) / (precision_manual + recall_manual) if (precision_manual + recall_manual) > 0 else 0
    
    print(f"  Accuracy:  {accuracy_manual:.3f}")
    print(f"  Precision: {precision_manual:.3f}")
    print(f"  Recall:    {recall_manual:.3f}")
    print(f"  F1-Score:  {f1_manual:.3f}")


def exercise_3_regression_metrics():
    """Exercise 3: Regression Metrics"""
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Dataset: Diabetes Progression")
    print(f"Test samples: {len(y_test)}")
    print(f"Target range: {y_test.min():.1f} to {y_test.max():.1f}")
    
    print(f"\nRegression Metrics:")
    print(f"  MAE:  {mae:.2f} - Average error")
    print(f"  MSE:  {mse:.2f} - Squared error (penalizes large errors)")
    print(f"  RMSE: {rmse:.2f} - Root squared error (same units as target)")
    print(f"  R²:   {r2:.3f} - Variance explained (0 to 1)")
    
    print(f"\nInterpretation:")
    if r2 > 0.7:
        quality = "Good"
    elif r2 > 0.5:
        quality = "Moderate"
    else:
        quality = "Poor"
    print(f"  R² of {r2:.3f} indicates {quality} model")
    print(f"  Model explains {r2*100:.1f}% of variance")
    print(f"  Average prediction error: {mae:.2f} units")


def exercise_4_imbalanced_data():
    """Exercise 4: Handle Imbalanced Data"""
    np.random.seed(42)
    X = np.random.randn(1000, 5)
    y = np.array([0]*900 + [1]*100)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"Imbalanced Dataset:")
    print(f"  Class 0: {(y_train == 0).sum()} samples (90%)")
    print(f"  Class 1: {(y_train == 1).sum()} samples (10%)")
    
    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.3f} (MISLEADING!)")
    print(f"  Precision: {precision:.3f} (for minority class)")
    print(f"  Recall:    {recall:.3f} (for minority class)")
    print(f"  F1-Score:  {f1:.3f} (for minority class)")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))
    
    print(f"\nWhy Accuracy is Misleading:")
    print(f"  A model predicting all 0s would get {0.9:.1%} accuracy")
    print(f"  But it would miss all minority class samples!")
    print(f"  Use precision, recall, F1 for imbalanced data")


def exercise_5_metric_selection():
    """Exercise 5: Choose Appropriate Metrics"""
    scenarios = {
        "spam_detection": {
            "description": "Email spam filter",
            "cost": "False positives (good email marked spam) are very costly",
            "metric": "Precision",
            "reason": "Minimize false positives - don't want to lose important emails"
        },
        "cancer_screening": {
            "description": "Cancer detection",
            "cost": "False negatives (missing cancer) are very costly",
            "metric": "Recall",
            "reason": "Catch all cancer cases - false negatives are dangerous"
        },
        "house_price": {
            "description": "Predict house prices",
            "cost": "Need interpretable error in dollars",
            "metric": "MAE",
            "reason": "Average error in dollars - easy to understand"
        },
        "fraud_detection": {
            "description": "Credit card fraud (1% fraud rate)",
            "cost": "Imbalanced, need to catch fraud",
            "metric": "F1-Score or Recall",
            "reason": "Balance catching fraud while minimizing false alarms"
        },
        "sales_forecast": {
            "description": "Predict monthly sales",
            "cost": "Need to explain variance",
            "metric": "R²",
            "reason": "Shows how well model explains sales variation"
        }
    }
    
    print("Metric Selection for Different Scenarios:\n")
    
    for name, info in scenarios.items():
        print(f"{name.replace('_', ' ').title()}:")
        print(f"  Description: {info['description']}")
        print(f"  Cost: {info['cost']}")
        print(f"  Best Metric: {info['metric']}")
        print(f"  Reason: {info['reason']}")
        print()


if __name__ == "__main__":
    print("Day 60: Model Evaluation Metrics - Solutions\n")
    
    print("=" * 60)
    print("Exercise 1: Classification Metrics")
    print("=" * 60)
    exercise_1_classification_metrics()
    
    print("\n" + "=" * 60)
    print("Exercise 2: Confusion Matrix")
    print("=" * 60)
    exercise_2_confusion_matrix()
    
    print("\n" + "=" * 60)
    print("Exercise 3: Regression Metrics")
    print("=" * 60)
    exercise_3_regression_metrics()
    
    print("\n" + "=" * 60)
    print("Exercise 4: Imbalanced Data")
    print("=" * 60)
    exercise_4_imbalanced_data()
    
    print("\n" + "=" * 60)
    print("Exercise 5: Metric Selection")
    print("=" * 60)
    exercise_5_metric_selection()
