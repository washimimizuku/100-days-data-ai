"""
Day 62: Classification & Regression - Solutions
"""
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, r2_score
import numpy as np


def exercise_1_classification_comparison():
    """Exercise 1: Classification Algorithm Comparison"""
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': (LogisticRegression(max_iter=1000), True),
        'Decision Tree': (DecisionTreeClassifier(random_state=42), False),
        'Random Forest': (RandomForestClassifier(n_estimators=100, random_state=42), False),
        'SVM': (SVC(random_state=42), True),
        'KNN': (KNeighborsClassifier(), True)
    }
    
    print(f"Dataset: Breast Cancer")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"\n{'Algorithm':<25} {'Accuracy':<10} {'Needs Scaling':<15}")
    print("-" * 50)
    
    results = {}
    
    for name, (model, needs_scaling) in models.items():
        X_tr = X_train_scaled if needs_scaling else X_train
        X_te = X_test_scaled if needs_scaling else X_test
        
        model.fit(X_tr, y_train)
        accuracy = model.score(X_te, y_test)
        results[name] = accuracy
        
        scaling_str = "Yes" if needs_scaling else "No"
        print(f"{name:<25} {accuracy:<10.3f} {scaling_str:<15}")
    
    best = max(results, key=results.get)
    worst = min(results, key=results.get)
    
    print("-" * 50)
    print(f"Best:  {best} ({results[best]:.3f})")
    print(f"Worst: {worst} ({results[worst]:.3f})")


def exercise_2_regression_comparison():
    """Exercise 2: Regression Algorithm Comparison"""
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=1.0),
        'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    print(f"Dataset: Diabetes")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"\n{'Algorithm':<25} {'R² Score':<10}")
    print("-" * 35)
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        r2 = model.score(X_test, y_test)
        results[name] = r2
        print(f"{name:<25} {r2:<10.3f}")
    
    best = max(results, key=results.get)
    
    print("-" * 35)
    print(f"Best: {best} ({results[best]:.3f})")
    
    print(f"\nAnalysis:")
    print(f"  Random Forest typically performs best")
    print(f"  Linear models good for interpretability")
    print(f"  Lasso performs feature selection")


def exercise_3_algorithm_selection():
    """Exercise 3: Algorithm Selection for Scenarios"""
    scenarios = {
        "email_spam": {
            "description": "Email spam classification",
            "requirements": "Fast, interpretable, text data",
            "algorithm": "Naive Bayes or Logistic Regression",
            "reason": "Fast, works well with high-dimensional text features"
        },
        "house_price": {
            "description": "House price prediction",
            "requirements": "Interpretable, linear relationships",
            "algorithm": "Linear Regression or Ridge",
            "reason": "Interpretable coefficients, captures linear relationships"
        },
        "image_classification": {
            "description": "Image classification (high dimensions)",
            "requirements": "High accuracy, many features",
            "algorithm": "Random Forest or SVM",
            "reason": "Handles high dimensions well, high accuracy"
        },
        "customer_churn": {
            "description": "Predict customer churn",
            "requirements": "High accuracy, feature importance",
            "algorithm": "Random Forest",
            "reason": "High accuracy, provides feature importance for insights"
        },
        "stock_price": {
            "description": "Stock price prediction",
            "requirements": "Non-linear patterns, time series",
            "algorithm": "Random Forest or Gradient Boosting",
            "reason": "Captures non-linear patterns, robust to noise"
        }
    }
    
    print("Algorithm Selection Guide:\n")
    
    for name, info in scenarios.items():
        print(f"{name.replace('_', ' ').title()}:")
        print(f"  Description: {info['description']}")
        print(f"  Requirements: {info['requirements']}")
        print(f"  Best Algorithm: {info['algorithm']}")
        print(f"  Reason: {info['reason']}")
        print()


def exercise_4_hyperparameter_impact():
    """Exercise 4: Analyze Hyperparameter Impact"""
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    print("Impact of n_estimators on Random Forest:")
    print(f"{'n_estimators':<15} {'Accuracy':<10}")
    print("-" * 25)
    
    for n_est in [10, 50, 100, 200]:
        model = RandomForestClassifier(n_estimators=n_est, random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        print(f"{n_est:<15} {accuracy:<10.3f}")
    
    print("\nImpact of max_depth on Decision Tree:")
    print(f"{'max_depth':<15} {'Accuracy':<10}")
    print("-" * 25)
    
    for depth in [3, 5, 10, None]:
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        depth_str = str(depth) if depth else "None"
        print(f"{depth_str:<15} {accuracy:<10.3f}")
    
    print("\nObservations:")
    print("  - More trees generally improve performance (diminishing returns)")
    print("  - Deeper trees can overfit (None = unlimited depth)")
    print("  - Optimal values depend on dataset")


def exercise_5_ensemble_methods():
    """Exercise 5: Build Ensemble Models"""
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    svm = SVC(probability=True, random_state=42)
    
    print("Individual Model Performance:")
    print(f"{'Model':<25} {'Accuracy':<10}")
    print("-" * 35)
    
    lr.fit(X_train_scaled, y_train)
    lr_acc = lr.score(X_test_scaled, y_test)
    print(f"{'Logistic Regression':<25} {lr_acc:<10.3f}")
    
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)
    print(f"{'Random Forest':<25} {rf_acc:<10.3f}")
    
    svm.fit(X_train_scaled, y_train)
    svm_acc = svm.score(X_test_scaled, y_test)
    print(f"{'SVM':<25} {svm_acc:<10.3f}")
    
    ensemble = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000)),
            ('svm', SVC(probability=True, random_state=42))
        ],
        voting='soft'
    )
    
    ensemble.fit(X_train_scaled, y_train)
    ensemble_acc = ensemble.score(X_test_scaled, y_test)
    
    print("-" * 35)
    print(f"{'Voting Ensemble':<25} {ensemble_acc:<10.3f}")
    
    print("\nEnsemble Benefits:")
    print("  - Combines strengths of multiple models")
    print("  - Often more robust than individual models")
    print("  - Reduces variance and overfitting")


if __name__ == "__main__":
    print("Day 62: Classification & Regression - Solutions\n")
    
    print("=" * 60)
    print("Exercise 1: Classification Comparison")
    print("=" * 60)
    exercise_1_classification_comparison()
    
    print("\n" + "=" * 60)
    print("Exercise 2: Regression Comparison")
    print("=" * 60)
    exercise_2_regression_comparison()
    
    print("\n" + "=" * 60)
    print("Exercise 3: Algorithm Selection")
    print("=" * 60)
    exercise_3_algorithm_selection()
    
    print("\n" + "=" * 60)
    print("Exercise 4: Hyperparameter Impact")
    print("=" * 60)
    exercise_4_hyperparameter_impact()
    
    print("\n" + "=" * 60)
    print("Exercise 5: Ensemble Methods")
    print("=" * 60)
    exercise_5_ensemble_methods()
