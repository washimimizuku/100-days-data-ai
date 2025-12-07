"""
Day 59: Scikit-learn Fundamentals - Solutions
"""
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import numpy as np
import joblib
import os


def exercise_1_classification():
    """Exercise 1: Classification with Iris Dataset"""
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Dataset: Iris (3 classes, {len(X)} samples)")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"\nModel: Logistic Regression")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-Score:  {f1:.3f}")


def exercise_2_regression():
    """Exercise 2: Regression with Diabetes Dataset"""
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    ridge_pred = ridge_model.predict(X_test)
    
    print(f"Dataset: Diabetes ({len(X)} samples, {X.shape[1]} features)")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    print("\nLinear Regression:")
    print(f"  MAE:  {mean_absolute_error(y_test, lr_pred):.2f}")
    print(f"  MSE:  {mean_squared_error(y_test, lr_pred):.2f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, lr_pred)):.2f}")
    print(f"  R²:   {r2_score(y_test, lr_pred):.3f}")
    
    print("\nRidge Regression:")
    print(f"  MAE:  {mean_absolute_error(y_test, ridge_pred):.2f}")
    print(f"  MSE:  {mean_squared_error(y_test, ridge_pred):.2f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, ridge_pred)):.2f}")
    print(f"  R²:   {r2_score(y_test, ridge_pred):.3f}")
    
    lr_r2 = r2_score(y_test, lr_pred)
    ridge_r2 = r2_score(y_test, ridge_pred)
    better = "Ridge" if ridge_r2 > lr_r2 else "Linear"
    print(f"\nBetter model: {better} Regression")


def exercise_3_pipeline():
    """Exercise 3: Create Pipeline"""
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=200))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Pipeline Steps:")
    print("  1. StandardScaler (preprocessing)")
    print("  2. LogisticRegression (classifier)")
    print(f"\nAccuracy: {accuracy:.3f}")
    print("\nBenefits:")
    print("  - Prevents data leakage")
    print("  - Cleaner code")
    print("  - Easier deployment")


def exercise_4_model_comparison():
    """Exercise 4: Compare Multiple Models"""
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=200),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    
    print(f"Dataset: Iris ({len(X_train)} train, {len(X_test)} test)")
    print("\nModel Comparison:")
    print(f"{'Model':<25} {'Accuracy':<10}")
    print("-" * 35)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        results[name] = accuracy
        print(f"{name:<25} {accuracy:.3f}")
    
    best_model = max(results, key=results.get)
    best_accuracy = results[best_model]
    print("-" * 35)
    print(f"Best: {best_model} ({best_accuracy:.3f})")


def exercise_5_model_persistence():
    """Exercise 5: Save and Load Models"""
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    original_pred = model.predict(X_test)
    original_accuracy = accuracy_score(y_test, original_pred)
    
    filename = 'iris_model.pkl'
    joblib.dump(model, filename)
    print(f"Model saved to '{filename}'")
    
    loaded_model = joblib.load(filename)
    print(f"Model loaded from '{filename}'")
    
    loaded_pred = loaded_model.predict(X_test)
    loaded_accuracy = accuracy_score(y_test, loaded_pred)
    
    print(f"\nOriginal model accuracy: {original_accuracy:.3f}")
    print(f"Loaded model accuracy:   {loaded_accuracy:.3f}")
    print(f"Predictions match: {np.array_equal(original_pred, loaded_pred)}")
    
    if os.path.exists(filename):
        os.remove(filename)
        print(f"\nCleaned up: '{filename}' deleted")


if __name__ == "__main__":
    print("Day 59: Scikit-learn Fundamentals - Solutions\n")
    
    print("=" * 60)
    print("Exercise 1: Classification")
    print("=" * 60)
    exercise_1_classification()
    
    print("\n" + "=" * 60)
    print("Exercise 2: Regression")
    print("=" * 60)
    exercise_2_regression()
    
    print("\n" + "=" * 60)
    print("Exercise 3: Pipeline")
    print("=" * 60)
    exercise_3_pipeline()
    
    print("\n" + "=" * 60)
    print("Exercise 4: Model Comparison")
    print("=" * 60)
    exercise_4_model_comparison()
    
    print("\n" + "=" * 60)
    print("Exercise 5: Model Persistence")
    print("=" * 60)
    exercise_5_model_persistence()
