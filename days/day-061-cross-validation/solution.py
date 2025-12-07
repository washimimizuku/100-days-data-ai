"""
Day 61: Cross-Validation & Hyperparameter Tuning - Solutions
"""
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
import numpy as np
import time


def exercise_1_kfold_cv():
    """Exercise 1: K-Fold Cross-Validation"""
    iris = load_iris()
    X, y = iris.data, iris.target
    
    model = LogisticRegression(max_iter=200)
    
    k_values = [3, 5, 10]
    
    print(f"Dataset: Iris ({len(X)} samples)")
    print(f"\nComparing K-Fold Cross-Validation:\n")
    print(f"{'K':<5} {'Mean':<10} {'Std':<10} {'Range':<15}")
    print("-" * 40)
    
    for k in k_values:
        scores = cross_val_score(model, X, y, cv=k)
        mean_score = scores.mean()
        std_score = scores.std()
        score_range = f"{scores.min():.3f}-{scores.max():.3f}"
        
        print(f"{k:<5} {mean_score:<10.3f} {std_score:<10.3f} {score_range:<15}")
    
    print("\nRecommendation:")
    print("  K=5 is most common (good balance)")
    print("  K=10 more reliable but slower")
    print("  K=3 faster but more variance")


def exercise_2_grid_search():
    """Exercise 2: Grid Search"""
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'max_iter': [100, 200, 500]
    }
    
    grid_search = GridSearchCV(
        LogisticRegression(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    print(f"Dataset: Breast Cancer")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"\nParameter Grid:")
    print(f"  C: {param_grid['C']}")
    print(f"  max_iter: {param_grid['max_iter']}")
    print(f"  Total combinations: {len(param_grid['C']) * len(param_grid['max_iter'])}")
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    print(f"\nGrid Search Results:")
    print(f"  Best parameters: {grid_search.best_params_}")
    print(f"  Best CV score: {grid_search.best_score_:.3f}")
    print(f"  Time taken: {elapsed_time:.2f}s")
    
    test_score = grid_search.score(X_test, y_test)
    print(f"\nTest Set Performance:")
    print(f"  Test score: {test_score:.3f}")


def exercise_3_random_search():
    """Exercise 3: Random Search"""
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    param_dist = {
        'n_estimators': randint(50, 200),
        'max_depth': randint(5, 20),
        'min_samples_split': randint(2, 11)
    }
    
    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_dist,
        n_iter=20,
        cv=5,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )
    
    print(f"Dataset: Breast Cancer")
    print(f"Training samples: {len(X_train)}")
    print(f"\nParameter Distributions:")
    print(f"  n_estimators: 50-200")
    print(f"  max_depth: 5-20")
    print(f"  min_samples_split: 2-11")
    print(f"  Random iterations: 20")
    
    start_time = time.time()
    random_search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    print(f"\nRandom Search Results:")
    print(f"  Best parameters: {random_search.best_params_}")
    print(f"  Best CV score: {random_search.best_score_:.3f}")
    print(f"  Time taken: {elapsed_time:.2f}s")
    
    test_score = random_search.score(X_test, y_test)
    print(f"\nTest Set Performance:")
    print(f"  Test score: {test_score:.3f}")
    
    print(f"\nAdvantage:")
    print(f"  Random search explores parameter space efficiently")
    print(f"  Often finds good solutions faster than grid search")


def exercise_4_pipeline_cv():
    """Exercise 4: Pipeline with Cross-Validation"""
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(random_state=42))
    ])
    
    param_grid = {
        'svm__C': [0.1, 1, 10],
        'svm__kernel': ['linear', 'rbf']
    }
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    print("Pipeline Steps:")
    print("  1. StandardScaler (preprocessing)")
    print("  2. SVC (classifier)")
    
    print(f"\nParameter Grid:")
    print(f"  svm__C: {param_grid['svm__C']}")
    print(f"  svm__kernel: {param_grid['svm__kernel']}")
    print(f"  Total combinations: {len(param_grid['svm__C']) * len(param_grid['svm__kernel'])}")
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nResults:")
    print(f"  Best parameters: {grid_search.best_params_}")
    print(f"  Best CV score: {grid_search.best_score_:.3f}")
    
    test_score = grid_search.score(X_test, y_test)
    print(f"  Test score: {test_score:.3f}")
    
    print(f"\nBenefit of Pipeline:")
    print(f"  Scaling is done inside CV folds (no data leakage)")
    print(f"  Cleaner code and easier deployment")


def exercise_5_model_comparison():
    """Exercise 5: Compare Models with Cross-Validation"""
    iris = load_iris()
    X, y = iris.data, iris.target
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=200),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42)
    }
    
    print(f"Dataset: Iris ({len(X)} samples)")
    print(f"Cross-Validation: 5-fold\n")
    print(f"{'Model':<25} {'Mean':<10} {'Std':<10} {'Range':<15}")
    print("-" * 60)
    
    results = {}
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5)
        mean_score = scores.mean()
        std_score = scores.std()
        score_range = f"{scores.min():.3f}-{scores.max():.3f}"
        
        results[name] = mean_score
        print(f"{name:<25} {mean_score:<10.3f} {std_score:<10.3f} {score_range:<15}")
    
    best_model = max(results, key=results.get)
    best_score = results[best_model]
    
    print("-" * 60)
    print(f"Best Model: {best_model} ({best_score:.3f})")
    
    print(f"\nInterpretation:")
    print(f"  All models perform well on Iris dataset")
    print(f"  Small std indicates stable performance")
    print(f"  Choose based on interpretability and speed needs")


if __name__ == "__main__":
    print("Day 61: Cross-Validation & Hyperparameter Tuning - Solutions\n")
    
    print("=" * 60)
    print("Exercise 1: K-Fold Cross-Validation")
    print("=" * 60)
    exercise_1_kfold_cv()
    
    print("\n" + "=" * 60)
    print("Exercise 2: Grid Search")
    print("=" * 60)
    exercise_2_grid_search()
    
    print("\n" + "=" * 60)
    print("Exercise 3: Random Search")
    print("=" * 60)
    exercise_3_random_search()
    
    print("\n" + "=" * 60)
    print("Exercise 4: Pipeline with CV")
    print("=" * 60)
    exercise_4_pipeline_cv()
    
    print("\n" + "=" * 60)
    print("Exercise 5: Model Comparison")
    print("=" * 60)
    exercise_5_model_comparison()
