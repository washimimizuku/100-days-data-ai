"""
Day 61: Cross-Validation & Hyperparameter Tuning - Exercises
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


def exercise_1_kfold_cv():
    """
    Exercise 1: K-Fold Cross-Validation
    
    Compare different K values for cross-validation.
    """
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # TODO: Create LogisticRegression model
    # TODO: Try K=3, 5, 10 for cross-validation
    # TODO: Calculate mean and std for each K
    # TODO: Print results and compare
    # TODO: Which K gives most reliable estimate?
    pass


def exercise_2_grid_search():
    """
    Exercise 2: Grid Search
    
    Tune hyperparameters using grid search.
    """
    # Load breast cancer dataset
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    # TODO: Define parameter grid for LogisticRegression
    #       C: [0.01, 0.1, 1, 10, 100]
    #       max_iter: [100, 200, 500]
    # TODO: Create GridSearchCV with cv=5
    # TODO: Fit on training data
    # TODO: Print best parameters and best score
    # TODO: Evaluate on test set
    pass


def exercise_3_random_search():
    """
    Exercise 3: Random Search
    
    Use random search for efficient hyperparameter tuning.
    """
    # Load breast cancer dataset
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    # TODO: Define parameter distributions for RandomForestClassifier
    #       n_estimators: randint(50, 200)
    #       max_depth: randint(5, 20)
    #       min_samples_split: randint(2, 11)
    # TODO: Create RandomizedSearchCV with n_iter=20, cv=5
    # TODO: Fit on training data
    # TODO: Print best parameters and best score
    # TODO: Compare time with grid search
    pass


def exercise_4_pipeline_cv():
    """
    Exercise 4: Pipeline with Cross-Validation
    
    Combine preprocessing and model tuning in pipeline.
    """
    # Load breast cancer dataset
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    # TODO: Create pipeline with StandardScaler and SVC
    # TODO: Define parameter grid for SVC in pipeline
    #       svm__C: [0.1, 1, 10]
    #       svm__kernel: ['linear', 'rbf']
    # TODO: Create GridSearchCV with pipeline
    # TODO: Fit and evaluate
    # TODO: Print best parameters and scores
    pass


def exercise_5_model_comparison():
    """
    Exercise 5: Compare Models with Cross-Validation
    
    Compare multiple models using cross-validation.
    """
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # TODO: Create dictionary of models:
    #       - LogisticRegression
    #       - RandomForestClassifier
    #       - SVC
    # TODO: For each model, run 5-fold cross-validation
    # TODO: Calculate mean and std of scores
    # TODO: Print comparison table
    # TODO: Identify best model
    pass


if __name__ == "__main__":
    print("Day 61: Cross-Validation & Hyperparameter Tuning - Exercises\n")
    
    print("=" * 60)
    print("Exercise 1: K-Fold Cross-Validation")
    print("=" * 60)
    # exercise_1_kfold_cv()
    
    print("\n" + "=" * 60)
    print("Exercise 2: Grid Search")
    print("=" * 60)
    # exercise_2_grid_search()
    
    print("\n" + "=" * 60)
    print("Exercise 3: Random Search")
    print("=" * 60)
    # exercise_3_random_search()
    
    print("\n" + "=" * 60)
    print("Exercise 4: Pipeline with CV")
    print("=" * 60)
    # exercise_4_pipeline_cv()
    
    print("\n" + "=" * 60)
    print("Exercise 5: Model Comparison")
    print("=" * 60)
    # exercise_5_model_comparison()
