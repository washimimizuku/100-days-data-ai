"""
Day 62: Classification & Regression - Exercises
"""
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, r2_score
import numpy as np


def exercise_1_classification_comparison():
    """
    Exercise 1: Classification Algorithm Comparison
    
    Compare multiple classification algorithms on same dataset.
    """
    # Load breast cancer dataset
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    # TODO: Create dictionary of classification models:
    #       - LogisticRegression
    #       - DecisionTreeClassifier
    #       - RandomForestClassifier
    #       - SVC
    #       - KNeighborsClassifier
    # TODO: Train each model
    # TODO: Calculate accuracy on test set
    # TODO: Print comparison table
    # TODO: Identify best and worst performers
    pass


def exercise_2_regression_comparison():
    """
    Exercise 2: Regression Algorithm Comparison
    
    Compare regression algorithms.
    """
    # Load diabetes dataset
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    # TODO: Create dictionary of regression models:
    #       - LinearRegression
    #       - Ridge
    #       - Lasso
    #       - DecisionTreeRegressor
    #       - RandomForestRegressor
    # TODO: Train each model
    # TODO: Calculate R² score on test set
    # TODO: Print comparison table
    # TODO: Analyze which performs best
    pass


def exercise_3_algorithm_selection():
    """
    Exercise 3: Algorithm Selection for Scenarios
    
    Choose appropriate algorithms for different scenarios.
    """
    scenarios = {
        "email_spam": {
            "description": "Email spam classification",
            "requirements": "Fast, interpretable, text data",
            "algorithm": None  # TODO: Choose algorithm
        },
        "house_price": {
            "description": "House price prediction",
            "requirements": "Interpretable, linear relationships",
            "algorithm": None  # TODO: Choose algorithm
        },
        "image_classification": {
            "description": "Image classification (high dimensions)",
            "requirements": "High accuracy, many features",
            "algorithm": None  # TODO: Choose algorithm
        },
        "customer_churn": {
            "description": "Predict customer churn",
            "requirements": "High accuracy, feature importance",
            "algorithm": None  # TODO: Choose algorithm
        },
        "stock_price": {
            "description": "Stock price prediction",
            "requirements": "Non-linear patterns, time series",
            "algorithm": None  # TODO: Choose algorithm
        }
    }
    
    # TODO: Fill in appropriate algorithms
    # TODO: Print scenarios with chosen algorithms
    # TODO: Explain reasoning for each choice
    pass


def exercise_4_hyperparameter_impact():
    """
    Exercise 4: Analyze Hyperparameter Impact
    
    See how hyperparameters affect model performance.
    """
    # Load breast cancer dataset
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    # TODO: Test RandomForestClassifier with different n_estimators
    #       Try: 10, 50, 100, 200
    # TODO: Test DecisionTreeClassifier with different max_depth
    #       Try: 3, 5, 10, None
    # TODO: Plot or print how performance changes
    # TODO: Identify optimal values
    pass


def exercise_5_ensemble_methods():
    """
    Exercise 5: Build Ensemble Models
    
    Create and evaluate ensemble classifiers.
    """
    # Load breast cancer dataset
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    # TODO: Train individual models:
    #       - LogisticRegression
    #       - RandomForestClassifier
    #       - SVC (with probability=True)
    # TODO: Create VotingClassifier with these models
    # TODO: Train voting classifier
    # TODO: Compare ensemble vs individual models
    # TODO: Print results
    pass


if __name__ == "__main__":
    print("Day 62: Classification & Regression - Exercises\n")
    
    print("=" * 60)
    print("Exercise 1: Classification Comparison")
    print("=" * 60)
    # exercise_1_classification_comparison()
    
    print("\n" + "=" * 60)
    print("Exercise 2: Regression Comparison")
    print("=" * 60)
    # exercise_2_regression_comparison()
    
    print("\n" + "=" * 60)
    print("Exercise 3: Algorithm Selection")
    print("=" * 60)
    # exercise_3_algorithm_selection()
    
    print("\n" + "=" * 60)
    print("Exercise 4: Hyperparameter Impact")
    print("=" * 60)
    # exercise_4_hyperparameter_impact()
    
    print("\n" + "=" * 60)
    print("Exercise 5: Ensemble Methods")
    print("=" * 60)
    # exercise_5_ensemble_methods()
