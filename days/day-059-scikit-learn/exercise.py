"""
Day 59: Scikit-learn Fundamentals - Exercises
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


def exercise_1_classification():
    """
    Exercise 1: Classification with Iris Dataset
    
    Train and evaluate multiple classification models.
    """
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # TODO: Split data 80/20
    # TODO: Train LogisticRegression model
    # TODO: Make predictions
    # TODO: Calculate accuracy, precision, recall, f1
    # TODO: Print results
    pass


def exercise_2_regression():
    """
    Exercise 2: Regression with Diabetes Dataset
    
    Build and compare regression models.
    """
    # Load diabetes dataset
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    
    # TODO: Split data 80/20
    # TODO: Train LinearRegression model
    # TODO: Train Ridge regression model
    # TODO: Make predictions with both models
    # TODO: Calculate MAE, MSE, RMSE, R² for both
    # TODO: Compare which model performs better
    pass


def exercise_3_pipeline():
    """
    Exercise 3: Create Pipeline
    
    Build preprocessing and modeling pipeline.
    """
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # TODO: Split data 80/20
    # TODO: Create pipeline with StandardScaler and LogisticRegression
    # TODO: Fit pipeline on training data
    # TODO: Predict on test data
    # TODO: Calculate accuracy
    # TODO: Print results
    pass


def exercise_4_model_comparison():
    """
    Exercise 4: Compare Multiple Models
    
    Train and compare different algorithms.
    """
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # TODO: Split data 80/20
    # TODO: Create dictionary of models:
    #       - LogisticRegression
    #       - DecisionTreeClassifier
    #       - RandomForestClassifier
    #       - KNeighborsClassifier
    # TODO: Train each model
    # TODO: Calculate accuracy for each
    # TODO: Print comparison table
    # TODO: Identify best model
    pass


def exercise_5_model_persistence():
    """
    Exercise 5: Save and Load Models
    
    Practice model persistence with joblib.
    """
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # TODO: Split data 80/20
    # TODO: Create and train a RandomForestClassifier
    # TODO: Save model to 'iris_model.pkl'
    # TODO: Load model from file
    # TODO: Make predictions with loaded model
    # TODO: Verify predictions match
    # TODO: Clean up (delete saved file)
    pass


if __name__ == "__main__":
    print("Day 59: Scikit-learn Fundamentals - Exercises\n")
    
    print("=" * 60)
    print("Exercise 1: Classification")
    print("=" * 60)
    # exercise_1_classification()
    
    print("\n" + "=" * 60)
    print("Exercise 2: Regression")
    print("=" * 60)
    # exercise_2_regression()
    
    print("\n" + "=" * 60)
    print("Exercise 3: Pipeline")
    print("=" * 60)
    # exercise_3_pipeline()
    
    print("\n" + "=" * 60)
    print("Exercise 4: Model Comparison")
    print("=" * 60)
    # exercise_4_model_comparison()
    
    print("\n" + "=" * 60)
    print("Exercise 5: Model Persistence")
    print("=" * 60)
    # exercise_5_model_persistence()
