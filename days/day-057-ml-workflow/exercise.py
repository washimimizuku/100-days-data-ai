"""
Day 57: ML Workflow Overview - Exercises
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def exercise_1_problem_classification():
    """
    Exercise 1: Problem Classification
    
    For each scenario, identify:
    - Problem type (classification/regression/clustering)
    - Supervised or unsupervised
    - Appropriate metric
    """
    # TODO: Classify these problems
    problems = {
        "email_spam": {
            "description": "Detect if email is spam",
            "type": None,  # classification or regression?
            "learning": None,  # supervised or unsupervised?
            "metric": None  # accuracy, MSE, etc.?
        },
        "house_price": {
            "description": "Predict house sale price",
            "type": None,
            "learning": None,
            "metric": None
        },
        "customer_segments": {
            "description": "Group customers by behavior",
            "type": None,
            "learning": None,
            "metric": None
        },
        "loan_default": {
            "description": "Predict if loan will default",
            "type": None,
            "learning": None,
            "metric": None
        },
        "temperature": {
            "description": "Forecast tomorrow's temperature",
            "type": None,
            "learning": None,
            "metric": None
        }
    }
    
    # TODO: Fill in the problem classifications
    # TODO: Print results
    pass


def exercise_2_train_test_split():
    """
    Exercise 2: Train/Test Split
    
    Practice splitting data correctly.
    """
    # Sample data
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = np.random.randn(100)
    
    # TODO: Split data 80/20 train/test
    # TODO: Print shapes of train and test sets
    # TODO: Verify split is correct
    pass


def exercise_3_simple_model():
    """
    Exercise 3: Simple Linear Regression
    
    Build a model to predict house prices based on size.
    """
    # Sample data: house size (sqft) → price ($)
    sizes = np.array([800, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000])
    prices = np.array([150000, 180000, 210000, 250000, 290000, 320000, 350000, 390000, 430000, 470000])
    
    # TODO: Reshape data for sklearn
    # TODO: Split into train/test (80/20)
    # TODO: Create and train LinearRegression model
    # TODO: Make predictions on test set
    # TODO: Print predictions vs actual
    pass


def exercise_4_evaluation():
    """
    Exercise 4: Model Evaluation
    
    Calculate and interpret evaluation metrics.
    """
    # Sample predictions vs actual
    y_true = np.array([100, 150, 200, 250, 300])
    y_pred = np.array([110, 140, 210, 240, 310])
    
    # TODO: Calculate MAE (Mean Absolute Error)
    # TODO: Calculate MSE (Mean Squared Error)
    # TODO: Calculate RMSE (Root Mean Squared Error)
    # TODO: Calculate R² score
    # TODO: Interpret results (which model is better?)
    pass


def exercise_5_complete_workflow():
    """
    Exercise 5: Complete ML Workflow
    
    Implement end-to-end workflow for salary prediction.
    """
    # Sample data: years of experience → salary
    data = {
        'years_experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
                            11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'salary': [40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000,
                  90000, 95000, 100000, 105000, 110000, 115000, 120000, 125000, 130000, 135000]
    }
    df = pd.DataFrame(data)
    
    # TODO: 1. Explore data (print head, describe, check for nulls)
    # TODO: 2. Prepare features (X) and target (y)
    # TODO: 3. Split data (80/20)
    # TODO: 4. Train LinearRegression model
    # TODO: 5. Make predictions on test set
    # TODO: 6. Calculate metrics (MAE, MSE, R²)
    # TODO: 7. Predict salary for 12 years experience
    # TODO: 8. Print all results
    pass


if __name__ == "__main__":
    print("Day 57: ML Workflow - Exercises\n")
    
    print("=" * 60)
    print("Exercise 1: Problem Classification")
    print("=" * 60)
    # exercise_1_problem_classification()
    
    print("\n" + "=" * 60)
    print("Exercise 2: Train/Test Split")
    print("=" * 60)
    # exercise_2_train_test_split()
    
    print("\n" + "=" * 60)
    print("Exercise 3: Simple Model")
    print("=" * 60)
    # exercise_3_simple_model()
    
    print("\n" + "=" * 60)
    print("Exercise 4: Evaluation")
    print("=" * 60)
    # exercise_4_evaluation()
    
    print("\n" + "=" * 60)
    print("Exercise 5: Complete Workflow")
    print("=" * 60)
    # exercise_5_complete_workflow()
