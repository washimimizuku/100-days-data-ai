"""
Day 57: ML Workflow Overview - Solutions
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def exercise_1_problem_classification():
    """Exercise 1: Problem Classification"""
    problems = {
        "email_spam": {
            "description": "Detect if email is spam",
            "type": "classification",
            "learning": "supervised",
            "metric": "accuracy, precision, recall"
        },
        "house_price": {
            "description": "Predict house sale price",
            "type": "regression",
            "learning": "supervised",
            "metric": "MAE, MSE, R²"
        },
        "customer_segments": {
            "description": "Group customers by behavior",
            "type": "clustering",
            "learning": "unsupervised",
            "metric": "silhouette score"
        },
        "loan_default": {
            "description": "Predict if loan will default",
            "type": "classification",
            "learning": "supervised",
            "metric": "precision, recall, F1"
        },
        "temperature": {
            "description": "Forecast tomorrow's temperature",
            "type": "regression",
            "learning": "supervised",
            "metric": "MAE, RMSE"
        }
    }
    
    for name, info in problems.items():
        print(f"\n{name}:")
        print(f"  Description: {info['description']}")
        print(f"  Type: {info['type']}")
        print(f"  Learning: {info['learning']}")
        print(f"  Metric: {info['metric']}")


def exercise_2_train_test_split():
    """Exercise 2: Train/Test Split"""
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = np.random.randn(100)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Original data: X={X.shape}, y={y.shape}")
    print(f"Training set: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Test set: X_test={X_test.shape}, y_test={y_test.shape}")
    print(f"\nSplit ratio: {len(X_train)/len(X):.0%} train, {len(X_test)/len(X):.0%} test")


def exercise_3_simple_model():
    """Exercise 3: Simple Linear Regression"""
    sizes = np.array([800, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000])
    prices = np.array([150000, 180000, 210000, 250000, 290000, 320000, 350000, 390000, 430000, 470000])
    
    X = sizes.reshape(-1, 1)
    y = prices
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("House Price Predictions:")
    print(f"{'Size (sqft)':<15} {'Actual Price':<15} {'Predicted Price':<15}")
    print("-" * 45)
    for size, actual, pred in zip(X_test.flatten(), y_test, y_pred):
        print(f"{size:<15.0f} ${actual:<14,.0f} ${pred:<14,.0f}")
    
    print(f"\nModel: price = {model.coef_[0]:.2f} * size + {model.intercept_:.2f}")


def exercise_4_evaluation():
    """Exercise 4: Model Evaluation"""
    y_true = np.array([100, 150, 200, 250, 300])
    y_pred = np.array([110, 140, 210, 240, 310])
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print("Evaluation Metrics:")
    print(f"MAE:  {mae:.2f} (average error)")
    print(f"MSE:  {mse:.2f} (squared error)")
    print(f"RMSE: {rmse:.2f} (root squared error)")
    print(f"R²:   {r2:.4f} (variance explained)")
    
    print("\nInterpretation:")
    print(f"- On average, predictions are off by {mae:.2f}")
    print(f"- Model explains {r2*100:.2f}% of variance")
    if r2 > 0.9:
        print("- Excellent model performance!")
    elif r2 > 0.7:
        print("- Good model performance")
    else:
        print("- Model needs improvement")


def exercise_5_complete_workflow():
    """Exercise 5: Complete ML Workflow"""
    data = {
        'years_experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
                            11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'salary': [40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000,
                  90000, 95000, 100000, 105000, 110000, 115000, 120000, 125000, 130000, 135000]
    }
    df = pd.DataFrame(data)
    
    print("1. Data Exploration:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print("\nStatistics:")
    print(df.describe())
    
    print("\n2. Data Preparation:")
    X = df[['years_experience']]
    y = df['salary']
    print(f"Features: {X.columns.tolist()}")
    print(f"Target: salary")
    
    print("\n3. Train/Test Split:")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    print("\n4. Model Training:")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model trained successfully")
    
    print("\n5. Predictions:")
    y_pred = model.predict(X_test)
    print(f"{'Years':<10} {'Actual':<15} {'Predicted':<15} {'Error':<10}")
    print("-" * 50)
    for years, actual, pred in zip(X_test.values.flatten(), y_test, y_pred):
        error = abs(actual - pred)
        print(f"{years:<10.0f} ${actual:<14,.0f} ${pred:<14,.0f} ${error:<9,.0f}")
    
    print("\n6. Model Evaluation:")
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MAE:  ${mae:,.2f}")
    print(f"MSE:  ${mse:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R²:   {r2:.4f}")
    
    print("\n7. New Prediction:")
    new_experience = np.array([[12]])
    predicted_salary = model.predict(new_experience)[0]
    print(f"Predicted salary for 12 years experience: ${predicted_salary:,.2f}")
    
    print("\n8. Model Equation:")
    print(f"salary = {model.coef_[0]:,.2f} * years + {model.intercept_:,.2f}")


if __name__ == "__main__":
    print("Day 57: ML Workflow - Solutions\n")
    
    print("=" * 60)
    print("Exercise 1: Problem Classification")
    print("=" * 60)
    exercise_1_problem_classification()
    
    print("\n" + "=" * 60)
    print("Exercise 2: Train/Test Split")
    print("=" * 60)
    exercise_2_train_test_split()
    
    print("\n" + "=" * 60)
    print("Exercise 3: Simple Model")
    print("=" * 60)
    exercise_3_simple_model()
    
    print("\n" + "=" * 60)
    print("Exercise 4: Evaluation")
    print("=" * 60)
    exercise_4_evaluation()
    
    print("\n" + "=" * 60)
    print("Exercise 5: Complete Workflow")
    print("=" * 60)
    exercise_5_complete_workflow()
