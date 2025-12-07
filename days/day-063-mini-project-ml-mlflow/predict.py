"""
Day 63: ML Pipeline with MLflow
Make predictions with saved model
"""
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


def load_best_model():
    """Load best model from MLflow"""
    print("Loading best model from MLflow...")
    
    mlflow.set_experiment("breast_cancer_classification")
    
    runs = mlflow.search_runs(
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )
    
    if len(runs) == 0:
        raise ValueError("No runs found. Train models first with ml_pipeline.py")
    
    best_run = runs.iloc[0]
    run_id = best_run['run_id']
    model_name = best_run['tags.model_type']
    accuracy = best_run['metrics.accuracy']
    
    print(f"Best Model: {model_name}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Run ID: {run_id}")
    
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    
    return model, model_name


def make_predictions(model, X_test, y_test):
    """Make predictions and evaluate"""
    print("\nMaking predictions...")
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    accuracy = (y_pred == y_test).mean()
    
    print(f"Test Accuracy: {accuracy:.3f}")
    
    print("\nSample Predictions:")
    print(f"{'Actual':<10} {'Predicted':<10} {'Probability':<15}")
    print("-" * 35)
    
    for i in range(min(10, len(y_test))):
        actual = "Malignant" if y_test[i] == 0 else "Benign"
        predicted = "Malignant" if y_pred[i] == 0 else "Benign"
        
        if y_proba is not None:
            prob = y_proba[i][y_pred[i]]
            print(f"{actual:<10} {predicted:<10} {prob:<15.3f}")
        else:
            print(f"{actual:<10} {predicted:<10} {'N/A':<15}")
    
    return y_pred, y_proba


def main():
    """Main prediction function"""
    print("=" * 60)
    print("ML Predictions with MLflow")
    print("=" * 60)
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model, model_name = load_best_model()
    
    needs_scaling = model_name in ['Logistic Regression', 'SVM', 'KNN']
    X_test_final = X_test_scaled if needs_scaling else X_test
    
    y_pred, y_proba = make_predictions(model, X_test_final, y_test)
    
    print("\n" + "=" * 60)
    print("Predictions completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
