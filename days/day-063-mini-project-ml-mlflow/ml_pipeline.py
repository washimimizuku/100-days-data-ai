"""
Day 63: ML Pipeline with MLflow
Main pipeline orchestrator
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import numpy as np


def load_data():
    """Load and split breast cancer dataset"""
    print("Loading data...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Dataset: Breast Cancer")
    print(f"Samples: {len(X)}, Features: {X.shape[1]}")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, data.feature_names


def preprocess_data(X_train, X_test):
    """Scale features"""
    print("\nPreprocessing data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features scaled with StandardScaler")
    return X_train_scaled, X_test_scaled, scaler


def get_models():
    """Define models to train"""
    return {
        'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'SVM': SVC(C=1.0, kernel='rbf', probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5, weights='uniform')
    }


def train_model(model, model_name, X_train, y_train, X_test, y_test, needs_scaling):
    """Train single model with MLflow tracking"""
    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("dataset", "breast_cancer")
        mlflow.set_tag("task", "classification")
        
        params = model.get_params()
        for param, value in params.items():
            mlflow.log_param(param, value)
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        mlflow.sklearn.log_model(model, "model")
        
        run_id = mlflow.active_run().info.run_id
        
        return run_id, {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }


def train_all_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    """Train all models and track with MLflow"""
    print("\nTraining models...")
    
    models = get_models()
    results = {}
    
    scaling_required = {
        'Logistic Regression': True,
        'Decision Tree': False,
        'Random Forest': False,
        'SVM': True,
        'KNN': True
    }
    
    for i, (name, model) in enumerate(models.items(), 1):
        print(f"[{i}/{len(models)}] Training {name}...")
        
        needs_scaling = scaling_required[name]
        X_tr = X_train_scaled if needs_scaling else X_train
        X_te = X_test_scaled if needs_scaling else X_test
        
        run_id, metrics = train_model(model, name, X_tr, y_train, X_te, y_test, needs_scaling)
        
        results[name] = {
            'run_id': run_id,
            'metrics': metrics
        }
        
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Run ID: {run_id}")
    
    return results


def select_best_model(results):
    """Select best model based on accuracy"""
    print("\nModel Comparison:")
    print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 65)
    
    best_model = None
    best_accuracy = 0
    
    for name, result in results.items():
        metrics = result['metrics']
        print(f"{name:<25} {metrics['accuracy']:<10.3f} {metrics['precision']:<10.3f} "
              f"{metrics['recall']:<10.3f} {metrics['f1_score']:<10.3f}")
        
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            best_model = name
    
    print("-" * 65)
    print(f"Best Model: {best_model} (Accuracy: {best_accuracy:.3f})")
    
    return best_model, results[best_model]['run_id']


def main():
    """Main pipeline"""
    print("=" * 60)
    print("ML Pipeline with MLflow")
    print("=" * 60)
    
    mlflow.set_experiment("breast_cancer_classification")
    
    X_train, X_test, y_train, y_test, feature_names = load_data()
    
    X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)
    
    results = train_all_models(
        X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled
    )
    
    best_model_name, best_run_id = select_best_model(results)
    
    print(f"\nBest model saved with run_id: {best_run_id}")
    print("\nView results:")
    print("  mlflow ui")
    print("  Open: http://localhost:5000")
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
