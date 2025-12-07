"""
Day 65: MLflow Tracking - Solutions
"""
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt


def exercise_1_advanced_logging():
    """Exercise 1: Advanced Logging"""
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    mlflow.set_experiment("mlflow_tracking_demo")
    
    with mlflow.start_run(run_name="advanced_logging_example"):
        mlflow.set_tags({
            "model_type": "RandomForest",
            "dataset": "breast_cancer",
            "developer": "student"
        })
        
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        mlflow.log_params(model.get_params())
        
        y_pred = model.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }
        mlflow.log_metrics(metrics)
        
        importances = model.feature_importances_
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances[:10])), importances[:10])
        plt.title("Top 10 Feature Importances")
        plt.xlabel("Feature Index")
        plt.ylabel("Importance")
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        plt.close()
        
        mlflow.sklearn.log_model(model, "model")
        
        run_id = mlflow.active_run().info.run_id
        print(f"Run ID: {run_id}")
        print(f"Metrics: {metrics}")


def exercise_2_parent_child_runs():
    """Exercise 2: Parent-Child Runs"""
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    mlflow.set_experiment("mlflow_tracking_demo")
    
    with mlflow.start_run(run_name="cross_validation"):
        n_splits = 5
        mlflow.log_param("n_splits", n_splits)
        mlflow.log_param("model_type", "RandomForest")
        
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            with mlflow.start_run(run_name=f"fold_{fold}", nested=True):
                X_train_fold = X[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X[val_idx]
                y_val_fold = y[val_idx]
                
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train_fold, y_train_fold)
                
                score = model.score(X_val_fold, y_val_fold)
                mlflow.log_metric("accuracy", score)
                fold_scores.append(score)
                
                print(f"Fold {fold}: {score:.3f}")
        
        avg_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        mlflow.log_metric("avg_accuracy", avg_score)
        mlflow.log_metric("std_accuracy", std_score)
        
        print(f"\nCross-Validation Results:")
        print(f"Average Accuracy: {avg_score:.3f} ± {std_score:.3f}")


def exercise_3_search_compare():
    """Exercise 3: Search and Compare Runs"""
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    mlflow.set_experiment("mlflow_tracking_demo")
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest_50": RandomForestClassifier(n_estimators=50, random_state=42),
        "RandomForest_100": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("model_type", name)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            mlflow.log_metric("accuracy", accuracy)
            print(f"{name}: {accuracy:.3f}")
    
    runs_df = mlflow.search_runs(
        filter_string="",
        order_by=["metrics.accuracy DESC"]
    )
    
    print("\nAll Runs:")
    print(runs_df[["run_id", "tags.model_type", "metrics.accuracy"]].head())
    
    high_accuracy_runs = mlflow.search_runs(
        filter_string="metrics.accuracy > 0.95"
    )
    print(f"\nRuns with accuracy > 0.95: {len(high_accuracy_runs)}")
    
    best_run = runs_df.iloc[0]
    print(f"\nBest Run:")
    print(f"  Model: {best_run['tags.model_type']}")
    print(f"  Accuracy: {best_run['metrics.accuracy']:.3f}")
    print(f"  Run ID: {best_run['run_id']}")


def exercise_4_custom_metrics():
    """Exercise 4: Custom Metrics"""
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    mlflow.set_experiment("mlflow_tracking_demo")
    
    with mlflow.start_run(run_name="custom_metrics"):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        mlflow.log_metrics({
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn)
        })
        
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        
        mlflow.log_metrics({
            "specificity": specificity,
            "sensitivity": sensitivity
        })
        
        print("Custom Metrics:")
        print(f"  True Positives: {tp}")
        print(f"  False Positives: {fp}")
        print(f"  True Negatives: {tn}")
        print(f"  False Negatives: {fn}")
        print(f"  Specificity: {specificity:.3f}")
        print(f"  Sensitivity: {sensitivity:.3f}")


def exercise_5_autologging():
    """Exercise 5: Autologging"""
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    mlflow.sklearn.autolog()
    
    mlflow.set_experiment("mlflow_tracking_demo")
    
    with mlflow.start_run(run_name="autologging_example"):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        score = model.score(X_test, y_test)
        
        run_id = mlflow.active_run().info.run_id
        
        print("Autologging Results:")
        print(f"  Run ID: {run_id}")
        print(f"  Test Accuracy: {score:.3f}")
        print("\nAutomatically logged:")
        print("  - Model parameters")
        print("  - Training score")
        print("  - Model artifact")
        print("\nView in MLflow UI: mlflow ui")


if __name__ == "__main__":
    print("Day 65: MLflow Tracking - Solutions\n")
    
    print("=" * 60)
    print("Exercise 1: Advanced Logging")
    print("=" * 60)
    exercise_1_advanced_logging()
    
    print("\n" + "=" * 60)
    print("Exercise 2: Parent-Child Runs")
    print("=" * 60)
    exercise_2_parent_child_runs()
    
    print("\n" + "=" * 60)
    print("Exercise 3: Search and Compare")
    print("=" * 60)
    exercise_3_search_compare()
    
    print("\n" + "=" * 60)
    print("Exercise 4: Custom Metrics")
    print("=" * 60)
    exercise_4_custom_metrics()
    
    print("\n" + "=" * 60)
    print("Exercise 5: Autologging")
    print("=" * 60)
    exercise_5_autologging()
