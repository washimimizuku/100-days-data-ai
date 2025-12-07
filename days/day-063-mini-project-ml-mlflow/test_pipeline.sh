#!/bin/bash

echo "=========================================="
echo "Testing ML Pipeline with MLflow"
echo "=========================================="

# Test 1: Data loading
echo ""
echo "Test 1: Data Loading"
python -c "
from ml_pipeline import load_data
X_train, X_test, y_train, y_test, feature_names = load_data()
assert len(X_train) > 0, 'Training data is empty'
assert len(X_test) > 0, 'Test data is empty'
print('✓ Data loading successful')
"

# Test 2: Preprocessing
echo ""
echo "Test 2: Preprocessing"
python -c "
from ml_pipeline import load_data, preprocess_data
import numpy as np
X_train, X_test, y_train, y_test, _ = load_data()
X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)
assert np.abs(X_train_scaled.mean()) < 0.1, 'Scaling failed'
print('✓ Preprocessing successful')
"

# Test 3: Model training
echo ""
echo "Test 3: Model Training"
python ml_pipeline.py

# Test 4: MLflow tracking
echo ""
echo "Test 4: MLflow Tracking"
python -c "
import mlflow
mlflow.set_experiment('breast_cancer_classification')
runs = mlflow.search_runs()
assert len(runs) >= 5, f'Expected at least 5 runs, got {len(runs)}'
print(f'✓ MLflow tracking successful ({len(runs)} runs)')
"

# Test 5: Predictions
echo ""
echo "Test 5: Predictions"
python predict.py

echo ""
echo "=========================================="
echo "All tests passed!"
echo "=========================================="
echo ""
echo "View MLflow UI:"
echo "  mlflow ui"
echo "  Open: http://localhost:5000"
