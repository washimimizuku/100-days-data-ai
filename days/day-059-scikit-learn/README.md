# Day 59: Scikit-learn Fundamentals

## 📖 Learning Objectives

By the end of this session, you will:
- Understand the scikit-learn API and workflow
- Use common classification and regression algorithms
- Apply the fit/predict pattern
- Evaluate models with built-in metrics
- Use pipelines for cleaner code
- Understand model persistence

**Time**: 1 hour  
**Level**: Beginner

---

## What is Scikit-learn?

**Scikit-learn** is Python's most popular machine learning library, providing:
- Simple and consistent API
- Wide range of algorithms
- Preprocessing and evaluation tools
- Excellent documentation

**Installation**:
```bash
pip install scikit-learn
```

---

## The Scikit-learn API

All estimators follow the same pattern:

```python
from sklearn.some_module import SomeModel

# 1. Create model
model = SomeModel(hyperparameter=value)

# 2. Train model
model.fit(X_train, y_train)

# 3. Make predictions
y_pred = model.predict(X_test)

# 4. Evaluate
score = model.score(X_test, y_test)
```

---

## Classification Algorithms

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Logistic Regression - linear boundaries, interpretable
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
print(f"Logistic Regression: {model.score(X_test, y_test):.3f}")

# Decision Tree - non-linear, interpretable, max_depth prevents overfitting
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)
print(f"Decision Tree: {model.score(X_test, y_test):.3f}")

# Random Forest - ensemble, high accuracy, n_estimators=number of trees
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(f"Random Forest: {model.score(X_test, y_test):.3f}")

# KNN - distance-based, simple baseline, n_neighbors=5 is common
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
print(f"KNN: {model.score(X_test, y_test):.3f}")
```

---

## Regression Algorithms

```python
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes

# Load data
diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42
)

# Linear Regression - simple, interpretable, assumes linear relationship
model = LinearRegression()
model.fit(X_train, y_train)
print(f"Linear Regression: {model.score(X_test, y_test):.3f}")

# Ridge Regression - adds L2 regularization, alpha controls strength
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
print(f"Ridge: {model.score(X_test, y_test):.3f}")

# Decision Tree - non-linear, max_depth prevents overfitting
model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train, y_train)
print(f"Decision Tree: {model.score(X_test, y_test):.3f}")

# Random Forest - ensemble method, robust to overfitting
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(f"Random Forest: {model.score(X_test, y_test):.3f}")
```

---

## Model Evaluation

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,  # Classification
    mean_absolute_error, mean_squared_error, r2_score          # Regression
)
import numpy as np

# Classification metrics
y_pred = classifier.predict(X_test)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred, average='weighted'):.3f}")
print(f"F1-Score:  {f1_score(y_test, y_pred, average='weighted'):.3f}")

# Regression metrics
y_pred = regressor.predict(X_test)
print(f"MAE:  {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²:   {r2_score(y_test, y_pred):.3f}")
```

---

## Pipelines

Combine preprocessing and modeling:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Train (automatically scales then trains)
pipeline.fit(X_train, y_train)

# Predict (automatically scales then predicts)
y_pred = pipeline.predict(X_test)
```

**Benefits**:
- Prevents data leakage
- Cleaner code
- Easier deployment

---

## Complete Example

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load data
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Preprocess
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 5. Predict
y_pred = model.predict(X_test_scaled)

# 6. Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

---

## Model Persistence

### Save Model

```python
import joblib

# Save model
joblib.dump(model, 'model.pkl')

# Save pipeline
joblib.dump(pipeline, 'pipeline.pkl')
```

### Load Model

```python
# Load model
loaded_model = joblib.load('model.pkl')

# Use loaded model
predictions = loaded_model.predict(X_new)
```

---

## Algorithm Selection Guide

| Algorithm | Use Case | Key Hyperparameters |
|-----------|----------|---------------------|
| Logistic Regression | Binary/multiclass, interpretable, linear | `C` (regularization), `max_iter` |
| Decision Tree | Interpretable, non-linear, mixed features | `max_depth`, `min_samples_split` |
| Random Forest | High accuracy, feature importance | `n_estimators`, `max_depth` |
| KNN | Small datasets, simple baseline | `n_neighbors`, `weights` |
| Linear Regression | Continuous target, linear relationships | None (no hyperparameters) |
| Ridge/Lasso | Many features, prevent overfitting | `alpha` (regularization strength) |

---

## Built-in Datasets

```python
from sklearn.datasets import (
    load_iris,          # Classification (3 classes)
    load_breast_cancer, # Binary classification
    load_wine,          # Classification (3 classes)
    load_diabetes,      # Regression
    load_boston,        # Regression (deprecated)
    make_classification,# Generate synthetic data
    make_regression
)

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names
```

---

## Best Practices & Common Patterns

**Key Rules**:
1. Always split data before preprocessing
2. Fit transformers on training data only
3. Use pipelines to prevent data leakage
4. Set `random_state` for reproducibility
5. Scale features for distance-based algorithms (KNN, SVM)

```python
# Compare multiple models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name}: {model.score(X_test, y_test):.3f}")

# Feature importance (tree-based models)
model = RandomForestClassifier()
model.fit(X_train, y_train)
for name, importance in zip(feature_names, model.feature_importances_):
    print(f"{name}: {importance:.3f}")
```

---

## 💻 Exercises

Complete the exercises in `exercise.py`:

### Exercise 1: Classification
Train and evaluate classification models.

### Exercise 2: Regression
Build regression models and compare.

### Exercise 3: Pipeline
Create preprocessing and modeling pipeline.

### Exercise 4: Model Comparison
Compare multiple algorithms.

### Exercise 5: Model Persistence
Save and load trained models.

---

## ✅ Quiz

Test your understanding in `quiz.md`.

---

## 🎯 Key Takeaways

- Scikit-learn provides consistent API: fit, predict, score
- Classification algorithms: LogisticRegression, DecisionTree, RandomForest, KNN
- Regression algorithms: LinearRegression, Ridge, DecisionTree, RandomForest
- Always fit transformers on training data only
- Use pipelines to prevent data leakage
- Evaluate with appropriate metrics (accuracy, precision, recall, MAE, R²)
- Save models with joblib for deployment

---

## 📚 Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Choosing the Right Estimator](https://scikit-learn.org/stable/tutorial/machine_learning_map/)

---

## Tomorrow: Day 60 - Model Evaluation Metrics

Deep dive into evaluation metrics and when to use each.
