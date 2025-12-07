# Day 61: Cross-Validation & Hyperparameter Tuning

## 📖 Learning Objectives

By the end of this session, you will:
- Understand cross-validation and why it's important
- Implement k-fold cross-validation
- Tune hyperparameters with Grid Search
- Use Random Search for efficiency
- Apply best practices for model selection
- Avoid common pitfalls in validation

**Time**: 1 hour  
**Level**: Beginner

---

## Why Cross-Validation?

**Problem with single train/test split**:
- Results depend on which samples are in test set
- May get lucky or unlucky with split
- Unreliable performance estimate

**Solution**: Cross-validation
- Use multiple train/test splits
- Average results for reliable estimate
- Better use of limited data

---

## K-Fold Cross-Validation

### How it Works

```
Dataset split into K folds (e.g., K=5):

Fold 1: [Test] [Train] [Train] [Train] [Train]
Fold 2: [Train] [Test] [Train] [Train] [Train]
Fold 3: [Train] [Train] [Test] [Train] [Train]
Fold 4: [Train] [Train] [Train] [Test] [Train]
Fold 5: [Train] [Train] [Train] [Train] [Test]

Average scores across all folds
```

### Implementation

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Create model
model = LogisticRegression(max_iter=200)

# 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5)

print(f"Scores: {scores}")
print(f"Mean: {scores.mean():.3f}")
print(f"Std:  {scores.std():.3f}")
```

### Choosing K

**Common values**:
- K=5: Good balance (most common)
- K=10: More reliable, slower
- K=n (LOOCV): Maximum data use, very slow

**Trade-off**:
- Larger K: More reliable, slower
- Smaller K: Faster, more variance

---

## Cross-Validation Strategies

```python
from sklearn.model_selection import (
    StratifiedKFold, TimeSeriesSplit, LeaveOneOut, cross_val_score
)

# Stratified K-Fold - maintains class distribution (use for imbalanced classes)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf)

# Time Series Split - respects temporal order (use for time series)
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=tscv)

# Leave-One-Out - each sample is test set once (use for very small datasets)
loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo)
```

---

## Hyperparameters

**Hyperparameters** are settings chosen before training (not learned from data):

```python
LogisticRegression(C=1.0, max_iter=100)  # C: regularization strength
RandomForestClassifier(n_estimators=100, max_depth=10)  # trees, depth
KNeighborsClassifier(n_neighbors=5)  # number of neighbors
```

---

## Grid Search

**Exhaustive search** over hyperparameter grid:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

# Create grid search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Fit (tries all combinations)
grid_search.fit(X_train, y_train)

# Best parameters
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")

# Use best model
best_model = grid_search.best_estimator_
```

**Total combinations**: 3 × 4 × 3 = 36 models trained

---

## Random Search

**Random sampling** from hyperparameter distributions:

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter distributions
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 11),
    'max_features': uniform(0.1, 0.9)
}

# Create random search
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=20,  # Try 20 random combinations
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

# Fit
random_search.fit(X_train, y_train)

print(f"Best params: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_:.3f}")
```

**Advantages**:
- Faster than grid search
- Can explore more hyperparameters
- Often finds good solutions

---

## Complete Example

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# Load data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

# Define parameter grid
param_grid = {
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf'],
    'svm__gamma': ['scale', 'auto']
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Fit
grid_search.fit(X_train, y_train)

# Results
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

# Test set performance
test_score = grid_search.score(X_test, y_test)
print(f"Test score: {test_score:.3f}")
```

---

## Nested Cross-Validation

**Problem**: Using same data for tuning and evaluation

**Solution**: Nested CV
- Outer loop: Model evaluation
- Inner loop: Hyperparameter tuning

```python
from sklearn.model_selection import cross_val_score, GridSearchCV

# Inner CV: Hyperparameter tuning
inner_cv = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3
)

# Outer CV: Model evaluation
outer_scores = cross_val_score(
    inner_cv,
    X, y,
    cv=5
)

print(f"Nested CV scores: {outer_scores}")
print(f"Mean: {outer_scores.mean():.3f}")
```

---

## Scoring Metrics

```python
# Classification: 'accuracy', 'f1', 'roc_auc', 'precision', 'recall'
GridSearchCV(model, param_grid, scoring='f1')

# Regression: 'neg_mean_squared_error', 'r2', 'neg_mean_absolute_error'
GridSearchCV(model, param_grid, scoring='r2')

# Multiple metrics (refit specifies which to use for best_estimator_)
GridSearchCV(model, param_grid, scoring=['accuracy', 'f1'], refit='accuracy')
```

---

## Best Practices

```python
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

# 1. Always use cross-validation (not single split)
scores = cross_val_score(model, X_train, y_train, cv=5)

# 2. Separate test set for final evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
grid_search.fit(X_train, y_train)  # Tune on training only
test_score = grid_search.score(X_test, y_test)  # Final evaluation

# 3. Use pipelines to prevent data leakage
pipeline = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression())])
grid_search = GridSearchCV(pipeline, param_grid, cv=5)

# 4. Start with random search (fast), then grid search (fine-tune)
random_search = RandomizedSearchCV(model, param_dist, n_iter=20)
```

---

## Common Pitfalls

```python
# 1. Data Leakage - WRONG: scale before split
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled)

# RIGHT: scale after split
X_train, X_test = train_test_split(X)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Using test set for tuning - WRONG
best_params = tune_on_test_set(X_test, y_test)

# RIGHT: tune on training set with CV
grid_search.fit(X_train, y_train)

# 3. Not setting random_state - always set for reproducibility
GridSearchCV(model, param_grid, cv=5, random_state=42)
```

---

## 💻 Exercises

Complete the exercises in `exercise.py`:

### Exercise 1: K-Fold Cross-Validation
Implement and compare different K values.

### Exercise 2: Grid Search
Tune hyperparameters with grid search.

### Exercise 3: Random Search
Use random search for efficiency.

### Exercise 4: Pipeline with CV
Combine preprocessing and tuning.

### Exercise 5: Model Comparison
Compare models with cross-validation.

---

## ✅ Quiz

Test your understanding in `quiz.md`.

---

## 🎯 Key Takeaways

- Cross-validation provides reliable performance estimates
- K-fold CV uses multiple train/test splits
- Grid search exhaustively tries all combinations
- Random search samples randomly (faster)
- Always separate test set for final evaluation
- Use pipelines to prevent data leakage
- Stratified CV for imbalanced classification
- Time series split for temporal data

---

## 📚 Resources

- [Scikit-learn Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Grid Search Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [Random Search Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)

---

## Tomorrow: Day 62 - Classification & Regression

Deep dive into classification and regression algorithms.
