# Day 62: Classification & Regression

## 📖 Learning Objectives

By the end of this session, you will:
- Understand classification algorithms in depth
- Master regression algorithms
- Know when to use each algorithm
- Apply algorithms to real problems
- Understand algorithm strengths and weaknesses
- Make informed algorithm choices

**Time**: 1 hour  
**Level**: Beginner

---

## Classification Algorithms

### Logistic Regression

**Linear model with sigmoid function**. Fast, interpretable, probabilistic. Assumes linear boundaries.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=1.0, max_iter=100, random_state=42)
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)  # Get probabilities
```

**Use for**: Interpretability, linear relationships, baseline model

---

### Decision Tree

**Splits data based on feature values**. Interpretable, handles non-linearity, no scaling needed. Prone to overfitting.

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
model.fit(X_train, y_train)
importances = model.feature_importances_  # Feature importance
```

**Use for**: Interpretability, non-linear patterns, mixed feature types

---

### Random Forest

**Ensemble of decision trees**. High accuracy, robust to overfitting, provides feature importance. Less interpretable, slower.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
```

**Use for**: High accuracy, non-linear patterns, feature importance

---

### Support Vector Machine (SVM)

**Finds optimal hyperplane to separate classes**. Effective in high dimensions, versatile kernels. Requires scaling, slow on large data.

```python
from sklearn.svm import SVC

model = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42)
model.fit(X_train, y_train)
```

**Use for**: High-dimensional data, clear margin, small-medium datasets

---

### K-Nearest Neighbors (KNN)

**Classifies based on K nearest neighbors**. Simple, no training phase. Slow on large data, requires scaling.

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5, weights='uniform')
model.fit(X_train, y_train)
```

**Use for**: Small datasets, non-linear boundaries, simple baseline

---

### Naive Bayes

**Probabilistic classifier based on Bayes' theorem**. Very fast, good for text. Assumes feature independence.

```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
```

**Use for**: Text classification, real-time prediction, high-dimensional data

---

## Regression Algorithms

### Linear Regression

**Fits linear relationship between features and target**. Fast, interpretable. Assumes linearity, sensitive to outliers.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
print(f"Coefficients: {model.coef_}, Intercept: {model.intercept_}")
```

**Use for**: Linear relationships, interpretability, baseline model

---

### Ridge Regression

**Linear regression with L2 regularization**. Handles multicollinearity, prevents overfitting. Retains all features.

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0, random_state=42)
model.fit(X_train, y_train)
```

**Use for**: Many correlated features, prevent overfitting

---

### Lasso Regression

**Linear regression with L1 regularization**. Performs feature selection, sparse solutions. Can be unstable with correlated features.

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=1.0, random_state=42)
model.fit(X_train, y_train)
selected = X.columns[model.coef_ != 0]  # Selected features
```

**Use for**: Feature selection, many irrelevant features

---

### Decision Tree Regressor

**Splits data to minimize variance**. Handles non-linearity, no scaling needed. Prone to overfitting.

```python
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(max_depth=5, min_samples_split=10, random_state=42)
model.fit(X_train, y_train)
```

**Use for**: Non-linear relationships, feature interactions

---

### Random Forest Regressor

**Ensemble of decision tree regressors**. High accuracy, robust, provides feature importance. Can't extrapolate.

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
```

**Use for**: High accuracy, non-linear patterns, feature importance

---

## Algorithm Selection Guide

| Requirement | Classification | Regression |
|-------------|---------------|------------|
| Interpretability | Logistic Regression, Decision Tree | Linear Regression |
| Linear patterns | Logistic Regression, SVM (linear) | Linear/Ridge/Lasso |
| Non-linear patterns | Random Forest, SVM (rbf) | Random Forest, Decision Tree |
| High accuracy | Random Forest | Random Forest |
| High dimensions | SVM, Naive Bayes | Ridge, Lasso |
| Feature selection | - | Lasso |
| Small dataset | KNN, SVM | Linear Regression |
| Fast prediction | Logistic Regression, Naive Bayes | Linear Regression |

---

## Practical Tips

```python
# 1. Start simple, then increase complexity
baseline = LogisticRegression().fit(X_train, y_train)
complex = RandomForestClassifier().fit(X_train, y_train)
print(f"Baseline: {baseline.score(X_test, y_test):.3f}")
print(f"Complex: {complex.score(X_test, y_test):.3f}")

# 2. Feature scaling needed: LogisticRegression, SVC, KNN
# No scaling needed: DecisionTree, RandomForest

# 3. Ensemble methods combine multiple models
from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier(
    estimators=[('lr', LogisticRegression()), ('rf', RandomForestClassifier())],
    voting='soft'
)
ensemble.fit(X_train, y_train)
```

---

## 💻 Exercises

Complete the exercises in `exercise.py`:

### Exercise 1: Classification Comparison
Compare classification algorithms on same dataset.

### Exercise 2: Regression Comparison
Compare regression algorithms.

### Exercise 3: Algorithm Selection
Choose appropriate algorithm for scenarios.

### Exercise 4: Hyperparameter Impact
Analyze hyperparameter effects.

### Exercise 5: Ensemble Methods
Build and evaluate ensemble models.

---

## ✅ Quiz

Test your understanding in `quiz.md`.

---

## 🎯 Key Takeaways

- Logistic Regression: Fast, interpretable, linear boundaries
- Decision Tree: Interpretable, non-linear, prone to overfitting
- Random Forest: High accuracy, robust, less interpretable
- SVM: Effective in high dimensions, needs scaling
- KNN: Simple, slow on large data, needs scaling
- Linear Regression: Fast, interpretable, assumes linearity
- Ridge/Lasso: Regularized linear models, handle multicollinearity
- Start simple, then increase complexity
- Choose based on problem requirements

---

## 📚 Resources

- [Scikit-learn Algorithm Cheat Sheet](https://scikit-learn.org/stable/tutorial/machine_learning_map/)
- [Choosing the Right Estimator](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
- [Algorithm Comparison](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)

---

## Tomorrow: Day 63 - Mini Project: ML with MLflow

Build complete ML project with experiment tracking.
