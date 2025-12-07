# Day 60: Model Evaluation Metrics

## 📖 Learning Objectives

By the end of this session, you will:
- Understand classification metrics (accuracy, precision, recall, F1)
- Learn regression metrics (MAE, MSE, RMSE, R²)
- Interpret confusion matrices
- Choose appropriate metrics for different problems
- Handle imbalanced datasets
- Use classification reports

**Time**: 1 hour  
**Level**: Beginner

---

## Why Metrics Matter

**Accuracy alone is not enough!**

Example: Cancer detection (1% have cancer)
- Model predicts "no cancer" for everyone
- Accuracy: 99% ✓
- But misses all cancer cases! ✗

**Solution**: Use multiple metrics

---

## Classification Metrics

### Confusion Matrix
```
            Predicted
            Pos    Neg
Actual Pos  TP     FN
       Neg  FP     TN
```

### Core Metrics

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

# Accuracy = (TP + TN) / Total - use for balanced datasets
accuracy = accuracy_score(y_true, y_pred)  # 0.80

# Precision = TP / (TP + FP) - minimize false positives (spam detection)
precision = precision_score(y_true, y_pred)

# Recall = TP / (TP + FN) - minimize false negatives (disease detection)
recall = recall_score(y_true, y_pred)

# F1 = 2 * (Precision * Recall) / (Precision + Recall) - balance both
f1 = f1_score(y_true, y_pred)

print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}, F1: {f1:.2f}")
```

---

## Complete Classification Example

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Load data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate metrics
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

---

## Multiclass & Visualization

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score
import matplotlib.pyplot as plt

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()

# Multiclass averaging: 'macro' (unweighted), 'weighted' (by support), 'micro' (global)
precision = precision_score(y_test, y_pred, average='weighted')
```

---

## Regression Metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_true = [100, 150, 200, 250]
y_pred = [110, 140, 210, 240]

# MAE = average absolute error (interpretable, same units as target)
mae = mean_absolute_error(y_true, y_pred)  # 10.00

# MSE = average squared error (penalizes large errors)
mse = mean_squared_error(y_true, y_pred)

# RMSE = √MSE (same units as target, penalizes large errors)
rmse = np.sqrt(mse)

# R² = proportion of variance explained (range: -∞ to 1, 1 is perfect)
r2 = r2_score(y_true, y_pred)

print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
```

---

## Complete Regression Example

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)
import numpy as np

# Load data
data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE:  {mae:.2f}")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.3f}")
```

---

## Choosing the Right Metric

| Problem | Scenario | Metric |
|---------|----------|--------|
| Classification | Balanced classes | Accuracy |
| Classification | Imbalanced classes | Precision, Recall, F1 |
| Classification | False positives costly | Precision |
| Classification | False negatives costly | Recall |
| Regression | Interpretable error | MAE |
| Regression | Penalize large errors | RMSE |
| Regression | Variance explained | R² |

---

## Imbalanced Datasets

### Problem

```python
# 95% class 0, 5% class 1
y = [0]*95 + [1]*5

# Model predicts all 0s
y_pred = [0]*100

# Accuracy is 95%! But useless
accuracy = accuracy_score(y, y_pred)  # 0.95
```

### Solutions

**1. Use appropriate metrics**:
```python
# Precision, recall, F1 for minority class
precision = precision_score(y, y_pred, pos_label=1)
recall = recall_score(y, y_pred, pos_label=1)
```

**2. Class weights**:
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced')
```

**3. Resampling**:
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
```

---

## ROC Curve and AUC

**ROC**: Receiver Operating Characteristic  
**AUC**: Area Under Curve

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Get probability predictions
y_proba = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# Calculate AUC
auc = roc_auc_score(y_test, y_proba)

# Plot
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

**AUC Interpretation**:
- 1.0: Perfect classifier
- 0.9-1.0: Excellent
- 0.8-0.9: Good
- 0.7-0.8: Fair
- 0.5: Random guessing

---

## Precision-Recall Trade-off

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]

# Adjust threshold to balance precision vs recall
y_pred_default = (y_proba >= 0.5).astype(int)  # Default
y_pred_high = (y_proba >= 0.7).astype(int)     # Higher precision, lower recall
y_pred_low = (y_proba >= 0.3).astype(int)      # Lower precision, higher recall
```

---

## Best Practices

1. Use multiple metrics (don't rely on accuracy alone)
2. Understand problem context (what errors are costly?)
3. Check class distribution (imbalanced data needs special metrics)
4. Use cross-validation for reliable estimates
5. Compare to baseline (random or simple model)
6. Visualize results (confusion matrix, ROC curve)

---

## 💻 Exercises

Complete the exercises in `exercise.py`:

### Exercise 1: Classification Metrics
Calculate and interpret classification metrics.

### Exercise 2: Confusion Matrix
Analyze confusion matrix for insights.

### Exercise 3: Regression Metrics
Compare regression models using metrics.

### Exercise 4: Imbalanced Data
Handle imbalanced classification problem.

### Exercise 5: Metric Selection
Choose appropriate metrics for scenarios.

---

## ✅ Quiz

Test your understanding in `quiz.md`.

---

## 🎯 Key Takeaways

- Accuracy is not enough, especially for imbalanced data
- Precision: Of predicted positives, how many are correct?
- Recall: Of actual positives, how many did we find?
- F1-Score: Balance between precision and recall
- MAE: Average absolute error (interpretable)
- RMSE: Penalizes large errors more
- R²: Proportion of variance explained (0 to 1)
- Choose metrics based on problem and costs

---

## 📚 Resources

- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Classification Metrics Guide](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
- [Regression Metrics Guide](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)

---

## Tomorrow: Day 61 - Cross-Validation & Hyperparameter Tuning

Learn to validate models properly and optimize hyperparameters.
