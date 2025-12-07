# Day 57: ML Workflow Overview

## 📖 Learning Objectives

By the end of this session, you will:
- Understand the machine learning workflow
- Identify problem types (classification, regression, clustering)
- Learn the model development lifecycle
- Understand train/test splits
- Know when to use ML vs traditional programming

**Time**: 1 hour  
**Level**: Beginner

---

## What is Machine Learning?

**Machine Learning** is teaching computers to learn patterns from data without being explicitly programmed.

**Traditional Programming**:
```
Rules + Data → Output
```

**Machine Learning**:
```
Data + Output → Rules (Model)
```

---

## ML Workflow Stages

### 1. Problem Definition
- What are you trying to predict?
- What data do you have?
- What's the success metric?

### 2. Data Collection
- Gather relevant data
- Ensure data quality
- Check for sufficient quantity

### 3. Data Exploration (EDA)
- Understand data distribution
- Find patterns and correlations
- Identify missing values and outliers

### 4. Data Preparation
- Clean data
- Handle missing values
- Feature engineering
- Split train/test sets

### 5. Model Selection
- Choose appropriate algorithm
- Consider problem type
- Balance complexity vs interpretability

### 6. Model Training
- Fit model to training data
- Tune hyperparameters
- Validate performance

### 7. Model Evaluation
- Test on unseen data
- Calculate metrics
- Compare with baseline

### 8. Model Deployment
- Integrate into production
- Monitor performance
- Retrain as needed

---

## Problem Types

### Supervised Learning

**Classification**: Predict categories
```python
# Examples:
- Email: spam or not spam
- Image: cat or dog
- Transaction: fraud or legitimate
- Customer: will churn or stay
```

**Regression**: Predict continuous values
```python
# Examples:
- House price prediction
- Sales forecasting
- Temperature prediction
- Stock price prediction
```

### Unsupervised Learning

**Clustering**: Group similar items
```python
# Examples:
- Customer segmentation
- Document grouping
- Anomaly detection
```

**Dimensionality Reduction**: Reduce features
```python
# Examples:
- Data visualization
- Feature extraction
- Noise reduction
```

---

## Train/Test Split

```python
from sklearn.model_selection import train_test_split

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Why split?**
- Train: Learn patterns
- Test: Evaluate on unseen data
- Prevents overfitting

---

## Simple Example

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Data: house size → price
X = np.array([[1000], [1500], [2000], [2500], [3000]])
y = np.array([200000, 300000, 400000, 500000, 600000])

# Split data
X_train, X_test = X[:4], X[4:]
y_train, y_test = y[:4], y[4:]

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
prediction = model.predict([[2200]])
print(f"Predicted price: ${prediction[0]:,.0f}")
```

---

## Model Evaluation Basics

### Classification Metrics
- **Accuracy**: % correct predictions
- **Precision**: % of positive predictions that are correct
- **Recall**: % of actual positives found
- **F1-Score**: Balance of precision and recall

### Regression Metrics
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of determination

---

## Overfitting vs Underfitting

**Underfitting**: Model too simple
- High training error
- High test error
- Solution: More complex model

**Good Fit**: Just right
- Low training error
- Low test error
- Generalizes well

**Overfitting**: Model too complex
- Very low training error
- High test error
- Solution: Regularization, more data

---

## When to Use ML

**Use ML when**:
- Pattern is complex
- Rules are hard to define
- Data is available
- Pattern changes over time

**Don't use ML when**:
- Simple rules work
- No data available
- Interpretability critical
- Deterministic solution exists

---

## ML Libraries

```python
# Core libraries
import numpy as np           # Numerical computing
import pandas as pd          # Data manipulation
import matplotlib.pyplot as plt  # Visualization

# ML library
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
```

---

## Complete Workflow Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load data
df = pd.read_csv('house_prices.csv')

# 2. Explore data
print(df.head())
print(df.describe())

# 3. Prepare data
X = df[['size', 'bedrooms', 'age']]
y = df['price']

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.2f}")
```

---

## Best Practices

1. **Start simple**: Begin with simple models
2. **Baseline first**: Compare against simple baseline
3. **Validate properly**: Use train/test split
4. **Iterate**: Improve incrementally
5. **Document**: Track experiments
6. **Monitor**: Watch for data drift
7. **Retrain**: Update models regularly

---

## Common Pitfalls

- **Data leakage**: Test data in training
- **Overfitting**: Model memorizes training data
- **Wrong metric**: Optimizing wrong thing
- **Insufficient data**: Not enough examples
- **Biased data**: Unrepresentative samples
- **Ignoring domain knowledge**: Not using expertise

---

## 💻 Exercises

Complete the exercises in `exercise.py`:

### Exercise 1: Problem Classification
Identify problem types for various scenarios.

### Exercise 2: Train/Test Split
Practice splitting data correctly.

### Exercise 3: Simple Model
Build a basic linear regression model.

### Exercise 4: Evaluation
Calculate and interpret metrics.

### Exercise 5: Workflow
Implement complete ML workflow.

---

## ✅ Quiz

Test your understanding in `quiz.md`.

---

## 🎯 Key Takeaways

- ML learns patterns from data without explicit programming
- Workflow: Define → Collect → Explore → Prepare → Train → Evaluate → Deploy
- Classification predicts categories, regression predicts numbers
- Always split data into train and test sets
- Start simple, iterate, and validate properly
- Overfitting is memorizing, underfitting is too simple
- Choose appropriate metrics for your problem

---

## 📚 Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
- [Kaggle Learn](https://www.kaggle.com/learn)

---

## Tomorrow: Day 58 - Feature Engineering

Learn to create and transform features for better models.
