# Day 58: Feature Engineering

## 📖 Learning Objectives

By the end of this session, you will:
- Understand what feature engineering is and why it matters
- Handle missing values effectively
- Encode categorical variables
- Scale and normalize numerical features
- Create new features from existing ones
- Apply feature engineering best practices

**Time**: 1 hour  
**Level**: Beginner

---

## What is Feature Engineering?

**Feature Engineering** transforms raw data into features that better represent the underlying problem to predictive models.

**Impact**: Good features → Better performance | Bad features → Poor predictions | Often more important than algorithm choice

---

## Handling Missing Values

```python
import pandas as pd
from sklearn.impute import SimpleImputer

# Detection
df.isnull().sum()  # Count missing
df.isnull().sum() / len(df) * 100  # Percentage

# Strategy 1: Remove
df.dropna()  # Remove rows
df.dropna(thresh=len(df)*0.5, axis=1)  # Remove columns with >50% missing

# Strategy 2: Impute with statistics
SimpleImputer(strategy='mean')  # Mean (for normal data)
SimpleImputer(strategy='median')  # Median (better for outliers)
SimpleImputer(strategy='most_frequent')  # Mode (for categorical)

# Strategy 3: Forward/backward fill
df['price'].fillna(method='ffill')  # Use previous value
df['price'].fillna(method='bfill')  # Use next value

# Strategy 4: Constant value
df['discount'].fillna(0)  # Fill with specific value
```

---

## Encoding Categorical Variables

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

# Label Encoding: For ordinal (order matters) - Education: HS < Bachelor < Master < PhD
encoder = LabelEncoder()
encoded = encoder.fit_transform(['High School', 'Bachelor', 'Master', 'PhD'])  # [0, 1, 2, 3]

# One-Hot Encoding: For nominal (no order) - Color: Red, Blue, Green
pd.get_dummies(df, columns=['color'], prefix='color')  # Creates binary columns
# Or with sklearn
OneHotEncoder(sparse=False, drop='first').fit_transform(df[['color']])  # drop='first' avoids multicollinearity
```

---

## Feature Scaling

**Why Scale?** Algorithms sensitive to magnitude (KNN, SVM, Neural Networks), features on different scales can dominate, gradient descent converges faster

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Standardization (Z-score): mean=0, std=1 | Use when: normally distributed
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # (x - mean) / std

# Normalization (Min-Max): range [0, 1] | Use when: need bounded range
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)  # (x - min) / (max - min)

# Robust Scaling: uses median and IQR | Use when: data has outliers
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```

---

## Creating New Features

```python
import numpy as np
import pandas as pd

# Mathematical transformations
df['age_squared'] = df['age'] ** 2  # Polynomial
df['log_income'] = np.log1p(df['income'])  # Log (for skewed data)
df['sqrt_area'] = np.sqrt(df['area'])  # Square root

# Combining features
df['price_per_sqft'] = df['price'] / df['sqft']  # Ratios
df['age_diff'] = df['current_year'] - df['birth_year']  # Differences
df['total_rooms'] = df['bedrooms'] + df['bathrooms']  # Sums

# Date/time features
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['days_since'] = (pd.Timestamp.now() - df['date']).dt.days

# Binning
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100], 
                         labels=['Child', 'Young', 'Middle', 'Senior'])
df['income_bracket'] = pd.qcut(df['income'], q=4, 
                               labels=['Low', 'Medium', 'High', 'Very High'])
```

---

## Complete Example

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Load data
df = pd.DataFrame({
    'age': [25, 30, np.nan, 45, 50],
    'income': [50000, 60000, 55000, np.nan, 80000],
    'education': ['Bachelor', 'Master', 'Bachelor', 'PhD', 'Master'],
    'city': ['NYC', 'LA', 'NYC', 'SF', 'LA']
})

# 1. Handle missing values
imputer = SimpleImputer(strategy='median')
df[['age', 'income']] = imputer.fit_transform(df[['age', 'income']])

# 2. Encode ordinal feature
education_map = {'Bachelor': 1, 'Master': 2, 'PhD': 3}
df['education_encoded'] = df['education'].map(education_map)

# 3. One-hot encode nominal feature
df = pd.get_dummies(df, columns=['city'], prefix='city')

# 4. Create new features
df['income_per_age'] = df['income'] / df['age']

# 5. Scale numerical features
scaler = StandardScaler()
df[['age', 'income']] = scaler.fit_transform(df[['age', 'income']])

print(df.head())
```

---

## Feature Engineering Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Create pipeline (prevents data leakage, ensures consistency, easier to deploy)
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

X_transformed = pipeline.fit_transform(X)
```

---

## Best Practices & Common Pitfalls

```python
# 1. Fit on training data only (avoid data leakage)
scaler.fit(X_train)  # ✅ Correct
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Handle missing values before scaling
X = imputer.fit_transform(X)
X = scaler.fit_transform(X)

# 3. Keep original features
df['log_income'] = np.log1p(df['income'])  # Don't overwrite

# 4. Document transformations
feature_info = {'age': 'StandardScaler', 'income': 'log + StandardScaler'}
```

**Common Pitfalls**: Data leakage (using test data), wrong encoding (one-hot for ordinal), scaling before split, ignoring outliers, over-engineering

---

## 💻 Exercises

Complete the exercises in `exercise.py`:

### Exercise 1: Missing Values
Handle missing data with different strategies.

### Exercise 2: Categorical Encoding
Encode categorical variables appropriately.

### Exercise 3: Feature Scaling
Apply different scaling techniques.

### Exercise 4: Feature Creation
Create new features from existing ones.

### Exercise 5: Complete Pipeline
Build end-to-end feature engineering pipeline.

---

## ✅ Quiz

Test your understanding in `quiz.md`.

---

## 🎯 Key Takeaways

- Feature engineering often improves models more than algorithm choice
- Handle missing values: remove, impute, or fill strategically
- Label encoding for ordinal, one-hot for nominal categories
- Scale features for distance-based algorithms
- Create features: transformations, combinations, date/time extraction
- Always fit transformations on training data only
- Use pipelines to prevent data leakage

---

## 📚 Resources

- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

---

## Tomorrow: Day 59 - Scikit-learn Fundamentals

Learn the core scikit-learn API and common algorithms.
