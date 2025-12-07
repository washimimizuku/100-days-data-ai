# Day 58: Feature Engineering - Quiz

Test your understanding of feature engineering concepts.

---

## Questions

### Question 1
What is the primary goal of feature engineering?

A) To reduce the size of the dataset  
B) To transform raw data into features that better represent the problem  
C) To remove all missing values  
D) To make the model train faster

**Answer: B**

Feature engineering transforms raw data into features that better represent the underlying problem to predictive models. Good features can significantly improve model performance, often more than choosing a different algorithm.

---

### Question 2
Which strategy is best for imputing missing values in a feature with many outliers?

A) Mean imputation  
B) Mode imputation  
C) Median imputation  
D) Forward fill

**Answer: C**

Median imputation is robust to outliers because it uses the middle value rather than the average. Mean imputation would be affected by extreme values, making it less suitable for data with outliers.

---

### Question 3
When should you use one-hot encoding instead of label encoding?

A) For ordinal categorical variables  
B) For nominal categorical variables with no inherent order  
C) For numerical variables  
D) For variables with missing values

**Answer: B**

One-hot encoding is appropriate for nominal categorical variables (like color, city) where there's no inherent order. Label encoding should be used for ordinal variables (like education level) where order matters.

---

### Question 4
What is the correct order for feature engineering steps?

A) Scale → Impute → Encode  
B) Encode → Scale → Impute  
C) Impute → Encode → Scale  
D) Scale → Encode → Impute

**Answer: C**

The correct order is: 1) Impute missing values first, 2) Encode categorical variables, 3) Scale numerical features last. This prevents errors and ensures proper transformations.

---

### Question 5
What does StandardScaler do to features?

A) Scales to range [0, 1]  
B) Transforms to mean=0 and std=1  
C) Removes outliers  
D) Fills missing values

**Answer: B**

StandardScaler (z-score normalization) transforms features to have mean=0 and standard deviation=1 using the formula: (x - mean) / std. MinMaxScaler scales to [0, 1].

---

### Question 6
Why must you fit the scaler only on training data?

A) To make training faster  
B) To prevent data leakage from test set  
C) To reduce memory usage  
D) To improve model accuracy

**Answer: B**

Fitting the scaler on all data (including test set) causes data leakage, where information from the test set influences the training process. Always fit on training data only, then transform both train and test sets.

---

### Question 7
Which feature creation technique is best for highly skewed income data?

A) Squaring the values  
B) Taking the square root  
C) Log transformation  
D) Dividing by the mean

**Answer: C**

Log transformation (np.log1p) is effective for highly skewed data because it compresses large values more than small values, making the distribution more normal and easier for models to learn.

---

### Question 8
What is the purpose of the drop='first' parameter in OneHotEncoder?

A) To remove the first row of data  
B) To avoid multicollinearity by dropping one category  
C) To delete missing values  
D) To speed up encoding

**Answer: B**

drop='first' removes one category to avoid multicollinearity (perfect correlation between features). If you have n categories, you only need n-1 binary features because the last category is implied when all others are 0.

---

### Question 9
Which scaling method is most robust to outliers?

A) StandardScaler  
B) MinMaxScaler  
C) RobustScaler  
D) Normalizer

**Answer: C**

RobustScaler uses the median and interquartile range (IQR) instead of mean and standard deviation, making it robust to outliers. StandardScaler and MinMaxScaler are both affected by extreme values.

---

### Question 10
When creating date/time features, which is NOT a useful feature to extract?

A) Day of week  
B) Month  
C) Is weekend  
D) Exact millisecond

**Answer: D**

Exact milliseconds are usually too granular and add noise rather than signal. Useful date/time features include year, month, day, day of week, is_weekend, hour, and time differences, which capture meaningful patterns.

---

## Scoring

- **10/10**: Perfect! You understand feature engineering
- **8-9/10**: Excellent! Minor review needed
- **6-7/10**: Good! Review key concepts
- **4-5/10**: Fair - revisit the material
- **0-3/10**: Needs review - go through the lesson again

---

## Key Concepts to Remember

1. **Feature engineering** transforms raw data into better representations
2. **Missing values**: Impute with mean/median/mode or forward/backward fill
3. **Label encoding** for ordinal (ordered) categories
4. **One-hot encoding** for nominal (unordered) categories
5. **StandardScaler**: mean=0, std=1 (for normal distributions)
6. **MinMaxScaler**: range [0,1] (for bounded ranges)
7. **RobustScaler**: Uses median/IQR (robust to outliers)
8. **Always fit on training data only** to prevent data leakage
9. **Create features**: ratios, differences, products, date/time components
10. **Order matters**: Impute → Encode → Scale
