# Day 61: Cross-Validation & Hyperparameter Tuning - Quiz

Test your understanding of cross-validation and hyperparameter tuning.

---

## Questions

### Question 1
Why is cross-validation better than a single train/test split?

A) It trains the model faster  
B) It provides more reliable performance estimates by using multiple splits  
C) It requires less data  
D) It automatically tunes hyperparameters

**Answer: B**

Cross-validation uses multiple train/test splits and averages the results, providing a more reliable estimate of model performance. A single split can be lucky or unlucky depending on which samples end up in the test set.

---

### Question 2
In 5-fold cross-validation, what percentage of data is used for testing in each fold?

A) 10%  
B) 20%  
C) 50%  
D) 80%

**Answer: B**

In K-fold cross-validation, the data is split into K equal parts. Each fold uses 1/K of the data for testing. For K=5, that's 1/5 = 20% for testing and 80% for training in each fold.

---

### Question 3
What is a hyperparameter?

A) A parameter learned during training  
B) A setting chosen before training that controls the learning process  
C) The final model weights  
D) The training data

**Answer: B**

Hyperparameters are settings chosen before training (like learning rate, number of trees, regularization strength) that control how the model learns. They are not learned from data like model parameters (weights).

---

### Question 4
What is the main advantage of Random Search over Grid Search?

A) It always finds better parameters  
B) It's more accurate  
C) It explores the parameter space more efficiently and is faster  
D) It requires less memory

**Answer: C**

Random Search samples randomly from parameter distributions, allowing it to explore more hyperparameters efficiently. It often finds good solutions faster than Grid Search, which exhaustively tries all combinations.

---

### Question 5
When should you use Stratified K-Fold instead of regular K-Fold?

A) For regression problems  
B) For classification with imbalanced classes  
C) For time series data  
D) For very large datasets

**Answer: B**

Stratified K-Fold maintains the class distribution in each fold, which is important for imbalanced classification problems. It ensures each fold has a representative sample of each class.

---

### Question 6
What is the purpose of a separate test set when using cross-validation?

A) To train the model  
B) To tune hyperparameters  
C) To provide an unbiased final evaluation  
D) To speed up training

**Answer: C**

The test set provides an unbiased final evaluation of the model. Cross-validation is used on the training set for model selection and hyperparameter tuning, while the test set is held out for final performance assessment.

---

### Question 7
What does GridSearchCV do?

A) Searches for the best features  
B) Tries all combinations of hyperparameters and selects the best  
C) Splits data into train and test sets  
D) Visualizes model performance

**Answer: B**

GridSearchCV exhaustively tries all combinations of hyperparameters from a specified grid, evaluates each using cross-validation, and selects the combination with the best performance.

---

### Question 8
Why should you use a Pipeline with GridSearchCV?

A) To make training faster  
B) To prevent data leakage by ensuring preprocessing happens inside CV folds  
C) To reduce memory usage  
D) To visualize results

**Answer: B**

Pipelines ensure that preprocessing (like scaling) is fit only on training data within each CV fold, preventing data leakage. Without a pipeline, you might accidentally use information from the validation fold during preprocessing.

---

### Question 9
What is nested cross-validation used for?

A) Training multiple models simultaneously  
B) Getting an unbiased estimate of model performance when tuning hyperparameters  
C) Reducing training time  
D) Handling missing data

**Answer: B**

Nested cross-validation uses an outer loop for model evaluation and an inner loop for hyperparameter tuning. This provides an unbiased performance estimate because the same data isn't used for both tuning and evaluation.

---

### Question 10
Which cross-validation strategy should you use for time series data?

A) K-Fold  
B) Stratified K-Fold  
C) TimeSeriesSplit  
D) Leave-One-Out

**Answer: C**

TimeSeriesSplit respects temporal order by always using past data for training and future data for testing. Regular K-Fold would cause data leakage by using future information to predict the past.

---

## Scoring

- **10/10**: Perfect! You understand cross-validation and tuning
- **8-9/10**: Excellent! Minor review needed
- **6-7/10**: Good! Review key concepts
- **4-5/10**: Fair - revisit the material
- **0-3/10**: Needs review - go through the lesson again

---

## Key Concepts to Remember

1. **Cross-validation** provides reliable performance estimates using multiple splits
2. **K-fold CV** splits data into K parts, using each as test set once
3. **Hyperparameters** are settings chosen before training
4. **Grid Search** exhaustively tries all parameter combinations
5. **Random Search** samples randomly (faster, often as good)
6. **Stratified K-Fold** maintains class distribution (for imbalanced data)
7. **TimeSeriesSplit** respects temporal order (for time series)
8. **Pipelines** prevent data leakage in cross-validation
9. **Separate test set** for final unbiased evaluation
10. **Nested CV** for unbiased performance when tuning
