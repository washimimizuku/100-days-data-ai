# Day 57: ML Workflow Overview - Quiz

Test your understanding of machine learning workflow concepts.

---

## Questions

### Question 1
What is the main difference between traditional programming and machine learning?

A) Traditional programming is faster  
B) Traditional programming uses rules + data → output, ML uses data + output → rules  
C) ML doesn't need data  
D) Traditional programming can't solve complex problems

**Answer: B**

Traditional programming requires explicit rules to be written, while machine learning learns the rules (model) from data and expected outputs. This is the fundamental paradigm shift that makes ML powerful for pattern recognition tasks.

---

### Question 2
Which problem type should you use to predict house prices?

A) Classification  
B) Clustering  
C) Regression  
D) Dimensionality reduction

**Answer: C**

House prices are continuous numerical values, making this a regression problem. Classification predicts categories, clustering groups similar items, and dimensionality reduction reduces features.

---

### Question 3
Why do we split data into training and test sets?

A) To make the dataset smaller  
B) To evaluate model performance on unseen data  
C) To train the model faster  
D) To reduce overfitting during training

**Answer: B**

The test set provides unseen data to evaluate how well the model generalizes. This helps detect overfitting and gives a realistic estimate of production performance. The split doesn't reduce overfitting itself, but helps detect it.

---

### Question 4
What is overfitting?

A) Model is too simple and performs poorly  
B) Model memorizes training data and performs poorly on new data  
C) Model trains too quickly  
D) Model has too few parameters

**Answer: B**

Overfitting occurs when a model learns the training data too well, including noise and outliers, resulting in poor generalization to new data. It's characterized by very low training error but high test error.

---

### Question 5
Which metric is appropriate for evaluating a regression model?

A) Accuracy  
B) Precision  
C) Mean Squared Error (MSE)  
D) F1-Score

**Answer: C**

MSE measures the average squared difference between predictions and actual values, making it suitable for regression. Accuracy, precision, and F1-score are classification metrics.

---

### Question 6
What is the typical train/test split ratio?

A) 50/50  
B) 60/40  
C) 80/20  
D) 95/5

**Answer: C**

80/20 is the most common split, providing enough training data while reserving sufficient test data for reliable evaluation. The exact ratio can vary based on dataset size and requirements.

---

### Question 7
Which scenario is NOT suitable for machine learning?

A) Predicting customer churn with historical data  
B) Calculating tax based on fixed rules  
C) Detecting fraud in transactions  
D) Recommending products based on user behavior

**Answer: B**

Tax calculation follows deterministic rules that can be explicitly programmed. ML is best for complex patterns, not simple rule-based calculations. The other scenarios involve pattern recognition from data.

---

### Question 8
What does R² (R-squared) measure?

A) The number of errors  
B) The proportion of variance explained by the model  
C) The training time  
D) The number of features

**Answer: B**

R² ranges from 0 to 1 and indicates how much of the target variable's variance is explained by the model. An R² of 0.8 means the model explains 80% of the variance.

---

### Question 9
In the ML workflow, what comes after data exploration?

A) Model deployment  
B) Model training  
C) Data preparation  
D) Problem definition

**Answer: C**

The workflow is: Define → Collect → Explore → Prepare → Train → Evaluate → Deploy. After exploring data to understand patterns, you prepare it (clean, transform, split) before training.

---

### Question 10
What is underfitting?

A) Model is too complex  
B) Model is too simple and can't capture patterns  
C) Model trains too slowly  
D) Model has too many features

**Answer: B**

Underfitting occurs when a model is too simple to capture the underlying patterns in the data, resulting in poor performance on both training and test sets. The solution is to use a more complex model or add features.

---

## Scoring

- **10/10**: Perfect! You understand ML workflow fundamentals
- **8-9/10**: Excellent! Minor review needed
- **6-7/10**: Good! Review key concepts
- **4-5/10**: Fair - revisit the material
- **0-3/10**: Needs review - go through the lesson again

---

## Key Concepts to Remember

1. **ML learns patterns from data** without explicit programming
2. **Workflow stages**: Define → Collect → Explore → Prepare → Train → Evaluate → Deploy
3. **Classification** predicts categories, **regression** predicts numbers
4. **Train/test split** (typically 80/20) evaluates generalization
5. **Overfitting** = memorizing training data (low train error, high test error)
6. **Underfitting** = too simple (high train error, high test error)
7. **Regression metrics**: MAE, MSE, RMSE, R²
8. **Classification metrics**: Accuracy, precision, recall, F1-score
9. Use ML when patterns are complex and data is available
10. Start simple, validate properly, and iterate
