# Day 60: Model Evaluation Metrics - Quiz

Test your understanding of model evaluation metrics.

---

## Questions

### Question 1
What is the main problem with using only accuracy for imbalanced datasets?

A) Accuracy is too slow to calculate  
B) A model predicting only the majority class can have high accuracy  
C) Accuracy doesn't work with neural networks  
D) Accuracy requires too much data

**Answer: B**

For imbalanced data (e.g., 95% class 0, 5% class 1), a model that always predicts class 0 achieves 95% accuracy while being completely useless for detecting class 1. This is why precision, recall, and F1-score are better for imbalanced datasets.

---

### Question 2
What does precision measure?

A) Of all actual positives, how many were found  
B) Of all predicted positives, how many are correct  
C) The overall accuracy of the model  
D) The variance explained by the model

**Answer: B**

Precision = TP / (TP + FP) measures the proportion of positive predictions that are actually correct. It answers: "When the model predicts positive, how often is it right?"

---

### Question 3
When should you prioritize recall over precision?

A) When false positives are very costly  
B) When false negatives are very costly  
C) When the dataset is balanced  
D) When training time is limited

**Answer: B**

Recall should be prioritized when false negatives (missing positive cases) are costly, such as in disease detection where missing a cancer diagnosis is dangerous. Precision is prioritized when false positives are costly.

---

### Question 4
What is the F1-score?

A) The average of precision and recall  
B) The harmonic mean of precision and recall  
C) The product of precision and recall  
D) The difference between precision and recall

**Answer: B**

F1-score is the harmonic mean: F1 = 2 * (Precision * Recall) / (Precision + Recall). The harmonic mean gives more weight to lower values, making it a balanced metric when you care about both precision and recall.

---

### Question 5
What does an R² score of 0.8 mean?

A) The model is 80% accurate  
B) The model explains 80% of the variance  
C) The model has 80% error  
D) The model predicts 80% of samples correctly

**Answer: B**

R² (coefficient of determination) measures the proportion of variance in the target variable that's explained by the model. An R² of 0.8 means the model explains 80% of the variance, with 1.0 being perfect.

---

### Question 6
Which metric penalizes large errors more than small errors?

A) Mean Absolute Error (MAE)  
B) Mean Squared Error (MSE)  
C) Accuracy  
D) R² Score

**Answer: B**

MSE squares the errors, which penalizes large errors more heavily than small ones. MAE treats all errors equally. This makes MSE useful when large errors are particularly undesirable.

---

### Question 7
In a confusion matrix, what is a False Negative (FN)?

A) Predicted negative, actually negative  
B) Predicted positive, actually negative  
C) Predicted negative, actually positive  
D) Predicted positive, actually positive

**Answer: C**

False Negative means the model predicted negative but the actual value was positive. This is particularly dangerous in medical diagnosis where it means missing a disease case.

---

### Question 8
Which metric is most interpretable for regression problems?

A) Mean Squared Error (MSE)  
B) Mean Absolute Error (MAE)  
C) R² Score  
D) Root Mean Squared Error (RMSE)

**Answer: B**

MAE is the most interpretable because it's the average error in the same units as the target variable. If predicting house prices, MAE of $10,000 means predictions are off by $10,000 on average.

---

### Question 9
What does an AUC (Area Under ROC Curve) of 0.5 indicate?

A) Perfect classifier  
B) Good classifier  
C) Random guessing  
D) Worst possible classifier

**Answer: C**

AUC of 0.5 indicates the model performs no better than random guessing. AUC of 1.0 is perfect, 0.9-1.0 is excellent, 0.8-0.9 is good, and 0.5 is random.

---

### Question 10
For spam detection, which metric should you prioritize?

A) Recall (catch all spam)  
B) Precision (avoid marking good emails as spam)  
C) Accuracy (overall correctness)  
D) F1-Score (balance both equally)

**Answer: B**

For spam detection, precision is more important because false positives (marking good emails as spam) are very costly - users might miss important emails. It's better to let some spam through than to lose legitimate emails.

---

## Scoring

- **10/10**: Perfect! You understand evaluation metrics
- **8-9/10**: Excellent! Minor review needed
- **6-7/10**: Good! Review key concepts
- **4-5/10**: Fair - revisit the material
- **0-3/10**: Needs review - go through the lesson again

---

## Key Concepts to Remember

1. **Accuracy** is misleading for imbalanced datasets
2. **Precision**: Of predicted positives, % correct (minimize false positives)
3. **Recall**: Of actual positives, % found (minimize false negatives)
4. **F1-Score**: Harmonic mean of precision and recall
5. **MAE**: Average absolute error (interpretable)
6. **MSE**: Squared error (penalizes large errors)
7. **RMSE**: Root of MSE (same units as target)
8. **R²**: Proportion of variance explained (0 to 1)
9. **Choose metrics** based on problem costs and requirements
10. **Use multiple metrics** for comprehensive evaluation
