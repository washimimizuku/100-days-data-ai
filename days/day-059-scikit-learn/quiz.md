# Day 59: Scikit-learn Fundamentals - Quiz

Test your understanding of scikit-learn concepts.

---

## Questions

### Question 1
What is the correct order of the scikit-learn workflow?

A) predict → fit → score  
B) fit → predict → score  
C) score → fit → predict  
D) fit → score → predict

**Answer: B**

The scikit-learn workflow is: 1) fit (train the model on training data), 2) predict (make predictions on new data), 3) score (evaluate model performance). You must train before predicting.

---

### Question 2
Which algorithm is best for interpretable binary classification with linear decision boundaries?

A) Random Forest  
B) K-Nearest Neighbors  
C) Logistic Regression  
D) Neural Network

**Answer: C**

Logistic Regression is highly interpretable, works well with linear decision boundaries, and is specifically designed for classification. Random Forest is less interpretable, and KNN doesn't provide feature importance.

---

### Question 3
What does the fit() method do?

A) Makes predictions on new data  
B) Trains the model on training data  
C) Evaluates model performance  
D) Saves the model to disk

**Answer: B**

The fit() method trains (or "fits") the model on the provided training data. It learns the patterns and parameters from X_train and y_train. Predictions are made with predict().

---

### Question 4
Why should you use a Pipeline in scikit-learn?

A) To make training faster  
B) To prevent data leakage and ensure consistent transformations  
C) To reduce memory usage  
D) To visualize the model

**Answer: B**

Pipelines prevent data leakage by ensuring transformations (like scaling) are fit only on training data and applied consistently to both train and test sets. They also make code cleaner and deployment easier.

---

### Question 5
Which metric is returned by the score() method for regression models?

A) Mean Absolute Error  
B) Mean Squared Error  
C) R² (coefficient of determination)  
D) Root Mean Squared Error

**Answer: C**

For regression models, score() returns R² (R-squared), which measures the proportion of variance explained by the model. Values range from 0 to 1, with 1 being perfect prediction.

---

### Question 6
What is the purpose of random_state parameter?

A) To make the model train faster  
B) To ensure reproducible results  
C) To improve model accuracy  
D) To reduce overfitting

**Answer: B**

random_state ensures reproducibility by fixing the random seed. This makes results consistent across runs, which is important for debugging, comparison, and sharing results. It doesn't affect accuracy or speed.

---

### Question 7
Which algorithm is an ensemble method?

A) Logistic Regression  
B) Decision Tree  
C) Random Forest  
D) K-Nearest Neighbors

**Answer: C**

Random Forest is an ensemble method that combines multiple decision trees. Ensemble methods typically achieve better performance by aggregating predictions from multiple models. Logistic Regression, Decision Tree, and KNN are single models.

---

### Question 8
When should you scale features before training?

A) Always, for all algorithms  
B) For distance-based algorithms (KNN, SVM)  
C) Never, it doesn't matter  
D) Only for classification problems

**Answer: B**

Distance-based algorithms (KNN, SVM, Neural Networks) are sensitive to feature scales because they use distances or gradients. Tree-based algorithms (Decision Tree, Random Forest) don't require scaling as they use splits, not distances.

---

### Question 9
What does joblib.dump() do?

A) Removes the model from memory  
B) Saves the model to a file  
C) Prints model parameters  
D) Evaluates the model

**Answer: B**

joblib.dump() serializes and saves the trained model to a file (usually .pkl). This allows you to load and use the model later without retraining. joblib.load() loads the saved model back.

---

### Question 10
Which is NOT a valid scikit-learn estimator method?

A) fit()  
B) predict()  
C) score()  
D) train()

**Answer: D**

Scikit-learn uses fit() to train models, not train(). The standard API methods are fit() (train), predict() (make predictions), and score() (evaluate). There is no train() method in scikit-learn.

---

## Scoring

- **10/10**: Perfect! You understand scikit-learn fundamentals
- **8-9/10**: Excellent! Minor review needed
- **6-7/10**: Good! Review key concepts
- **4-5/10**: Fair - revisit the material
- **0-3/10**: Needs review - go through the lesson again

---

## Key Concepts to Remember

1. **Scikit-learn API**: fit() → predict() → score()
2. **Classification algorithms**: LogisticRegression, DecisionTree, RandomForest, KNN
3. **Regression algorithms**: LinearRegression, Ridge, DecisionTree, RandomForest
4. **Pipelines** prevent data leakage and ensure consistent transformations
5. **Always fit on training data only** (transformers and models)
6. **random_state** ensures reproducibility
7. **Scaling** is important for distance-based algorithms
8. **score()** returns accuracy for classification, R² for regression
9. **joblib** for model persistence (save/load)
10. **Start simple** with baseline models before complex ones
