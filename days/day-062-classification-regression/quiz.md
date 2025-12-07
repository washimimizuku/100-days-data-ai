# Day 62: Classification & Regression - Quiz

Test your understanding of classification and regression algorithms.

---

## Questions

### Question 1
Which algorithm is best when you need an interpretable model with linear decision boundaries?

A) Random Forest  
B) Logistic Regression  
C) K-Nearest Neighbors  
D) SVM with RBF kernel

**Answer: B**

Logistic Regression provides interpretable coefficients showing feature importance and works well with linear decision boundaries. Random Forest is less interpretable, KNN doesn't provide feature importance, and SVM with RBF kernel handles non-linear boundaries.

---

### Question 2
What is the main advantage of Random Forest over a single Decision Tree?

A) Faster training  
B) More interpretable  
C) Less prone to overfitting  
D) Requires less memory

**Answer: C**

Random Forest is an ensemble of decision trees that reduces overfitting through averaging. Single decision trees are prone to overfitting. Random Forest is slower, less interpretable, and uses more memory than a single tree.

---

### Question 3
Which algorithms require feature scaling?

A) Decision Tree and Random Forest  
B) Logistic Regression and SVM  
C) Random Forest and Gradient Boosting  
D) Decision Tree and Naive Bayes

**Answer: B**

Distance-based and gradient-based algorithms like Logistic Regression, SVM, and KNN require feature scaling. Tree-based algorithms (Decision Tree, Random Forest) don't require scaling as they use splits, not distances.

---

### Question 4
When should you use Lasso Regression instead of Ridge Regression?

A) When you want to keep all features  
B) When you need feature selection  
C) When features are uncorrelated  
D) When you have few features

**Answer: B**

Lasso (L1 regularization) performs feature selection by driving some coefficients to exactly zero. Ridge (L2 regularization) shrinks coefficients but keeps all features. Use Lasso when you have many features and want automatic feature selection.

---

### Question 5
What is the main weakness of K-Nearest Neighbors (KNN)?

A) Can't handle multiclass problems  
B) Requires extensive training  
C) Slow prediction on large datasets  
D) Can't handle non-linear boundaries

**Answer: C**

KNN has no training phase but requires computing distances to all training samples during prediction, making it slow on large datasets. It handles multiclass naturally and works well with non-linear boundaries.

---

### Question 6
Which algorithm is best for high-dimensional data with clear margin of separation?

A) Decision Tree  
B) K-Nearest Neighbors  
C) Support Vector Machine (SVM)  
D) Naive Bayes

**Answer: C**

SVM is effective in high-dimensional spaces and works well when there's a clear margin of separation between classes. It's memory efficient and can use different kernels for flexibility.

---

### Question 7
What does the max_depth hyperparameter control in Decision Trees?

A) Number of features to consider  
B) Maximum depth of the tree  
C) Minimum samples per leaf  
D) Number of trees in the forest

**Answer: B**

max_depth controls the maximum depth of the decision tree. Limiting depth prevents overfitting by stopping the tree from becoming too complex. None (unlimited depth) can lead to overfitting.

---

### Question 8
Which regression algorithm is best when you have many correlated features?

A) Linear Regression  
B) Ridge Regression  
C) Decision Tree  
D) K-Nearest Neighbors

**Answer: B**

Ridge Regression handles multicollinearity (correlated features) well through L2 regularization. Linear Regression can have unstable coefficients with correlated features. Ridge shrinks coefficients while keeping all features.

---

### Question 9
What is the purpose of a Voting Classifier?

A) To select the best single model  
B) To combine predictions from multiple models  
C) To vote on which features to use  
D) To determine the best hyperparameters

**Answer: B**

Voting Classifier is an ensemble method that combines predictions from multiple models. It can use hard voting (majority vote) or soft voting (average probabilities) to make final predictions, often improving robustness.

---

### Question 10
Which algorithm naturally provides feature importance?

A) Logistic Regression  
B) K-Nearest Neighbors  
C) Random Forest  
D) Support Vector Machine

**Answer: C**

Random Forest (and other tree-based models) naturally provide feature importance based on how much each feature reduces impurity. This helps understand which features are most important for predictions.

---

## Scoring

- **10/10**: Perfect! You understand classification and regression
- **8-9/10**: Excellent! Minor review needed
- **6-7/10**: Good! Review key concepts
- **4-5/10**: Fair - revisit the material
- **0-3/10**: Needs review - go through the lesson again

---

## Key Concepts to Remember

1. **Logistic Regression**: Fast, interpretable, linear boundaries
2. **Decision Tree**: Interpretable, non-linear, prone to overfitting
3. **Random Forest**: High accuracy, robust, ensemble method
4. **SVM**: Effective in high dimensions, needs scaling
5. **KNN**: Simple, slow on large data, needs scaling
6. **Linear Regression**: Fast, interpretable, assumes linearity
7. **Ridge**: L2 regularization, handles multicollinearity
8. **Lasso**: L1 regularization, performs feature selection
9. **Scaling needed**: Logistic Regression, SVM, KNN
10. **Start simple**, then increase complexity as needed
