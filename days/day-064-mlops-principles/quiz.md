# Day 64: MLOps Principles - Quiz

Test your understanding of MLOps principles and practices.

---

## Questions

### Question 1
What is the main goal of MLOps?

A) To make models more accurate  
B) To deploy and maintain ML systems in production reliably  
C) To reduce training time  
D) To collect more data

**Answer: B**

MLOps aims to deploy and maintain ML systems in production reliably and efficiently by combining ML, DevOps, and Data Engineering practices. It focuses on the operational aspects of ML systems, not just model accuracy.

---

### Question 2
Which of the following should be version controlled in MLOps?

A) Only code  
B) Only models  
C) Code, data, and models  
D) Only data

**Answer: C**

In MLOps, everything should be version controlled: code (Git), data (DVC), and models (MLflow Model Registry). This ensures reproducibility and allows tracking changes across the entire ML pipeline.

---

### Question 3
What is model drift?

A) When the model training takes too long  
B) When model performance degrades over time due to changing data patterns  
C) When the model is too complex  
D) When the model uses too much memory

**Answer: B**

Model drift occurs when the statistical properties of the target variable or input data change over time, causing model performance to degrade. This requires monitoring and potentially retraining the model.

---

### Question 4
What is the purpose of a Model Registry?

A) To train models faster  
B) To store and version models with metadata  
C) To collect training data  
D) To monitor model performance

**Answer: B**

A Model Registry (like MLflow Model Registry) stores trained models with versioning, metadata, and stage transitions (Staging, Production). It provides a centralized repository for managing model lifecycle.

---

### Question 5
In CI/CD for ML, what does CI typically include?

A) Only code testing  
B) Data validation, model training, and testing  
C) Only model deployment  
D) Only monitoring

**Answer: B**

Continuous Integration for ML includes data validation (checking data quality), model training (automated training pipeline), and testing (unit tests, model performance tests) before deployment.

---

### Question 6
What is the difference between data drift and concept drift?

A) There is no difference  
B) Data drift is input distribution change, concept drift is X-y relationship change  
C) Data drift is faster than concept drift  
D) Concept drift only affects classification

**Answer: B**

Data drift occurs when the input feature distribution changes (P(X) changes), while concept drift occurs when the relationship between features and target changes (P(y|X) changes). Both require different detection and handling strategies.

---

### Question 7
Which MLOps maturity level includes automated retraining triggers?

A) Level 0: Manual Process  
B) Level 1: ML Pipeline Automation  
C) Level 2: CI/CD Pipeline Automation  
D) Level 3: Full MLOps Automation

**Answer: D**

Level 3 (Full MLOps Automation) includes automated retraining triggers based on performance degradation or drift detection, along with feature stores and advanced monitoring.

---

### Question 8
What is A/B testing in model deployment?

A) Testing two different datasets  
B) Deploying two model versions and comparing performance with real traffic  
C) Training two models simultaneously  
D) Testing model accuracy twice

**Answer: B**

A/B testing deploys two model versions (e.g., current and new) to production, routing a portion of traffic to each, and comparing their performance with real user data before full rollout.

---

### Question 9
Why is monitoring important in production ML systems?

A) To detect model drift and performance degradation  
B) To make models train faster  
C) To reduce data storage costs  
D) To improve model accuracy automatically

**Answer: A**

Monitoring is crucial to detect model drift, performance degradation, data quality issues, and system health in production. It enables timely interventions like retraining or rollback.

---

### Question 10
What should trigger an automatic model rollback?

A) New data arrives  
B) Model accuracy drops below threshold  
C) Training completes  
D) A new model version is registered

**Answer: B**

Automatic rollback should be triggered when model performance (accuracy, latency, error rate) drops below predefined thresholds, indicating the new model is performing worse than the previous version.

---

## Scoring

- **10/10**: Perfect! You understand MLOps principles
- **8-9/10**: Excellent! Minor review needed
- **6-7/10**: Good! Review key concepts
- **4-5/10**: Fair - revisit the material
- **0-3/10**: Needs review - go through the lesson again

---

## Key Concepts to Remember

1. **MLOps** combines ML, DevOps, and Data Engineering
2. **Version control** everything: code, data, models
3. **Model drift** requires continuous monitoring
4. **Model Registry** manages model versions and stages
5. **CI/CD** automates testing and deployment
6. **Data drift**: Input distribution changes
7. **Concept drift**: X-y relationship changes
8. **A/B testing** validates new models with real traffic
9. **Monitoring** detects issues in production
10. **Automated rollback** protects against bad deployments
