# Day 66: Model Monitoring - Quiz

Test your understanding of model monitoring concepts.

---

## Questions

### Question 1
Why is model monitoring critical in production?

A) To make models train faster  
B) To detect performance degradation and data drift early  
C) To reduce model size  
D) To improve model accuracy automatically

**Answer: B**

Model monitoring is critical to detect performance degradation, data drift, and other issues early before they significantly impact business outcomes. Without monitoring, models can fail silently in production.

---

### Question 2
What is data drift?

A) When the model trains slowly  
B) When input data distribution changes over time  
C) When the model is too complex  
D) When predictions are wrong

**Answer: B**

Data drift occurs when the statistical properties of input features change over time. This can cause model performance to degrade as the model was trained on different data distributions.

---

### Question 3
Which test is commonly used to detect data drift?

A) T-test  
B) ANOVA  
C) Kolmogorov-Smirnov (KS) test  
D) Chi-square test

**Answer: C**

The Kolmogorov-Smirnov (KS) test compares two distributions and is commonly used to detect data drift by comparing reference and current data distributions. A low p-value indicates significant drift.

---

### Question 4
What should you monitor for a classification model in production?

A) Only accuracy  
B) Accuracy, precision, recall, and prediction distribution  
C) Only training time  
D) Only model size

**Answer: B**

Monitor multiple metrics including accuracy, precision, recall, F1-score, prediction distribution, confidence scores, and data quality. Relying on a single metric can miss important issues.

---

### Question 5
When should an alert be triggered?

A) After every prediction  
B) When metrics fall below predefined thresholds  
C) Once per day  
D) Never

**Answer: B**

Alerts should be triggered when metrics fall below predefined thresholds (e.g., accuracy < 0.85, latency > 100ms). Thresholds should be based on business requirements and acceptable performance levels.

---

### Question 6
What is the purpose of gradual rollout?

A) To train models faster  
B) To test new models with a small percentage of traffic before full deployment  
C) To reduce model size  
D) To collect more data

**Answer: B**

Gradual rollout deploys new models to a small percentage of traffic first (e.g., 10%), monitors performance, and gradually increases traffic if metrics are good. This reduces risk of widespread issues.

---

### Question 7
What indicates that a model should be retrained?

A) Model was deployed yesterday  
B) Accuracy drops below threshold or significant drift detected  
C) New data is available  
D) The model is too old

**Answer: B**

Retraining should be triggered when performance drops below acceptable thresholds or significant data/concept drift is detected. Simply having new data or time passing isn't sufficient reason alone.

---

### Question 8
What is prediction drift?

A) When predictions take too long  
B) When the distribution of model outputs changes  
C) When the model is retrained  
D) When input data changes

**Answer: B**

Prediction drift occurs when the distribution of model outputs (predictions) changes over time. This can indicate issues even when input data appears stable and should be monitored separately.

---

### Question 9
What should be included in a monitoring dashboard?

A) Only current accuracy  
B) Current metrics, trends, alerts, and data quality indicators  
C) Only training logs  
D) Only model parameters

**Answer: B**

A comprehensive monitoring dashboard should include current performance metrics, historical trends, active alerts, data quality indicators, prediction distributions, and confidence scores for holistic monitoring.

---

### Question 10
What is automatic rollback?

A) Retraining the model automatically  
B) Reverting to the previous model version when new model performs poorly  
C) Deleting old models  
D) Updating model parameters

**Answer: B**

Automatic rollback reverts to the previous model version when the new model's performance falls below acceptable thresholds. This protects against deploying poorly performing models to production.

---

## Scoring

- **10/10**: Perfect! You understand model monitoring
- **8-9/10**: Excellent! Minor review needed
- **6-7/10**: Good! Review key concepts
- **4-5/10**: Fair - revisit the material
- **0-3/10**: Needs review - go through the lesson again

---

## Key Concepts to Remember

1. **Monitor continuously** to detect issues early
2. **Data drift**: Input distribution changes (KS test)
3. **Concept drift**: X-y relationship changes
4. **Prediction drift**: Output distribution changes
5. **Track multiple metrics**: accuracy, precision, recall, confidence
6. **Set thresholds** based on business requirements
7. **Alert on violations** for timely intervention
8. **Gradual rollout** reduces deployment risk
9. **Automatic rollback** protects production
10. **Trigger retraining** when performance degrades or drift detected
