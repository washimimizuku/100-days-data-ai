# Day 66: Model Monitoring

## 📖 Learning Objectives

By the end of this session, you will:
- Understand why model monitoring is critical
- Track model performance in production
- Monitor data quality and drift
- Implement alerting systems
- Handle model degradation
- Use monitoring tools and dashboards
- Apply monitoring best practices

**Time**: 1 hour  
**Level**: Intermediate

---

## Why Monitor Models?

**Production ML models can fail silently**:
- Performance degrades over time
- Data distribution changes
- Business logic changes
- Infrastructure issues
- Data quality problems

**Without monitoring**:
- Silent failures go undetected
- Poor predictions affect business
- User trust erodes
- Revenue loss

**With monitoring**:
- Early detection of issues
- Proactive intervention
- Maintain model quality
- Build trust

---

## What to Monitor

### 1. Model Performance

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error, r2_score

# Classification monitoring
def monitor_classification(y_true, y_pred):
    metrics = {'accuracy': accuracy_score(y_true, y_pred), 
               'precision': precision_score(y_true, y_pred, average='weighted'),
               'recall': recall_score(y_true, y_pred, average='weighted')}
    for metric, value in metrics.items():
        log_metric(metric, value)
        if metric == 'accuracy' and value < 0.85: alert(f"Accuracy dropped to {value:.3f}")
    return metrics

# Regression monitoring
def monitor_regression(y_true, y_pred):
    mae, r2 = mean_absolute_error(y_true, y_pred), r2_score(y_true, y_pred)
    log_metric('mae', mae); log_metric('r2', r2)
    if r2 < 0.7: alert(f"R² dropped to {r2:.3f}")
    return {'mae': mae, 'r2': r2}
```

### 2. Prediction Distribution

```python
import numpy as np

def monitor_predictions(predictions, reference_predictions):
    """Monitor prediction distribution"""
    
    # Current distribution
    current_dist = np.bincount(predictions) / len(predictions)
    reference_dist = np.bincount(reference_predictions) / len(reference_predictions)
    
    # Calculate drift
    from scipy.stats import chisquare
    statistic, p_value = chisquare(current_dist, reference_dist)
    
    log_metric('prediction_drift_pvalue', p_value)
    
    if p_value < 0.05:
        alert("Prediction distribution has drifted!")
    
    return p_value
```

### 3. Data Quality

```python
def monitor_data_quality(df):
    """Monitor input data quality"""
    
    # Missing values
    missing_pct = df.isnull().sum() / len(df) * 100
    for col, pct in missing_pct.items():
        log_metric(f'missing_{col}', pct)
        if pct > 5:
            alert(f"High missing values in {col}: {pct:.1f}%")
    
    # Outliers
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
        outlier_pct = outliers / len(df) * 100
        
        log_metric(f'outliers_{col}', outlier_pct)
        if outlier_pct > 10:
            alert(f"High outliers in {col}: {outlier_pct:.1f}%")
```

### 4. Data Drift & Latency

```python
from scipy.stats import ks_2samp
import time

# Data drift detection using KS test
def monitor_data_drift(reference_data, current_data, threshold=0.05):
    drift_detected = {}
    for col in reference_data.columns:
        if reference_data[col].dtype in [np.float64, np.int64]:
            statistic, p_value = ks_2samp(reference_data[col].dropna(), current_data[col].dropna())
            drift_detected[col] = p_value < threshold
            log_metric(f'drift_pvalue_{col}', p_value)
            if drift_detected[col]: alert(f"Data drift detected in {col}!")
    return drift_detected

# Latency monitoring
def monitor_latency(predict_func, X):
    start_time = time.time()
    predictions = predict_func(X)
    latency_ms = (time.time() - start_time) * 1000
    log_metric('latency_ms', latency_ms)
    if latency_ms > 100: alert(f"High latency: {latency_ms:.1f}ms")
    return latency_ms
```

---

## Monitoring Dashboard

### Key Metrics to Display

```python
class ModelMonitor:
    def __init__(self, model, reference_data):
        self.model, self.reference_data, self.metrics_history = model, reference_data, []
    
    def monitor_batch(self, X, y_true=None):
        """Monitor a batch of predictions"""
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X) if hasattr(self.model, 'predict_proba') else None
        
        metrics = {'timestamp': time.time(), 'n_samples': len(X)}
        if y_true is not None: metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Prediction distribution
        pred_dist = np.bincount(y_pred) / len(y_pred)
        metrics['pred_class_0'], metrics['pred_class_1'] = pred_dist[0], pred_dist[1] if len(pred_dist) > 1 else 0
        
        # Confidence
        if y_proba is not None:
            metrics['avg_confidence'] = y_proba.max(axis=1).mean()
            metrics['low_confidence_pct'] = (y_proba.max(axis=1) < 0.7).mean() * 100
        
        # Data drift
        for col_idx, col_name in enumerate(self.reference_data.columns):
            _, p_value = ks_2samp(self.reference_data.iloc[:, col_idx], X[:, col_idx])
            metrics[f'drift_{col_name}'] = p_value < 0.05
        
        self.metrics_history.append(metrics)
        return metrics
    
    def get_dashboard_data(self):
        df = pd.DataFrame(self.metrics_history)
        return {'current_accuracy': df['accuracy'].iloc[-1] if 'accuracy' in df else None,
                'accuracy_trend': df['accuracy'].tolist() if 'accuracy' in df else [],
                'avg_confidence': df['avg_confidence'].mean(),
                'drift_features': [col for col in df.columns if col.startswith('drift_') and df[col].iloc[-1]],
                'total_predictions': df['n_samples'].sum()}
```

---

## Alerting System

```python
class AlertSystem:
    def __init__(self):
        self.thresholds = {'accuracy': 0.85, 'latency_ms': 100, 'missing_pct': 5, 'drift_pvalue': 0.05}
        self.alerts = []
    
    def check_metric(self, metric_name, value):
        if metric_name == 'accuracy' and value < self.thresholds['accuracy']:
            self.trigger_alert('HIGH', f"Accuracy dropped to {value:.3f}", metric_name, value)
        elif metric_name == 'latency_ms' and value > self.thresholds['latency_ms']:
            self.trigger_alert('MEDIUM', f"High latency: {value:.1f}ms", metric_name, value)
    
    def trigger_alert(self, severity, message, metric, value):
        alert = {'timestamp': time.time(), 'severity': severity, 'message': message, 'metric': metric, 'value': value}
        self.alerts.append(alert)
        self.send_notification(alert)
    
    def send_notification(self, alert):
        print(f"[{alert['severity']}] {alert['message']}")
        # In production: send email, post to Slack, create PagerDuty incident, log to monitoring system
```

---

## Handling Model Degradation

```python
# 1. Automatic Rollback - deploy new model only if it meets thresholds
def deploy_with_monitoring(new_model, old_model, X_test, y_test, threshold=0.85):
    new_accuracy, old_accuracy = new_model.score(X_test, y_test), old_model.score(X_test, y_test)
    if new_accuracy >= old_accuracy and new_accuracy >= threshold:
        print(f"Deploying new model (accuracy: {new_accuracy:.3f})")
        return new_model
    else:
        print(f"Rolling back to old model (new: {new_accuracy:.3f}, old: {old_accuracy:.3f})")
        return old_model

# 2. Gradual Rollout - route percentage of traffic to new model
class GradualRollout:
    def __init__(self, new_model, old_model, rollout_pct=10):
        self.new_model, self.old_model, self.rollout_pct = new_model, old_model, rollout_pct
        self.new_model_metrics, self.old_model_metrics = [], []
    
    def predict(self, X):
        if random.random() * 100 < self.rollout_pct:
            prediction = self.new_model.predict(X)
            self.new_model_metrics.append(prediction)
        else:
            prediction = self.old_model.predict(X)
            self.old_model_metrics.append(prediction)
        return prediction
    
    def increase_rollout(self, increment=10):
        self.rollout_pct = min(100, self.rollout_pct + increment)
        print(f"Rollout increased to {self.rollout_pct}%")
```

### 3. Retraining Trigger

```python
class RetrainingTrigger:
    def __init__(self, accuracy_threshold=0.85, drift_threshold=0.05):
        self.accuracy_threshold, self.drift_threshold, self.metrics_window = accuracy_threshold, drift_threshold, []
    
    def should_retrain(self, accuracy, drift_pvalue):
        self.metrics_window.append({'accuracy': accuracy, 'drift_pvalue': drift_pvalue})
        if len(self.metrics_window) > 100: self.metrics_window.pop(0)
        
        # Check conditions on recent window
        recent_accuracy = np.mean([m['accuracy'] for m in self.metrics_window[-10:]])
        recent_drift = np.mean([m['drift_pvalue'] for m in self.metrics_window[-10:]])
        
        if recent_accuracy < self.accuracy_threshold:
            print(f"Retraining triggered: accuracy {recent_accuracy:.3f} < {self.accuracy_threshold}")
            return True
        if recent_drift < self.drift_threshold:
            print(f"Retraining triggered: drift detected (p={recent_drift:.4f})")
            return True
        return False
```

---

## Monitoring Tools

```python
# Prometheus metrics for tracking
from prometheus_client import Counter, Histogram, Gauge
predictions_total = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Latency')
model_accuracy = Gauge('model_accuracy', 'Current accuracy')

def predict_with_metrics(model, X):
    predictions_total.inc()
    with prediction_latency.time():
        return model.predict(X)

# Logging for alerts and predictions
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('model_monitor')

def log_prediction(prediction, confidence, latency):
    logger.info(f"Prediction: {prediction}, Confidence: {confidence:.3f}, Latency: {latency:.3f}ms")

def log_alert(severity, message):
    (logger.error if severity == 'HIGH' else logger.warning)(f"ALERT: {message}")
```

---

## Best Practices

```python
# 1. Monitor continuously - track every prediction
for batch in data_stream:
    monitor.track(batch, model.predict(batch))

# 2. Set appropriate thresholds based on business requirements
thresholds = {'accuracy': 0.85, 'latency': 100, 'drift_pvalue': 0.05}

# 3. Track trends, not just point values
def check_trend(metrics, window=10):
    recent = metrics[-window:]
    return np.polyfit(range(len(recent)), recent, 1)[0] < 0  # Declining trend

# 4. Automate responses - alert, log, retrain
if accuracy < threshold:
    send_alert(); increase_logging(); trigger_retraining()
```

---

## 💻 Exercises

Complete the exercises in `exercise.py`:

### Exercise 1: Performance Monitoring
Track model performance metrics.

### Exercise 2: Data Drift Detection
Implement drift detection system.

### Exercise 3: Alerting System
Build alert system with thresholds.

### Exercise 4: Dashboard Metrics
Create monitoring dashboard.

### Exercise 5: Retraining Trigger
Implement automatic retraining logic.

---

## ✅ Quiz

Test your understanding in `quiz.md`.

---

## 🎯 Key Takeaways

- Monitor model performance continuously
- Track data quality and drift
- Set up alerting for critical issues
- Monitor prediction distribution and confidence
- Track latency and system health
- Implement automatic rollback
- Use gradual rollout for new models
- Trigger retraining when needed

---

## 📚 Resources

- [Evidently AI](https://www.evidentlyai.com/)
- [WhyLabs](https://whylabs.ai/)
- [Prometheus](https://prometheus.io/)
- [Grafana](https://grafana.com/)

---

## Tomorrow: Day 67 - PyTorch Tensors

Start learning PyTorch for deep learning.
