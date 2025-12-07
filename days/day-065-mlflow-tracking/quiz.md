# Day 65: MLflow Tracking - Quiz

Test your understanding of MLflow tracking concepts.

---

## Questions

### Question 1
What is the difference between a parameter and a metric in MLflow?

A) Parameters are outputs, metrics are inputs  
B) Parameters are inputs (hyperparameters), metrics are outputs (performance)  
C) They are the same thing  
D) Parameters are for classification, metrics are for regression

**Answer: B**

Parameters are input values like hyperparameters (learning_rate, batch_size) that you set before training. Metrics are output values like performance measures (accuracy, loss) that result from training.

---

### Question 2
How do you create a nested (child) run in MLflow?

A) Use mlflow.start_run() with nested=False  
B) Use mlflow.start_run(nested=True) inside another run  
C) Use mlflow.create_child_run()  
D) Nested runs are not supported

**Answer: B**

To create a nested run, call mlflow.start_run(nested=True) inside an existing run context. This creates a parent-child relationship useful for tracking sub-experiments like cross-validation folds.

---

### Question 3
What does mlflow.log_artifact() do?

A) Logs a metric value  
B) Logs a parameter  
C) Logs a file (plot, model, data)  
D) Logs a tag

**Answer: C**

mlflow.log_artifact() logs files like plots, models, datasets, or any other artifacts. These files are stored and can be retrieved later from the MLflow tracking server.

---

### Question 4
How do you search for runs with accuracy greater than 0.9?

A) mlflow.search_runs(filter="accuracy > 0.9")  
B) mlflow.search_runs(filter_string="metrics.accuracy > 0.9")  
C) mlflow.find_runs(accuracy=0.9)  
D) mlflow.get_runs(min_accuracy=0.9)

**Answer: B**

Use mlflow.search_runs(filter_string="metrics.accuracy > 0.9") to filter runs. The filter string uses the format "metrics.<metric_name>" or "params.<param_name>" with comparison operators.

---

### Question 5
What is the purpose of mlflow.set_experiment()?

A) To start a new run  
B) To set the active experiment for subsequent runs  
C) To delete an experiment  
D) To log an experiment

**Answer: B**

mlflow.set_experiment() sets the active experiment that subsequent runs will be logged to. It creates the experiment if it doesn't exist. All runs created after this call will belong to that experiment.

---

### Question 6
What does autologging do in MLflow?

A) Automatically starts runs  
B) Automatically logs parameters, metrics, and models for supported frameworks  
C) Automatically deploys models  
D) Automatically creates experiments

**Answer: B**

Autologging (e.g., mlflow.sklearn.autolog()) automatically logs parameters, metrics, and models for supported ML frameworks without manual logging calls. It simplifies tracking by capturing common information automatically.

---

### Question 7
How do you log a metric at a specific step (e.g., epoch)?

A) mlflow.log_metric("loss", value, epoch=5)  
B) mlflow.log_metric("loss", value, step=5)  
C) mlflow.log_metric("loss", value, at=5)  
D) mlflow.log_metric_step("loss", value, 5)

**Answer: B**

Use mlflow.log_metric("metric_name", value, step=step_number) to log metrics at specific steps. This is useful for tracking training progress over epochs or iterations.

---

### Question 8
What is the purpose of tags in MLflow?

A) To log metrics  
B) To add metadata for organization and filtering  
C) To log parameters  
D) To create experiments

**Answer: B**

Tags add metadata to runs for organization and filtering (e.g., "model_type", "developer", "environment"). They help categorize and search runs but don't affect model training.

---

### Question 9
How do you retrieve the best run from an experiment?

A) mlflow.get_best_run()  
B) mlflow.search_runs(order_by=["metrics.accuracy DESC"]).iloc[0]  
C) mlflow.find_best()  
D) mlflow.max_run()

**Answer: B**

Use mlflow.search_runs() with order_by parameter to sort runs by a metric, then select the first row (.iloc[0]) to get the best run. You can order by any metric in ascending or descending order.

---

### Question 10
What information is stored in an MLflow run?

A) Only parameters  
B) Only metrics  
C) Parameters, metrics, artifacts, tags, and metadata  
D) Only the model

**Answer: C**

An MLflow run stores comprehensive information: parameters (inputs), metrics (outputs), artifacts (files), tags (metadata), source code version, start/end time, and run status. This enables full reproducibility.

---

## Scoring

- **10/10**: Perfect! You understand MLflow tracking
- **8-9/10**: Excellent! Minor review needed
- **6-7/10**: Good! Review key concepts
- **4-5/10**: Fair - revisit the material
- **0-3/10**: Needs review - go through the lesson again

---

## Key Concepts to Remember

1. **Parameters** are inputs (hyperparameters), **metrics** are outputs (performance)
2. **Nested runs** created with nested=True for parent-child relationships
3. **log_artifact()** logs files (plots, models, data)
4. **search_runs()** filters and searches runs programmatically
5. **set_experiment()** sets active experiment for runs
6. **Autologging** automatically captures parameters, metrics, and models
7. **step parameter** tracks metrics over time (epochs, iterations)
8. **Tags** add metadata for organization and filtering
9. **Order runs** by metrics to find best performers
10. **Runs store** parameters, metrics, artifacts, tags, and metadata
