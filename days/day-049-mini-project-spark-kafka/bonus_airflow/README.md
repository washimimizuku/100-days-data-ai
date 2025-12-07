# Bonus: Airflow Orchestration

**Prerequisites**:
- Completed core project
- Airflow knowledge from other bootcamp
- Docker installed

**Time**: 30 minutes

---

## Overview

This bonus section shows how to orchestrate the real-time analytics pipeline with Airflow. While the streaming application runs continuously, Airflow can:

1. **Start/Stop Pipeline**: Control streaming jobs
2. **Monitor Health**: Check pipeline status
3. **Batch Processing**: Run complementary batch jobs
4. **Alerting**: Send notifications on failures

---

## Architecture

```
Airflow DAG
├── Start Kafka
├── Create Topics
├── Start Data Generator
├── Start Streaming App
├── Monitor Pipeline (sensor)
├── Verify Outputs
└── Generate Reports
```

---

## Setup

### 1. Install Airflow

```bash
# Using Docker Compose (recommended)
cd bonus_airflow
docker-compose up -d

# Wait for Airflow to be ready
sleep 30

# Access UI: http://localhost:8080
# Username: airflow
# Password: airflow
```

### 2. Configure Connection

In Airflow UI:
1. Go to Admin → Connections
2. Add new connection:
   - Conn Id: `kafka_default`
   - Conn Type: `Generic`
   - Host: `localhost`
   - Port: `9092`

---

## DAG Overview

The `streaming_pipeline_dag.py` orchestrates:

### Tasks

1. **check_kafka**: Verify Kafka is running
2. **create_topics**: Ensure topics exist
3. **start_generator**: Launch data generator
4. **start_streaming**: Launch Spark Streaming app
5. **monitor_pipeline**: Check pipeline health
6. **verify_outputs**: Validate results
7. **generate_report**: Create summary report
8. **stop_pipeline**: Graceful shutdown

### Schedule

- **Trigger**: Manual or scheduled
- **Frequency**: On-demand (for demos) or daily (for production)
- **Timeout**: 2 hours
- **Retries**: 3 with exponential backoff

---

## Running

### 1. Deploy DAG

```bash
# Copy DAG to Airflow
cp streaming_pipeline_dag.py ~/airflow/dags/

# Or if using Docker
docker cp streaming_pipeline_dag.py airflow-webserver:/opt/airflow/dags/
```

### 2. Trigger DAG

```bash
# Via CLI
airflow dags trigger streaming_pipeline

# Or via UI
# Go to DAGs → streaming_pipeline → Trigger DAG
```

### 3. Monitor

```bash
# View logs
airflow tasks logs streaming_pipeline start_streaming <execution_date>

# Check status
airflow dags state streaming_pipeline <execution_date>
```

---

## Production Considerations

### 1. Resource Management

```python
# In DAG
default_args = {
    'pool': 'streaming_pool',  # Limit concurrent streaming jobs
    'queue': 'streaming_queue',  # Dedicated queue
}
```

### 2. Monitoring

```python
# Add SLAs
sla = timedelta(minutes=30)

# Add callbacks
on_failure_callback = send_alert
on_success_callback = log_metrics
```

### 3. Error Handling

```python
# Retry configuration
retries = 3
retry_delay = timedelta(minutes=5)
retry_exponential_backoff = True
```

---

## Integration with Batch Layer

Combine streaming with batch processing:

```python
# Lambda architecture
streaming_dag >> batch_dag

# Batch tasks
- aggregate_historical_data
- train_ml_models
- generate_reports
```

---

## Alerting

Configure alerts for:
- Pipeline failures
- High latency (> 30s)
- Low throughput (< 100 events/s)
- Data quality issues

```python
def check_pipeline_health(**context):
    metrics = get_streaming_metrics()
    if metrics['latency'] > 30:
        send_alert("High latency detected")
```

---

## Learning Outcomes

After completing this bonus:
- Understand orchestration patterns for streaming
- Learn to monitor streaming jobs
- Implement error handling and retries
- Combine streaming with batch processing

---

## Resources

- [Airflow Streaming Guide](https://airflow.apache.org/docs/)
- [Spark on Airflow](https://airflow.apache.org/docs/apache-airflow-providers-apache-spark/)
- [Production Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
