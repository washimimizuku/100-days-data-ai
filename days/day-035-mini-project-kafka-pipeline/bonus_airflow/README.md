# Bonus: Airflow Orchestration

**Prerequisites**: 
- Completed core Kafka pipeline project
- Airflow knowledge from other bootcamp
- Docker and Docker Compose installed

**Time**: 30 minutes

---

## Overview

This bonus section shows how to orchestrate the Kafka pipeline with Apache Airflow. The DAG will:
- Start/stop Kafka infrastructure
- Run pipeline components in sequence
- Monitor component health
- Handle failures with retries
- Schedule regular pipeline runs

**Note**: This assumes you learned Airflow in the other bootcamp. If not, you can skip this section and complete it later.

---

## Architecture

```
Airflow Scheduler
       ↓
   DAG: kafka_pipeline
       ↓
   ┌─────────────────┐
   │ start_kafka     │
   └────────┬────────┘
            ↓
   ┌─────────────────┐
   │ create_topics   │
   └────────┬────────┘
            ↓
   ┌─────────────────────────────────┐
   │  run_generator                  │
   │  run_processor  (parallel)      │
   │  run_analytics                  │
   │  run_sink                       │
   └────────┬────────────────────────┘
            ↓
   ┌─────────────────┐
   │ verify_output   │
   └────────┬────────┘
            ↓
   ┌─────────────────┐
   │ cleanup         │
   └─────────────────┘
```

---

## Setup

### 1. Install Airflow

```bash
# Create virtual environment
python -m venv airflow_venv
source airflow_venv/bin/activate

# Install Airflow
pip install apache-airflow==2.7.0

# Initialize database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
```

### 2. Configure Airflow

```bash
# Set Airflow home
export AIRFLOW_HOME=~/airflow

# Copy DAG to Airflow
cp pipeline_dag.py $AIRFLOW_HOME/dags/

# Update paths in DAG
# Edit pipeline_dag.py and set PROJECT_DIR to your project path
```

### 3. Start Airflow

```bash
# Start webserver
airflow webserver --port 8080 &

# Start scheduler
airflow scheduler &
```

---

## DAG Configuration

The `pipeline_dag.py` includes:

### Default Arguments
```python
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email_on_retry': False
}
```

### Schedule
- **Interval**: `@hourly` (runs every hour)
- **Catchup**: `False` (don't backfill)

### Tasks
1. **start_kafka**: Start Kafka container
2. **create_topics**: Create required topics
3. **run_generator**: Generate events (60 seconds)
4. **run_processor**: Process orders
5. **run_analytics**: Calculate metrics
6. **run_sink**: Write to file
7. **verify_output**: Check results
8. **cleanup**: Stop components

---

## Running the DAG

### Via Web UI

1. Open http://localhost:8080
2. Login with admin/admin
3. Find `kafka_pipeline` DAG
4. Toggle to enable
5. Click "Trigger DAG"
6. Monitor execution in Graph View

### Via CLI

```bash
# Test DAG
airflow dags test kafka_pipeline 2024-01-01

# Trigger manually
airflow dags trigger kafka_pipeline

# Check status
airflow dags list-runs -d kafka_pipeline
```

---

## Monitoring

### Check Task Logs

```bash
# View logs for specific task
airflow tasks logs kafka_pipeline run_generator 2024-01-01
```

### Monitor Metrics

The DAG includes custom metrics:
- Events generated
- Orders processed
- Metrics calculated
- Pipeline duration

View in Airflow UI under "Browse > Task Instances"

---

## Error Handling

### Retry Logic

Tasks automatically retry on failure:
- Max retries: 2
- Retry delay: 5 minutes
- Exponential backoff: Enabled

### Failure Notifications

Configure email alerts:

```python
# In pipeline_dag.py
default_args = {
    'email': ['team@example.com'],
    'email_on_failure': True
}
```

### Manual Recovery

If a task fails:
1. Check logs in Airflow UI
2. Fix the issue
3. Clear failed task
4. Re-run from that point

---

## Advanced Features

### 1. Dynamic Task Generation

Generate tasks based on configuration:

```python
products = ['prod-1', 'prod-2', 'prod-3', 'prod-4', 'prod-5']

for product in products:
    task = BashOperator(
        task_id=f'analyze_{product}',
        bash_command=f'python analyze.py --product {product}'
    )
```

### 2. Sensors

Wait for data availability:

```python
from airflow.sensors.filesystem import FileSensor

wait_for_data = FileSensor(
    task_id='wait_for_data',
    filepath='/path/to/data.json',
    poke_interval=30,
    timeout=600
)
```

### 3. XComs

Share data between tasks:

```python
# Push data
ti.xcom_push(key='event_count', value=1000)

# Pull data
event_count = ti.xcom_pull(key='event_count', task_ids='run_generator')
```

---

## Production Considerations

### 1. Resource Management

```python
# Limit concurrent tasks
dag = DAG(
    'kafka_pipeline',
    max_active_runs=1,
    concurrency=4
)
```

### 2. SLA Monitoring

```python
# Set SLA for tasks
task = BashOperator(
    task_id='run_generator',
    sla=timedelta(minutes=5)
)
```

### 3. Connection Pooling

Configure Kafka connection in Airflow:
- Admin > Connections
- Add new connection
- Type: Generic
- Host: localhost:9092

---

## Troubleshooting

### Issue: DAG not appearing

**Solution**:
```bash
# Check DAG file for errors
python $AIRFLOW_HOME/dags/pipeline_dag.py

# Refresh DAGs
airflow dags list
```

### Issue: Tasks failing

**Solution**:
- Check task logs in UI
- Verify Kafka is running
- Check file paths in DAG
- Ensure Python dependencies installed

### Issue: Scheduler not running

**Solution**:
```bash
# Check scheduler status
ps aux | grep airflow

# Restart scheduler
airflow scheduler
```

---

## Learning Outcomes

After completing this bonus section, you will:
- Orchestrate Kafka pipelines with Airflow
- Implement error handling and retries
- Monitor pipeline execution
- Schedule recurring data jobs
- Handle task dependencies
- Use Airflow best practices

---

## Resources

- [Airflow Documentation](https://airflow.apache.org/docs/)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Kafka + Airflow Integration](https://airflow.apache.org/docs/apache-airflow-providers-apache-kafka/)

---

## Next Steps

1. Add data quality checks as Airflow tasks
2. Implement alerting for pipeline failures
3. Create dashboard for pipeline metrics
4. Add integration tests
5. Deploy to production environment

