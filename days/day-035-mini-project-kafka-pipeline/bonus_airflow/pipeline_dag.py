"""
Airflow DAG for Kafka Pipeline Orchestration

This DAG orchestrates the real-time Kafka pipeline:
1. Starts Kafka infrastructure
2. Creates topics
3. Runs pipeline components
4. Verifies output
5. Cleans up resources
"""
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import os
import json

# Configuration
PROJECT_DIR = '/path/to/day-035-mini-project-kafka-pipeline'  # UPDATE THIS
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')

default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['team@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'kafka_pipeline',
    default_args=default_args,
    description='Real-time Kafka data pipeline',
    schedule_interval='@hourly',
    catchup=False,
    max_active_runs=1,
    tags=['kafka', 'streaming', 'data-engineering']
)


# Task 1: Start Kafka
start_kafka = BashOperator(
    task_id='start_kafka',
    bash_command="""
    docker start kafka 2>/dev/null || \
    docker run -d --name kafka -p 9092:9092 apache/kafka:latest
    sleep 15
    """,
    dag=dag
)


# Task 2: Create Topics
create_topics = BashOperator(
    task_id='create_topics',
    bash_command="""
    docker exec kafka kafka-topics.sh --create --topic orders \
        --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1 \
        --if-not-exists
    
    docker exec kafka kafka-topics.sh --create --topic processed-orders \
        --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1 \
        --if-not-exists
    
    docker exec kafka kafka-topics.sh --create --topic product-metrics \
        --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1 \
        --if-not-exists
    
    docker exec kafka kafka-topics.sh --create --topic orders-dlq \
        --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1 \
        --if-not-exists
    """,
    dag=dag
)


# Task 3: Run Event Generator
run_generator = BashOperator(
    task_id='run_generator',
    bash_command=f'cd {PROJECT_DIR} && python event_generator.py',
    dag=dag
)


# Task 4: Run Order Processor
run_processor = BashOperator(
    task_id='run_processor',
    bash_command=f'cd {PROJECT_DIR} && timeout 65 python order_processor.py',
    dag=dag
)


# Task 5: Run Analytics Engine
run_analytics = BashOperator(
    task_id='run_analytics',
    bash_command=f'cd {PROJECT_DIR} && timeout 65 python analytics_engine.py',
    dag=dag
)


# Task 6: Run Data Sink
run_sink = BashOperator(
    task_id='run_sink',
    bash_command=f'cd {PROJECT_DIR} && timeout 65 python data_sink.py',
    dag=dag
)


# Task 7: Verify Output
def verify_output(**context):
    """Verify pipeline output and push metrics to XCom"""
    output_file = os.path.join(OUTPUT_DIR, 'metrics.json')
    
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Output file not found: {output_file}")
    
    # Count metrics
    with open(output_file, 'r') as f:
        metrics = [json.loads(line) for line in f]
    
    metric_count = len(metrics)
    
    if metric_count == 0:
        raise ValueError("No metrics written to output file")
    
    # Calculate totals
    total_orders = sum(m['order_count'] for m in metrics)
    total_revenue = sum(m['total_revenue'] for m in metrics)
    
    # Push to XCom
    context['ti'].xcom_push(key='metric_count', value=metric_count)
    context['ti'].xcom_push(key='total_orders', value=total_orders)
    context['ti'].xcom_push(key='total_revenue', value=total_revenue)
    
    print(f"✓ Verification passed")
    print(f"  Metrics written: {metric_count}")
    print(f"  Total orders: {total_orders}")
    print(f"  Total revenue: ${total_revenue:.2f}")
    
    return {
        'metric_count': metric_count,
        'total_orders': total_orders,
        'total_revenue': total_revenue
    }


verify_output_task = PythonOperator(
    task_id='verify_output',
    python_callable=verify_output,
    provide_context=True,
    dag=dag
)


# Task 8: Cleanup
cleanup = BashOperator(
    task_id='cleanup',
    bash_command="""
    # Stop Python processes
    pkill -f event_generator.py 2>/dev/null || true
    pkill -f order_processor.py 2>/dev/null || true
    pkill -f analytics_engine.py 2>/dev/null || true
    pkill -f data_sink.py 2>/dev/null || true
    
    # Keep Kafka running for next run
    echo "Cleanup complete"
    """,
    trigger_rule='all_done',  # Run even if upstream fails
    dag=dag
)


# Task Dependencies
start_kafka >> create_topics

# Run components in parallel after topics are created
create_topics >> [run_generator, run_processor, run_analytics, run_sink]

# Verify after all components complete
[run_generator, run_processor, run_analytics, run_sink] >> verify_output_task

# Cleanup after verification
verify_output_task >> cleanup


# Optional: Add monitoring task
def send_metrics_to_monitoring(**context):
    """Send pipeline metrics to monitoring system"""
    ti = context['ti']
    
    metric_count = ti.xcom_pull(key='metric_count', task_ids='verify_output')
    total_orders = ti.xcom_pull(key='total_orders', task_ids='verify_output')
    total_revenue = ti.xcom_pull(key='total_revenue', task_ids='verify_output')
    
    # Send to monitoring system (e.g., Prometheus, Datadog)
    # This is a placeholder - implement based on your monitoring setup
    print(f"Sending metrics to monitoring system:")
    print(f"  kafka.pipeline.metrics_count: {metric_count}")
    print(f"  kafka.pipeline.total_orders: {total_orders}")
    print(f"  kafka.pipeline.total_revenue: {total_revenue}")


monitor_task = PythonOperator(
    task_id='send_monitoring_metrics',
    python_callable=send_metrics_to_monitoring,
    provide_context=True,
    dag=dag
)

verify_output_task >> monitor_task >> cleanup
