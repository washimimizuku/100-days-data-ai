"""
Airflow DAG for Real-Time Analytics Pipeline Orchestration

This DAG orchestrates the Spark + Kafka streaming pipeline with:
- Health checks
- Pipeline startup/shutdown
- Monitoring
- Alerting
"""
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime, timedelta
import subprocess
import json
import os


default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'streaming_pipeline',
    default_args=default_args,
    description='Orchestrate real-time analytics pipeline',
    schedule_interval=None,  # Manual trigger
    catchup=False,
    tags=['streaming', 'kafka', 'spark'],
)


def check_kafka_health(**context):
    """Check if Kafka is running and healthy"""
    try:
        result = subprocess.run(
            ['docker', 'ps', '--filter', 'name=kafka', '--format', '{{.Status}}'],
            capture_output=True,
            text=True,
            check=True
        )
        if 'Up' in result.stdout:
            print("Kafka is healthy")
            return True
        else:
            raise Exception("Kafka is not running")
    except Exception as e:
        raise Exception(f"Kafka health check failed: {e}")


def verify_topics(**context):
    """Verify required Kafka topics exist"""
    required_topics = ['clickstream', 'transactions', 'analytics-results']
    
    try:
        result = subprocess.run(
            ['docker', 'exec', 'kafka-day49', '/opt/kafka/bin/kafka-topics.sh',
             '--list', '--bootstrap-server', 'localhost:9092'],
            capture_output=True,
            text=True,
            check=True
        )
        
        existing_topics = result.stdout.strip().split('\n')
        missing = [t for t in required_topics if t not in existing_topics]
        
        if missing:
            raise Exception(f"Missing topics: {missing}")
        
        print(f"All required topics exist: {required_topics}")
        return True
    except Exception as e:
        raise Exception(f"Topic verification failed: {e}")


def monitor_pipeline_metrics(**context):
    """Monitor pipeline metrics and check for issues"""
    output_dir = '/opt/airflow/dags/output'
    
    # Check if output files exist
    if not os.path.exists(f"{output_dir}/analytics"):
        raise Exception("No output files found")
    
    # Count output files
    file_count = len([f for f in os.listdir(f"{output_dir}/analytics/revenue")
                     if f.endswith('.parquet')])
    
    print(f"Output files: {file_count}")
    
    if file_count == 0:
        raise Exception("No data processed")
    
    # Check generator stats
    if os.path.exists(f"{output_dir}/generator.log"):
        with open(f"{output_dir}/generator.log") as f:
            log = f.read()
            if 'errors' in log.lower():
                print("Warning: Errors detected in generator log")
    
    return {"file_count": file_count}


def generate_summary_report(**context):
    """Generate summary report of pipeline execution"""
    ti = context['ti']
    metrics = ti.xcom_pull(task_ids='monitor_pipeline')
    
    report = {
        'execution_date': context['execution_date'].isoformat(),
        'duration': str(datetime.now() - context['execution_date']),
        'metrics': metrics,
        'status': 'success'
    }
    
    # Save report
    output_dir = '/opt/airflow/dags/output'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/pipeline_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Report generated: {report}")
    return report


# Task 1: Check Kafka health
check_kafka = PythonOperator(
    task_id='check_kafka',
    python_callable=check_kafka_health,
    dag=dag,
)

# Task 2: Verify topics
verify_topics_task = PythonOperator(
    task_id='verify_topics',
    python_callable=verify_topics,
    dag=dag,
)

# Task 3: Start data generator
start_generator = BashOperator(
    task_id='start_generator',
    bash_command='cd /opt/airflow/dags && python data_generator.py --duration 60 --rate 50 > output/generator.log 2>&1 &',
    dag=dag,
)

# Task 4: Wait for data to flow
wait_for_data = BashOperator(
    task_id='wait_for_data',
    bash_command='sleep 10',
    dag=dag,
)

# Task 5: Start streaming application
start_streaming = BashOperator(
    task_id='start_streaming',
    bash_command='cd /opt/airflow/dags && timeout 45 python streaming_analytics.py || true',
    dag=dag,
)

# Task 6: Monitor pipeline
monitor_pipeline = PythonOperator(
    task_id='monitor_pipeline',
    python_callable=monitor_pipeline_metrics,
    dag=dag,
)

# Task 7: Generate report
generate_report = PythonOperator(
    task_id='generate_report',
    python_callable=generate_summary_report,
    dag=dag,
)

# Task 8: Cleanup
cleanup = BashOperator(
    task_id='cleanup',
    bash_command='pkill -f data_generator.py || true',
    dag=dag,
)

# Define task dependencies
check_kafka >> verify_topics_task >> start_generator >> wait_for_data
wait_for_data >> start_streaming >> monitor_pipeline >> generate_report >> cleanup
