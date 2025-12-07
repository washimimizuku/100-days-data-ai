"""
Day 34: Kafka Connect - Solutions

Prerequisites:
docker run -d --name kafka-connect -p 8083:8083 confluentinc/cp-kafka-connect:latest
pip install requests kafka-python
"""

import requests
import json
import time
from kafka import KafkaProducer, KafkaConsumer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONNECT_URL = "http://localhost:8083"
BOOTSTRAP_SERVERS = ['localhost:9092']


def exercise_1_file_source_connector():
    """File source connector"""
    print("Creating file source connector...")
    
    config = {
        "name": "file-source",
        "config": {
            "connector.class": "FileStreamSource",
            "tasks.max": "1",
            "file": "/tmp/kafka-input.txt",
            "topic": "file-input"
        }
    }
    
    try:
        response = requests.post(
            f"{CONNECT_URL}/connectors",
            json=config,
            headers={"Content-Type": "application/json"}
        )
        print(f"✓ Connector created: {response.json()}")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()


def exercise_2_file_sink_connector():
    """File sink connector"""
    print("Creating file sink connector...")
    
    config = {
        "name": "file-sink",
        "config": {
            "connector.class": "FileStreamSink",
            "tasks.max": "1",
            "file": "/tmp/kafka-output.txt",
            "topics": "file-input"
        }
    }
    
    try:
        response = requests.post(f"{CONNECT_URL}/connectors", json=config)
        print(f"✓ Connector created: {response.json()['name']}")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()


def exercise_3_jdbc_source_connector():
    """JDBC source connector"""
    print("Creating JDBC source connector...")
    
    config = {
        "name": "jdbc-source",
        "config": {
            "connector.class": "io.confluent.connect.jdbc.JdbcSourceConnector",
            "tasks.max": "1",
            "connection.url": "jdbc:postgresql://localhost:5432/mydb",
            "connection.user": "postgres",
            "connection.password": "password",
            "table.whitelist": "users",
            "mode": "incrementing",
            "incrementing.column.name": "id",
            "topic.prefix": "db-"
        }
    }
    
    print("Configuration:", json.dumps(config, indent=2))
    print("Note: Requires JDBC connector plugin and PostgreSQL")
    print()


def exercise_4_jdbc_sink_connector():
    """JDBC sink connector"""
    print("Creating JDBC sink connector...")
    
    config = {
        "name": "jdbc-sink",
        "config": {
            "connector.class": "io.confluent.connect.jdbc.JdbcSinkConnector",
            "tasks.max": "1",
            "connection.url": "jdbc:postgresql://localhost:5432/targetdb",
            "connection.user": "postgres",
            "connection.password": "password",
            "topics": "orders",
            "auto.create": "true",
            "insert.mode": "upsert",
            "pk.mode": "record_key",
            "pk.fields": "id"
        }
    }
    
    print("Configuration:", json.dumps(config, indent=2))
    print("Note: Requires JDBC connector plugin and PostgreSQL")
    print()


def exercise_5_connector_management():
    """Connector management via REST API"""
    print("Managing connectors...")
    
    try:
        # List connectors
        response = requests.get(f"{CONNECT_URL}/connectors")
        connectors = response.json()
        print(f"✓ Connectors: {connectors}")
        
        if connectors:
            connector_name = connectors[0]
            
            # Get status
            status = requests.get(f"{CONNECT_URL}/connectors/{connector_name}/status").json()
            print(f"✓ Status: {status['connector']['state']}")
            
            # Pause
            requests.put(f"{CONNECT_URL}/connectors/{connector_name}/pause")
            print(f"✓ Paused: {connector_name}")
            
            time.sleep(1)
            
            # Resume
            requests.put(f"{CONNECT_URL}/connectors/{connector_name}/resume")
            print(f"✓ Resumed: {connector_name}")
            
    except Exception as e:
        print(f"✗ Error: {e}")
    print()


def exercise_6_single_message_transforms():
    """SMTs configuration"""
    print("Configuring Single Message Transforms...")
    
    config = {
        "name": "smt-example",
        "config": {
            "connector.class": "FileStreamSource",
            "file": "/tmp/input.txt",
            "topic": "transformed-topic",
            "transforms": "InsertField,MaskField",
            "transforms.InsertField.type": "org.apache.kafka.connect.transforms.InsertField$Value",
            "transforms.InsertField.timestamp.field": "processed_at",
            "transforms.MaskField.type": "org.apache.kafka.connect.transforms.MaskField$Value",
            "transforms.MaskField.fields": "ssn,credit_card",
            "transforms.MaskField.replacement": "****"
        }
    }
    
    print("SMT Configuration:")
    print("- InsertField: Adds 'processed_at' timestamp")
    print("- MaskField: Masks 'ssn' and 'credit_card' fields")
    print()


def exercise_7_custom_source_connector():
    """Custom source connector"""
    print("Custom source connector implementation...")
    
    class CustomSourceConnector:
        def __init__(self, config):
            self.producer = KafkaProducer(
                bootstrap_servers=config['bootstrap.servers'],
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            self.topic = config['topic']
            self.running = True
        
        def poll(self):
            """Fetch data from source"""
            count = 0
            while self.running and count < 10:
                # Simulate fetching from external source
                data = {
                    'id': count,
                    'message': f'Custom source message {count}',
                    'timestamp': time.time()
                }
                
                self.producer.send(self.topic, data)
                print(f"✓ Sent: {data}")
                
                count += 1
                time.sleep(0.5)
            
            self.producer.flush()
        
        def stop(self):
            self.running = False
            self.producer.close()
    
    # Test custom connector
    config = {
        'bootstrap.servers': BOOTSTRAP_SERVERS,
        'topic': 'custom-source-topic'
    }
    
    connector = CustomSourceConnector(config)
    connector.poll()
    connector.stop()
    print()


def exercise_8_monitoring_error_handling():
    """Monitoring and error handling"""
    print("Monitoring connectors...")
    
    try:
        # Get all connectors
        connectors = requests.get(f"{CONNECT_URL}/connectors").json()
        
        for connector_name in connectors:
            # Get detailed status
            status = requests.get(f"{CONNECT_URL}/connectors/{connector_name}/status").json()
            
            print(f"\nConnector: {connector_name}")
            print(f"  State: {status['connector']['state']}")
            print(f"  Worker: {status['connector']['worker_id']}")
            
            # Check tasks
            for task in status['tasks']:
                print(f"  Task {task['id']}: {task['state']}")
                if task['state'] == 'FAILED':
                    print(f"    Error: {task.get('trace', 'No trace')}")
        
        # Error handling configuration example
        error_config = {
            "errors.tolerance": "all",
            "errors.log.enable": "true",
            "errors.log.include.messages": "true",
            "errors.deadletterqueue.topic.name": "dlq-topic",
            "errors.deadletterqueue.topic.replication.factor": "1"
        }
        
        print("\nError Handling Config:")
        for key, value in error_config.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    print()


if __name__ == "__main__":
    print("Day 34: Kafka Connect - Solutions\n")
    print("=" * 60)
    print("\nNote: Requires Kafka Connect running on localhost:8083\n")
    
    try:
        print("\nExercise 1: File Source Connector")
        print("-" * 60)
        exercise_1_file_source_connector()
        
        print("\nExercise 2: File Sink Connector")
        print("-" * 60)
        exercise_2_file_sink_connector()
        
        print("\nExercise 3: JDBC Source Connector")
        print("-" * 60)
        exercise_3_jdbc_source_connector()
        
        print("\nExercise 4: JDBC Sink Connector")
        print("-" * 60)
        exercise_4_jdbc_sink_connector()
        
        print("\nExercise 5: Connector Management")
        print("-" * 60)
        exercise_5_connector_management()
        
        print("\nExercise 6: Single Message Transforms")
        print("-" * 60)
        exercise_6_single_message_transforms()
        
        print("\nExercise 7: Custom Source Connector")
        print("-" * 60)
        exercise_7_custom_source_connector()
        
        print("\nExercise 8: Monitoring and Error Handling")
        print("-" * 60)
        exercise_8_monitoring_error_handling()
        
        print("=" * 60)
        print("All exercises completed!")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure Kafka Connect is running on localhost:8083")
