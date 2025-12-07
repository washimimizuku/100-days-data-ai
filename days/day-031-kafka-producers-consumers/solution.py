"""
Day 31: Kafka Producers - Real Implementation Solutions

Prerequisites:
docker run -d --name kafka -p 9092:9092 apache/kafka:latest
pip install kafka-python
"""

from kafka import KafkaProducer
from kafka.errors import KafkaError, KafkaTimeoutError
import json
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOOTSTRAP_SERVERS = ['localhost:9092']


def exercise_1_basic_producer():
    """Basic producer with JSON serialization"""
    print("Creating basic producer...")
    
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks='all'
    )
    
    for i in range(5):
        message = {'id': i, 'message': f'Hello Kafka {i}', 'timestamp': time.time()}
        future = producer.send('test-topic', message)
        metadata = future.get(timeout=10)
        print(f"Sent message {i} to partition {metadata.partition} at offset {metadata.offset}")
    
    producer.flush()
    producer.close()
    print("Producer closed\n")


def exercise_2_producer_with_keys():
    """Producer with keys for partition assignment"""
    print("Creating producer with keys...")
    
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        key_serializer=lambda k: k.encode('utf-8'),
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks='all'
    )
    
    partition_map = {}
    
    for i in range(1, 11):
        key = f'user-{i}'
        message = {'user_id': key, 'action': 'login', 'timestamp': time.time()}
        future = producer.send('user-events', key=key, value=message)
        metadata = future.get(timeout=10)
        
        partition_map[key] = metadata.partition
        print(f"Key '{key}' → Partition {metadata.partition}")
    
    # Verify same keys go to same partition
    print("\nVerifying partition consistency...")
    for key in ['user-1', 'user-5', 'user-10']:
        future = producer.send('user-events', key=key, value={'test': 'verify'})
        metadata = future.get(timeout=10)
        print(f"Key '{key}' → Partition {metadata.partition} (expected {partition_map[key]})")
    
    producer.close()
    print()


def exercise_3_async_callbacks():
    """Async producer with callbacks"""
    print("Creating async producer with callbacks...")
    
    success_count = 0
    error_count = 0
    
    def on_success(metadata):
        nonlocal success_count
        success_count += 1
        print(f"✓ Sent to {metadata.topic}[{metadata.partition}] @ {metadata.offset}")
    
    def on_error(e):
        nonlocal error_count
        error_count += 1
        print(f"✗ Error: {e}")
    
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks='all'
    )
    
    # Send async with callbacks
    for i in range(10):
        message = {'id': i, 'data': f'async-message-{i}'}
        future = producer.send('async-topic', message)
        future.add_callback(on_success)
        future.add_errback(on_error)
    
    producer.flush()
    print(f"\nResults: {success_count} success, {error_count} errors")
    producer.close()
    print()


def exercise_4_transactional_producer():
    """Transactional producer for exactly-once"""
    print("Creating transactional producer...")
    
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        transactional_id='my-transactional-id',
        enable_idempotence=True,
        acks='all',
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    producer.init_transactions()
    
    try:
        producer.begin_transaction()
        
        # Send to orders topic
        for i in range(5):
            producer.send('orders', {'order_id': i, 'amount': 100 + i})
        
        # Send to inventory topic
        for i in range(5):
            producer.send('inventory', {'product_id': i, 'quantity': -1})
        
        producer.commit_transaction()
        print("Transaction committed successfully")
        
    except Exception as e:
        print(f"Error: {e}")
        producer.abort_transaction()
        print("Transaction aborted")
    
    producer.close()
    print()


def exercise_5_error_handling_dlq():
    """Error handling with dead letter queue"""
    print("Creating producer with DLQ...")
    
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks='all'
    )
    
    def send_with_retry(topic, message, max_retries=3):
        for attempt in range(max_retries):
            try:
                future = producer.send(topic, message)
                metadata = future.get(timeout=10)
                print(f"✓ Sent to {topic}: {message}")
                return metadata
            except KafkaTimeoutError:
                print(f"Timeout on attempt {attempt + 1}")
                if attempt == max_retries - 1:
                    send_to_dlq(topic, message, "Max retries exceeded")
            except KafkaError as e:
                print(f"Error: {e}")
                send_to_dlq(topic, message, str(e))
                break
    
    def send_to_dlq(original_topic, message, error):
        dlq_message = {
            'original_topic': original_topic,
            'message': message,
            'error': error,
            'timestamp': time.time()
        }
        producer.send(f'{original_topic}.dlq', dlq_message)
        print(f"✗ Sent to DLQ: {error}")
    
    # Test with valid messages
    for i in range(3):
        send_with_retry('events', {'id': i, 'data': f'message-{i}'})
    
    producer.close()
    print()


def exercise_6_performance_optimized():
    """High-throughput producer"""
    print("Creating performance-optimized producer...")
    
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks='all',
        batch_size=32768,
        linger_ms=100,
        compression_type='lz4',
        buffer_memory=67108864
    )
    
    start_time = time.time()
    message_count = 10000
    
    for i in range(message_count):
        message = {
            'id': i,
            'data': f'performance-test-{i}',
            'timestamp': time.time()
        }
        producer.send('performance-topic', message)
    
    producer.flush()
    elapsed = time.time() - start_time
    throughput = message_count / elapsed
    
    print(f"Sent {message_count} messages in {elapsed:.2f}s")
    print(f"Throughput: {throughput:.0f} messages/sec")
    
    producer.close()
    print()


def exercise_7_avro_serialization():
    """Avro serialization (requires confluent-kafka)"""
    print("Avro serialization example...")
    print("Note: Requires confluent-kafka and schema registry")
    print("Install: pip install confluent-kafka")
    print()
    
    try:
        from confluent_kafka import Producer
        from confluent_kafka.schema_registry import SchemaRegistryClient
        from confluent_kafka.schema_registry.avro import AvroSerializer
        from confluent_kafka.serialization import SerializationContext, MessageField
        
        schema_str = """
        {
          "type": "record",
          "name": "User",
          "fields": [
            {"name": "name", "type": "string"},
            {"name": "email", "type": "string"},
            {"name": "age", "type": "int"}
          ]
        }
        """
        
        # Note: Requires schema registry running
        print("Schema defined (requires schema registry to run)")
        
    except ImportError:
        print("confluent-kafka not installed")
        print("Install with: pip install confluent-kafka")
    print()


def exercise_8_production_ready_class():
    """Production-ready producer class"""
    print("Creating production-ready producer class...")
    
    class ProductionProducer:
        def __init__(self, bootstrap_servers, topic):
            self.topic = topic
            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks='all',
                retries=3,
                enable_idempotence=True,
                compression_type='lz4',
                batch_size=32768,
                linger_ms=100
            )
            self.success_count = 0
            self.error_count = 0
        
        def send(self, message, key=None):
            try:
                future = self.producer.send(
                    self.topic,
                    key=key.encode() if key else None,
                    value=message
                )
                future.add_callback(self._on_success)
                future.add_errback(self._on_error)
            except Exception as e:
                logger.error(f"Failed to send: {e}")
                self._send_to_dlq(message, str(e))
        
        def _on_success(self, metadata):
            self.success_count += 1
            logger.info(f"Sent to {metadata.topic}[{metadata.partition}] @ {metadata.offset}")
        
        def _on_error(self, e):
            self.error_count += 1
            logger.error(f"Send failed: {e}")
        
        def _send_to_dlq(self, message, error):
            dlq_message = {
                'original_message': message,
                'error': error,
                'timestamp': time.time()
            }
            self.producer.send(f'{self.topic}.dlq', dlq_message)
        
        def get_metrics(self):
            return {
                'success': self.success_count,
                'errors': self.error_count,
                'total': self.success_count + self.error_count
            }
        
        def close(self):
            self.producer.flush()
            self.producer.close()
    
    # Test the class
    producer = ProductionProducer(BOOTSTRAP_SERVERS, 'production-events')
    
    for i in range(100):
        producer.send({'event_id': i, 'data': f'event-{i}'}, key=f'key-{i % 10}')
    
    time.sleep(2)  # Wait for async sends
    metrics = producer.get_metrics()
    print(f"Metrics: {metrics}")
    
    producer.close()
    print()


if __name__ == "__main__":
    print("Day 31: Kafka Producers - Real Implementation Solutions\n")
    print("=" * 60)
    print("\nNote: Requires Kafka running on localhost:9092")
    print("Start with: docker run -d --name kafka -p 9092:9092 apache/kafka:latest\n")
    
    try:
        print("\nExercise 1: Basic Producer")
        print("-" * 60)
        exercise_1_basic_producer()
        
        print("\nExercise 2: Producer with Keys")
        print("-" * 60)
        exercise_2_producer_with_keys()
        
        print("\nExercise 3: Async Callbacks")
        print("-" * 60)
        exercise_3_async_callbacks()
        
        print("\nExercise 4: Transactional Producer")
        print("-" * 60)
        exercise_4_transactional_producer()
        
        print("\nExercise 5: Error Handling and DLQ")
        print("-" * 60)
        exercise_5_error_handling_dlq()
        
        print("\nExercise 6: Performance Optimized")
        print("-" * 60)
        exercise_6_performance_optimized()
        
        print("\nExercise 7: Avro Serialization")
        print("-" * 60)
        exercise_7_avro_serialization()
        
        print("\nExercise 8: Production-Ready Class")
        print("-" * 60)
        exercise_8_production_ready_class()
        
        print("=" * 60)
        print("All exercises completed!")
        
    except KafkaError as e:
        print(f"\nKafka Error: {e}")
        print("Make sure Kafka is running on localhost:9092")
