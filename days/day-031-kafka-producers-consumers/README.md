# Day 31: Kafka Producers - Real Implementation

## 📖 Learning Objectives (15 min)

**Time**: 1 hour

- Implement production-ready Kafka producers in Python
- Master producer configurations and performance tuning
- Handle serialization with JSON, Avro, and custom formats
- Implement error handling, retries, and idempotence
- Use callbacks for asynchronous message delivery
- Apply batching and compression for throughput

---

## Theory

### Kafka Python Client

The `kafka-python` library provides Python bindings for Kafka.

```bash
pip install kafka-python
```

### Producer Basics

```python
from kafka import KafkaProducer
import json

# Create producer
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Synchronous send
future = producer.send('my-topic', {'key': 'value'})
metadata = future.get(timeout=10)
print(f"Sent to partition {metadata.partition} at offset {metadata.offset}")

# With key (same key → same partition)
producer.send('orders', key=b'user123', value={'order_id': 1, 'amount': 99.99})

# Asynchronous with callbacks
future = producer.send('my-topic', {'data': 'value'})
future.add_callback(lambda m: print(f"Success: {m.partition}"))
future.add_errback(lambda e: print(f"Error: {e}"))

producer.flush()
producer.close()
```

### Producer Configuration

```python
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    
    # Serialization
    key_serializer=lambda k: k.encode('utf-8'),
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    
    # Acknowledgments
    acks='all',  # Wait for all replicas (safest)
    # acks=1     # Wait for leader only (default)
    # acks=0     # Don't wait (fastest, least safe)
    
    # Retries
    retries=3,
    retry_backoff_ms=100,
    
    # Batching (for throughput)
    batch_size=16384,  # Bytes
    linger_ms=10,      # Wait up to 10ms to batch
    
    # Compression
    compression_type='gzip',  # or 'snappy', 'lz4', 'zstd'
    
    # Idempotence (exactly-once)
    enable_idempotence=True,
    
    # Timeouts
    request_timeout_ms=30000,
)
```

### Advanced Producer Patterns

```python
# Custom partitioner
class CustomPartitioner:
    def __call__(self, key, all_partitions, available_partitions):
        return hash(key) % len(all_partitions) if key else available_partitions[0]

# Transactional producer (exactly-once)
producer = KafkaProducer(transactional_id='my-tx-id', enable_idempotence=True, acks='all')
producer.init_transactions()
try:
    producer.begin_transaction()
    producer.send('topic1', {'data': 'value1'})
    producer.send('topic2', {'data': 'value2'})
    producer.commit_transaction()
except Exception:
    producer.abort_transaction()

# Message headers (metadata)
producer.send('my-topic', value={'data': 'value'},
              headers=[('source', b'api'), ('version', b'1.0')])
```

### Error Handling & Retries

```python
from kafka.errors import KafkaError, KafkaTimeoutError

# Retry logic
def send_with_retry(producer, topic, message, max_retries=3):
    for attempt in range(max_retries):
        try:
            return producer.send(topic, message).get(timeout=10)
        except KafkaTimeoutError:
            if attempt == max_retries - 1: raise

# Dead Letter Queue pattern
def send_with_dlq(producer, topic, message):
    try:
        producer.send(topic, message).get(timeout=10)
    except Exception as e:
        dlq_msg = {'original_topic': topic, 'message': message, 'error': str(e)}
        producer.send(f'{topic}.dlq', dlq_msg)
```

### Advanced Serialization

```python
# Avro with Schema Registry
from confluent_kafka import Producer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer

schema_str = '{"type": "record", "name": "User", "fields": [{"name": "name", "type": "string"}]}'
schema_client = SchemaRegistryClient({'url': 'http://localhost:8081'})
avro_serializer = AvroSerializer(schema_client, schema_str)

producer = Producer({'bootstrap.servers': 'localhost:9092'})
producer.produce('users', value=avro_serializer({'name': 'Alice'}, None))

# Custom serialization
import pickle
producer = KafkaProducer(value_serializer=lambda v: pickle.dumps(v))
producer.send('events', {'event_id': '123', 'data': {}})
```

### Performance Optimization

#### Batching for Throughput

```python
producer = KafkaProducer(
    # Batch settings
    batch_size=32768,      # 32KB batches
    linger_ms=100,         # Wait up to 100ms to fill batch
    buffer_memory=67108864, # 64MB buffer
    
    # Compression
    compression_type='lz4',  # Fast compression
    
    # Network
    max_in_flight_requests_per_connection=5
)

# Send many messages - they'll be batched automatically
for i in range(10000):
    producer.send('my-topic', {'id': i, 'data': f'message-{i}'})

producer.flush()  # Ensure all sent
```

#### Monitoring Producer Metrics

```python
# Get producer metrics
metrics = producer.metrics()

for metric_name, metric in metrics.items():
    if 'record-send-rate' in metric_name:
        print(f"{metric_name}: {metric.value}")
```

### Production Best Practices

**Configuration**:
- `enable_idempotence=True` - Prevents duplicates
- `acks='all'` - Wait for all replicas (durability)
- `retries=MAX_INT` - Retry indefinitely
- `max.in.flight.requests.per.connection=5` - With idempotence
- `compression_type='lz4'` - Fast compression

**Error Handling**:
- Implement retry logic with exponential backoff
- Use dead letter queues for failed messages
- Log all errors with context
- Monitor producer metrics

**Performance**:
- Batch messages with `linger_ms` and `batch_size`
- Use compression for large messages
- Async sends with callbacks for throughput
- Reuse producer instances (thread-safe)

**Security**:
- Use SSL/TLS for encryption
- Implement SASL authentication
- Never log sensitive data
- Validate input before sending

### Real-World Producer Example

```python
import logging
from kafka import KafkaProducer
from kafka.errors import KafkaError
import json
import time

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
        self.logger = logging.getLogger(__name__)
    
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
            self.logger.error(f"Failed to send: {e}")
            self._send_to_dlq(message, str(e))
    
    def _on_success(self, metadata):
        self.logger.info(
            f"Sent to {metadata.topic}[{metadata.partition}] @ {metadata.offset}"
        )
    
    def _on_error(self, e):
        self.logger.error(f"Send failed: {e}")
    
    def _send_to_dlq(self, message, error):
        dlq_message = {
            'original_message': message,
            'error': error,
            'timestamp': time.time()
        }
        self.producer.send(f'{self.topic}.dlq', dlq_message)
    
    def close(self):
        self.producer.flush()
        self.producer.close()

# Usage
producer = ProductionProducer(['localhost:9092'], 'events')
producer.send({'event': 'user_login', 'user_id': 123}, key='user-123')
producer.close()
```

---

## 💻 Exercises (40 min)

### Exercise 1: Basic Producer
Create a producer that sends JSON messages with proper configuration.

### Exercise 2: Producer with Keys and Partitioning
Send messages with keys to control partition assignment.

### Exercise 3: Async Producer with Callbacks
Implement asynchronous sending with success/error callbacks.

### Exercise 4: Transactional Producer
Create a producer with exactly-once semantics using transactions.

### Exercise 5: Error Handling and DLQ
Implement comprehensive error handling with dead letter queue.

### Exercise 6: Performance-Optimized Producer
Build a high-throughput producer with batching and compression.

### Exercise 7: Avro Serialization
Implement producer with Avro schema serialization.

### Exercise 8: Production-Ready Producer Class
Create a reusable producer class with monitoring and best practices.

---

## ✅ Quiz (5 min)

Test your understanding in `quiz.md`.

---

## 🎯 Key Takeaways

- **Idempotence** (`enable_idempotence=True`) prevents duplicate messages
- **acks='all'** ensures messages are replicated before acknowledgment
- **Batching** (`batch_size`, `linger_ms`) dramatically improves throughput
- **Compression** (`lz4`, `gzip`) reduces network bandwidth
- **Async sends** with callbacks provide high performance with feedback
- **Transactions** enable exactly-once semantics across multiple messages
- **Error handling** with retries and DLQ prevents data loss
- **Serialization** (JSON, Avro) must match consumer expectations
- **Partitioning** with keys ensures message ordering per key
- **Monitoring** producer metrics is essential for production systems

---

## 📚 Resources

- [kafka-python Documentation](https://kafka-python.readthedocs.io/)
- [Kafka Producer Configs](https://kafka.apache.org/documentation/#producerconfigs)
- [Kafka Consumer Configs](https://kafka.apache.org/documentation/#consumerconfigs)
- [Confluent Python Client](https://docs.confluent.io/kafka-clients/python/current/overview.html)

---

## Tomorrow: Day 32 - Kafka Consumers & Consumer Groups

Learn how to implement production-ready Kafka consumers with consumer groups, offset management, and rebalancing strategies.
