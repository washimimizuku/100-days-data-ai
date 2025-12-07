# Day 32: Kafka Consumers & Consumer Groups

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of this session, you will:
- Implement production-ready Kafka consumers in Python
- Understand consumer groups and parallel processing
- Master offset management strategies
- Handle consumer rebalancing
- Implement at-least-once and at-most-once semantics
- Monitor consumer lag and performance

---

## Why Consumer Groups?

Consumer groups enable **parallel processing** and **fault tolerance**:

- **Scalability**: Multiple consumers process partitions in parallel
- **Fault tolerance**: If consumer fails, partitions reassigned to others
- **Load balancing**: Partitions distributed across consumers
- **Ordering**: Messages in same partition processed in order

---

## Consumer Basics

```python
from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'my-topic',
    bootstrap_servers=['localhost:9092'],
    group_id='my-group',
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='earliest',
    enable_auto_commit=False,
    max_poll_interval_ms=300000,
    session_timeout_ms=10000,
    heartbeat_interval_ms=3000
)

for message in consumer:
    print(f"P{message.partition}@{message.offset}: {message.value}")
```

---

## Consumer Groups

### How Consumer Groups Work

```
Topic: orders (3 partitions)
Consumer Group: order-processors

Partition 0 → Consumer 1
Partition 1 → Consumer 2
Partition 2 → Consumer 3
```

**Rules**:
- Each partition assigned to ONE consumer in group
- One consumer can handle MULTIPLE partitions
- If consumers > partitions, some consumers idle

### Creating Consumer Group

```python
# Multiple consumers with same group_id share partitions
consumer1 = KafkaConsumer('orders', group_id='order-processors')
consumer2 = KafkaConsumer('orders', group_id='order-processors')

# Check assigned partitions
partitions = consumer.assignment()
for p in partitions:
    print(f"Assigned: {p.topic}[{p.partition}] @ offset {consumer.position(p)}")
```

---

## Offset Management

### Offset Management Strategies

```python
# Auto-commit (simple, may lose/duplicate)
consumer = KafkaConsumer('topic', enable_auto_commit=True, auto_commit_interval_ms=5000)

# Manual commit (recommended, at-least-once)
consumer = KafkaConsumer('topic', enable_auto_commit=False)
for message in consumer:
    process(message.value)
    consumer.commit()

# Batch commit (better throughput)
batch = []
for message in consumer:
    batch.append(message)
    if len(batch) >= 100:
        process_batch(batch)
        consumer.commit()
        batch = []

# Commit specific offset
from kafka import TopicPartition
tp = TopicPartition(message.topic, message.partition)
consumer.commit({tp: message.offset + 1})
```

---

## Rebalancing

### What is Rebalancing?

When consumers join/leave group, partitions are **reassigned**:

```
Before: 3 consumers, 3 partitions
Consumer 1: [P0]
Consumer 2: [P1]
Consumer 3: [P2]

Consumer 2 crashes...

After rebalance: 2 consumers, 3 partitions
Consumer 1: [P0, P1]
Consumer 3: [P2]
```

### Rebalance Listener

```python
from kafka import ConsumerRebalanceListener

class RebalanceListener(ConsumerRebalanceListener):
    def on_partitions_revoked(self, revoked):
        print(f"Partitions revoked: {revoked}")
        # Commit offsets before losing partitions
        consumer.commit()
    
    def on_partitions_assigned(self, assigned):
        print(f"Partitions assigned: {assigned}")
        # Initialize state for new partitions

consumer = KafkaConsumer(
    'my-topic',
    group_id='my-group'
)

consumer.subscribe(['my-topic'], listener=RebalanceListener())
```

### Avoiding Rebalances

```python
# Increase timeouts for slow processing
consumer = KafkaConsumer(
    'my-topic', group_id='my-group',
    max_poll_interval_ms=600000,  # 10 min
    session_timeout_ms=30000       # 30 sec
)
```

---

## Delivery Semantics

### Delivery Semantics

```python
# At-most-once (may lose): auto-commit before processing
consumer = KafkaConsumer('topic', enable_auto_commit=True)

# At-least-once (may duplicate): commit after processing
consumer = KafkaConsumer('topic', enable_auto_commit=False)
for msg in consumer:
    process(msg.value)
    consumer.commit()

# Exactly-once: idempotent processing
consumer = KafkaConsumer('topic', enable_auto_commit=False, isolation_level='read_committed')
for msg in consumer:
    if not already_processed(msg.offset):
        process(msg.value)
        mark_processed(msg.offset)
    consumer.commit()
```

---

## Seeking and Replaying

### Seeking and Replaying

```python
from kafka import TopicPartition

tp = TopicPartition('my-topic', 0)
consumer.seek(tp, 100)              # Seek to offset 100
consumer.seek_to_beginning(tp)      # Seek to start
consumer.seek_to_end(tp)            # Seek to end

# Replay last 1000 messages
for p in consumer.assignment():
    current = consumer.position(p)
    consumer.seek(p, max(0, current - 1000))
```

---

## Monitoring Consumer Lag

### Check Lag

```python
from kafka import TopicPartition

def get_consumer_lag(consumer):
    partitions = consumer.assignment()
    lag_info = {}
    
    for partition in partitions:
        # Current position
        current_offset = consumer.position(partition)
        
        # Latest offset in partition
        end_offsets = consumer.end_offsets([partition])
        latest_offset = end_offsets[partition]
        
        # Calculate lag
        lag = latest_offset - current_offset
        lag_info[partition.partition] = {
            'current': current_offset,
            'latest': latest_offset,
            'lag': lag
        }
    
    return lag_info

# Monitor lag
lag = get_consumer_lag(consumer)
for partition, info in lag.items():
    print(f"Partition {partition}: lag = {info['lag']}")
```

---

## Error Handling

### Retry Pattern

```python
from kafka.errors import KafkaError
import time

def consume_with_retry(consumer, max_retries=3):
    for message in consumer:
        retries = 0
        while retries < max_retries:
            try:
                process(message.value)
                consumer.commit()
                break
            except Exception as e:
                retries += 1
                print(f"Retry {retries}/{max_retries}: {e}")
                time.sleep(2 ** retries)  # Exponential backoff
                
                if retries == max_retries:
                    send_to_dlq(message)
                    consumer.commit()  # Skip bad message
```

### Dead Letter Queue

```python
from kafka import KafkaProducer

dlq_producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def send_to_dlq(message, error):
    dlq_message = {
        'original_topic': message.topic,
        'partition': message.partition,
        'offset': message.offset,
        'key': message.key,
        'value': message.value,
        'error': str(error),
        'timestamp': time.time()
    }
    dlq_producer.send(f'{message.topic}.dlq', dlq_message)
```

---

## Production Best Practices

**Configuration**:
- Use `group_id` for parallel processing
- Set `enable_auto_commit=False` for control
- Increase `max_poll_interval_ms` for slow processing
- Set `session_timeout_ms` appropriately

**Processing**:
- Commit after successful processing (at-least-once)
- Make processing idempotent (exactly-once)
- Handle errors with retries and DLQ
- Monitor consumer lag

**Performance**:
- Batch processing for throughput
- Multiple consumers for parallelism
- Tune fetch settings (`fetch_min_bytes`, `max_poll_records`)
- Avoid long processing in poll loop

**Reliability**:
- Implement rebalance listeners
- Handle consumer shutdown gracefully
- Monitor and alert on lag
- Test failure scenarios

---

## 💻 Exercises (40 min)

### Exercise 1: Basic Consumer
Create a consumer that reads messages and prints them.

### Exercise 2: Consumer Group
Create multiple consumers in same group for parallel processing.

### Exercise 3: Manual Offset Management
Implement consumer with manual commits after processing.

### Exercise 4: Batch Processing
Process messages in batches for better throughput.

### Exercise 5: Rebalance Handling
Implement rebalance listener to handle partition changes.

### Exercise 6: Consumer Lag Monitoring
Monitor and report consumer lag for each partition.

### Exercise 7: Error Handling with DLQ
Implement retry logic and dead letter queue.

### Exercise 8: Production Consumer Class
Create reusable consumer class with best practices.

---

## ✅ Quiz (5 min)

Test your understanding in `quiz.md`.

---

## 🎯 Key Takeaways

- **Consumer groups** enable parallel processing and fault tolerance
- **Manual commits** provide at-least-once guarantee
- **Rebalancing** redistributes partitions when consumers change
- **Offset management** is critical for reliability
- **Consumer lag** indicates processing backlog
- **Idempotent processing** enables exactly-once semantics
- **Batch processing** improves throughput
- **Error handling** with retries and DLQ prevents data loss
- **Monitoring** lag and metrics is essential
- **Graceful shutdown** prevents message loss

---

## 📚 Resources

- [kafka-python Consumer](https://kafka-python.readthedocs.io/en/master/apidoc/KafkaConsumer.html)
- [Kafka Consumer Configs](https://kafka.apache.org/documentation/#consumerconfigs)
- [Consumer Groups](https://kafka.apache.org/documentation/#consumergroups)
- [Offset Management](https://kafka.apache.org/documentation/#offsetmanagement)

---

## Tomorrow: Day 33 - Kafka Streams

Learn Kafka Streams for real-time stream processing with stateless and stateful transformations.
