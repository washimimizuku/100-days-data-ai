# Day 30: Kafka Fundamentals

## 📖 Learning Objectives (15 min)

**Time**: 1 hour

- Understand Apache Kafka architecture and core concepts
- Learn about topics, partitions, and brokers
- Understand producers, consumers, and consumer groups
- Grasp Kafka's role in streaming architectures

---

## Theory

### What is Apache Kafka?

**Apache Kafka** is a distributed streaming platform designed for high-throughput, fault-tolerant, real-time data pipelines.

**Key Characteristics**:
- **Distributed**: Runs on clusters of servers
- **Persistent**: Stores data on disk (not just in-memory)
- **Scalable**: Handles millions of messages per second
- **Fault-tolerant**: Replicates data across brokers

**Common Use Cases**:
- Event streaming (clickstreams, logs, metrics)
- Message queue (decoupling services)
- Stream processing (real-time analytics)
- Data integration (CDC, ETL pipelines)

### Core Concepts

#### 1. Topics

A **topic** is a category or feed name to which records are published.

```
Topic: "user-events"
├── Event 1: {"user": "alice", "action": "login"}
├── Event 2: {"user": "bob", "action": "purchase"}
└── Event 3: {"user": "alice", "action": "logout"}
```

**Characteristics**:
- Topics are append-only logs
- Messages are immutable once written
- Messages have offsets (position in log)
- Topics can have multiple partitions

#### 2. Partitions

Topics are divided into **partitions** for parallelism and scalability.

```
Topic: "orders" (3 partitions)

Partition 0: [msg0, msg3, msg6, msg9]
Partition 1: [msg1, msg4, msg7, msg10]
Partition 2: [msg2, msg5, msg8, msg11]
```

**Why Partitions?**
- **Parallelism**: Multiple consumers can read simultaneously
- **Scalability**: Distribute load across brokers
- **Ordering**: Messages within a partition are ordered

**Partition Key**:
```python
# Messages with same key go to same partition
producer.send("orders", key="user123", value=order_data)
# All orders from user123 go to same partition → ordering guaranteed
```

#### 3. Brokers

A **broker** is a Kafka server that stores data and serves clients.

```
Kafka Cluster
├── Broker 1 (leader for partition 0)
├── Broker 2 (leader for partition 1)
└── Broker 3 (leader for partition 2)
```

**Replication**:
- Each partition has replicas across brokers
- One replica is the leader (handles reads/writes)
- Others are followers (backup)

```
Partition 0:
├── Leader: Broker 1
├── Follower: Broker 2
└── Follower: Broker 3
```

#### 4. Producers

**Producers** publish messages to topics.

```python
from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Send message
producer.send('user-events', {'user': 'alice', 'action': 'login'})
producer.flush()
```

**Producer Guarantees**:
- `acks=0`: Fire and forget (fastest, no guarantee)
- `acks=1`: Leader acknowledges (default)
- `acks=all`: All replicas acknowledge (slowest, safest)

#### 5. Consumers

**Consumers** read messages from topics.

```python
from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'user-events',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='earliest',
    group_id='analytics-group'
)

for message in consumer:
    print(f"Received: {message.value}")
```

**Consumer Offsets**:
- Kafka tracks which messages each consumer has read
- Offset = position in partition
- Consumers commit offsets to track progress

```
Partition 0: [msg0, msg1, msg2, msg3, msg4]
                           ↑
                    Consumer offset = 2
                    (next read: msg3)
```

#### 6. Consumer Groups

**Consumer groups** enable parallel consumption and load balancing.

```
Topic: "orders" (3 partitions)

Consumer Group "processors":
├── Consumer 1 → reads Partition 0
├── Consumer 2 → reads Partition 1
└── Consumer 3 → reads Partition 2
```

**Rules**:
- Each partition assigned to one consumer in group
- Multiple groups can read same topic independently
- If consumer fails, partition reassigned to another

**Example: Multiple Groups**
```
Topic: "user-events"

Group "analytics" → Consumer A (all partitions)
Group "alerts"    → Consumer B (all partitions)
Group "storage"   → Consumer C (all partitions)

All groups read same data independently!
```

### Kafka Architecture

```
┌─────────────────────────────────────────────┐
│           Kafka Cluster                      │
│                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Broker 1 │  │ Broker 2 │  │ Broker 3 │  │
│  │          │  │          │  │          │  │
│  │ Part 0   │  │ Part 1   │  │ Part 2   │  │
│  │ (leader) │  │ (leader) │  │ (leader) │  │
│  └──────────┘  └──────────┘  └──────────┘  │
│                                              │
└─────────────────────────────────────────────┘
       ↑                              ↓
   Producers                      Consumers
```

### Message Structure

```python
message = {
    "key": "user123",           # Optional, for partitioning
    "value": {                  # Actual data
        "user_id": "user123",
        "action": "purchase",
        "amount": 99.99
    },
    "timestamp": 1704470400000, # When produced
    "headers": {                # Optional metadata
        "source": "web-app",
        "version": "1.0"
    }
}
```

### Kafka vs Traditional Message Queues

| Feature | Kafka | RabbitMQ/SQS |
|---------|-------|--------------|
| **Model** | Pub/Sub + Log | Queue |
| **Persistence** | Disk (days/weeks) | Memory (minutes) |
| **Replay** | Yes (seek to offset) | No |
| **Throughput** | Very high (millions/sec) | Moderate |
| **Ordering** | Per partition | Per queue |
| **Use Case** | Event streaming, logs | Task queues, RPC |

### Retention and Compaction

**Time-based Retention**:
```
# Keep messages for 7 days
retention.ms=604800000
```

**Size-based Retention**:
```
# Keep up to 1GB per partition
retention.bytes=1073741824
```

**Log Compaction**:
- Keeps only latest value per key
- Useful for state snapshots

```
Before compaction:
key=user1, value={"name": "Alice"}
key=user2, value={"name": "Bob"}
key=user1, value={"name": "Alice Smith"}  ← Latest

After compaction:
key=user1, value={"name": "Alice Smith"}  ← Only latest kept
key=user2, value={"name": "Bob"}
```

### Kafka Guarantees

**Ordering**:
- ✅ Guaranteed within a partition
- ❌ Not guaranteed across partitions

**Durability**:
- Messages replicated across brokers
- Configurable replication factor (typically 3)

**Delivery Semantics**:
- **At-most-once**: May lose messages (acks=0)
- **At-least-once**: May duplicate (acks=1, default)
- **Exactly-once**: No loss, no duplicates (acks=all + idempotence)

### Common Kafka Commands

```bash
# Create topic
kafka-topics.sh --create --topic my-topic \
  --bootstrap-server localhost:9092 \
  --partitions 3 --replication-factor 2

# List topics
kafka-topics.sh --list --bootstrap-server localhost:9092

# Describe topic
kafka-topics.sh --describe --topic my-topic \
  --bootstrap-server localhost:9092

# Produce messages (console)
kafka-console-producer.sh --topic my-topic \
  --bootstrap-server localhost:9092

# Consume messages (console)
kafka-console-consumer.sh --topic my-topic \
  --bootstrap-server localhost:9092 --from-beginning

# Delete topic
kafka-topics.sh --delete --topic my-topic \
  --bootstrap-server localhost:9092
```

---

## 💻 Exercises (40 min)

### Exercise 1: Kafka Concepts Quiz
Answer conceptual questions about Kafka architecture.

### Exercise 2: Partition Assignment
Calculate partition assignment for messages with keys.

### Exercise 3: Consumer Group Simulation
Simulate consumer group behavior with partition assignment.

### Exercise 4: Offset Management
Track consumer offsets and simulate offset commits.

### Exercise 5: Message Ordering
Analyze ordering guarantees in different scenarios.

### Exercise 6: Architecture Design
Design a Kafka-based architecture for a use case.

---

## ✅ Quiz (5 min)

Test your understanding in `quiz.md`.

---

## 🎯 Key Takeaways

- **Kafka** is a distributed streaming platform for high-throughput event streaming
- **Topics** are categories; **partitions** enable parallelism and ordering
- **Brokers** store data; **replication** provides fault tolerance
- **Producers** write messages; **consumers** read messages
- **Consumer groups** enable parallel consumption with load balancing
- **Ordering** guaranteed within partition, not across partitions
- **Offsets** track consumer position in partition
- Kafka persists data on disk (unlike traditional message queues)

---

## 📚 Resources

- [Kafka Documentation](https://kafka.apache.org/documentation/)
- [Kafka: The Definitive Guide](https://www.confluent.io/resources/kafka-the-definitive-guide/)
- [Confluent Kafka Tutorials](https://kafka-tutorials.confluent.io/)
- [Kafka Python Client](https://kafka-python.readthedocs.io/)

---

## Tomorrow: Day 31 - Kafka Producers & Consumers

Hands-on with Kafka Python client: producing and consuming messages, handling errors, and best practices.
