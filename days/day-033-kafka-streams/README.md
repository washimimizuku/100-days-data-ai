# Day 33: Kafka Streams

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of this session, you will:
- Understand Kafka Streams architecture and topology
- Implement stateless transformations (map, filter, flatMap)
- Implement stateful transformations (aggregations, joins)
- Understand KStream vs KTable concepts
- Use windowing for time-based aggregations
- Build real-time stream processing applications
- Handle state stores and fault tolerance

---

## Why Kafka Streams?

Kafka Streams is a **client library** for building stream processing applications:

- **No separate cluster**: Runs as part of your application
- **Exactly-once semantics**: Built-in support
- **Fault-tolerant**: Automatic state recovery
- **Scalable**: Parallel processing across instances
- **Stateful**: Built-in state stores
- **Real-time**: Low-latency processing

**Use Cases**:
- Real-time analytics and aggregations
- Data enrichment and transformation
- Fraud detection and alerting
- Event-driven microservices
- Stream joins and correlations

---

## Kafka Streams Architecture

### Stream Processing Topology

```
Source Topic → Processor 1 → Processor 2 → Sink Topic
                    ↓
              State Store
```

**Components**:
- **Source**: Read from Kafka topics
- **Processor**: Transform, filter, aggregate
- **Sink**: Write to Kafka topics
- **State Store**: Persistent local storage

### KStream vs KTable

**KStream** (Event Stream):
- Represents a stream of records
- Each record is an independent event
- Unbounded, append-only
- Example: Click events, transactions

**KTable** (Changelog Stream):
- Represents a table (latest value per key)
- Updates replace previous values
- Bounded by keys
- Example: User profiles, inventory

```python
# KStream: All events
user-123: login
user-123: click
user-123: logout

# KTable: Latest state per key
user-123: {status: "offline", last_seen: "2024-01-01"}
```

---

## Stateless Transformations

### Common Operations

```python
from kafka import KafkaConsumer, KafkaProducer

consumer = KafkaConsumer('input-topic')
producer = KafkaProducer()

for message in consumer:
    value = message.value
    
    # Map: Transform values
    transformed = {**value, 'processed': True}
    
    # Filter: Select records
    if value['amount'] > 1000:
        producer.send('high-value-topic', transformed)
    
    # FlatMap: One-to-many (split text into words)
    if 'text' in value:
        for word in value['text'].split():
            producer.send('words-topic', {'word': word})
    
    # Branch: Route by category
    category = value.get('category', 'info')
    producer.send(f'{category}-topic', value)
```

---

## Stateful Transformations

### Aggregations (Count, Sum, Reduce)

```python
from collections import defaultdict

# Stateful: Count events per user
state = defaultdict(int)

for message in consumer:
    user_id = message.value['user_id']
    state[user_id] += 1
    
    output = {'user_id': user_id, 'count': state[user_id]}
    producer.send('counts-topic', output)
```

### GroupBy and Aggregate

```python
# GroupBy: Aggregate by key
state = defaultdict(lambda: {'count': 0, 'total': 0})

for message in consumer:
    product_id = message.value['product_id']
    amount = message.value['amount']
    
    state[product_id]['count'] += 1
    state[product_id]['total'] += amount
    state[product_id]['avg'] = state[product_id]['total'] / state[product_id]['count']
    
    producer.send('product-stats-topic', {
        'product_id': product_id,
        **state[product_id]
    })
```

### Joins (Stream-Stream, Stream-Table)

```python
# Stream-Table Join: Enrich events with user data
user_table = {}  # Simulated KTable

# Load user table
for message in user_consumer:
    user_id = message.key
    user_table[user_id] = message.value

# Join events with user data
for message in event_consumer:
    user_id = message.value['user_id']
    user_data = user_table.get(user_id, {})
    
    enriched = {**message.value, 'user_name': user_data.get('name')}
    producer.send('enriched-events-topic', enriched)
```

---

## Windowing

### Tumbling Windows (Fixed, Non-Overlapping)

```python
import time
from collections import defaultdict

# Tumbling window: 1-minute windows
window_size = 60  # seconds
windows = defaultdict(lambda: defaultdict(int))

for message in consumer:
    timestamp = message.timestamp / 1000  # Convert to seconds
    window_start = int(timestamp // window_size) * window_size
    
    key = message.value['key']
    windows[window_start][key] += 1
    
    # Emit window results
    producer.send('windowed-counts-topic', {
        'window_start': window_start,
        'key': key,
        'count': windows[window_start][key]
    })
```

### Session Windows (Dynamic, Activity-Based)

```python
# Session window: Group events within 5-min inactivity gap
session_gap = 300
sessions = {}

for message in consumer:
    user_id = message.value['user_id']
    timestamp = message.timestamp / 1000
    
    if user_id not in sessions or timestamp - sessions[user_id]['end'] > session_gap:
        if user_id in sessions:
            producer.send('sessions-topic', sessions[user_id])
        sessions[user_id] = {'start': timestamp, 'end': timestamp, 'count': 0}
    
    sessions[user_id]['end'] = timestamp
    sessions[user_id]['count'] += 1
```

---

## State Stores

### Local State Store

```python
import shelve

# Persistent state store
state_store = shelve.open('kafka_streams_state.db')

for message in consumer:
    key = message.key.decode()
    value = message.value
    
    # Read from state
    current_state = state_store.get(key, {})
    
    # Update state
    current_state['count'] = current_state.get('count', 0) + 1
    current_state['last_seen'] = time.time()
    
    # Write to state
    state_store[key] = current_state
    
    # Emit result
    producer.send('output-topic', current_state)

state_store.close()
```

---

## Python Alternative: Faust

Python doesn't have official Kafka Streams, but **Faust** provides similar functionality:

```python
import faust

app = faust.App('myapp', broker='kafka://localhost:9092')

class Order(faust.Record):
    order_id: str
    amount: float

orders_topic = app.topic('orders', value_type=Order)

@app.agent(orders_topic)
async def process_orders(orders):
    async for order in orders:
        if order.amount > 1000:
            print(f'High-value order: {order.order_id}')
```

---

## Production Best Practices

**Configuration**:
- Set `application.id` uniquely per application
- Configure `num.stream.threads` for parallelism
- Set `commit.interval.ms` for state commits
- Enable `exactly_once` semantics

**State Management**:
- Use RocksDB for large state stores
- Configure state store retention
- Enable changelog topics for recovery
- Monitor state store size

**Performance**:
- Partition input topics appropriately
- Use appropriate window sizes
- Tune buffer sizes and batch settings
- Monitor lag and throughput

**Fault Tolerance**:
- Enable standby replicas
- Configure replication factor
- Implement error handling
- Monitor application health

---

## 💻 Exercises (40 min)

### Exercise 1: Stateless Transformations
Implement map, filter, and flatMap operations.

### Exercise 2: Word Count
Build classic word count with aggregation.

### Exercise 3: Stream Enrichment
Join event stream with reference data.

### Exercise 4: Tumbling Window Aggregation
Count events in 1-minute tumbling windows.

### Exercise 5: Session Detection
Detect user sessions with inactivity gaps.

### Exercise 6: Real-Time Analytics
Calculate running statistics (count, sum, avg).

### Exercise 7: Stream Branching
Route events to different topics by category.

### Exercise 8: Stateful Processing with Store
Implement stateful processor with persistent state.

---

## ✅ Quiz (5 min)

Test your understanding in `quiz.md`.

---

## 🎯 Key Takeaways

- **Kafka Streams** is a library for stream processing, not a separate cluster
- **KStream** represents event streams, **KTable** represents state
- **Stateless** transformations (map, filter) don't require state
- **Stateful** transformations (aggregate, join) use state stores
- **Windowing** enables time-based aggregations
- **Exactly-once** semantics prevent duplicates
- **State stores** provide fault-tolerant local storage
- **Topology** defines the processing graph
- **Parallelism** achieved through partitions and threads
- **Faust** is a Python alternative to Kafka Streams

---

## 📚 Resources

- [Kafka Streams Documentation](https://kafka.apache.org/documentation/streams/)
- [Kafka Streams Tutorial](https://kafka.apache.org/documentation/streams/tutorial)
- [Faust Documentation](https://faust.readthedocs.io/)
- [Kafka Streams Examples](https://github.com/confluentinc/kafka-streams-examples)

---

## Tomorrow: Day 34 - Kafka Connect

Learn Kafka Connect for integrating Kafka with external systems using source and sink connectors.
