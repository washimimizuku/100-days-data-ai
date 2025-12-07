# Real-Time Analytics with Spark + Kafka - Project Specification

## Project Architecture

### Data Model

**Clickstream Event**:
```json
{
  "event_id": "evt_1234567890",
  "user_id": "user_123",
  "session_id": "sess_abc",
  "product_id": "prod_001",
  "category": "electronics",
  "action": "view",  // view, click, add_to_cart
  "timestamp": "2024-01-01T10:00:00Z",
  "page_url": "/product/prod_001",
  "referrer": "google.com"
}
```

**Transaction Event**:
```json
{
  "transaction_id": "txn_9876543210",
  "user_id": "user_123",
  "session_id": "sess_abc",
  "product_id": "prod_001",
  "category": "electronics",
  "amount": 299.99,
  "quantity": 1,
  "timestamp": "2024-01-01T10:05:00Z",
  "payment_method": "credit_card"
}
```

---

## Implementation Details

### 1. Data Generator (`data_generator.py`)

```python
from kafka import KafkaProducer
import json
import random
import time
from datetime import datetime
import uuid

class DataGenerator:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all'
        )
        
        self.products = [
            {"id": "prod_001", "category": "electronics", "price": 299.99},
            {"id": "prod_002", "category": "electronics", "price": 499.99},
            {"id": "prod_003", "category": "clothing", "price": 49.99},
            {"id": "prod_004", "category": "books", "price": 19.99},
            {"id": "prod_005", "category": "home", "price": 79.99}
        ]
        
        self.actions = ["view", "click", "add_to_cart"]
        self.users = [f"user_{i}" for i in range(1, 101)]
        
    def generate_clickstream(self):
        """Generate clickstream event"""
        user_id = random.choice(self.users)
        product = random.choice(self.products)
        
        return {
            "event_id": f"evt_{uuid.uuid4().hex[:12]}",
            "user_id": user_id,
            "session_id": f"sess_{user_id}_{int(time.time() / 1800)}",
            "product_id": product["id"],
            "category": product["category"],
            "action": random.choice(self.actions),
            "timestamp": datetime.now().isoformat(),
            "page_url": f"/product/{product['id']}",
            "referrer": random.choice(["google.com", "facebook.com", "direct"])
        }
    
    def generate_transaction(self):
        """Generate transaction event (20% of clicks)"""
        user_id = random.choice(self.users)
        product = random.choice(self.products)
        
        return {
            "transaction_id": f"txn_{uuid.uuid4().hex[:12]}",
            "user_id": user_id,
            "session_id": f"sess_{user_id}_{int(time.time() / 1800)}",
            "product_id": product["id"],
            "category": product["category"],
            "amount": product["price"] * random.randint(1, 3),
            "quantity": random.randint(1, 3),
            "timestamp": datetime.now().isoformat(),
            "payment_method": random.choice(["credit_card", "debit_card", "paypal"])
        }
    
    def run(self, duration=120, events_per_second=100):
        """Run generator"""
        start = time.time()
        count = {"clicks": 0, "transactions": 0}
        
        while time.time() - start < duration:
            # 80% clickstream, 20% transactions
            if random.random() < 0.8:
                event = self.generate_clickstream()
                self.producer.send('clickstream', event)
                count["clicks"] += 1
            else:
                event = self.generate_transaction()
                self.producer.send('transactions', event)
                count["transactions"] += 1
            
            if (count["clicks"] + count["transactions"]) % 1000 == 0:
                print(f"Generated: {count['clicks']} clicks, {count['transactions']} transactions")
            
            time.sleep(1.0 / events_per_second)
        
        self.producer.flush()
        print(f"Total: {count['clicks']} clicks, {count['transactions']} transactions")
```

---

### 2. Streaming Analytics (`streaming_analytics.py`)

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.streaming import GroupState, GroupStateTimeout

# Configuration
KAFKA_BOOTSTRAP = "localhost:9092"
CHECKPOINT_DIR = "output/checkpoint"

# Initialize Spark
spark = SparkSession.builder \
    .appName("RealTimeAnalytics") \
    .config("spark.sql.shuffle.partitions", "50") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Schemas
clickstream_schema = StructType([
    StructField("event_id", StringType()),
    StructField("user_id", StringType()),
    StructField("session_id", StringType()),
    StructField("product_id", StringType()),
    StructField("category", StringType()),
    StructField("action", StringType()),
    StructField("timestamp", TimestampType()),
    StructField("page_url", StringType()),
    StructField("referrer", StringType())
])

transaction_schema = StructType([
    StructField("transaction_id", StringType()),
    StructField("user_id", StringType()),
    StructField("session_id", StringType()),
    StructField("product_id", StringType()),
    StructField("category", StringType()),
    StructField("amount", DoubleType()),
    StructField("quantity", IntegerType()),
    StructField("timestamp", TimestampType()),
    StructField("payment_method", StringType())
])

# Read streams
clicks = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP) \
    .option("subscribe", "clickstream") \
    .option("startingOffsets", "latest") \
    .load() \
    .select(from_json(col("value").cast("string"), clickstream_schema).alias("data")) \
    .select("data.*") \
    .withWatermark("timestamp", "5 minutes")

transactions = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP) \
    .option("subscribe", "transactions") \
    .option("startingOffsets", "latest") \
    .load() \
    .select(from_json(col("value").cast("string"), transaction_schema).alias("data")) \
    .select("data.*") \
    .withWatermark("timestamp", "5 minutes")

# 1. Page Views per Product (1-minute windows)
page_views = clicks.groupBy(
    window("timestamp", "1 minute"),
    "product_id",
    "category"
).agg(
    count("*").alias("view_count"),
    countDistinct("user_id").alias("unique_users")
)

# 2. Revenue per Category (5-minute windows)
revenue = transactions.groupBy(
    window("timestamp", "5 minutes"),
    "category"
).agg(
    count("*").alias("transaction_count"),
    sum("amount").alias("total_revenue"),
    avg("amount").alias("avg_transaction")
)

# 3. Conversion Rate (Join clicks + transactions)
conversions = clicks.alias("c").join(
    transactions.alias("t"),
    expr("""
        c.user_id = t.user_id AND
        c.product_id = t.product_id AND
        t.timestamp >= c.timestamp AND
        t.timestamp <= c.timestamp + interval 10 minutes
    """),
    "left_outer"
).groupBy(
    window("c.timestamp", "5 minutes"),
    "c.product_id"
).agg(
    count("c.event_id").alias("clicks"),
    count("t.transaction_id").alias("purchases"),
    (count("t.transaction_id") / count("c.event_id")).alias("conversion_rate")
)

# 4. Session Analytics (Stateful)
session_schema = StructType([
    StructField("session_id", StringType()),
    StructField("event_count", IntegerType()),
    StructField("duration_seconds", IntegerType()),
    StructField("status", StringType())
])

state_schema = StructType([
    StructField("start_time", LongType()),
    StructField("last_time", LongType()),
    StructField("event_count", IntegerType())
])

def track_sessions(session_id, events, state):
    """Track user sessions with 30-minute timeout"""
    if state.hasTimedOut:
        session = state.get
        state.remove()
        duration = session.last_time - session.start_time
        return iter([Row(
            session_id=session_id,
            event_count=session.event_count,
            duration_seconds=int(duration),
            status="completed"
        )])
    
    if state.exists:
        session = state.get
    else:
        session = None
    
    for event in events:
        event_time = int(event.timestamp.timestamp())
        if session is None:
            session = Row(
                start_time=event_time,
                last_time=event_time,
                event_count=1
            )
        else:
            session = Row(
                start_time=session.start_time,
                last_time=event_time,
                event_count=session.event_count + 1
            )
    
    if session:
        state.update(session)
        state.setTimeoutDuration("30 minutes")
    
    return iter([])

sessions = clicks.groupByKey(lambda x: x.session_id) \
    .flatMapGroupsWithState(
        track_sessions,
        session_schema,
        state_schema,
        GroupStateTimeout.ProcessingTimeTimeout,
        outputMode="append"
    )

# Write outputs
query1 = page_views.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", False) \
    .trigger(processingTime="10 seconds") \
    .start()

query2 = revenue.writeStream \
    .outputMode("append") \
    .format("parquet") \
    .option("path", "output/revenue") \
    .option("checkpointLocation", f"{CHECKPOINT_DIR}/revenue") \
    .trigger(processingTime="10 seconds") \
    .start()

query3 = conversions.writeStream \
    .outputMode("append") \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP) \
    .option("topic", "analytics-results") \
    .option("checkpointLocation", f"{CHECKPOINT_DIR}/conversions") \
    .trigger(processingTime="10 seconds") \
    .start()

# Wait for termination
spark.streams.awaitAnyTermination()
```

---

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Latency | < 10s | Event to output |
| Throughput | > 1000 events/s | Input rate |
| State Size | < 1GB | Memory usage |
| CPU Usage | < 80% | Average |
| Success Rate | > 99.9% | No data loss |

---

## Testing Strategy

1. **Unit Tests**: Test individual functions
2. **Integration Tests**: Test end-to-end flow
3. **Performance Tests**: Load testing
4. **Failure Tests**: Simulate failures
5. **Data Quality**: Validate outputs

---

## Monitoring

Track these metrics:
- Processing rate (events/second)
- Batch duration
- State size
- Watermark lag
- Error rate
- Resource usage

---

## Deployment

1. Package application
2. Configure resources
3. Deploy to cluster
4. Monitor and alert
5. Scale as needed
