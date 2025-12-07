# Day 50: Review Exercise Solutions

Reference answers for self-assessment. Compare your answers with these.

---

## Section 1: Data Formats (10 points)

### Exercise 1.1: Format Selection
```python
answers = {
    "A": "Avro",      # Real-time streaming - schema evolution, compact
    "B": "Parquet",   # 100TB analytics - columnar, compressed
    "C": "JSON",      # API response - human-readable, flexible
    "D": "Parquet",   # ML training - efficient reads, columnar
}
```

### Exercise 1.2: CSV to Parquet
```python
import pandas as pd

df = pd.read_csv("data.csv")
df.to_parquet("data.parquet", compression="snappy", index=False)

# Or with PyArrow for better performance
import pyarrow.parquet as pq
import pyarrow.csv as csv

table = csv.read_csv("data.csv")
pq.write_table(table, "data.parquet", compression="snappy")
```

### Exercise 1.3: Compression Algorithms
```python
use_cases = {
    "snappy": "Fast compression/decompression, moderate ratio. Good for Spark/Hadoop.",
    "zstd": "Best compression ratio, slower. Good for cold storage, archival.",
    "lz4": "Fastest compression, lower ratio. Good for real-time streaming.",
}
```

---

## Section 2: Table Formats (10 points)

### Exercise 2.1: Iceberg vs Delta Lake
```python
differences = [
    "Iceberg: Multi-engine support (Spark, Flink, Trino). Delta: Spark-focused.",
    "Iceberg: Hidden partitioning. Delta: Explicit partitioning.",
    "Iceberg: Metadata stored separately. Delta: Transaction log in _delta_log.",
]
```

### Exercise 2.2: Time Travel Query
```sql
-- Iceberg
SELECT * FROM sales.orders
FOR SYSTEM_TIME AS OF TIMESTAMP '2024-01-01 10:00:00';

-- Or by snapshot ID
SELECT * FROM sales.orders
FOR SYSTEM_VERSION AS OF 12345;
```

### Exercise 2.3: ACID Properties
```python
acid = {
    "Atomicity": "All operations in transaction succeed or all fail. No partial updates.",
    "Consistency": "Data remains in valid state. Constraints maintained.",
    "Isolation": "Concurrent transactions don't interfere. Serializable isolation.",
    "Durability": "Committed changes persist. Survive failures.",
}
```

---

## Section 3: Architecture (15 points)

### Exercise 3.1: Medallion Architecture
```python
medallion = {
    "bronze": "Raw data as-is. JSON logs, CSV files. No transformations. Append-only.",
    "silver": "Cleaned, validated data. Deduplicated. Type conversions. Quality checks.",
    "gold": "Business-level aggregates. Denormalized. Optimized for queries. Star schema.",
}
```

### Exercise 3.2: Star Schema
```python
schema = {
    "fact_sales": [
        "sale_id", "date_key", "product_key", "customer_key",
        "quantity", "amount", "discount"
    ],
    "dim_product": [
        "product_key", "product_id", "name", "category", "price"
    ],
    "dim_customer": [
        "customer_key", "customer_id", "name", "email", "segment"
    ],
    "dim_date": [
        "date_key", "date", "year", "quarter", "month", "day", "day_of_week"
    ],
}
```

### Exercise 3.3: SCD Type 2
```python
def update_customer(existing_df, new_df):
    from pyspark.sql.functions import current_timestamp, lit
    
    # Close existing records
    updated_existing = existing_df.join(
        new_df, "customer_id", "left_anti"
    ).union(
        existing_df.join(new_df, "customer_id", "inner")
        .select(existing_df["*"])
        .withColumn("end_date", current_timestamp())
        .withColumn("is_current", lit(False))
    )
    
    # Add new records
    new_records = new_df.withColumn("effective_date", current_timestamp()) \
        .withColumn("end_date", lit(None)) \
        .withColumn("is_current", lit(True))
    
    return updated_existing.union(new_records)
```

---

## Section 4: Spark (20 points)

### Exercise 4.1: Spark Pipeline
```python
def exercise_4_1(spark):
    df = spark.read.parquet("sales.parquet")
    
    result = df.filter(col("amount") > 100) \
        .groupBy("category") \
        .agg(sum("amount").alias("total_sales"))
    
    result.write.mode("overwrite").parquet("results.parquet")
```

### Exercise 4.2: Broadcast Join
```python
optimized = """
from pyspark.sql.functions import broadcast

df1 = spark.read.parquet("large_table")  # 1TB
df2 = spark.read.parquet("small_table")  # 1MB

# Broadcast small table to avoid shuffle
result = df1.join(broadcast(df2), "id")
"""
```

### Exercise 4.3: Shuffle Operations
```python
shuffle_operations = [
    "groupBy / groupByKey - Requires data redistribution by key",
    "join - Needs matching keys on same partition",
    "repartition / coalesce - Explicitly redistributes data",
]
```

### Exercise 4.4: Partitioning
```python
df.write \
    .partitionBy("date", "region") \
    .mode("overwrite") \
    .parquet("output/sales")
```

---

## Section 5: Kafka (20 points)

### Exercise 5.1: Kafka Producer
```python
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    acks='all',  # Wait for all replicas
    retries=3,
    max_in_flight_requests_per_connection=1,  # Exactly-once
    enable_idempotence=True,  # Exactly-once
)
```

### Exercise 5.2: Kafka Consumer
```python
consumer = KafkaConsumer(
    'topic-name',
    bootstrap_servers=['localhost:9092'],
    group_id='analytics',
    enable_auto_commit=False,  # Manual commit
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
)

# Manual commit after processing
for message in consumer:
    process(message.value)
    consumer.commit()
```

### Exercise 5.3: Consumer Group Rebalancing
```python
explanation = {
    "triggers": [
        "Consumer joins/leaves group",
        "Topic partition count changes",
        "Consumer heartbeat timeout"
    ],
    "process": "Coordinator reassigns partitions to consumers. Stop-the-world event.",
    "impact": "Brief processing pause. Duplicate messages possible if not committed.",
}
```

### Exercise 5.4: Topic Configuration
```python
config = {
    "partitions": 50,  # High parallelism for throughput
    "replication_factor": 3,  # Fault tolerance
    "retention_ms": 604800000,  # 7 days
    "min_insync_replicas": 2,  # Exactly-once guarantee
}
```

---

## Section 6: Data Quality (10 points)

### Exercise 6.1: Quality Dimensions
```python
dimensions = [
    "Accuracy - Correct values",
    "Completeness - No missing data",
    "Consistency - Same across systems",
    "Timeliness - Available when needed",
    "Validity - Conforms to rules",
    "Uniqueness - No duplicates",
]
```

### Exercise 6.2: Great Expectations
```python
expectations = """
import great_expectations as gx

context = gx.get_context()
suite = context.add_expectation_suite("user_validation")

# Age between 0 and 120
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeBetween(
        column="age", min_value=0, max_value=120
    )
)

# Email pattern
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToMatchRegex(
        column="email", regex=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    )
)

# No nulls in user_id
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToNotBeNull(column="user_id")
)
"""
```

### Exercise 6.3: Data Lineage
```python
lineage = {
    "source": {
        "system": "S3",
        "path": "s3://bucket/raw/data.csv",
        "timestamp": "2024-01-01T10:00:00Z"
    },
    "transformations": [
        {"step": "clean", "operation": "remove_nulls", "records_dropped": 100},
        {"step": "aggregate", "operation": "group_by_date", "records_out": 365},
    ],
    "outputs": {
        "system": "Redshift",
        "table": "analytics.daily_report",
        "timestamp": "2024-01-01T10:15:00Z"
    },
}
```

---

## Section 7: Streaming (15 points)

### Exercise 7.1: Structured Streaming
```python
def exercise_7_1(spark):
    schema = StructType([
        StructField("event_id", StringType()),
        StructField("user_id", StringType()),
        StructField("amount", DoubleType()),
        StructField("timestamp", TimestampType())
    ])
    
    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "events") \
        .load() \
        .select(from_json(col("value").cast("string"), schema).alias("data")) \
        .select("data.*") \
        .withWatermark("timestamp", "5 minutes")
    
    windowed = df.groupBy(
        window("timestamp", "1 minute")
    ).agg(
        count("*").alias("event_count"),
        sum("amount").alias("total_amount")
    )
    
    query = windowed.writeStream \
        .outputMode("append") \
        .format("console") \
        .trigger(processingTime="10 seconds") \
        .start()
    
    return query
```

### Exercise 7.2: Stream-to-Stream Join
```python
def exercise_7_2(spark):
    clicks = spark.readStream.format("kafka")...withWatermark("timestamp", "10 minutes")
    purchases = spark.readStream.format("kafka")...withWatermark("timestamp", "10 minutes")
    
    joined = clicks.alias("c").join(
        purchases.alias("t"),
        expr("""
            c.user_id = t.user_id AND
            c.product_id = t.product_id AND
            t.timestamp >= c.timestamp AND
            t.timestamp <= c.timestamp + interval 10 minutes
        """)
    )
    
    return joined
```

### Exercise 7.3: Watermarking
```python
answers = {
    "definition": "Threshold for how late data can arrive. Events older than watermark are dropped.",
    "choosing_delay": "Balance completeness vs latency. Analyze data arrival patterns. Start conservative.",
    "late_data": "Data arriving after watermark is dropped. Can track with metrics for monitoring.",
}
```

### Exercise 7.4: Streaming Optimization
```python
optimizations = [
    "Increase parallelism: More Kafka partitions, more Spark executors",
    "Tune trigger interval: Larger batches reduce overhead",
    "Optimize state: Reduce state size, use appropriate timeout",
]
```

---

## Scoring Guide

**Total Points**: 100

- **90-100**: Excellent understanding
- **75-89**: Good, minor gaps
- **60-74**: Fair, review needed
- **< 60**: Significant review needed

Compare your answers with these solutions and score honestly.
