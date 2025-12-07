# Day 4: Avro

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of today, you will:
- Understand Apache Avro and its use cases
- Learn why Avro is used with Kafka
- Read and write Avro files in Python
- Understand schema evolution basics
- Compare Avro with Parquet and JSON

---

## Theory

### What is Apache Avro?

Apache Avro is a **row-based data serialization format** designed for streaming data and schema evolution.

**Key Features:**
- Row-based format (reads entire records)
- Schema stored with data
- Schema evolution support
- Compact binary format
- Perfect for Kafka streaming
- Language-neutral

### Why Avro + Kafka?

**The Problem:**
Kafka moves data between systems in real-time. What format should the data be in?

**Why Not JSON?**
- ❌ No schema enforcement
- ❌ Larger message size
- ❌ No schema evolution
- ❌ Slower serialization

**Why Avro is Perfect:**
- ✅ Schema Registry validates data
- ✅ 70% smaller than JSON
- ✅ Schema evolution without breaking consumers
- ✅ Type safety

### Kafka + Avro Architecture

```
Producer → [Avro Serialize] → Schema Registry → Kafka → Consumer → [Avro Deserialize]
```

**Schema Registry:**
- Stores all schemas centrally
- Assigns schema IDs
- Validates compatibility
- Enables safe schema changes

### Avro Schema Example

```python
schema = {
    "type": "record",
    "name": "User",
    "fields": [
        {"name": "name", "type": "string"},
        {"name": "age", "type": "int"},
        {"name": "email", "type": ["null", "string"], "default": null}
    ]
}
```

### Writing Avro Files

```python
from avro.datafile import DataFileWriter
from avro.io import DatumWriter
import avro.schema

# Define schema
schema = avro.schema.parse('''
{
    "type": "record",
    "name": "User",
    "fields": [
        {"name": "name", "type": "string"},
        {"name": "age", "type": "int"}
    ]
}
''')

# Write data
with DataFileWriter(open("users.avro", "wb"), DatumWriter(), schema) as writer:
    writer.append({"name": "Alice", "age": 25})
    writer.append({"name": "Bob", "age": 30})
```

### Reading Avro Files

```python
from avro.datafile import DataFileReader
from avro.io import DatumReader

with DataFileReader(open("users.avro", "rb"), DatumReader()) as reader:
    for user in reader:
        print(user)
```

### Kafka Producer with Avro

```python
from confluent_kafka.avro import AvroProducer
import avro.schema

schema = avro.schema.parse('''
{
    "type": "record",
    "name": "Event",
    "fields": [
        {"name": "user_id", "type": "int"},
        {"name": "action", "type": "string"}
    ]
}
''')

producer = AvroProducer({
    'bootstrap.servers': 'localhost:9092',
    'schema.registry.url': 'http://localhost:8081'
}, default_value_schema=schema)

# Send event
producer.produce(
    topic='events',
    value={'user_id': 123, 'action': 'click'}
)
producer.flush()
```

### Kafka Consumer with Avro

```python
from confluent_kafka.avro import AvroConsumer

consumer = AvroConsumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'my-group',
    'schema.registry.url': 'http://localhost:8081'
})

consumer.subscribe(['events'])

while True:
    msg = consumer.poll(1.0)
    if msg is None:
        continue
    
    event = msg.value()  # Automatically deserialized!
    print(f"User {event['user_id']} did {event['action']}")
```

### Schema Evolution

**V1 Schema:**
```json
{
    "fields": [
        {"name": "user_id", "type": "int"},
        {"name": "action", "type": "string"}
    ]
}
```

**V2 Schema (add optional field):**
```json
{
    "fields": [
        {"name": "user_id", "type": "int"},
        {"name": "action", "type": "string"},
        {"name": "device", "type": ["null", "string"], "default": null}
    ]
}
```

**Result:**
- Old producers work (device = null)
- New producers send device
- Old consumers ignore device
- New consumers read device
- **No breaking changes!**

### Avro vs Parquet vs JSON

| Feature | Avro | Parquet | JSON |
|---------|------|---------|------|
| **Storage** | Row-based | Columnar | Row-based |
| **Schema** | Required | Optional | None |
| **Evolution** | Excellent | Limited | None |
| **Size** | Small | Smallest | Large |
| **Use Case** | Streaming | Analytics | APIs |

### When to Use What

**Use Avro:**
- Kafka streaming
- Schema changes frequently
- Need schema validation
- Microservices communication

**Use Parquet:**
- Data warehouse/lake
- Analytics queries
- Read-heavy workloads

**Use JSON:**
- REST APIs
- Human-readable needed
- Simple data exchange

---

## 💻 Exercises (40 min)

Open `exercise.py` and complete the tasks.

### Exercise 1: Basic Avro
- Define schema for employee data
- Write 5 records to Avro file
- Read and print all records

### Exercise 2: Schema Evolution
- Create V1 schema (3 fields)
- Write data with V1
- Create V2 schema (add optional field)
- Read V1 data with V2 schema

### Exercise 3: Avro vs JSON Size
- Write same data to Avro and JSON
- Compare file sizes
- Calculate size difference

### Exercise 4: Kafka Simulation
- Simulate Kafka producer (write to file)
- Simulate Kafka consumer (read from file)
- Use Avro serialization

### Exercise 5: Complex Schema
- Create schema with nested record
- Create schema with array field
- Write and read complex data

---

## ✅ Quiz (5 min)

Answer these questions in `quiz.md`:

1. Is Avro row-based or columnar?
2. Why is Avro used with Kafka?
3. What is Schema Registry?
4. What is schema evolution?
5. Is Avro smaller than JSON?
6. When should you use Avro over Parquet?
7. Can Avro schemas have optional fields?
8. Is Avro human-readable?

---

## 🎯 Key Takeaways

- **Avro is row-based** - good for streaming
- **Kafka standard** - most common format for Kafka
- **Schema Registry** - centralized schema management
- **Schema evolution** - change schemas safely
- **70% smaller than JSON** - saves bandwidth
- **Type safety** - validates data at write time
- **Use for streaming**, Parquet for analytics

---

## 📚 Additional Resources

- [Apache Avro Documentation](https://avro.apache.org/docs/)
- [Confluent Schema Registry](https://docs.confluent.io/platform/current/schema-registry/)
- [Avro vs Parquet](https://www.databricks.com/glossary/what-is-avro)

---

## Tomorrow: Day 5 - Data Serialization Comparison

We'll compare all formats (CSV, JSON, Parquet, Avro, Arrow) and learn when to use each.
