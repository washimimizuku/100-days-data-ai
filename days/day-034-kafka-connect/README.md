# Day 34: Kafka Connect

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of this session, you will:
- Understand Kafka Connect architecture
- Configure source connectors to pull data into Kafka
- Configure sink connectors to push data from Kafka
- Use built-in connectors (File, JDBC, Elasticsearch)
- Understand connector configuration and management
- Implement custom connectors
- Handle data transformations with SMTs
- Monitor connector health and performance

---

## Why Kafka Connect?

Kafka Connect is a **framework for integrating Kafka with external systems**:

- **Scalable**: Distributed, fault-tolerant architecture
- **Reusable**: Pre-built connectors for common systems
- **Declarative**: JSON configuration, no code needed
- **Fault-tolerant**: Automatic recovery and retries
- **Exactly-once**: Supported for compatible connectors
- **Transformations**: Built-in data transformations (SMTs)

**Use Cases**:
- Stream database changes to Kafka (CDC)
- Load data from files, APIs, message queues
- Sink data to databases, data warehouses, search engines
- Real-time data integration pipelines
- Event-driven architectures

---

## Kafka Connect Architecture

```
Source System → Source Connector → Kafka Topics
Kafka Topics → Sink Connector → Target System
```

**Components**:
- **Workers**: Run connectors (standalone or distributed)
- **Connectors**: High-level abstraction (source or sink)
- **Tasks**: Parallel units of work
- **Converters**: Serialize/deserialize data (JSON, Avro, etc.)
- **Transforms**: Modify data in-flight (SMTs)

### Standalone vs Distributed Mode

**Standalone**:
- Single process
- Good for development/testing
- Configuration in properties file

**Distributed** (Production):
- Multiple workers
- Fault-tolerant and scalable
- Configuration via REST API
- Automatic rebalancing

---

## Source Connectors

Pull data **into** Kafka from external systems.

### Common Source Connectors

```json
// File Source
{"name": "file-source", "config": {
  "connector.class": "FileStreamSource",
  "file": "/tmp/input.txt", "topic": "file-topic"
}}

// JDBC Source (incrementing mode)
{"name": "jdbc-source", "config": {
  "connector.class": "io.confluent.connect.jdbc.JdbcSourceConnector",
  "connection.url": "jdbc:postgresql://localhost:5432/mydb",
  "table.whitelist": "users,orders",
  "mode": "incrementing",
  "incrementing.column.name": "id"
}}

// Debezium CDC
{"name": "debezium-postgres", "config": {
  "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
  "database.hostname": "localhost",
  "database.dbname": "mydb",
  "table.include.list": "public.users"
}}
```

---

## Sink Connectors

Push data **from** Kafka to external systems.

### Common Sink Connectors

```json
// File Sink
{"name": "file-sink", "config": {
  "connector.class": "FileStreamSink",
  "file": "/tmp/output.txt", "topics": "my-topic"
}}

// JDBC Sink (upsert mode)
{"name": "jdbc-sink", "config": {
  "connector.class": "io.confluent.connect.jdbc.JdbcSinkConnector",
  "connection.url": "jdbc:postgresql://localhost:5432/targetdb",
  "topics": "orders",
  "insert.mode": "upsert",
  "pk.fields": "id"
}}

// Elasticsearch Sink
{"name": "elasticsearch-sink", "config": {
  "connector.class": "io.confluent.connect.elasticsearch.ElasticsearchSinkConnector",
  "topics": "logs",
  "connection.url": "http://localhost:9200"
}}
```

---

## Connector Management

### REST API & Python Management

```bash
# List/Status/Create/Delete connectors
curl http://localhost:8083/connectors
curl http://localhost:8083/connectors/my-connector/status
curl -X POST http://localhost:8083/connectors -d @config.json
curl -X DELETE http://localhost:8083/connectors/my-connector
```

```python
import requests

CONNECT_URL = "http://localhost:8083"

def create_connector(name, config):
    return requests.post(f"{CONNECT_URL}/connectors",
                        json={"name": name, "config": config}).json()

def get_status(name):
    return requests.get(f"{CONNECT_URL}/connectors/{name}/status").json()
```

---

## Single Message Transforms (SMTs)

Modify data in-flight without custom code.

### Common Transforms

```json
{
  "transforms": "InsertField,MaskField",
  "transforms.InsertField.type": "org.apache.kafka.connect.transforms.InsertField$Value",
  "transforms.InsertField.timestamp.field": "processed_at",
  "transforms.MaskField.type": "org.apache.kafka.connect.transforms.MaskField$Value",
  "transforms.MaskField.fields": "ssn,credit_card"
}
```

**Available SMTs**:
- `InsertField`: Add fields (timestamp, topic, partition)
- `ReplaceField`: Rename or exclude fields
- `MaskField`: Mask sensitive data
- `ValueToKey`: Copy value fields to key
- `ExtractField`: Extract nested field
- `Cast`: Change field types
- `TimestampConverter`: Convert timestamp formats
- `Filter`: Drop records based on predicate

---

## Converters

Control serialization format.

```json
{
  "key.converter": "org.apache.kafka.connect.json.JsonConverter",
  "value.converter": "org.apache.kafka.connect.json.JsonConverter",
  "key.converter.schemas.enable": "false",
  "value.converter.schemas.enable": "false"
}
```

**Common Converters**:
- `JsonConverter`: JSON with optional schema
- `AvroConverter`: Avro with Schema Registry
- `StringConverter`: Plain strings
- `ByteArrayConverter`: Raw bytes

---

## Custom Connectors

### Source Connector Example

```python
from kafka import KafkaProducer
import json
import time

class CustomSourceConnector:
    def __init__(self, config):
        self.producer = KafkaProducer(
            bootstrap_servers=config['bootstrap.servers'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.topic = config['topic']
        self.running = True
    
    def poll(self):
        """Pull data from source system"""
        while self.running:
            # Fetch data from external system
            data = self.fetch_from_source()
            
            # Send to Kafka
            for record in data:
                self.producer.send(self.topic, record)
            
            time.sleep(1)
    
    def fetch_from_source(self):
        # Implement source-specific logic
        return [{'id': 1, 'data': 'example'}]
    
    def stop(self):
        self.running = False
        self.producer.close()
```

### Sink Connector Example

```python
from kafka import KafkaConsumer
import json

class CustomSinkConnector:
    def __init__(self, config):
        self.consumer = KafkaConsumer(
            config['topics'],
            bootstrap_servers=config['bootstrap.servers'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        self.running = True
    
    def run(self):
        """Push data to target system"""
        for message in self.consumer:
            if not self.running:
                break
            
            # Write to external system
            self.write_to_target(message.value)
    
    def write_to_target(self, record):
        # Implement target-specific logic
        print(f"Writing to target: {record}")
    
    def stop(self):
        self.running = False
        self.consumer.close()
```

---

## Production Best Practices

**Configuration**:
- Use distributed mode for production
- Set `tasks.max` based on parallelism needs
- Configure error handling and retries
- Enable monitoring and logging

**Error Handling**:
- Set `errors.tolerance` to `all` or `none`
- Configure `errors.deadletterqueue.topic.name`
- Set `errors.retry.timeout` and `errors.retry.delay.max.ms`

**Performance**:
- Tune `batch.size` and `linger.ms`
- Adjust `max.poll.records` and `max.poll.interval.ms`
- Use appropriate number of tasks
- Monitor lag and throughput

**Security**:
- Use SSL/TLS for connections
- Implement authentication (SASL)
- Encrypt sensitive configuration
- Use secrets management

---

## 💻 Exercises (40 min)

### Exercise 1: File Source Connector
Configure file source connector to read from file.

### Exercise 2: File Sink Connector
Configure file sink connector to write to file.

### Exercise 3: JDBC Source Connector
Pull data from PostgreSQL into Kafka.

### Exercise 4: JDBC Sink Connector
Push data from Kafka to PostgreSQL.

### Exercise 5: Connector Management
Use REST API to manage connectors.

### Exercise 6: Single Message Transforms
Apply SMTs to add timestamps and mask fields.

### Exercise 7: Custom Source Connector
Implement custom source connector in Python.

### Exercise 8: Monitoring and Error Handling
Monitor connector health and handle errors.

---

## ✅ Quiz (5 min)

Test your understanding in `quiz.md`.

---

## 🎯 Key Takeaways

- **Kafka Connect** integrates Kafka with external systems
- **Source connectors** pull data into Kafka
- **Sink connectors** push data from Kafka
- **Distributed mode** provides fault tolerance and scalability
- **REST API** manages connectors dynamically
- **SMTs** transform data without custom code
- **Converters** control serialization format
- **Custom connectors** extend functionality
- **Error handling** with DLQ and retries
- **Monitoring** is essential for production

---

## 📚 Resources

- [Kafka Connect Documentation](https://kafka.apache.org/documentation/#connect)
- [Confluent Hub](https://www.confluent.io/hub/) - Connector repository
- [Debezium](https://debezium.io/) - CDC connectors
- [Kafka Connect REST API](https://docs.confluent.io/platform/current/connect/references/restapi.html)

---

## Tomorrow: Day 35 - Mini Project: Real-time Kafka Pipeline

Build an end-to-end real-time data pipeline using Kafka producers, consumers, Kafka Streams, and Kafka Connect.
