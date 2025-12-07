# Day 34: Kafka Connect - Quiz

Test your understanding of Kafka Connect, connectors, and integration patterns.

---

## Questions

### Question 1
What is Kafka Connect?

A) A separate Kafka cluster  
B) A framework for integrating Kafka with external systems  
C) A monitoring tool  
D) A data format  

### Question 2
What is the difference between source and sink connectors?

A) Source is faster than sink  
B) Source pulls data into Kafka, sink pushes data from Kafka  
C) Sink is for databases only  
D) There is no difference  

### Question 3
What is the recommended mode for production deployments?

A) Standalone mode  
B) Distributed mode  
C) Embedded mode  
D) Cluster mode  

### Question 4
How do you manage connectors in Kafka Connect?

A) Only through configuration files  
B) Through REST API  
C) Through Kafka topics  
D) Through command line only  

### Question 5
What are Single Message Transforms (SMTs)?

A) Data compression algorithms  
B) Built-in transformations applied to messages without custom code  
C) Security protocols  
D) Monitoring metrics  

### Question 6
What does the JDBC source connector's "incrementing" mode do?

A) Increases message size  
B) Tracks new records using an auto-increment column  
C) Speeds up processing  
D) Compresses data  

### Question 7
What is the purpose of converters in Kafka Connect?

A) To convert currencies  
B) To control serialization/deserialization format (JSON, Avro, etc.)  
C) To convert time zones  
D) To compress messages  

### Question 8
What is Debezium?

A) A Kafka broker  
B) A CDC (Change Data Capture) connector for databases  
C) A monitoring tool  
D) A data format  

### Question 9
How does Kafka Connect handle errors?

A) Always crashes  
B) Configurable with error tolerance, logging, and dead letter queues  
C) Ignores all errors  
D) Retries indefinitely  

### Question 10
What is the benefit of using Kafka Connect over custom code?

A) Faster processing  
B) Reusable connectors, fault tolerance, and no code needed for common integrations  
C) Smaller message size  
D) Better compression  

---

## Answer Key

### Question 1: B
**Correct Answer: B) A framework for integrating Kafka with external systems**

Explanation: Kafka Connect is a framework that provides a scalable, fault-tolerant way to integrate Kafka with external systems like databases, file systems, and APIs. It's not a separate cluster but runs as part of your Kafka infrastructure.

### Question 2: B
**Correct Answer: B) Source pulls data into Kafka, sink pushes data from Kafka**

Explanation: Source connectors pull data from external systems into Kafka topics. Sink connectors push data from Kafka topics to external systems. This bidirectional capability enables complete data integration pipelines.

### Question 3: B
**Correct Answer: B) Distributed mode**

Explanation: Distributed mode is recommended for production as it provides fault tolerance, scalability, and automatic rebalancing. Standalone mode is simpler but only suitable for development and testing.

### Question 4: B
**Correct Answer: B) Through REST API**

Explanation: Kafka Connect provides a REST API for managing connectors dynamically. You can create, delete, pause, resume, and monitor connectors through HTTP endpoints without restarting the Connect cluster.

### Question 5: B
**Correct Answer: B) Built-in transformations applied to messages without custom code**

Explanation: Single Message Transforms (SMTs) are lightweight transformations that modify messages in-flight. Common SMTs include InsertField, MaskField, ReplaceField, and Cast, allowing data transformation without writing custom code.

### Question 6: B
**Correct Answer: B) Tracks new records using an auto-increment column**

Explanation: Incrementing mode tracks which records have been processed using an auto-incrementing column (like an ID). It only pulls new records where the ID is greater than the last processed ID, avoiding full table scans.

### Question 7: B
**Correct Answer: B) To control serialization/deserialization format (JSON, Avro, etc.)**

Explanation: Converters control how data is serialized when written to Kafka and deserialized when read from Kafka. Common converters include JsonConverter, AvroConverter, and StringConverter.

### Question 8: B
**Correct Answer: B) A CDC (Change Data Capture) connector for databases**

Explanation: Debezium is a set of CDC connectors that capture row-level changes from databases (inserts, updates, deletes) and stream them to Kafka. It supports PostgreSQL, MySQL, MongoDB, SQL Server, and more.

### Question 9: B
**Correct Answer: B) Configurable with error tolerance, logging, and dead letter queues**

Explanation: Kafka Connect provides configurable error handling including error tolerance levels (none/all), error logging, retry policies, and dead letter queues for failed messages. This prevents connector failures from blocking the entire pipeline.

### Question 10: B
**Correct Answer: B) Reusable connectors, fault tolerance, and no code needed for common integrations**

Explanation: Kafka Connect provides pre-built connectors for common systems, eliminating the need to write custom integration code. It also provides built-in fault tolerance, scalability, exactly-once semantics, and operational features like monitoring and management.

---

## Scoring

- **10/10**: Kafka Connect expert! You understand integration patterns.
- **8-9/10**: Strong knowledge. Review SMTs and error handling.
- **6-7/10**: Good foundation. Practice connector configuration.
- **4-5/10**: Basic understanding. Review source vs sink and modes.
- **0-3/10**: Review the material and practice with real connectors.

---

## Key Concepts to Remember

1. **Kafka Connect** integrates Kafka with external systems
2. **Source connectors** pull data into Kafka
3. **Sink connectors** push data from Kafka
4. **Distributed mode** for production deployments
5. **REST API** for dynamic connector management
6. **SMTs** transform data without custom code
7. **Converters** control serialization format
8. **Incrementing mode** tracks new records efficiently
9. **Debezium** provides CDC for databases
10. **Error handling** with DLQ prevents pipeline failures
