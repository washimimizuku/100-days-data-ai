# Day 33: Kafka Streams - Quiz

Test your understanding of Kafka Streams, stream processing, and stateful transformations.

---

## Questions

### Question 1
What is Kafka Streams?

A) A separate cluster for stream processing  
B) A client library for building stream processing applications  
C) A database for storing streams  
D) A monitoring tool for Kafka  

### Question 2
What is the difference between KStream and KTable?

A) KStream is faster than KTable  
B) KStream represents events, KTable represents state (latest value per key)  
C) KTable is for batch processing  
D) There is no difference  

### Question 3
Which operation is stateless?

A) Aggregation  
B) Join  
C) Filter  
D) Count  

### Question 4
What is a tumbling window?

A) A window that moves continuously  
B) Fixed-size, non-overlapping time windows  
C) A window based on event count  
D) A window that never closes  

### Question 5
What is the purpose of a state store in Kafka Streams?

A) To store Kafka topics  
B) To provide fault-tolerant local storage for stateful operations  
C) To cache messages  
D) To compress data  

### Question 6
What does flatMap do?

A) Flattens nested data structures  
B) Transforms one input record into zero or more output records  
C) Filters records  
D) Aggregates records  

### Question 7
What is a session window?

A) Fixed-size window  
B) Window based on user login/logout  
C) Dynamic window based on inactivity gaps  
D) Window that processes one event at a time  

### Question 8
How does Kafka Streams achieve fault tolerance?

A) By replicating data to multiple brokers  
B) By using changelog topics to backup state stores  
C) By storing everything in memory  
D) By using a separate database  

### Question 9
What is the benefit of exactly-once semantics in Kafka Streams?

A) Faster processing  
B) Prevents duplicate processing of messages  
C) Reduces memory usage  
D) Simplifies code  

### Question 10
What is Faust?

A) A Kafka broker  
B) A Python stream processing library similar to Kafka Streams  
C) A monitoring tool  
D) A data format  

---

## Answer Key

### Question 1: B
**Correct Answer: B) A client library for building stream processing applications**

Explanation: Kafka Streams is a client library that runs as part of your application, not a separate cluster. It provides APIs for building stream processing applications that read from and write to Kafka topics.

### Question 2: B
**Correct Answer: B) KStream represents events, KTable represents state (latest value per key)**

Explanation: KStream is an unbounded stream of records where each record is an independent event. KTable represents a changelog stream where each record updates the state for a key, keeping only the latest value per key.

### Question 3: C
**Correct Answer: C) Filter**

Explanation: Filter is a stateless operation that selects records based on a predicate without maintaining any state. Aggregation, join, and count are stateful operations that require maintaining state across multiple records.

### Question 4: B
**Correct Answer: B) Fixed-size, non-overlapping time windows**

Explanation: Tumbling windows are fixed-size, non-overlapping time windows. For example, 1-minute tumbling windows would be [0-60s], [60-120s], [120-180s], etc. Each event belongs to exactly one window.

### Question 5: B
**Correct Answer: B) To provide fault-tolerant local storage for stateful operations**

Explanation: State stores provide persistent local storage for stateful operations like aggregations and joins. They are backed by changelog topics in Kafka for fault tolerance, allowing state recovery after failures.

### Question 6: B
**Correct Answer: B) Transforms one input record into zero or more output records**

Explanation: FlatMap transforms each input record into zero, one, or multiple output records. For example, splitting a sentence into words produces multiple output records (one per word) from a single input record.

### Question 7: C
**Correct Answer: C) Dynamic window based on inactivity gaps**

Explanation: Session windows are dynamic windows that group events based on activity. A session closes after a specified inactivity gap (e.g., 5 minutes of no events). They're useful for tracking user sessions or activity periods.

### Question 8: B
**Correct Answer: B) By using changelog topics to backup state stores**

Explanation: Kafka Streams achieves fault tolerance by backing up state stores to changelog topics in Kafka. If an instance fails, another instance can restore the state from the changelog topic and continue processing.

### Question 9: B
**Correct Answer: B) Prevents duplicate processing of messages**

Explanation: Exactly-once semantics ensure that each message is processed exactly once, even in the presence of failures and retries. This prevents duplicate results in aggregations and other stateful operations.

### Question 10: B
**Correct Answer: B) A Python stream processing library similar to Kafka Streams**

Explanation: Faust is a Python library that provides stream processing capabilities similar to Kafka Streams (which is Java-based). It allows building stream processing applications in Python with similar concepts like agents, tables, and windowing.

---

## Scoring

- **10/10**: Kafka Streams expert! You understand stream processing concepts.
- **8-9/10**: Strong knowledge. Review windowing and state stores.
- **6-7/10**: Good foundation. Practice stateful transformations.
- **4-5/10**: Basic understanding. Review KStream vs KTable and operations.
- **0-3/10**: Review the material and practice stream processing patterns.

---

## Key Concepts to Remember

1. **Kafka Streams** is a library, not a separate cluster
2. **KStream** represents events, **KTable** represents state
3. **Stateless** operations (map, filter) don't require state
4. **Stateful** operations (aggregate, join) use state stores
5. **Tumbling windows** are fixed-size, non-overlapping
6. **Session windows** are dynamic, based on inactivity
7. **State stores** provide fault-tolerant local storage
8. **Changelog topics** backup state for recovery
9. **Exactly-once** semantics prevent duplicates
10. **Faust** is the Python alternative to Kafka Streams
