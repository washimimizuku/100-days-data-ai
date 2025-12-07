# Day 31: Kafka Producers - Quiz

Test your understanding of Kafka producers and production best practices.

---

## Questions

### Question 1
What does the `acks` parameter control in a Kafka producer?

A) The number of partitions to create  
B) How many broker acknowledgments are required before considering a send successful  
C) The serialization format  
D) The consumer group size  

### Question 2
What is the safest `acks` setting for durability?

A) acks=0 (no acknowledgment, fastest)  
B) acks=1 (leader acknowledgment only)  
C) acks='all' (all in-sync replicas, safest)  
D) acks='auto' (automatic selection)  

### Question 3
What does `enable_idempotence=True` prevent?

A) Message loss  
B) Duplicate messages from retries  
C) Slow performance  
D) Partition rebalancing  

### Question 4
What is the purpose of `linger_ms` in producer configuration?

A) How long to wait before timing out  
B) How long to wait to batch messages for better throughput  
C) How long to retry failed sends  
D) How long to keep connections open  

### Question 5
What happens when you send a message with a key in Kafka?

A) The message is encrypted  
B) The message goes to a specific partition based on key hash  
C) The message is compressed  
D) The message is sent faster  

### Question 6
What is a dead letter queue (DLQ) used for?

A) Storing all messages permanently  
B) Storing messages that failed processing after retries  
C) Improving performance  
D) Compressing messages  

### Question 7
What does `compression_type='lz4'` do?

A) Encrypts messages  
B) Compresses messages to reduce network bandwidth  
C) Speeds up serialization  
D) Increases durability  

### Question 8
What is the benefit of transactional producers?

A) Faster message sending  
B) Exactly-once semantics across multiple topics  
C) Automatic partitioning  
D) Better compression  

### Question 9
What does `batch_size` control?

A) Number of consumers  
B) Maximum bytes of messages to batch before sending  
C) Number of partitions  
D) Number of retries  

### Question 10
What is the recommended way to handle producer errors in production?

A) Ignore them  
B) Retry with exponential backoff and send to DLQ on final failure  
C) Crash the application  
D) Log and continue without retry  

---

## Answer Key

### Question 1: B
**Correct Answer: B) How many broker acknowledgments are required before considering a send successful**

Explanation: The `acks` parameter determines how many broker acknowledgments the producer requires before considering a message successfully sent. This controls the durability vs. performance tradeoff.

### Question 2: C
**Correct Answer: C) acks='all' (all in-sync replicas, safest)**

Explanation: `acks='all'` waits for all in-sync replicas to acknowledge the message, providing the highest durability guarantee. `acks=0` is fastest but least safe, `acks=1` waits only for the leader.

### Question 3: B
**Correct Answer: B) Duplicate messages from retries**

Explanation: Idempotence ensures that retrying a message send won't create duplicates. The broker tracks producer IDs and sequence numbers to detect and ignore duplicate sends.

### Question 4: B
**Correct Answer: B) How long to wait to batch messages for better throughput**

Explanation: `linger_ms` specifies how long the producer waits before sending a batch, allowing more messages to accumulate for better throughput. Setting it to 0 sends immediately, higher values increase batching.

### Question 5: B
**Correct Answer: B) The message goes to a specific partition based on key hash**

Explanation: Messages with the same key are hashed to the same partition, ensuring ordering for messages with the same key. This is crucial for maintaining order in event streams.

### Question 6: B
**Correct Answer: B) Storing messages that failed processing after retries**

Explanation: A dead letter queue (DLQ) stores messages that failed to send or process after all retry attempts. This prevents data loss and allows manual investigation of failures.

### Question 7: B
**Correct Answer: B) Compresses messages to reduce network bandwidth**

Explanation: Compression reduces the size of messages sent over the network. LZ4 provides fast compression with good ratios. Other options include gzip (better compression, slower) and snappy.

### Question 8: B
**Correct Answer: B) Exactly-once semantics across multiple topics**

Explanation: Transactional producers allow atomic writes across multiple topics/partitions. Either all messages in the transaction are committed or none are, providing exactly-once semantics.

### Question 9: B
**Correct Answer: B) Maximum bytes of messages to batch before sending**

Explanation: `batch_size` sets the maximum bytes of messages to batch together before sending to the broker. Larger batches improve throughput but increase latency.

### Question 10: B
**Correct Answer: B) Retry with exponential backoff and send to DLQ on final failure**

Explanation: Production systems should retry failed sends with exponential backoff to handle transient errors. After max retries, send to a DLQ for manual investigation to prevent data loss.

---

## Scoring

- **10/10**: Kafka producer expert! You understand production best practices.
- **8-9/10**: Strong knowledge. Review idempotence and transactions.
- **6-7/10**: Good foundation. Practice error handling and performance tuning.
- **4-5/10**: Basic understanding. Review producer configurations and patterns.
- **0-3/10**: Review the material and practice with real Kafka implementations.

---

## Key Concepts to Remember

1. **acks='all'** provides strongest durability guarantee
2. **enable_idempotence=True** prevents duplicates from retries
3. **Batching** (batch_size, linger_ms) improves throughput
4. **Compression** (lz4, gzip) reduces network bandwidth
5. **Keys** ensure messages go to same partition for ordering
6. **Transactions** provide exactly-once semantics
7. **DLQ** pattern prevents data loss on failures
8. **Async sends** with callbacks provide high performance
9. **Retries** with exponential backoff handle transient errors
10. **Monitoring** producer metrics is essential for production
