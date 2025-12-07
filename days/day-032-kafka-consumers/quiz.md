# Day 32: Kafka Consumers & Consumer Groups - Quiz

Test your understanding of Kafka consumers, consumer groups, and offset management.

---

## Questions

### Question 1
What is the purpose of a consumer group in Kafka?

A) To encrypt messages  
B) To enable parallel processing and fault tolerance  
C) To compress messages  
D) To create topics  

### Question 2
How many consumers in a group can read from the same partition?

A) Unlimited  
B) Exactly one  
C) Two for redundancy  
D) Depends on configuration  

### Question 3
What happens when `enable_auto_commit=False`?

A) Consumer won't read messages  
B) Offsets must be committed manually by the application  
C) Consumer runs faster  
D) Messages are deleted automatically  

### Question 4
What is consumer lag?

A) Network latency  
B) Difference between latest offset and consumer's current position  
C) Time to process a message  
D) Number of partitions  

### Question 5
What does `auto_offset_reset='earliest'` do?

A) Deletes old messages  
B) Starts reading from the beginning if no offset exists  
C) Commits offsets automatically  
D) Speeds up consumption  

### Question 6
When does consumer rebalancing occur?

A) Every hour  
B) When consumers join or leave the group  
C) When messages arrive  
D) Never  

### Question 7
What is the benefit of batch processing in consumers?

A) Better throughput by processing multiple messages at once  
B) Faster network speed  
C) Automatic error handling  
D) Smaller message size  

### Question 8
What does `max_poll_interval_ms` control?

A) How often to commit offsets  
B) Maximum time between poll() calls before consumer is considered dead  
C) Network timeout  
D) Message size limit  

### Question 9
What is a dead letter queue (DLQ) used for?

A) Storing all messages  
B) Storing messages that failed processing after retries  
C) Improving performance  
D) Compressing messages  

### Question 10
How do you achieve exactly-once semantics in a consumer?

A) Use auto-commit  
B) Make processing idempotent and use manual commits  
C) Increase batch size  
D) Use multiple consumer groups  

---

## Answer Key

### Question 1: B
**Correct Answer: B) To enable parallel processing and fault tolerance**

Explanation: Consumer groups allow multiple consumers to work together to process messages from a topic in parallel. Each partition is assigned to one consumer in the group, enabling horizontal scaling. If a consumer fails, its partitions are reassigned to other consumers in the group.

### Question 2: B
**Correct Answer: B) Exactly one**

Explanation: In a consumer group, each partition can only be consumed by one consumer at a time. This ensures message ordering within a partition. If you have more consumers than partitions, some consumers will be idle.

### Question 3: B
**Correct Answer: B) Offsets must be committed manually by the application**

Explanation: When `enable_auto_commit=False`, the application is responsible for committing offsets. This provides control over when offsets are committed, typically after successful message processing, enabling at-least-once delivery semantics.

### Question 4: B
**Correct Answer: B) Difference between latest offset and consumer's current position**

Explanation: Consumer lag is the number of messages the consumer is behind the latest message in the partition. It's calculated as (latest offset - current consumer position). High lag indicates the consumer can't keep up with the message rate.

### Question 5: B
**Correct Answer: B) Starts reading from the beginning if no offset exists**

Explanation: `auto_offset_reset='earliest'` tells the consumer to start reading from the beginning of the partition if no committed offset exists for the consumer group. The alternative is 'latest', which starts from the end.

### Question 6: B
**Correct Answer: B) When consumers join or leave the group**

Explanation: Rebalancing occurs when the consumer group membership changes (consumers join or leave) or when topic partitions change. During rebalancing, partitions are redistributed among available consumers, and no messages are consumed.

### Question 7: A
**Correct Answer: A) Better throughput by processing multiple messages at once**

Explanation: Batch processing improves throughput by processing multiple messages together and committing offsets once per batch instead of per message. This reduces the overhead of individual message processing and commits.

### Question 8: B
**Correct Answer: B) Maximum time between poll() calls before consumer is considered dead**

Explanation: `max_poll_interval_ms` sets the maximum time allowed between calls to poll(). If this time is exceeded, the consumer is considered dead and will be removed from the group, triggering a rebalance. Set this higher for slow processing.

### Question 9: B
**Correct Answer: B) Storing messages that failed processing after retries**

Explanation: A dead letter queue (DLQ) stores messages that failed processing after all retry attempts. This prevents blocking the consumer on bad messages while preserving them for later investigation and manual handling.

### Question 10: B
**Correct Answer: B) Make processing idempotent and use manual commits**

Explanation: Exactly-once semantics require idempotent processing (processing the same message multiple times has the same effect as once) combined with manual offset commits after successful processing. This ensures messages aren't lost or duplicated even with failures.

---

## Scoring

- **10/10**: Kafka consumer expert! You understand consumer groups and offset management.
- **8-9/10**: Strong knowledge. Review rebalancing and lag monitoring.
- **6-7/10**: Good foundation. Practice offset management and error handling.
- **4-5/10**: Basic understanding. Review consumer groups and delivery semantics.
- **0-3/10**: Review the material and practice with real Kafka consumers.

---

## Key Concepts to Remember

1. **Consumer groups** enable parallel processing with one partition per consumer
2. **Manual commits** provide at-least-once delivery guarantee
3. **Rebalancing** redistributes partitions when consumers change
4. **Consumer lag** indicates how far behind the consumer is
5. **auto_offset_reset** determines where to start reading
6. **Batch processing** improves throughput
7. **max_poll_interval_ms** prevents false dead consumer detection
8. **DLQ** pattern handles failed messages gracefully
9. **Idempotent processing** enables exactly-once semantics
10. **Monitoring lag** is critical for production systems
