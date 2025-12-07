# Day 30 Quiz: Kafka Fundamentals

Test your understanding of Apache Kafka architecture, topics, partitions, and consumer groups.

---

## Questions

**1. What is Apache Kafka primarily used for?**
   - A) Relational database management
   - B) Distributed streaming and event processing
   - C) Static file storage
   - D) Web server hosting

**Your answer:** 

---

**2. What is a Kafka topic?**
   - A) A physical server in the cluster
   - B) A category or feed name to which records are published
   - C) A consumer application
   - D) A database table

**Your answer:** 

---

**3. Why does Kafka divide topics into partitions?**
   - A) To save disk space
   - B) To enable parallelism, scalability, and ordering
   - C) To encrypt data
   - D) To compress messages

**Your answer:** 

---

**4. What is a Kafka broker?**
   - A) A message consumer
   - B) A server that stores data and serves clients
   - C) A partition key
   - D) A producer application

**Your answer:** 

---

**5. How does Kafka determine which partition a message goes to?**
   - A) Random selection
   - B) Round-robin across all partitions
   - C) Hash of the message key modulo number of partitions
   - D) Alphabetical order of topic name

**Your answer:** 

---

**6. What is a consumer group in Kafka?**
   - A) A set of topics
   - B) A group of brokers
   - C) A set of consumers that coordinate to consume a topic with load balancing
   - D) A partition replication strategy

**Your answer:** 

---

**7. Can multiple consumer groups read from the same Kafka topic?**
   - A) No, only one consumer group per topic
   - B) Yes, each group maintains independent offsets
   - C) Only if the topic has multiple partitions
   - D) Only with special permissions

**Your answer:** 

---

**8. What is an offset in Kafka?**
   - A) The time delay between producer and consumer
   - B) The position of a message within a partition
   - C) The number of replicas for a partition
   - D) The size of a message in bytes

**Your answer:** 

---

**9. What ordering guarantee does Kafka provide?**
   - A) Global ordering across all partitions
   - B) Ordering within a partition only
   - C) No ordering guarantees
   - D) Ordering only for the first 1000 messages

**Your answer:** 

---

**10. What happens if there are more consumers in a group than partitions?**
   - A) Kafka creates more partitions automatically
   - B) Some consumers will be idle with no partitions assigned
   - C) The topic is deleted
   - D) All consumers share all partitions

**Your answer:** 

---

## Answers

**1. B** - Distributed streaming and event processing
- Kafka is designed for high-throughput, fault-tolerant event streaming
- Used for real-time data pipelines, stream processing, and event-driven architectures

**2. B** - A category or feed name to which records are published
- Topics are logical categories like "orders", "user-events", "logs"
- Producers write to topics; consumers read from topics

**3. B** - To enable parallelism, scalability, and ordering
- Partitions allow multiple consumers to read in parallel
- Distribute load across brokers for scalability
- Maintain ordering within each partition

**4. B** - A server that stores data and serves clients
- Brokers are Kafka servers that form a cluster
- Each broker stores partitions and handles read/write requests

**5. C** - Hash of the message key modulo number of partitions
- Formula: `partition = hash(key) % num_partitions`
- Messages with same key always go to same partition (ordering!)
- If no key provided, round-robin or random assignment

**6. C** - A set of consumers that coordinate to consume a topic with load balancing
- Consumer groups enable parallel consumption
- Each partition assigned to one consumer in the group
- Automatic rebalancing when consumers join/leave

**7. B** - Yes, each group maintains independent offsets
- Multiple groups can read same topic independently
- Each group tracks its own progress (offsets)
- Common pattern: one group for processing, another for analytics

**8. B** - The position of a message within a partition
- Offsets are sequential integers: 0, 1, 2, 3...
- Consumers track which offset they've read up to
- Offsets enable replay and fault tolerance

**9. B** - Ordering within a partition only
- Messages in same partition are strictly ordered
- No ordering guarantee across different partitions
- Use partition keys to ensure related messages stay ordered

**10. B** - Some consumers will be idle with no partitions assigned
- Max useful consumers = number of partitions
- Extra consumers sit idle as backups
- If a consumer fails, idle consumer takes over its partitions

---

## Scoring

- **10 correct**: Excellent! You understand Kafka architecture thoroughly
- **8-9 correct**: Great job! Minor review on specific concepts
- **6-7 correct**: Good foundation, review partitions and consumer groups
- **4-5 correct**: Review the README and focus on core concepts
- **0-3 correct**: Revisit theory section and try exercises

---

## Key Concepts to Remember

1. **Topics** = logical categories, **Partitions** = physical divisions
2. **Brokers** = servers, **Replication** = fault tolerance
3. **Producers** write, **Consumers** read
4. **Consumer groups** = parallel consumption with load balancing
5. **Offsets** = position tracking for consumers
6. **Ordering** = guaranteed per partition, not globally
7. **Partition key** = determines which partition (same key → same partition)

---

## Next Steps

1. If you scored 8+: Move to Day 31 - Kafka Producers & Consumers
2. If you scored 6-7: Review partition and consumer group sections
3. If you scored below 6: Complete exercises and review theory
4. Practice: Draw a Kafka architecture diagram for a use case

---

**Tomorrow**: Day 31 - Kafka Producers & Consumers - Hands-on with Python Kafka client!
