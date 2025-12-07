# Day 48: Streaming Performance Optimization - Quiz

## Questions

**1. What is the trade-off when using smaller trigger intervals?**
A) Higher throughput and lower latency
B) Lower throughput but lower latency
C) Higher throughput and higher latency
D) No trade-off exists

**2. How many partitions should you typically use for optimal performance?**
A) Exactly 1 per executor
B) 2-3 times the number of cores
C) Always 1000 or more
D) As few as possible

**3. What is the primary cause of unbounded state growth?**
A) Too many partitions
B) Missing watermarks or timeouts
C) High event throughput
D) Complex query logic

**4. Where should checkpoints be stored in production environments?**
A) Local filesystem
B) In-memory only
C) HDFS or S3
D) Kafka topics

**5. What is the main advantage of Apache Flink over Spark Structured Streaming?**
A) Easier to learn
B) Lower latency (true streaming)
C) Better ecosystem
D) Unified batch/stream API

**6. When should you choose Spark Streaming over Flink?**
A) Need sub-second latency
B) Already using Spark for batch processing
C) Only simple transformations needed
D) Team has Flink expertise

**7. How should you handle skewed data in streaming?**
A) Increase executor memory
B) Add salt to skewed keys
C) Reduce partition count
D) Disable checkpointing

**8. What is the primary purpose of watermarks in streaming?**
A) Improve query performance
B) Limit state growth and enable cleanup
C) Enable parallel processing
D) Reduce output latency

**9. Which aggregation operation is most expensive in terms of memory?**
A) filter()
B) select()
C) collect_list()
D) count()

**10. How should you monitor streaming query performance?**
A) Spark UI only
B) Application logs only
C) query.lastProgress only
D) All available tools (UI, logs, metrics)

---

## Answers

**1. B - Lower throughput but lower latency**

Smaller trigger intervals mean more frequent micro-batches, which reduces latency (events are processed sooner) but also reduces throughput due to increased overhead from batch coordination and scheduling.

**2. B - 2-3 times the number of cores**

The rule of thumb is to use 2-3x the number of available cores. This provides good parallelism without excessive coordination overhead. For example, with 32 cores, use 64-96 partitions.

**3. B - Missing watermarks or timeouts**

Without watermarks or timeouts, Spark doesn't know when it's safe to drop old state. This causes state to accumulate indefinitely, eventually leading to out-of-memory errors.

**4. C - HDFS or S3**

Production checkpoints should be stored on distributed, fault-tolerant storage like HDFS or S3. Local filesystem is only suitable for testing, and in-memory storage doesn't provide fault tolerance.

**5. B - Lower latency (true streaming)**

Flink's main advantage is lower latency because it processes events one-by-one (true streaming) rather than in micro-batches. This can achieve millisecond latency vs Spark's second-level latency.

**6. B - Already using Spark for batch processing**

Choose Spark Streaming when you already use Spark for batch processing, as it provides a unified API and ecosystem. This reduces operational complexity and leverages existing team knowledge.

**7. B - Add salt to skewed keys**

To handle skewed data, add a random salt to skewed keys to distribute them across multiple partitions. For example, append "_0" through "_9" randomly to spread the load.

**8. B - Limit state growth and enable cleanup**

Watermarks tell Spark when data is "too late" and can be dropped, which limits state growth. They also determine when windows can be finalized and state can be cleaned up.

**9. C - collect_list()**

`collect_list()` is the most expensive because it stores all values in memory for each group. Operations like `count()`, `sum()`, and `avg()` only maintain running totals, using constant memory per group.

**10. D - All available tools (UI, logs, metrics)**

Effective monitoring requires using all available tools: Spark UI for visual insights, application logs for detailed debugging, query.lastProgress for programmatic metrics, and external monitoring systems for alerting.

---

## Scoring

- **10 correct**: Expert - You have mastered streaming performance optimization
- **8-9 correct**: Proficient - Strong understanding with minor gaps
- **6-7 correct**: Developing - Good foundation, review key concepts
- **Below 6**: Review needed - Revisit the material and practice more

---

## Key Concepts to Remember

1. **Trigger Trade-offs**: Smaller intervals = lower latency but lower throughput
2. **Partition Count**: Use 2-3x number of cores for optimal parallelism
3. **State Management**: Always set watermarks and timeouts to prevent unbounded growth
4. **Checkpointing**: Use distributed storage (HDFS/S3) in production
5. **Spark vs Flink**: Spark for unified ecosystem, Flink for ultra-low latency
6. **Skew Handling**: Add salt to distribute skewed keys
7. **Monitoring**: Use all available tools (UI, logs, metrics)
8. **Optimization**: Apply iteratively and measure each improvement
