# Day 43: Spark Structured Streaming - Quiz

## Questions

**1. What is the core abstraction in Spark Structured Streaming?**
A) RDD (Resilient Distributed Dataset)
B) Unbounded table that continuously grows
C) Stream buffer with fixed size
D) Event queue with FIFO ordering

**2. What is the default trigger interval in Structured Streaming?**
A) 1 second
B) 500 milliseconds
C) As soon as possible (0 seconds)
D) 10 seconds

**3. Which output mode writes only new rows added to the result table?**
A) Complete mode
B) Update mode
C) Append mode
D) Incremental mode

**4. What information is stored in the checkpoint location?**
A) Only processed offsets
B) Only query state
C) Offsets, metadata, and state information
D) Raw streaming data

**5. Which sink is best for testing and debugging streaming queries?**
A) File sink
B) Kafka sink
C) Console sink
D) Database sink

**6. What happens if the checkpoint directory is deleted while a query is running?**
A) Query continues normally
B) Query restarts from the beginning
C) Query fails immediately
D) Data is permanently lost

**7. Which output mode requires a watermark for aggregation queries?**
A) Append mode
B) Complete mode
C) Update mode
D) All modes require watermark

**8. What is micro-batch processing in Structured Streaming?**
A) Processing one record at a time
B) Processing small batches of data periodically
C) Processing the entire dataset at once
D) Processing data in parallel threads

**9. Which source is recommended for production streaming applications?**
A) Socket source
B) Rate source
C) Kafka source
D) Memory source

**10. How do you stop all active streaming queries in a Spark session?**
A) query.stop()
B) spark.stop()
C) spark.streams.stopAll()
D) spark.streams.stop()

---

## Answers

**1. B - Unbounded table that continuously grows**

Structured Streaming treats streaming data as an unbounded table that continuously grows over time. This abstraction allows you to use familiar DataFrame/SQL operations on streaming data.

**2. C - As soon as possible (0 seconds)**

The default trigger is `processingTime="0 seconds"`, which means the query processes data as soon as the previous micro-batch completes. This provides the lowest latency.

**3. C - Append mode**

Append mode only writes new rows that are added to the result table since the last trigger. It's the default mode and is suitable for queries where rows are never updated.

**4. C - Offsets, metadata, and state information**

The checkpoint location stores:
- Offsets of processed data
- Query metadata and configuration
- State information for stateful operations
This enables fault tolerance and exactly-once processing.

**5. C - Console sink**

The console sink is ideal for testing and debugging because it prints results directly to the console. It's not suitable for production but perfect for development.

**6. B - Query restarts from the beginning**

If the checkpoint directory is deleted, the query loses its progress information and will restart from the beginning (or the specified starting offset), potentially reprocessing data.

**7. A - Append mode**

Append mode requires a watermark when used with aggregation queries to determine when a window is complete and can be output. Complete and update modes don't require watermarks.

**8. B - Processing small batches of data periodically**

Micro-batch processing divides the streaming data into small batches and processes each batch using Spark's batch processing engine. This provides a balance between latency and throughput.

**9. C - Kafka source**

Kafka is the recommended source for production streaming applications because it provides:
- High throughput and scalability
- Fault tolerance and durability
- Exactly-once semantics
- Integration with the streaming ecosystem

**10. C - spark.streams.stopAll()**

`spark.streams.stopAll()` stops all active streaming queries in the current Spark session. Individual queries can be stopped with `query.stop()`.

---

## Scoring

- **10 correct**: Expert - You have mastered Structured Streaming fundamentals
- **8-9 correct**: Proficient - Strong understanding with minor gaps
- **6-7 correct**: Developing - Good foundation, review key concepts
- **Below 6**: Review needed - Revisit the material and practice more

---

## Key Concepts to Remember

1. **Unbounded Table Model**: Streams are treated as continuously growing tables
2. **Output Modes**: Append (new rows), Complete (entire table), Update (changed rows)
3. **Checkpointing**: Essential for fault tolerance and exactly-once semantics
4. **Triggers**: Control when micro-batches are processed
5. **Sources and Sinks**: Choose appropriate ones for your use case
