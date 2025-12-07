# Day 29 Quiz: Streaming Concepts

Test your understanding of streaming concepts, batch vs streaming, windowing, and watermarks.

---

## Questions

**1. What is the main difference between batch and stream processing?**
   - A) Batch is faster than streaming
   - B) Batch processes data in chunks at intervals; streaming processes continuously
   - C) Streaming is always cheaper than batch
   - D) Batch can only handle small datasets

**Your answer:** 

---

**2. When should you choose streaming over batch processing?**
   - A) When processing historical data for monthly reports
   - B) When real-time or near-real-time results are required
   - C) When cost optimization is the primary concern
   - D) When data arrives once per day

**Your answer:** 

---

**3. What is "event time" in stream processing?**
   - A) The time when the event is processed by the system
   - B) The time when the event actually occurred in the real world
   - C) The time when the event is written to storage
   - D) The time when the user sees the result

**Your answer:** 

---

**4. What is a tumbling window?**
   - A) A window that overlaps with adjacent windows
   - B) A window based on user activity gaps
   - C) A fixed-size, non-overlapping time window
   - D) A window that adjusts size dynamically

**Your answer:** 

---

**5. What is the purpose of a watermark in stream processing?**
   - A) To encrypt streaming data
   - B) To define how late data can arrive and still be processed
   - C) To compress streaming data
   - D) To partition data across nodes

**Your answer:** 

---

**6. Which architecture combines both batch and streaming layers?**
   - A) Kappa Architecture
   - B) Lambda Architecture
   - C) Delta Architecture
   - D) Sigma Architecture

**Your answer:** 

---

**7. What is backpressure in streaming systems?**
   - A) When data arrives faster than the system can process it
   - B) When the system runs out of memory
   - C) When network latency is high
   - D) When events arrive out of order

**Your answer:** 

---

**8. Which is a common streaming use case?**
   - A) Monthly financial reports
   - B) Annual tax calculations
   - C) Real-time fraud detection
   - D) Quarterly business reviews

**Your answer:** 

---

**9. What is a sliding window?**
   - A) A window that never closes
   - B) A window that moves forward in smaller increments, creating overlaps
   - C) A window that only processes the first event
   - D) A window that processes events in reverse order

**Your answer:** 

---

**10. What is "exactly-once" processing in streaming?**
   - A) Processing each event only one time, avoiding duplicates and data loss
   - B) Processing events as fast as possible
   - C) Processing only the first event in a stream
   - D) Processing events in chronological order

**Your answer:** 

---

## Answers

**1. B** - Batch processes data in chunks at intervals; streaming processes continuously
- Batch waits to collect data and processes it in scheduled intervals
- Streaming processes data continuously as it arrives

**2. B** - When real-time or near-real-time results are required
- Streaming is ideal for time-sensitive use cases like fraud detection, monitoring, alerts
- Batch is better for historical analysis and cost-sensitive workloads

**3. B** - The time when the event actually occurred in the real world
- Event time is when the event happened (e.g., when user clicked)
- Processing time is when the system processed it (can be different due to delays)

**4. C** - A fixed-size, non-overlapping time window
- Tumbling windows divide time into fixed intervals: [0-5min], [5-10min], [10-15min]
- Each event belongs to exactly one window

**5. B** - To define how late data can arrive and still be processed
- Watermarks handle out-of-order events by setting a threshold
- Events arriving after the watermark are considered "too late"

**6. B** - Lambda Architecture
- Lambda combines batch layer (accurate, slow) and speed layer (fast, approximate)
- Kappa is streaming-only (simpler but less flexible)

**7. A** - When data arrives faster than the system can process it
- Backpressure occurs when ingestion rate exceeds processing capacity
- Systems need mechanisms to handle this (buffering, throttling, dropping)

**8. C** - Real-time fraud detection
- Fraud detection requires immediate analysis of transactions
- Monthly reports, annual calculations, and quarterly reviews are batch use cases

**9. B** - A window that moves forward in smaller increments, creating overlaps
- Sliding windows overlap: [0-5min], [1-6min], [2-7min]
- Useful for moving averages and continuous monitoring

**10. A** - Processing each event only one time, avoiding duplicates and data loss
- Exactly-once ensures no duplicates (at-least-once) and no data loss (at-most-once)
- Critical for financial transactions and accurate counting

---

## Scoring

- **10 correct**: Excellent! You understand streaming concepts thoroughly
- **8-9 correct**: Great job! Minor review needed on specific topics
- **6-7 correct**: Good foundation, review windowing and watermarks
- **4-5 correct**: Review the README and focus on key concepts
- **0-3 correct**: Revisit the theory section and try exercises again

---

## Next Steps

1. If you scored 8+: Move to Day 30 - Kafka Fundamentals
2. If you scored 6-7: Review windowing and watermark sections
3. If you scored below 6: Complete the exercises and review theory
4. Practice: Try implementing a simple stream processor with windowing

---

**Tomorrow**: Day 30 - Kafka Fundamentals - Learn the most popular streaming platform!
