# Day 45: Watermarking and Late Data - Quiz

## Questions

**1. What is a watermark in Spark Structured Streaming?**
A) The maximum event time seen so far
B) A threshold that defines how late data can arrive before being dropped
C) A marker for processing time
D) The boundary of a window

**2. How is the watermark value calculated?**
A) Current processing time - watermark delay
B) Maximum event time seen - watermark delay
C) Minimum event time + watermark delay
D) Average event time across all events

**3. What happens to events with event_time older than the current watermark?**
A) They are processed normally
B) They are delayed until the next batch
C) They are dropped and ignored
D) They are stored in a separate table

**4. Which output mode requires a watermark for aggregation queries?**
A) Complete mode
B) Update mode
C) Append mode
D) All modes require watermarks

**5. What is the main purpose of watermarks in streaming applications?**
A) To improve query performance
B) To limit state growth and enable state cleanup
C) To enable parallel processing
D) To reduce output latency

**6. Can different streams in a join have different watermark delays?**
A) No, they must have the same delay
B) Yes, each stream can have its own delay
C) Only for inner joins
D) Only for aggregations

**7. What happens if you don't set a watermark on aggregation queries?**
A) The query fails immediately
B) State grows infinitely leading to memory issues
C) Query performance improves
D) Results become incorrect

**8. For stream-to-stream outer joins, watermarks are required on:**
A) The left stream only
B) The right stream only
C) Both streams
D) Neither stream (watermarks are optional)

**9. A longer watermark delay means:**
A) Less state in memory, faster output
B) More state in memory, slower output
C) No impact on state or output
D) Better performance overall

**10. Why should you use event time instead of processing time for business logic?**
A) Event time is faster to compute
B) Event time reflects when events actually occurred, giving accurate results
C) Event time uses less memory
D) Event time is easier to implement

---

## Answers

**1. B - A threshold that defines how late data can arrive before being dropped**

A watermark is a moving threshold that tells Spark how long to wait for late-arriving data. Events with event_time older than the watermark are considered "too late" and are dropped. This prevents infinite state growth.

**2. B - Maximum event time seen - watermark delay**

The watermark is calculated as: `Watermark = Max(event_time) - Watermark_Delay`. For example, if the latest event has timestamp 10:30:00 and the delay is 10 minutes, the watermark is 10:20:00.

**3. C - They are dropped and ignored**

Events with event_time older than the current watermark are considered too late and are automatically dropped by Spark. This is necessary to bound state growth and finalize windows.

**4. C - Append mode**

Append mode requires a watermark for aggregation queries because it needs to know when a window is complete and will never be updated again. Complete and update modes don't require watermarks (though they're recommended).

**5. B - To limit state growth and enable state cleanup**

The primary purpose of watermarks is to bound state growth. Without watermarks, Spark would need to keep state for all windows forever, eventually causing out-of-memory errors. Watermarks tell Spark when it's safe to drop old state.

**6. B - Yes, each stream can have its own delay**

Different streams can have different watermark delays based on their characteristics. For example, one stream might have more latency than another, so it would need a longer watermark delay.

**7. B - State grows infinitely leading to memory issues**

Without a watermark, Spark doesn't know when it's safe to drop state for old windows. It must keep all windows in memory forever, which eventually leads to OutOfMemoryError as state grows without bound.

**8. C - Both streams**

Stream-to-stream outer joins require watermarks on both input streams. The watermarks tell Spark when it's safe to output unmatched records (for left/right outer joins) or when to stop waiting for matches.

**9. B - More state in memory, slower output**

A longer watermark delay means Spark waits longer for late data, which means:
- More windows are kept in memory (more state)
- Windows are finalized later (slower output)
- But fewer events are dropped (more complete results)

**10. B - Event time reflects when events actually occurred, giving accurate results**

Event time represents when events actually happened in the real world, which is what matters for business logic. Processing time only tells you when Spark processed the event, which can vary due to network delays, system load, etc.

---

## Scoring

- **10 correct**: Expert - You have mastered watermarking and late data handling
- **8-9 correct**: Proficient - Strong understanding with minor gaps
- **6-7 correct**: Developing - Good foundation, review key concepts
- **Below 6**: Review needed - Revisit the material and practice more

---

## Key Concepts to Remember

1. **Watermark Formula**: `Max(event_time) - Watermark_Delay`
2. **Purpose**: Limit state growth and enable state cleanup
3. **Late Data**: Events older than watermark are dropped
4. **Append Mode**: Requires watermark for aggregations
5. **Outer Joins**: Require watermarks on both streams
6. **Trade-off**: Longer delay = more complete data but more state
7. **Event Time**: Use for business logic, not processing time
8. **State Management**: Watermarks prevent infinite state growth
