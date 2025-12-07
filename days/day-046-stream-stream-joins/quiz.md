# Day 46: Stream-to-Stream Joins - Quiz

## Questions

**1. What is required for stream-to-stream outer joins?**
A) Checkpoint location specified
B) Watermarks defined on both streams
C) Broadcast hint on one stream
D) Fixed trigger interval

**2. Why are time constraints necessary in stream-to-stream joins?**
A) To improve query performance
B) To limit state growth and prevent memory issues
C) To enable parallel processing
D) To reduce output latency

**3. In a left outer join, what happens when the right side doesn't match?**
A) The row is dropped completely
B) Right-side columns are filled with null values
C) The join operation fails
D) The row is delayed until a match arrives

**4. How is the join watermark calculated when streams have different watermark delays?**
A) Maximum of both watermarks
B) Minimum of both watermarks
C) Average of both watermarks
D) Sum of both watermarks

**5. Which join type does NOT require watermarks (though they're recommended)?**
A) Inner join
B) Left outer join
C) Right outer join
D) Full outer join

**6. What causes excessive state growth in stream-to-stream joins?**
A) Too many partitions
B) Missing watermarks or time constraints
C) High event throughput
D) Complex join conditions

**7. In a self-join to detect duplicates, how do you avoid matching an event with itself?**
A) Use different timestamp columns
B) Add condition: e1.event_id != e2.event_id
C) Use a watermark
D) Use different join keys

**8. What's the best way to handle unmatched records in production?**
A) Drop them silently
B) Retry indefinitely
C) Separate processing or output stream
D) Ignore them completely

**9. Using narrow time windows in joins results in:**
A) More state, slower processing
B) Less state, faster processing
C) No impact on performance
D) More matches found

**10. When debugging missing matches in joins, what should you check first?**
A) Memory usage and CPU
B) Watermarks and time constraints
C) Number of partitions
D) Trigger interval setting

---

## Answers

**1. B - Watermarks defined on both streams**

Stream-to-stream outer joins (left, right, full) require watermarks on both input streams. The watermarks tell Spark when it's safe to output unmatched records or stop waiting for potential matches.

**2. B - To limit state growth and prevent memory issues**

Without time constraints, Spark would need to buffer all events from both streams indefinitely, waiting for potential matches. Time constraints limit how long events are kept in state, preventing unbounded memory growth.

**3. B - Right-side columns are filled with null values**

In a left outer join, all rows from the left stream are output. When there's no matching row from the right stream, the right-side columns are filled with null values, but the left-side data is preserved.

**4. B - Minimum of both watermarks**

The effective watermark for a join is the minimum of the watermarks from both streams. For example, if stream1 has a 10-minute watermark and stream2 has a 5-minute watermark, the join uses a 5-minute watermark.

**5. A - Inner join**

Inner joins don't strictly require watermarks (though they're highly recommended). Outer joins (left, right, full) require watermarks on both streams to determine when to output unmatched records.

**6. B - Missing watermarks or time constraints**

Without watermarks and time constraints, Spark must keep all events in state forever, waiting for potential matches. This causes state to grow without bound, eventually leading to out-of-memory errors.

**7. B - Add condition: e1.event_id != e2.event_id**

In a self-join, you need to explicitly exclude self-matches by adding a condition that the event IDs (or unique identifiers) are different. Without this, every event would match with itself.

**8. C - Separate processing or output stream**

In production, unmatched records should be handled explicitly through separate processing logic or written to a different output stream (e.g., a separate Kafka topic or table) for monitoring, alerting, or retry logic.

**9. B - Less state, faster processing**

Narrow time windows mean Spark keeps events in state for a shorter period, resulting in less memory usage and faster state cleanup. However, this may result in fewer matches if events arrive late.

**10. B - Watermarks and time constraints**

When debugging missing matches, first verify that watermarks are set appropriately and time constraints are wide enough to accommodate actual data latency. Overly aggressive watermarks or narrow time windows are common causes of missing matches.

---

## Scoring

- **10 correct**: Expert - You have mastered stream-to-stream joins
- **8-9 correct**: Proficient - Strong understanding with minor gaps
- **6-7 correct**: Developing - Good foundation, review key concepts
- **Below 6**: Review needed - Revisit the material and practice more

---

## Key Concepts to Remember

1. **Outer Joins**: Require watermarks on both streams
2. **Time Constraints**: Essential to limit state growth
3. **Join Watermark**: Minimum of both stream watermarks
4. **State Management**: Monitor and optimize to prevent memory issues
5. **Unmatched Records**: Handle explicitly in production
6. **Self-Joins**: Exclude self-matches with ID comparison
7. **Debugging**: Check watermarks and time constraints first
8. **Performance**: Narrow windows = less state = faster processing
