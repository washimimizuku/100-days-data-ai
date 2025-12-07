# Day 44: Streaming Joins and Aggregations - Quiz

## Questions

**1. What is required for stream-to-stream outer joins in Spark Structured Streaming?**
A) Broadcast hint on both streams
B) Watermark defined on both streams
C) Checkpoint location specified
D) Fixed trigger interval

**2. Which join type is NOT supported for stream-to-static joins?**
A) Inner join
B) Left outer join
C) Right outer join
D) Full outer join

**3. What is the primary purpose of time constraints in stream-to-stream joins?**
A) Improve query performance
B) Limit state growth and memory usage
C) Enable parallel processing
D) Reduce output latency

**4. In a tumbling window of 10 minutes, how many windows does each event belong to?**
A) 0 windows
B) Exactly 1 window
C) 2 windows
D) Depends on the event timestamp

**5. What is the key difference between tumbling and sliding windows?**
A) Window size
B) Windows can overlap in sliding windows
C) Performance characteristics
D) Output mode requirements

**6. Which function is used to create session windows in Spark Structured Streaming?**
A) window()
B) session_window()
C) sliding_window()
D) tumbling_window()

**7. What should you broadcast in stream-to-static joins for optimal performance?**
A) The streaming DataFrame
B) The static DataFrame (if small enough)
C) Both DataFrames
D) Neither DataFrame

**8. How do you perform multiple aggregations on the same group?**
A) Call groupBy multiple times
B) Chain multiple agg() calls
C) Use single agg() with multiple aggregation functions
D) Create separate streaming queries

**9. What happens without a watermark in stream-to-stream joins?**
A) The join operation fails immediately
B) State grows indefinitely causing memory issues
C) Query performance improves
D) Output becomes incorrect

**10. Which aggregation function is typically most expensive in terms of memory?**
A) count()
B) sum()
C) collect_list()
D) avg()

---

## Answers

**1. B - Watermark defined on both streams**

Stream-to-stream outer joins require watermarks on both input streams to determine when to drop old state. Without watermarks, Spark cannot know when it's safe to remove state for events that will never match.

**2. D - Full outer join**

Full outer joins are NOT supported for stream-to-static joins. Only inner, left outer, and right outer joins are supported. Full outer joins are only available for stream-to-stream joins with watermarks.

**3. B - Limit state growth and memory usage**

Time constraints in stream-to-stream joins limit how long Spark needs to keep state for matching events. Without time bounds, state would grow indefinitely as Spark would need to remember all events forever.

**4. B - Exactly 1 window**

In tumbling windows, each event belongs to exactly one window. Tumbling windows are non-overlapping, fixed-size time intervals. For example, with 10-minute tumbling windows, an event at 10:05 belongs only to the 10:00-10:10 window.

**5. B - Windows can overlap in sliding windows**

The key difference is that sliding windows can overlap. For example, a 10-minute window with 5-minute slide means each event appears in 2 windows. Tumbling windows never overlap - they are consecutive, non-overlapping intervals.

**6. B - session_window()**

The `session_window()` function creates session windows based on inactivity gaps. For example, `session_window("timestamp", "5 minutes")` creates windows that close after 5 minutes of inactivity.

**7. B - The static DataFrame (if small enough)**

You should broadcast the static DataFrame in stream-to-static joins if it's small enough to fit in memory on each executor. This avoids shuffling the streaming data and significantly improves performance. Use `broadcast(static_df)`.

**8. C - Use single agg() with multiple aggregation functions**

The correct approach is to use a single `agg()` call with multiple aggregation functions passed as arguments. For example: `.agg(count("*"), sum("amount"), avg("amount"))`. This is more efficient than multiple groupBy operations.

**9. B - State grows indefinitely causing memory issues**

Without watermarks, Spark doesn't know when it's safe to drop old state from stream-to-stream joins. This causes state to grow indefinitely, eventually leading to out-of-memory errors. Watermarks tell Spark when data is "too late" and can be discarded.

**10. C - collect_list()**

`collect_list()` is the most expensive because it stores all values in memory for each group. Functions like `count()`, `sum()`, and `avg()` only need to maintain a running total, while `collect_list()` must keep every individual value.

---

## Scoring

- **10 correct**: Expert - You have mastered streaming joins and aggregations
- **8-9 correct**: Proficient - Strong understanding with minor gaps
- **6-7 correct**: Developing - Good foundation, review key concepts
- **Below 6**: Review needed - Revisit the material and practice more

---

## Key Concepts to Remember

1. **Stream-to-Static Joins**: Use broadcast for small static tables, no watermark needed
2. **Stream-to-Stream Joins**: Require time constraints and watermarks to limit state
3. **Tumbling Windows**: Non-overlapping, each event in exactly one window
4. **Sliding Windows**: Overlapping windows, events can appear in multiple windows
5. **Session Windows**: Dynamic windows based on inactivity gaps
6. **Watermarks**: Essential for managing state in stream-to-stream operations
7. **Multiple Aggregations**: Use single agg() call with multiple functions
8. **Performance**: Broadcast small tables, use appropriate window sizes, always set watermarks
