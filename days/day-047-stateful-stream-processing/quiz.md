# Day 47: Stateful Stream Processing - Quiz

## Questions

**1. What is stateful processing in streaming?**
A) Processing without any memory of previous events
B) Maintaining state information across multiple events
C) Processing events in parallel
D) Batch processing mode

**2. Which function produces exactly one output per group?**
A) flatMapGroupsWithState
B) mapGroupsWithState
C) groupBy
D) window

**3. What does GroupStateTimeout.NoTimeout mean?**
A) State expires immediately after processing
B) State never expires automatically
C) State expires after 1 hour
D) State expires based on watermark

**4. When should you use flatMapGroupsWithState instead of mapGroupsWithState?**
A) When you need exactly one output per group
B) When you need zero or multiple outputs per group
C) When you don't need state
D) For stateless operations

**5. How do you check if state has timed out in the update function?**
A) state.isExpired
B) state.hasTimedOut
C) state.timeout()
D) state.checkTimeout()

**6. What happens if you don't set timeouts on stateful operations?**
A) The query fails immediately
B) State grows unbounded causing memory issues
C) Query performance improves
D) State is automatically cleaned

**7. ProcessingTimeTimeout is based on:**
A) Event time from the data
B) Wall clock time (system time)
C) Watermark value
D) Batch processing time

**8. To limit state size in production, you should:**
A) Allocate more memory
B) Keep only recent data and set timeouts
C) Disable checkpointing
D) Increase partition count

**9. Where is state stored in Spark Structured Streaming?**
A) Only in memory
B) Only on disk
C) In checkpoint location with memory cache
D) In Kafka topics

**10. What output modes can mapGroupsWithState use?**
A) Append only
B) Complete only
C) Update only
D) Update or append

---

## Answers

**1. B - Maintaining state information across multiple events**

Stateful processing maintains information (state) across multiple events to compute results that depend on historical data. For example, calculating running totals, detecting patterns, or tracking sessions.

**2. B - mapGroupsWithState**

`mapGroupsWithState` produces exactly one output value per group in each micro-batch. If you need zero or multiple outputs per group, use `flatMapGroupsWithState` instead.

**3. B - State never expires automatically**

`GroupStateTimeout.NoTimeout` means the state will never expire automatically. You must manually remove state with `state.remove()` or it will persist indefinitely, which can lead to unbounded state growth.

**4. B - When you need zero or multiple outputs per group**

Use `flatMapGroupsWithState` when you need to produce zero, one, or multiple output records per group. For example, detecting patterns might produce no output until a pattern matches, then produce one or more results.

**5. B - state.hasTimedOut**

Check `state.hasTimedOut` to determine if the state has timed out. This is typically checked at the beginning of your update function to handle timeout logic separately from normal event processing.

**6. B - State grows unbounded causing memory issues**

Without timeouts, Spark keeps state for all keys forever, even if they're no longer active. This causes unbounded state growth and eventually leads to out-of-memory errors in production.

**7. B - Wall clock time (system time)**

`ProcessingTimeTimeout` is based on wall clock time (system time), not event time. The timeout is measured from when the state was last updated, using the processing time of the system.

**8. B - Keep only recent data and set timeouts**

To limit state size, keep only recent data (e.g., last N events or events within a time window) and always set appropriate timeouts. This prevents unbounded state growth while maintaining necessary historical information.

**9. C - In checkpoint location with memory cache**

State is stored in the checkpoint location for fault tolerance and recovery, with an in-memory cache for fast access. This provides both durability and performance.

**10. D - Update or append**

`mapGroupsWithState` can use either "update" mode (output all groups that received new data) or "append" mode (output only when explicitly specified). The choice depends on your use case and downstream requirements.

---

## Scoring

- **10 correct**: Expert - You have mastered stateful stream processing
- **8-9 correct**: Proficient - Strong understanding with minor gaps
- **6-7 correct**: Developing - Good foundation, review key concepts
- **Below 6**: Review needed - Revisit the material and practice more

---

## Key Concepts to Remember

1. **Stateful vs Stateless**: Stateful maintains information across events
2. **mapGroupsWithState**: One output per group
3. **flatMapGroupsWithState**: Zero or more outputs per group
4. **Timeouts**: Essential to prevent unbounded state growth
5. **State Storage**: Checkpoint location with memory cache
6. **Timeout Types**: NoTimeout, ProcessingTimeTimeout, EventTimeTimeout
7. **State Management**: Always limit state size and set timeouts
8. **Use Cases**: Running aggregations, session tracking, pattern detection, anomaly detection
