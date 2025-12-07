# Day 47: Stateful Stream Processing

## 📖 Learning Objectives

By the end of this session, you will:
- Understand stateful operations in streaming
- Use mapGroupsWithState for custom state logic
- Implement flatMapGroupsWithState for complex scenarios
- Manage state timeouts effectively
- Build session tracking and user analytics
- Optimize stateful operations

**Time**: 1 hour

---

## What is Stateful Processing?

**Stateful processing** maintains information across multiple events to compute results that depend on historical data.

```python
# Stateless: Each event processed independently
df.select("user_id", "amount").filter(col("amount") > 100)

# Stateful: Maintains state across events (running total)
df.groupBy("user_id").agg(sum("amount"))
```

**Built-in stateful operations**: `groupBy().agg()`, `window()`, stream joins, `dropDuplicates()`

**Custom stateful operations**:
- **mapGroupsWithState**: One output per group
- **flatMapGroupsWithState**: Multiple outputs per group

---

## mapGroupsWithState

Process groups with custom state logic, producing one output per group.

```python
from pyspark.sql.streaming import GroupState, GroupStateTimeout

# Basic structure
def update_state(key, values, state):
    # Get existing state or initialize
    old_state = state.get if state.exists else initial_value
    
    # Process new values and update state
    new_state = compute_new_state(old_state, values)
    state.update(new_state)
    
    return output_value

# Example: Running total per user
def update_total(user_id, values, state):
    total = state.get if state.exists else 0
    for value in values:
        total += value.amount
    state.update(total)
    return (user_id, total)

# Apply
result = df.groupByKey(lambda x: x.user_id) \
    .mapGroupsWithState(update_total, outputMode="update", timeoutConf=GroupStateTimeout.NoTimeout)
```

---

## flatMapGroupsWithState

Process groups with custom state logic, producing zero or more outputs per group.

```python
# Basic structure - returns iterator (0, 1, or many outputs)
def update_state_flat(key, values, state):
    outputs = []
    for value in values:
        if condition:
            outputs.append(result)
    state.update(new_state)
    return iter(outputs)

# Example: Session detection with 30-minute timeout
def detect_sessions(user_id, events, state):
    session = state.get if state.exists else {"start": None, "end": None, "count": 0}
    outputs = []
    
    for event in events:
        event_time = event.timestamp
        
        if session["start"] is None:
            # Start new session
            session = {"start": event_time, "end": event_time, "count": 1}
        elif (event_time - session["end"]).seconds > 1800:  # 30 min gap
            # Output completed session, start new one
            outputs.append({"user_id": user_id, "session_start": session["start"], 
                          "session_end": session["end"], "event_count": session["count"]})
            session = {"start": event_time, "end": event_time, "count": 1}
        else:
            # Continue session
            session["end"] = event_time
            session["count"] += 1
    
    state.update(session)
    return iter(outputs)

# Apply
result = df.groupByKey(lambda x: x.user_id) \
    .flatMapGroupsWithState(detect_sessions, outputMode="append", 
                           timeoutConf=GroupStateTimeout.ProcessingTimeTimeout)
```

---

## State Timeouts

Control when state expires and is removed.

```python
# 1. NoTimeout - state never expires
GroupStateTimeout.NoTimeout

# 2. ProcessingTimeTimeout - timeout based on processing time
def update_with_timeout(key, values, state):
    state.setTimeoutDuration("10 minutes")
    if state.hasTimedOut:
        state.remove()
        return (key, "timed_out")
    # Normal processing...

# 3. EventTimeTimeout - timeout based on event time watermark
def update_with_event_timeout(key, values, state):
    state.setTimeoutTimestamp(event_time + timeout_duration)
    if state.hasTimedOut:
        # Handle timeout...

# Example: Track active users with 5-minute inactivity timeout
def track_active_users(user_id, events, state):
    if state.hasTimedOut:
        last_activity = state.get
        state.remove()
        return iter([{"user_id": user_id, "status": "inactive", "last_seen": last_activity}])
    
    latest_time = max(event.timestamp for event in events)
    state.update(latest_time)
    state.setTimeoutDuration("5 minutes")
    return iter([{"user_id": user_id, "status": "active", "last_seen": latest_time}])
```

---

## Common Patterns

```python
# Pattern 1: Running aggregations
def running_stats(key, values, state):
    stats = state.get if state.exists else {"count": 0, "sum": 0, "min": float('inf'), "max": float('-inf')}
    for value in values:
        stats["count"] += 1
        stats["sum"] += value.amount
        stats["min"] = min(stats["min"], value.amount)
        stats["max"] = max(stats["max"], value.amount)
    stats["avg"] = stats["sum"] / stats["count"]
    state.update(stats)
    return (key, stats)

# Pattern 2: Event sequencing (detect view -> add_to_cart -> purchase)
def detect_sequence(key, events, state):
    sequence = state.get if state.exists else []
    outputs = []
    for event in events:
        sequence.append(event.event_type)
        if len(sequence) >= 3 and sequence[-3:] == ["view", "add_to_cart", "purchase"]:
            outputs.append({"user_id": key, "pattern": "conversion", "timestamp": event.timestamp})
            sequence = []
    state.update(sequence)
    return iter(outputs)

# Pattern 3: Anomaly detection (> 3 standard deviations)
def detect_anomalies(key, values, state):
    history = state.get if state.exists else {"values": [], "mean": 0, "stddev": 0}
    outputs = []
    for value in values:
        if history["stddev"] > 0:
            z_score = abs(value.amount - history["mean"]) / history["stddev"]
            if z_score > 3:
                outputs.append({"user_id": key, "amount": value.amount, "z_score": z_score})
        history["values"].append(value.amount)
        if len(history["values"]) > 100:
            history["values"].pop(0)
        history["mean"] = sum(history["values"]) / len(history["values"])
        variance = sum((x - history["mean"]) ** 2 for x in history["values"]) / len(history["values"])
        history["stddev"] = variance ** 0.5
    state.update(history)
    return iter(outputs)

# Pattern 4: State expiration with timeout
def expire_old_state(key, values, state):
    if state.hasTimedOut:
        state.remove()
        return iter([{"user_id": key, "status": "expired"}])
    for value in values:
        state.update(value)
        state.setTimeoutDuration("1 hour")
    return iter([{"user_id": key, "status": "active"}])
```

---

## Performance Optimization

```python
# 1. Limit state size - keep only recent data
def bounded_state(key, values, state):
    history = state.get if state.exists else []
    for value in values:
        history.append(value)
        if len(history) > 1000:  # Keep last 1000 items
            history = history[-1000:]
    state.update(history)
    return (key, len(history))

# 2. Always use timeouts to prevent unbounded state growth
state.setTimeoutDuration("1 hour")

# 3. Checkpoint frequently
query = df.groupByKey(...).mapGroupsWithState(...).writeStream \
    .option("checkpointLocation", "checkpoint/") \
    .trigger(processingTime="1 minute").start()

# 4. Monitor state size
progress = query.lastProgress
if progress and 'stateOperators' in progress:
    for op in progress['stateOperators']:
        print(f"State rows: {op.get('numRowsTotal', 0)}, Memory: {op.get('memoryUsedBytes', 0)}")
```

---

## 💻 Exercises

### Exercise 1: Running Total
Implement running total per user with mapGroupsWithState.

### Exercise 2: Session Detection
Detect user sessions with 30-minute timeout.

### Exercise 3: Event Sequencing
Detect specific event patterns.

### Exercise 4: Active User Tracking
Track active users with timeout.

### Exercise 5: Anomaly Detection
Detect anomalous values using historical stats.

---

## ✅ Quiz (5 min)

Test your understanding in `quiz.md`.

---

## 🎯 Key Takeaways

- Stateful operations maintain information across events
- mapGroupsWithState: one output per group
- flatMapGroupsWithState: multiple outputs per group
- Always set timeouts to prevent unbounded state growth
- Monitor state size in production
- Use checkpoints for fault tolerance
- Limit state size by keeping only recent data

---

## 📚 Resources

- [Arbitrary Stateful Operations](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html#arbitrary-stateful-operations)
- [mapGroupsWithState](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.streaming.DataStreamReader.html)

---

## Tomorrow: Day 48 - Streaming Performance Optimization

Learn techniques to optimize streaming query performance and resource usage.
