"""
Day 29: Streaming Concepts - Solutions
"""

from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict

# Exercise 1: Batch vs Streaming Decision
def exercise_1():
    """Analyze scenarios and decide: batch or streaming?"""
    scenarios = [
        {
            "name": "Daily sales report",
            "latency": "24 hours acceptable",
            "volume": "1M records/day",
            "pattern": "predictable",
            "decision": "BATCH",
            "reason": "Latency requirement is relaxed, predictable schedule, cost-effective"
        },
        {
            "name": "Fraud detection",
            "latency": "< 1 second required",
            "volume": "10K transactions/sec",
            "pattern": "continuous",
            "decision": "STREAMING",
            "reason": "Real-time requirement, continuous data, time-sensitive decisions"
        },
        {
            "name": "Monthly customer segmentation",
            "latency": "hours acceptable",
            "volume": "100M customer records",
            "pattern": "scheduled",
            "decision": "BATCH",
            "reason": "Large historical dataset, scheduled processing, no real-time need"
        },
        {
            "name": "IoT sensor monitoring",
            "latency": "< 5 seconds required",
            "volume": "1K sensors, continuous",
            "pattern": "real-time",
            "decision": "STREAMING",
            "reason": "Near real-time alerts needed, continuous sensor data"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\nScenario {i}: {scenario['name']}")
        print(f"  Latency: {scenario['latency']}")
        print(f"  Volume: {scenario['volume']}")
        print(f"  Decision: {scenario['decision']}")
        print(f"  Reason: {scenario['reason']}")


# Exercise 2: Event Time Simulation
def exercise_2():
    """Simulate events with event_time and processing_time"""
    base_time = datetime(2024, 1, 1, 10, 0, 0)
    
    events = [
        {
            "id": 1,
            "event_time": base_time,
            "processing_time": base_time + timedelta(seconds=1),
            "data": "purchase"
        },
        {
            "id": 2,
            "event_time": base_time + timedelta(seconds=5),
            "processing_time": base_time + timedelta(seconds=12),  # Late!
            "data": "click"
        },
        {
            "id": 3,
            "event_time": base_time + timedelta(seconds=10),
            "processing_time": base_time + timedelta(seconds=11),
            "data": "view"
        },
        {
            "id": 4,
            "event_time": base_time + timedelta(seconds=3),
            "processing_time": base_time + timedelta(seconds=15),  # Very late!
            "data": "signup"
        },
        {
            "id": 5,
            "event_time": base_time + timedelta(seconds=20),
            "processing_time": base_time + timedelta(seconds=21),
            "data": "logout"
        }
    ]
    
    print("\nEvent Analysis:")
    for event in events:
        delay = (event["processing_time"] - event["event_time"]).total_seconds()
        status = "LATE" if delay > 5 else "ON TIME"
        print(f"Event {event['id']}: {event['data']}")
        print(f"  Event time: {event['event_time'].strftime('%H:%M:%S')}")
        print(f"  Processing time: {event['processing_time'].strftime('%H:%M:%S')}")
        print(f"  Delay: {delay:.1f}s - {status}\n")


# Exercise 3: Windowing Implementation
def exercise_3():
    """Implement tumbling and sliding windows"""
    events = [
        {"time": datetime(2024, 1, 1, 10, 0, 0), "value": 10},
        {"time": datetime(2024, 1, 1, 10, 2, 0), "value": 20},
        {"time": datetime(2024, 1, 1, 10, 4, 0), "value": 15},
        {"time": datetime(2024, 1, 1, 10, 6, 0), "value": 25},
        {"time": datetime(2024, 1, 1, 10, 8, 0), "value": 30},
        {"time": datetime(2024, 1, 1, 10, 11, 0), "value": 40},
    ]
    
    # Tumbling windows (5-minute, non-overlapping)
    print("\nTumbling Windows (5 minutes):")
    tumbling = defaultdict(list)
    for event in events:
        # Round down to nearest 5-minute mark
        window_start = event["time"].replace(minute=(event["time"].minute // 5) * 5, second=0)
        tumbling[window_start].append(event)
    
    for window_start in sorted(tumbling.keys()):
        window_end = window_start + timedelta(minutes=5)
        values = [e["value"] for e in tumbling[window_start]]
        print(f"  [{window_start.strftime('%H:%M')} - {window_end.strftime('%H:%M')}): "
              f"values={values}, sum={sum(values)}")
    
    # Sliding windows (5-minute window, 1-minute slide)
    print("\nSliding Windows (5-minute window, 1-minute slide):")
    start_time = min(e["time"] for e in events)
    end_time = max(e["time"] for e in events)
    
    current = start_time
    while current <= end_time:
        window_end = current + timedelta(minutes=5)
        window_events = [e for e in events if current <= e["time"] < window_end]
        if window_events:
            values = [e["value"] for e in window_events]
            print(f"  [{current.strftime('%H:%M')} - {window_end.strftime('%H:%M')}): "
                  f"values={values}, sum={sum(values)}")
        current += timedelta(minutes=1)


# Exercise 4: Watermark Logic
def exercise_4():
    """Handle late-arriving events with watermarks"""
    current_time = datetime(2024, 1, 1, 10, 0, 0)
    watermark_delay = timedelta(seconds=10)
    watermark = current_time - watermark_delay
    
    events = [
        {"id": 1, "event_time": datetime(2024, 1, 1, 9, 59, 55), "value": 100},
        {"id": 2, "event_time": datetime(2024, 1, 1, 9, 59, 45), "value": 200},
        {"id": 3, "event_time": datetime(2024, 1, 1, 9, 59, 58), "value": 150},
        {"id": 4, "event_time": datetime(2024, 1, 1, 9, 59, 40), "value": 300},
    ]
    
    print(f"\nCurrent time: {current_time.strftime('%H:%M:%S')}")
    print(f"Watermark: {watermark.strftime('%H:%M:%S')} (current - 10s)")
    print(f"\nProcessing events:")
    
    accepted = []
    rejected = []
    
    for event in events:
        if event["event_time"] >= watermark:
            accepted.append(event)
            print(f"  Event {event['id']}: ACCEPTED (event_time={event['event_time'].strftime('%H:%M:%S')})")
        else:
            rejected.append(event)
            print(f"  Event {event['id']}: REJECTED - Too late (event_time={event['event_time'].strftime('%H:%M:%S')})")
    
    print(f"\nSummary: {len(accepted)} accepted, {len(rejected)} rejected")


# Exercise 5: Simple Stream Processor
def exercise_5():
    """Build a basic stream processor with state"""
    purchase_stream = [
        {"user_id": "user1", "amount": 100},
        {"user_id": "user2", "amount": 500},
        {"user_id": "user1", "amount": 300},
        {"user_id": "user1", "amount": 700},
        {"user_id": "user2", "amount": 600},
        {"user_id": "user3", "amount": 200},
    ]
    
    user_totals = defaultdict(int)
    threshold = 1000
    
    print("\nProcessing purchase stream:")
    for purchase in purchase_stream:
        user_id = purchase["user_id"]
        amount = purchase["amount"]
        user_totals[user_id] += amount
        
        print(f"  {user_id}: +${amount} → Total: ${user_totals[user_id]}")
        
        if user_totals[user_id] > threshold:
            print(f"    🚨 ALERT: {user_id} exceeded ${threshold}!")
    
    print(f"\nFinal state:")
    for user_id, total in sorted(user_totals.items()):
        print(f"  {user_id}: ${total}")


# Exercise 6: Use Case Design
def exercise_6():
    """Design a streaming architecture"""
    print("\nUse Case: Real-time Website Analytics")
    print("\nArchitecture Design:")
    print("""
    1. Data Source:
       - Web servers send clickstream events (page views, clicks, sessions)
       - Events include: user_id, page_url, timestamp, session_id, action
    
    2. Ingestion:
       - Events sent to Kafka topic 'web-events'
       - Partitioned by user_id for parallelism
    
    3. Processing Steps:
       a) Parse and validate events
       b) Enrich with user metadata (location, device)
       c) Apply windowing (5-minute tumbling windows)
       d) Aggregate metrics:
          - Page views per page
          - Unique users per page
          - Average session duration
          - Click-through rates
    
    4. Windowing Strategy:
       - Tumbling windows: 5 minutes for real-time dashboards
       - Session windows: Track user sessions (30-min timeout)
       - Watermark: 1 minute (allow 1-min late arrivals)
    
    5. Output/Sink:
       - Real-time metrics → Redis (for dashboard queries)
       - Aggregated data → PostgreSQL (for historical analysis)
       - Alerts → Slack/Email (for anomalies)
    
    6. Handling Late Data:
       - Accept events up to 1 minute late (watermark)
       - Store very late events in separate 'late-events' topic
       - Reprocess if needed for accuracy
    
    7. State Management:
       - Maintain session state in RocksDB (Flink state backend)
       - Checkpoint every 30 seconds for fault tolerance
    
    8. Scaling:
       - Horizontal scaling via Kafka partitions
       - Multiple stream processors for parallelism
       - Auto-scaling based on lag
    """)


if __name__ == "__main__":
    print("Day 29: Streaming Concepts - Solutions\n")
    
    print("=" * 50)
    print("Exercise 1: Batch vs Streaming Decision")
    print("=" * 50)
    exercise_1()
    
    print("\n" + "=" * 50)
    print("Exercise 2: Event Time Simulation")
    print("=" * 50)
    exercise_2()
    
    print("\n" + "=" * 50)
    print("Exercise 3: Windowing Implementation")
    print("=" * 50)
    exercise_3()
    
    print("\n" + "=" * 50)
    print("Exercise 4: Watermark Logic")
    print("=" * 50)
    exercise_4()
    
    print("\n" + "=" * 50)
    print("Exercise 5: Simple Stream Processor")
    print("=" * 50)
    exercise_5()
    
    print("\n" + "=" * 50)
    print("Exercise 6: Use Case Design")
    print("=" * 50)
    exercise_6()
