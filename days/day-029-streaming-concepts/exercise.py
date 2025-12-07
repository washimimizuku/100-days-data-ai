"""
Day 29: Streaming Concepts - Exercises
Batch vs streaming, windowing, watermarks
"""

from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict

# Exercise 1: Batch vs Streaming Decision
def exercise_1():
    """
    Analyze scenarios and decide: batch or streaming?
    
    TODO: For each scenario, determine if batch or streaming is better
    TODO: Print your decision and reasoning
    """
    scenarios = [
        {
            "name": "Daily sales report",
            "latency": "24 hours acceptable",
            "volume": "1M records/day",
            "pattern": "predictable"
        },
        {
            "name": "Fraud detection",
            "latency": "< 1 second required",
            "volume": "10K transactions/sec",
            "pattern": "continuous"
        },
        {
            "name": "Monthly customer segmentation",
            "latency": "hours acceptable",
            "volume": "100M customer records",
            "pattern": "scheduled"
        },
        {
            "name": "IoT sensor monitoring",
            "latency": "< 5 seconds required",
            "volume": "1K sensors, continuous",
            "pattern": "real-time"
        }
    ]
    
    # TODO: Analyze each scenario
    # TODO: Print: "Scenario X: [BATCH/STREAMING] - Reason: ..."
    pass


# Exercise 2: Event Time Simulation
def exercise_2():
    """
    Simulate events with event_time and processing_time.
    
    TODO: Create 5 events with timestamps
    TODO: Simulate some events arriving late (processing_time > event_time + 5 sec)
    TODO: Identify which events are late
    """
    events = []
    
    # TODO: Create events like:
    # {
    #     "id": 1,
    #     "event_time": datetime(...),
    #     "processing_time": datetime(...),
    #     "data": "some data"
    # }
    
    # TODO: Check for late arrivals (processing_time - event_time > 5 seconds)
    pass


# Exercise 3: Windowing Implementation
def exercise_3():
    """
    Implement tumbling and sliding windows.
    
    TODO: Given a list of events with timestamps, group them into:
    TODO: 1. Tumbling windows (5-minute non-overlapping)
    TODO: 2. Sliding windows (5-minute, sliding every 1 minute)
    """
    events = [
        {"time": datetime(2024, 1, 1, 10, 0, 0), "value": 10},
        {"time": datetime(2024, 1, 1, 10, 2, 0), "value": 20},
        {"time": datetime(2024, 1, 1, 10, 4, 0), "value": 15},
        {"time": datetime(2024, 1, 1, 10, 6, 0), "value": 25},
        {"time": datetime(2024, 1, 1, 10, 8, 0), "value": 30},
        {"time": datetime(2024, 1, 1, 10, 11, 0), "value": 40},
    ]
    
    # TODO: Implement tumbling_window(events, window_size_minutes=5)
    # TODO: Implement sliding_window(events, window_size_minutes=5, slide_minutes=1)
    # TODO: Print results showing which events fall in which windows
    pass


# Exercise 4: Watermark Logic
def exercise_4():
    """
    Handle late-arriving events with watermarks.
    
    TODO: Process events with a watermark of "current_time - 10 seconds"
    TODO: Accept events within watermark, reject events too late
    TODO: Track accepted vs rejected events
    """
    current_time = datetime(2024, 1, 1, 10, 0, 0)
    watermark_delay = timedelta(seconds=10)
    
    events = [
        {"id": 1, "event_time": datetime(2024, 1, 1, 9, 59, 55), "value": 100},  # 5 sec late
        {"id": 2, "event_time": datetime(2024, 1, 1, 9, 59, 45), "value": 200},  # 15 sec late
        {"id": 3, "event_time": datetime(2024, 1, 1, 9, 59, 58), "value": 150},  # 2 sec late
        {"id": 4, "event_time": datetime(2024, 1, 1, 9, 59, 40), "value": 300},  # 20 sec late
    ]
    
    # TODO: Calculate watermark = current_time - watermark_delay
    # TODO: For each event, check if event_time >= watermark
    # TODO: Print accepted and rejected events
    pass


# Exercise 5: Simple Stream Processor
def exercise_5():
    """
    Build a basic stream processor with state.
    
    TODO: Process a stream of purchase events
    TODO: Maintain running total per user
    TODO: Alert if user's total exceeds $1000
    """
    purchase_stream = [
        {"user_id": "user1", "amount": 100},
        {"user_id": "user2", "amount": 500},
        {"user_id": "user1", "amount": 300},
        {"user_id": "user1", "amount": 700},  # Should trigger alert
        {"user_id": "user2", "amount": 600},  # Should trigger alert
        {"user_id": "user3", "amount": 200},
    ]
    
    # TODO: Initialize state: user_totals = {}
    # TODO: For each purchase, update user's total
    # TODO: If total > 1000, print alert
    # TODO: Print final state
    pass


# Exercise 6: Use Case Design
def exercise_6():
    """
    Design a streaming architecture for a use case.
    
    TODO: Choose one use case:
    TODO: 1. Real-time website analytics (page views, clicks)
    TODO: 2. Stock price monitoring and alerts
    TODO: 3. Social media sentiment analysis
    TODO:
    TODO: Design and describe:
    TODO: - Data source
    TODO: - Processing steps
    TODO: - Windowing strategy
    TODO: - Output/sink
    TODO: - Handling late data
    """
    # TODO: Write your design as comments or print statements
    pass


if __name__ == "__main__":
    print("Day 29: Streaming Concepts - Exercises\n")
    
    print("=" * 50)
    print("Exercise 1: Batch vs Streaming Decision")
    print("=" * 50)
    # exercise_1()
    
    print("\n" + "=" * 50)
    print("Exercise 2: Event Time Simulation")
    print("=" * 50)
    # exercise_2()
    
    print("\n" + "=" * 50)
    print("Exercise 3: Windowing Implementation")
    print("=" * 50)
    # exercise_3()
    
    print("\n" + "=" * 50)
    print("Exercise 4: Watermark Logic")
    print("=" * 50)
    # exercise_4()
    
    print("\n" + "=" * 50)
    print("Exercise 5: Simple Stream Processor")
    print("=" * 50)
    # exercise_5()
    
    print("\n" + "=" * 50)
    print("Exercise 6: Use Case Design")
    print("=" * 50)
    # exercise_6()
