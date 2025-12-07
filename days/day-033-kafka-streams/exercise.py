"""
Day 33: Kafka Streams - Exercises

Prerequisites:
1. Start Kafka: docker run -d --name kafka -p 9092:9092 apache/kafka:latest
2. Install: pip install kafka-python faust-streaming
3. Run Day 31 producer to generate messages
"""

from kafka import KafkaConsumer, KafkaProducer
from collections import defaultdict
import json
import time


def exercise_1_stateless_transformations():
    """
    Exercise 1: Stateless Transformations
    
    Implement map, filter, and flatMap operations.
    
    TODO: Create consumer for 'events' topic
    TODO: Map: Add 'processed_at' timestamp to each event
    TODO: Filter: Only events with 'priority' == 'high'
    TODO: FlatMap: If event has 'tags' list, emit one message per tag
    TODO: Send results to 'processed-events' topic
    """
    pass


def exercise_2_word_count():
    """
    Exercise 2: Word Count
    
    Build classic word count with aggregation.
    
    TODO: Create consumer for 'sentences' topic
    TODO: Split each sentence into words
    TODO: Maintain count for each word in state
    TODO: Emit updated counts to 'word-counts' topic
    TODO: Print top 10 words periodically
    """
    pass


def exercise_3_stream_enrichment():
    """
    Exercise 3: Stream Enrichment
    
    Join event stream with reference data.
    
    TODO: Load user reference data into dictionary (KTable simulation)
    TODO: Create consumer for 'user-events' topic
    TODO: For each event, lookup user data by user_id
    TODO: Enrich event with user name, email, etc.
    TODO: Send enriched events to 'enriched-events' topic
    """
    pass


def exercise_4_tumbling_window():
    """
    Exercise 4: Tumbling Window Aggregation
    
    Count events in 1-minute tumbling windows.
    
    TODO: Create consumer for 'clicks' topic
    TODO: Group events into 1-minute windows
    TODO: Count events per window
    TODO: Emit window results when window closes
    TODO: Send to 'windowed-counts' topic
    """
    pass


def exercise_5_session_detection():
    """
    Exercise 5: Session Detection
    
    Detect user sessions with inactivity gaps.
    
    TODO: Create consumer for 'user-activity' topic
    TODO: Track sessions per user
    TODO: Close session after 5 minutes of inactivity
    TODO: Calculate session duration and event count
    TODO: Send completed sessions to 'sessions' topic
    """
    pass


def exercise_6_realtime_analytics():
    """
    Exercise 6: Real-Time Analytics
    
    Calculate running statistics (count, sum, avg).
    
    TODO: Create consumer for 'transactions' topic
    TODO: Maintain running stats per product_id:
    TODO:   - count (number of transactions)
    TODO:   - sum (total amount)
    TODO:   - avg (average amount)
    TODO:   - min/max amounts
    TODO: Emit updated stats after each transaction
    """
    pass


def exercise_7_stream_branching():
    """
    Exercise 7: Stream Branching
    
    Route events to different topics by category.
    
    TODO: Create consumer for 'all-events' topic
    TODO: Route events based on 'severity' field:
    TODO:   - 'critical' → 'critical-events' topic
    TODO:   - 'warning' → 'warning-events' topic
    TODO:   - 'info' → 'info-events' topic
    TODO: Count events per category
    """
    pass


def exercise_8_stateful_processing():
    """
    Exercise 8: Stateful Processing with Store
    
    Implement stateful processor with persistent state.
    
    TODO: Create consumer for 'orders' topic
    TODO: Use shelve for persistent state store
    TODO: Track order status per order_id
    TODO: Update state on each order event
    TODO: Emit state changes to 'order-status' topic
    TODO: Handle state recovery on restart
    """
    pass


if __name__ == "__main__":
    print("Day 33: Kafka Streams - Exercises\n")
    print("=" * 60)
    print("\nMake sure Kafka is running:")
    print("  docker run -d --name kafka -p 9092:9092 apache/kafka:latest\n")
    
    # Uncomment to run exercises
    # print("\nExercise 1: Stateless Transformations")
    # exercise_1_stateless_transformations()
    
    # print("\nExercise 2: Word Count")
    # exercise_2_word_count()
    
    # print("\nExercise 3: Stream Enrichment")
    # exercise_3_stream_enrichment()
    
    # print("\nExercise 4: Tumbling Window")
    # exercise_4_tumbling_window()
    
    # print("\nExercise 5: Session Detection")
    # exercise_5_session_detection()
    
    # print("\nExercise 6: Real-Time Analytics")
    # exercise_6_realtime_analytics()
    
    # print("\nExercise 7: Stream Branching")
    # exercise_7_stream_branching()
    
    # print("\nExercise 8: Stateful Processing")
    # exercise_8_stateful_processing()
