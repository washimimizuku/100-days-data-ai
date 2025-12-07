"""
Day 32: Kafka Consumers & Consumer Groups - Exercises

Prerequisites:
1. Start Kafka: docker run -d --name kafka -p 9092:9092 apache/kafka:latest
2. Install: pip install kafka-python
3. Run Day 31 producer to generate messages
"""

from kafka import KafkaConsumer, TopicPartition
from kafka.errors import KafkaError
import json
import time


def exercise_1_basic_consumer():
    """
    Exercise 1: Basic Consumer
    
    Create a consumer that reads messages and prints them.
    
    TODO: Create KafkaConsumer for 'test-topic'
    TODO: Set group_id='test-group'
    TODO: Configure JSON deserializer
    TODO: Set auto_offset_reset='earliest'
    TODO: Read and print 10 messages
    TODO: Close consumer
    """
    pass


def exercise_2_consumer_group():
    """
    Exercise 2: Consumer Group
    
    Create multiple consumers in same group for parallel processing.
    
    TODO: Create 3 consumers with same group_id
    TODO: Subscribe all to 'orders' topic
    TODO: Print which partitions each consumer gets
    TODO: Process messages in parallel (simulate with threads)
    TODO: Show load distribution
    """
    pass


def exercise_3_manual_offset_management():
    """
    Exercise 3: Manual Offset Management
    
    Implement consumer with manual commits after processing.
    
    TODO: Create consumer with enable_auto_commit=False
    TODO: Read messages one by one
    TODO: Process each message
    TODO: Commit offset after successful processing
    TODO: Handle errors without committing
    """
    pass


def exercise_4_batch_processing():
    """
    Exercise 4: Batch Processing
    
    Process messages in batches for better throughput.
    
    TODO: Create consumer
    TODO: Collect messages into batches of 50
    TODO: Process entire batch at once
    TODO: Commit after batch processing
    TODO: Measure throughput (messages/sec)
    """
    pass


def exercise_5_rebalance_handling():
    """
    Exercise 5: Rebalance Handling
    
    Implement rebalance listener to handle partition changes.
    
    TODO: Create ConsumerRebalanceListener class
    TODO: Implement on_partitions_revoked (commit offsets)
    TODO: Implement on_partitions_assigned (log assignment)
    TODO: Subscribe with listener
    TODO: Simulate rebalance by adding/removing consumers
    """
    pass


def exercise_6_consumer_lag_monitoring():
    """
    Exercise 6: Consumer Lag Monitoring
    
    Monitor and report consumer lag for each partition.
    
    TODO: Create consumer
    TODO: Get assigned partitions
    TODO: For each partition:
    TODO:   - Get current position
    TODO:   - Get end offset
    TODO:   - Calculate lag
    TODO: Print lag report
    TODO: Alert if lag > threshold
    """
    pass


def exercise_7_error_handling_dlq():
    """
    Exercise 7: Error Handling with DLQ
    
    Implement retry logic and dead letter queue.
    
    TODO: Create consumer
    TODO: Create DLQ producer
    TODO: Process messages with try/except
    TODO: Retry failed messages 3 times
    TODO: Send to DLQ after max retries
    TODO: Commit offset after DLQ send
    """
    pass


def exercise_8_production_consumer_class():
    """
    Exercise 8: Production Consumer Class
    
    Create reusable consumer class with best practices.
    
    TODO: Create ProductionConsumer class with:
    TODO: - __init__ with configuration
    TODO: - consume() method with error handling
    TODO: - _process_message() with retries
    TODO: - _handle_rebalance() listener
    TODO: - get_lag() for monitoring
    TODO: - close() for graceful shutdown
    TODO: Test with 100 messages
    """
    pass


if __name__ == "__main__":
    print("Day 32: Kafka Consumers & Consumer Groups - Exercises\n")
    print("=" * 60)
    print("\nMake sure Kafka is running:")
    print("  docker run -d --name kafka -p 9092:9092 apache/kafka:latest\n")
    
    # Uncomment to run exercises
    # print("\nExercise 1: Basic Consumer")
    # exercise_1_basic_consumer()
    
    # print("\nExercise 2: Consumer Group")
    # exercise_2_consumer_group()
    
    # print("\nExercise 3: Manual Offset Management")
    # exercise_3_manual_offset_management()
    
    # print("\nExercise 4: Batch Processing")
    # exercise_4_batch_processing()
    
    # print("\nExercise 5: Rebalance Handling")
    # exercise_5_rebalance_handling()
    
    # print("\nExercise 6: Consumer Lag Monitoring")
    # exercise_6_consumer_lag_monitoring()
    
    # print("\nExercise 7: Error Handling with DLQ")
    # exercise_7_error_handling_dlq()
    
    # print("\nExercise 8: Production Consumer Class")
    # exercise_8_production_consumer_class()
