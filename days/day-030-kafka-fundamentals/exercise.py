"""
Day 30: Kafka Fundamentals - Exercises
Topics, partitions, brokers, producers, consumers
"""

from typing import List, Dict
import hashlib

# Exercise 1: Kafka Concepts Quiz
def exercise_1():
    """
    Answer conceptual questions about Kafka.
    
    TODO: Answer these questions (write answers as comments or print)
    
    Q1: What is the difference between a topic and a partition?
    Q2: Why does Kafka use partitions?
    Q3: What is a broker in Kafka?
    Q4: What is the purpose of a consumer group?
    Q5: Can multiple consumer groups read from the same topic?
    """
    # TODO: Write your answers
    pass


# Exercise 2: Partition Assignment
def exercise_2():
    """
    Calculate which partition a message goes to based on key.
    
    TODO: Implement partition assignment logic
    TODO: Kafka uses: hash(key) % num_partitions
    TODO: Calculate partition for each message
    """
    num_partitions = 3
    
    messages = [
        {"key": "user123", "value": "login"},
        {"key": "user456", "value": "purchase"},
        {"key": "user123", "value": "logout"},
        {"key": "user789", "value": "signup"},
        {"key": "user456", "value": "view"},
    ]
    
    # TODO: For each message, calculate partition
    # TODO: Use hash function: hash(key) % num_partitions
    # TODO: Print: "Message with key 'user123' → Partition X"
    # TODO: Note which keys go to same partition (ordering preserved!)
    pass


# Exercise 3: Consumer Group Simulation
def exercise_3():
    """
    Simulate consumer group behavior with partition assignment.
    
    TODO: Given 6 partitions and different numbers of consumers,
    TODO: show how partitions are assigned to consumers
    """
    num_partitions = 6
    
    scenarios = [
        {"consumers": 2, "group": "group-A"},
        {"consumers": 3, "group": "group-B"},
        {"consumers": 6, "group": "group-C"},
        {"consumers": 8, "group": "group-D"},  # More consumers than partitions!
    ]
    
    # TODO: For each scenario, assign partitions to consumers
    # TODO: Rules:
    # TODO:   - Distribute partitions evenly
    # TODO:   - Each partition assigned to exactly one consumer
    # TODO:   - If more consumers than partitions, some consumers idle
    # TODO: Print assignment for each scenario
    pass


# Exercise 4: Offset Management
def exercise_4():
    """
    Track consumer offsets and simulate offset commits.
    
    TODO: Simulate a consumer reading messages and committing offsets
    TODO: Show what happens if consumer crashes and restarts
    """
    partition_messages = [
        {"offset": 0, "value": "msg0"},
        {"offset": 1, "value": "msg1"},
        {"offset": 2, "value": "msg2"},
        {"offset": 3, "value": "msg3"},
        {"offset": 4, "value": "msg4"},
        {"offset": 5, "value": "msg5"},
    ]
    
    # TODO: Simulate consumer reading messages
    # TODO: Consumer reads 3 messages, commits offset
    # TODO: Consumer crashes
    # TODO: Consumer restarts from last committed offset
    # TODO: Show which messages are read again (if any)
    pass


# Exercise 5: Message Ordering
def exercise_5():
    """
    Analyze ordering guarantees in different scenarios.
    
    TODO: For each scenario, determine if ordering is guaranteed
    """
    scenarios = [
        {
            "description": "All messages from user123 sent with key='user123'",
            "num_partitions": 3,
            "question": "Are messages from user123 ordered?"
        },
        {
            "description": "Messages sent without keys to 3 partitions",
            "num_partitions": 3,
            "question": "Are all messages globally ordered?"
        },
        {
            "description": "Single partition topic",
            "num_partitions": 1,
            "question": "Are all messages ordered?"
        },
        {
            "description": "2 consumers in same group reading 2 partitions",
            "num_partitions": 2,
            "question": "Does each consumer see messages in order?"
        }
    ]
    
    # TODO: For each scenario, answer the question
    # TODO: Explain why or why not
    pass


# Exercise 6: Architecture Design
def exercise_6():
    """
    Design a Kafka-based architecture for a use case.
    
    TODO: Choose one use case and design the architecture:
    TODO: 1. E-commerce order processing
    TODO: 2. Real-time log aggregation
    TODO: 3. IoT sensor data pipeline
    TODO:
    TODO: Specify:
    TODO: - Topic names and partition counts
    TODO: - Producer sources
    TODO: - Consumer groups and their purposes
    TODO: - Replication factor
    TODO: - Retention policy
    """
    # TODO: Write your design as comments or print statements
    pass


# Helper function for partition calculation
def calculate_partition(key: str, num_partitions: int) -> int:
    """
    Calculate partition for a given key.
    Mimics Kafka's default partitioner.
    """
    # TODO: Implement this helper
    # TODO: Use hash(key) % num_partitions
    pass


# Helper function for partition assignment
def assign_partitions(num_partitions: int, num_consumers: int) -> Dict[int, List[int]]:
    """
    Assign partitions to consumers in a consumer group.
    Returns dict: {consumer_id: [partition_ids]}
    """
    # TODO: Implement partition assignment logic
    # TODO: Distribute partitions evenly across consumers
    pass


if __name__ == "__main__":
    print("Day 30: Kafka Fundamentals - Exercises\n")
    
    print("=" * 50)
    print("Exercise 1: Kafka Concepts Quiz")
    print("=" * 50)
    # exercise_1()
    
    print("\n" + "=" * 50)
    print("Exercise 2: Partition Assignment")
    print("=" * 50)
    # exercise_2()
    
    print("\n" + "=" * 50)
    print("Exercise 3: Consumer Group Simulation")
    print("=" * 50)
    # exercise_3()
    
    print("\n" + "=" * 50)
    print("Exercise 4: Offset Management")
    print("=" * 50)
    # exercise_4()
    
    print("\n" + "=" * 50)
    print("Exercise 5: Message Ordering")
    print("=" * 50)
    # exercise_5()
    
    print("\n" + "=" * 50)
    print("Exercise 6: Architecture Design")
    print("=" * 50)
    # exercise_6()
