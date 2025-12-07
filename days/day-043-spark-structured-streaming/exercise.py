"""
Day 43: Spark Structured Streaming - Exercises
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, explode, window, count, avg, current_timestamp
import time


def exercise_1_socket_word_count():
    """
    Exercise 1: Socket Stream Word Count
    
    Create a streaming word count from socket input.
    
    Steps:
    1. Create SparkSession
    2. Read from socket (localhost:9999)
    3. Split lines into words
    4. Count words
    5. Write to console
    
    To test:
    - Terminal 1: nc -lk 9999
    - Terminal 2: python exercise.py
    - Type words in Terminal 1
    """
    # TODO: Create SparkSession
    
    # TODO: Read from socket
    
    # TODO: Split lines into words
    
    # TODO: Count words
    
    # TODO: Write to console with complete mode
    
    # TODO: Start query and wait
    
    pass


def exercise_2_file_stream():
    """
    Exercise 2: File Stream Processing
    
    Monitor directory and process new CSV files.
    
    Schema: timestamp, user_id, action, amount
    
    Tasks:
    1. Read CSV files from data/input/
    2. Filter for 'purchase' actions
    3. Calculate total amount per user
    4. Write to console
    """
    # TODO: Create SparkSession
    
    # TODO: Define schema
    
    # TODO: Read CSV stream
    
    # TODO: Filter for purchases
    
    # TODO: Group by user and sum amount
    
    # TODO: Write to console
    
    # TODO: Start and wait
    
    pass


def exercise_3_rate_source():
    """
    Exercise 3: Rate Source Aggregation
    
    Use rate source to generate test data.
    
    Tasks:
    1. Create rate source (10 rows/second)
    2. Add processed_at timestamp
    3. Calculate statistics per 10-second window
    4. Write to console
    """
    # TODO: Create SparkSession
    
    # TODO: Create rate source
    
    # TODO: Add processed_at column
    
    # TODO: Create 10-second tumbling window
    
    # TODO: Calculate count and avg(value)
    
    # TODO: Write to console
    
    # TODO: Start and wait
    
    pass


def exercise_4_output_modes():
    """
    Exercise 4: Compare Output Modes
    
    Create same query with different output modes.
    
    Tasks:
    1. Create rate source
    2. Count by value % 10 (creates 10 groups)
    3. Try append, complete, and update modes
    4. Observe differences
    """
    # TODO: Create SparkSession
    
    # TODO: Create rate source
    
    # TODO: Add group column (value % 10)
    
    # TODO: Count by group
    
    # TODO: Try complete mode first
    
    # TODO: Stop and try update mode
    
    # TODO: Compare outputs
    
    pass


def exercise_5_checkpoint_recovery():
    """
    Exercise 5: Checkpoint and Recovery
    
    Test fault tolerance with checkpointing.
    
    Tasks:
    1. Create rate source
    2. Write to parquet with checkpoint
    3. Stop query after 10 seconds
    4. Restart query (should resume from checkpoint)
    5. Verify no duplicate data
    """
    # TODO: Create SparkSession
    
    # TODO: Create rate source
    
    # TODO: Write to parquet with checkpoint
    
    # TODO: Start query
    
    # TODO: Wait 10 seconds
    
    # TODO: Stop query
    
    # TODO: Restart query (same checkpoint location)
    
    # TODO: Verify data continuity
    
    pass


if __name__ == "__main__":
    print("Day 43: Spark Structured Streaming Exercises\n")
    print("Uncomment exercises to run:\n")
    
    # Exercise 1: Socket word count
    # print("Exercise 1: Socket Word Count")
    # print("Start socket: nc -lk 9999")
    # exercise_1_socket_word_count()
    
    # Exercise 2: File stream
    # print("\nExercise 2: File Stream")
    # exercise_2_file_stream()
    
    # Exercise 3: Rate source
    # print("\nExercise 3: Rate Source")
    # exercise_3_rate_source()
    
    # Exercise 4: Output modes
    # print("\nExercise 4: Output Modes")
    # exercise_4_output_modes()
    
    # Exercise 5: Checkpoint recovery
    # print("\nExercise 5: Checkpoint Recovery")
    # exercise_5_checkpoint_recovery()
