"""
Day 45: Watermarking and Late Data - Exercises
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *


def exercise_1_basic_watermark():
    """
    Exercise 1: Basic Watermark
    
    Create streaming query with 5-minute watermark.
    
    Tasks:
    1. Create rate source
    2. Add watermark on timestamp
    3. Create windowed aggregation
    4. Write to console with append mode
    """
    # TODO: Create SparkSession
    
    # TODO: Create rate source
    
    # TODO: Add 5-minute watermark
    
    # TODO: Create 10-minute tumbling window aggregation
    
    # TODO: Write to console with append mode
    
    pass


def exercise_2_late_data_detection():
    """
    Exercise 2: Late Data Detection
    
    Identify and count late-arriving events.
    
    Tasks:
    1. Create rate source
    2. Calculate lateness (current_time - event_time)
    3. Tag events as on-time or late (threshold: 30 seconds)
    4. Count by late/on-time
    5. Write to console
    """
    # TODO: Create SparkSession
    
    # TODO: Create rate source
    
    # TODO: Add lateness calculation
    
    # TODO: Tag as late or on-time
    
    # TODO: Count by tag
    
    # TODO: Write to console
    
    pass


def exercise_3_watermark_aggregation():
    """
    Exercise 3: Watermark with Aggregation
    
    Implement windowed aggregation with watermark.
    
    Tasks:
    1. Create rate source with product_id
    2. Add 10-minute watermark
    3. Create 5-minute windows
    4. Calculate count and sum by product
    5. Compare append vs update mode
    """
    # TODO: Create SparkSession
    
    # TODO: Create rate source with product_id
    
    # TODO: Add watermark
    
    # TODO: Create windowed aggregation
    
    # TODO: Write with append mode
    
    # TODO: Try update mode and compare
    
    pass


def exercise_4_state_management():
    """
    Exercise 4: State Management
    
    Monitor state size with and without watermark.
    
    Tasks:
    1. Create rate source
    2. Run aggregation WITHOUT watermark
    3. Monitor state growth
    4. Run aggregation WITH watermark
    5. Compare state sizes
    """
    # TODO: Create SparkSession
    
    # TODO: Create rate source
    
    # TODO: Aggregation without watermark
    
    # TODO: Monitor state metrics
    
    # TODO: Aggregation with watermark
    
    # TODO: Compare state sizes
    
    pass


def exercise_5_join_with_watermark():
    """
    Exercise 5: Join with Watermark
    
    Implement stream-to-stream join with watermarks.
    
    Tasks:
    1. Create two rate sources (orders and shipments)
    2. Add watermarks to both (5 minutes)
    3. Join with time constraint (within 10 minutes)
    4. Write to console
    """
    # TODO: Create SparkSession
    
    # TODO: Create orders stream with watermark
    
    # TODO: Create shipments stream with watermark
    
    # TODO: Join with time constraint
    
    # TODO: Write to console
    
    pass


if __name__ == "__main__":
    print("Day 45: Watermarking and Late Data Exercises\n")
    print("Uncomment exercises to run:\n")
    
    # Exercise 1: Basic watermark
    # print("Exercise 1: Basic Watermark")
    # exercise_1_basic_watermark()
    
    # Exercise 2: Late data detection
    # print("\nExercise 2: Late Data Detection")
    # exercise_2_late_data_detection()
    
    # Exercise 3: Watermark aggregation
    # print("\nExercise 3: Watermark Aggregation")
    # exercise_3_watermark_aggregation()
    
    # Exercise 4: State management
    # print("\nExercise 4: State Management")
    # exercise_4_state_management()
    
    # Exercise 5: Join with watermark
    # print("\nExercise 5: Join with Watermark")
    # exercise_5_join_with_watermark()
