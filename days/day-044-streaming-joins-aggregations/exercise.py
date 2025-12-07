"""
Day 44: Streaming Joins and Aggregations - Exercises
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *


def exercise_1_stream_static_join():
    """
    Exercise 1: Stream-to-Static Enrichment
    
    Join streaming orders with static product catalog.
    
    Tasks:
    1. Create rate source for orders
    2. Load static products CSV
    3. Join to enrich orders with product info
    4. Write to console
    """
    # TODO: Create SparkSession
    
    # TODO: Create streaming orders (rate source)
    # Add columns: order_id, product_id (value % 5), amount
    
    # TODO: Create static products DataFrame
    # Schema: product_id, product_name, category, price
    
    # TODO: Join streams with static data
    
    # TODO: Write to console
    
    pass


def exercise_2_stream_stream_join():
    """
    Exercise 2: Stream-to-Stream Inner Join
    
    Join order and payment streams.
    
    Tasks:
    1. Create two rate sources (orders and payments)
    2. Add order_id to both
    3. Perform inner join
    4. Write results to console
    """
    # TODO: Create SparkSession
    
    # TODO: Create orders stream
    # Columns: timestamp, order_id, amount
    
    # TODO: Create payments stream
    # Columns: timestamp, order_id, payment_method
    
    # TODO: Join on order_id with time constraint
    
    # TODO: Write to console
    
    pass


def exercise_3_windowed_aggregations():
    """
    Exercise 3: Windowed Aggregations
    
    Calculate metrics per 5-minute window.
    
    Tasks:
    1. Create rate source
    2. Add product_id and amount columns
    3. Aggregate by 5-minute tumbling window
    4. Calculate count, sum, avg
    5. Write to console
    """
    # TODO: Create SparkSession
    
    # TODO: Create rate source with product_id and amount
    
    # TODO: Create 5-minute tumbling window
    
    # TODO: Group by window and product_id
    
    # TODO: Calculate aggregations
    
    # TODO: Write to console
    
    pass


def exercise_4_multiple_aggregations():
    """
    Exercise 4: Multiple Aggregations
    
    Compute various statistics by product and region.
    
    Tasks:
    1. Create rate source
    2. Add product_id, region, amount columns
    3. Group by window, product, region
    4. Calculate: count, sum, avg, min, max, stddev
    5. Write to console
    """
    # TODO: Create SparkSession
    
    # TODO: Create rate source with multiple columns
    
    # TODO: Create 10-minute window
    
    # TODO: Group by window, product_id, region
    
    # TODO: Calculate multiple aggregations
    
    # TODO: Write to console
    
    pass


def exercise_5_time_bounded_join():
    """
    Exercise 5: Time-Bounded Join
    
    Implement join with time constraints.
    
    Tasks:
    1. Create two streams with timestamps
    2. Add watermarks to both
    3. Join with time constraint (within 2 minutes)
    4. Write to console
    """
    # TODO: Create SparkSession
    
    # TODO: Create first stream with watermark
    
    # TODO: Create second stream with watermark
    
    # TODO: Join with time constraint expression
    
    # TODO: Write to console
    
    pass


if __name__ == "__main__":
    print("Day 44: Streaming Joins and Aggregations Exercises\n")
    print("Uncomment exercises to run:\n")
    
    # Exercise 1: Stream-to-static join
    # print("Exercise 1: Stream-to-Static Join")
    # exercise_1_stream_static_join()
    
    # Exercise 2: Stream-to-stream join
    # print("\nExercise 2: Stream-to-Stream Join")
    # exercise_2_stream_stream_join()
    
    # Exercise 3: Windowed aggregations
    # print("\nExercise 3: Windowed Aggregations")
    # exercise_3_windowed_aggregations()
    
    # Exercise 4: Multiple aggregations
    # print("\nExercise 4: Multiple Aggregations")
    # exercise_4_multiple_aggregations()
    
    # Exercise 5: Time-bounded join
    # print("\nExercise 5: Time-Bounded Join")
    # exercise_5_time_bounded_join()
