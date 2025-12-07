"""
Day 46: Stream-to-Stream Joins - Exercises
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *


def exercise_1_inner_join():
    """
    Exercise 1: Inner Join with Time Constraint
    
    Join orders and payments with 15-minute window.
    
    Tasks:
    1. Create orders stream with watermark
    2. Create payments stream with watermark
    3. Inner join with time constraint
    4. Write to console
    """
    # TODO: Create SparkSession
    
    # TODO: Create orders stream
    # Columns: order_id, amount, order_time
    # Watermark: 10 minutes
    
    # TODO: Create payments stream
    # Columns: order_id, payment_method, payment_time
    # Watermark: 10 minutes
    
    # TODO: Inner join with 15-minute time constraint
    
    # TODO: Write to console
    
    pass


def exercise_2_left_outer_join():
    """
    Exercise 2: Left Outer Join
    
    Find unpaid orders using left outer join.
    
    Tasks:
    1. Create orders and payments streams
    2. Left outer join
    3. Filter for unpaid orders (payment_id is null)
    4. Write to console
    """
    # TODO: Create SparkSession
    
    # TODO: Create orders stream with watermark
    
    # TODO: Create payments stream with watermark
    
    # TODO: Left outer join
    
    # TODO: Filter for unpaid orders
    
    # TODO: Write to console
    
    pass


def exercise_3_full_outer_join():
    """
    Exercise 3: Full Outer Join
    
    Complete reconciliation with full outer join.
    
    Tasks:
    1. Create orders and payments streams
    2. Full outer join
    3. Tag records as: matched, order_only, payment_only
    4. Write to console
    """
    # TODO: Create SparkSession
    
    # TODO: Create orders stream with watermark
    
    # TODO: Create payments stream with watermark
    
    # TODO: Full outer join
    
    # TODO: Add status column (matched/order_only/payment_only)
    
    # TODO: Write to console
    
    pass


def exercise_4_multi_stream_join():
    """
    Exercise 4: Multi-Stream Join
    
    Join three streams: orders, payments, shipments.
    
    Tasks:
    1. Create three streams with watermarks
    2. Join orders + payments
    3. Join result + shipments
    4. Write to console
    """
    # TODO: Create SparkSession
    
    # TODO: Create orders stream
    
    # TODO: Create payments stream
    
    # TODO: Create shipments stream
    
    # TODO: Join orders + payments
    
    # TODO: Join with shipments
    
    # TODO: Write to console
    
    pass


def exercise_5_self_join_duplicates():
    """
    Exercise 5: Self-Join for Duplicates
    
    Detect duplicate events within 1-minute window.
    
    Tasks:
    1. Create events stream
    2. Self-join with time constraint
    3. Exclude self-matches (e1.id != e2.id)
    4. Write duplicates to console
    """
    # TODO: Create SparkSession
    
    # TODO: Create events stream with watermark
    
    # TODO: Self-join (alias as e1 and e2)
    
    # TODO: Add condition to exclude self-matches
    
    # TODO: Write to console
    
    pass


if __name__ == "__main__":
    print("Day 46: Stream-to-Stream Joins Exercises\n")
    print("Uncomment exercises to run:\n")
    
    # Exercise 1: Inner join
    # print("Exercise 1: Inner Join")
    # exercise_1_inner_join()
    
    # Exercise 2: Left outer join
    # print("\nExercise 2: Left Outer Join")
    # exercise_2_left_outer_join()
    
    # Exercise 3: Full outer join
    # print("\nExercise 3: Full Outer Join")
    # exercise_3_full_outer_join()
    
    # Exercise 4: Multi-stream join
    # print("\nExercise 4: Multi-Stream Join")
    # exercise_4_multi_stream_join()
    
    # Exercise 5: Self-join duplicates
    # print("\nExercise 5: Self-Join for Duplicates")
    # exercise_5_self_join_duplicates()
