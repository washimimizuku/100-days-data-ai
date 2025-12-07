"""
Day 20: Slowly Changing Dimensions - Exercises
"""

from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

builder = SparkSession.builder.appName("Day20-SCD")
spark = configure_spark_with_delta_pip(builder).getOrCreate()


# Exercise 1: SCD Type 1
def exercise_1():
    """Implement SCD Type 1 (overwrite)"""
    # TODO: Create dim_customer (customer_key, customer_id, name, email)
    # TODO: Insert initial record (1, 'C001', 'Alice', 'old@email.com')
    # TODO: Update email to 'new@email.com' (overwrite)
    # TODO: Verify only new email exists (no history)
    pass


# Exercise 2: SCD Type 2
def exercise_2():
    """Implement SCD Type 2 (new row)"""
    # TODO: Create dim_customer with SCD columns
    #       (customer_key, customer_id, name, address, 
    #        effective_date, expiration_date, is_current)
    
    # TODO: Insert initial record
    #       (1, 'C001', 'Alice', '123 Old St', '2024-01-01', '9999-12-31', TRUE)
    
    # TODO: Update address to '456 New Ave'
    #       - Expire old record (set expiration_date, is_current=FALSE)
    #       - Insert new record with new address
    
    # TODO: Query current records (is_current = TRUE)
    # TODO: Query all history for customer
    pass


# Exercise 3: SCD Type 3
def exercise_3():
    """Implement SCD Type 3 (previous column)"""
    # TODO: Create dim_customer
    #       (customer_key, customer_id, name, 
    #        current_address, previous_address, address_change_date)
    
    # TODO: Insert initial record
    # TODO: Update address (move current to previous, set new current)
    # TODO: Compare with Type 2 (limited vs full history)
    pass


# Exercise 4: Point-in-Time Query
def exercise_4():
    """Query historical dimension state"""
    # TODO: Create fact_orders (order_id, customer_key, order_date, amount)
    # TODO: Create dim_customer with Type 2 (multiple versions)
    
    # TODO: Insert orders on different dates
    # TODO: Query orders with customer info as of order date
    # TODO: Verify correct historical customer data
    pass


# Exercise 5: SCD Merge Function
def exercise_5():
    """Implement generic SCD Type 2 merge"""
    # TODO: Create scd_type2_merge function
    #       Parameters: source_df, target_path, key_cols, compare_cols
    
    # TODO: Implement merge logic
    #       - Match on key columns
    #       - Compare change columns
    #       - Expire changed records
    #       - Insert new versions
    
    # TODO: Test with multiple updates
    # TODO: Handle edge cases (no changes, new records, duplicates)
    pass


if __name__ == "__main__":
    print("Day 20: Slowly Changing Dimensions\n")
    
    print("Exercise 1: SCD Type 1")
    exercise_1()
    
    print("\nExercise 2: SCD Type 2")
    exercise_2()
    
    print("\nExercise 3: SCD Type 3")
    exercise_3()
    
    print("\nExercise 4: Point-in-Time Query")
    exercise_4()
    
    print("\nExercise 5: SCD Merge Function")
    exercise_5()
    
    spark.stop()
