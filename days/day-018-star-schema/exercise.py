"""
Day 18: Star Schema - Exercises
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder.appName("Day18-StarSchema").getOrCreate()


# Exercise 1: Design Star Schema
def exercise_1():
    """Design star schema for retail business"""
    # TODO: Define fact table structure
    #       fact_sales: sale_id, date_key, customer_key, product_key, store_key
    #                   quantity, unit_price, total_amount, discount
    
    # TODO: Define dimension tables
    #       dim_date: date_key, date, year, quarter, month, week, day
    #       dim_customer: customer_key, customer_id, name, segment, city, country
    #       dim_product: product_key, product_id, name, category, brand, price
    #       dim_store: store_key, store_id, name, city, region
    
    # TODO: Define grain (one row per sale transaction)
    
    # TODO: Identify measures (quantity, amount, discount)
    pass


# Exercise 2: Create Dimensions
def exercise_2():
    """Create dimension tables with sample data"""
    # TODO: Create dim_date with date range
    #       Generate dates for 2024
    #       Add year, quarter, month, week, day_of_week
    
    # TODO: Create dim_customer
    #       customer_key (surrogate), customer_id (natural)
    #       name, email, segment, city, country
    
    # TODO: Create dim_product
    #       product_key (surrogate), product_id (natural)
    #       name, category, brand, unit_price
    
    # TODO: Write dimensions to Delta tables
    pass


# Exercise 3: Create Fact Table
def exercise_3():
    """Create fact table and load data"""
    # TODO: Create sample sales transactions
    #       Include foreign keys to dimensions
    #       Include measures (quantity, price, amount)
    
    # TODO: Join with dimensions to get surrogate keys
    
    # TODO: Create fact_sales table
    
    # TODO: Verify referential integrity
    pass


# Exercise 4: Query Star Schema
def exercise_4():
    """Write analytical queries"""
    # TODO: Query 1 - Revenue by month
    #       SELECT month, SUM(total_amount)
    #       FROM fact_sales JOIN dim_date
    #       GROUP BY month
    
    # TODO: Query 2 - Top 10 customers by revenue
    #       SELECT customer_name, SUM(total_amount)
    #       FROM fact_sales JOIN dim_customer
    #       GROUP BY customer_name
    #       ORDER BY revenue DESC LIMIT 10
    
    # TODO: Query 3 - Product performance by category
    #       SELECT category, COUNT(*), SUM(quantity), SUM(amount)
    #       FROM fact_sales JOIN dim_product
    #       GROUP BY category
    pass


# Exercise 5: Performance Comparison
def exercise_5():
    """Compare star schema vs normalized schema performance"""
    # TODO: Create normalized version (3NF)
    #       Separate city, state, country tables
    
    # TODO: Run same query on both schemas
    
    # TODO: Measure execution time
    
    # TODO: Compare query plans (explain())
    
    # TODO: Analyze join counts and complexity
    pass


if __name__ == "__main__":
    print("Day 18: Star Schema\n")
    
    print("Exercise 1: Design Star Schema")
    exercise_1()
    
    print("\nExercise 2: Create Dimensions")
    exercise_2()
    
    print("\nExercise 3: Create Fact Table")
    exercise_3()
    
    print("\nExercise 4: Query Star Schema")
    exercise_4()
    
    print("\nExercise 5: Performance Comparison")
    exercise_5()
    
    spark.stop()
