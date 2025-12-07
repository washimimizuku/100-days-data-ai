"""
Day 19: Snowflake Schema - Exercises
"""

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Day19-Snowflake").getOrCreate()


# Exercise 1: Design Snowflake Schema
def exercise_1():
    """Design normalized dimension hierarchies"""
    # TODO: Design normalized customer dimension
    #       dim_customer → dim_city → dim_state → dim_country
    
    # TODO: Design normalized product dimension
    #       dim_product → dim_subcategory → dim_category
    
    # TODO: Define foreign key relationships
    
    # TODO: Document hierarchy levels
    pass


# Exercise 2: Create Normalized Dimensions
def exercise_2():
    """Create normalized dimension tables"""
    # TODO: Create geography hierarchy
    #       dim_country (country_key, name, region)
    #       dim_state (state_key, name, country_key)
    #       dim_city (city_key, name, state_key)
    #       dim_customer (customer_key, name, city_key)
    
    # TODO: Create product hierarchy
    #       dim_category (category_key, name)
    #       dim_subcategory (subcategory_key, name, category_key)
    #       dim_product (product_key, name, subcategory_key)
    
    # TODO: Load sample data
    pass


# Exercise 3: Query Snowflake Schema
def exercise_3():
    """Write queries with multiple joins"""
    # TODO: Query revenue by country (4 joins)
    #       fact_sales → customer → city → state → country
    
    # TODO: Query sales by category (3 joins)
    #       fact_sales → product → subcategory → category
    
    # TODO: Compare with equivalent star schema query
    
    # TODO: Measure execution time difference
    pass


# Exercise 4: Hybrid Schema
def exercise_4():
    """Implement selective normalization"""
    # TODO: Identify large dimensions to normalize
    #       Customer geography (many cities/states)
    
    # TODO: Keep small dimensions denormalized
    #       Date dimension (365 rows)
    
    # TODO: Create hybrid schema
    
    # TODO: Justify normalization decisions
    pass


# Exercise 5: Optimization
def exercise_5():
    """Optimize snowflake schema performance"""
    # TODO: Create materialized view joining geography hierarchy
    #       mv_customer_full (customer_key, name, city, state, country)
    
    # TODO: Add indexes on foreign keys
    
    # TODO: Query using materialized view
    
    # TODO: Compare performance with raw joins
    pass


if __name__ == "__main__":
    print("Day 19: Snowflake Schema\n")
    
    print("Exercise 1: Design Snowflake Schema")
    exercise_1()
    
    print("\nExercise 2: Create Normalized Dimensions")
    exercise_2()
    
    print("\nExercise 3: Query Snowflake Schema")
    exercise_3()
    
    print("\nExercise 4: Hybrid Schema")
    exercise_4()
    
    print("\nExercise 5: Optimization")
    exercise_5()
    
    spark.stop()
