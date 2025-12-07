"""Day 23: Spark DataFrames - Exercises"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder.appName("Day23").getOrCreate()

def exercise_1():
    """Create DataFrames from various sources"""
    # TODO: Create from list of tuples
    # TODO: Create from dict
    # TODO: Create from pandas DataFrame
    # TODO: Print schema for each
    pass

def exercise_2():
    """Select and filter operations"""
    # TODO: Create sample DataFrame
    # TODO: Select name and age columns
    # TODO: Filter age > 25
    # TODO: Chain select and filter
    pass

def exercise_3():
    """Column transformations"""
    # TODO: Add new column (age + 10)
    # TODO: Rename column
    # TODO: Use when/otherwise for categories
    # TODO: Drop column
    pass

def exercise_4():
    """Aggregations and grouping"""
    # TODO: Group by city and count
    # TODO: Calculate avg, max, min age by city
    # TODO: Use window function for ranking
    pass

def exercise_5():
    """SQL queries"""
    # TODO: Create temp view
    # TODO: Write SQL query with WHERE and GROUP BY
    # TODO: Join two DataFrames using SQL
    pass

if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    spark.stop()
