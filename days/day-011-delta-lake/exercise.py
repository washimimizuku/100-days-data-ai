"""
Day 11: Delta Lake - Exercises
"""

from pyspark.sql import SparkSession
from delta import *

# Initialize Spark with Delta
builder = SparkSession.builder \
    .appName("Day11-DeltaLake") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

spark = configure_spark_with_delta_pip(builder).getOrCreate()


# Exercise 1: Create Delta Table
def exercise_1():
    """Create Delta table and verify"""
    # TODO: Create DataFrame with 3 employees (id, name, salary)
    # TODO: Write as Delta table to /tmp/employees
    # TODO: Read back and show data
    # TODO: Count records
    pass


# Exercise 2: CRUD Operations
def exercise_2():
    """Perform INSERT, UPDATE, DELETE, MERGE"""
    # TODO: INSERT 2 new employees
    # TODO: UPDATE salary for id=1 (increase by 10%)
    # TODO: DELETE employee with id=3
    # TODO: MERGE new data (update id=2, insert id=4)
    pass


# Exercise 3: Time Travel
def exercise_3():
    """Query historical versions"""
    # TODO: Create products table
    # TODO: Insert version 1 (3 products)
    # TODO: Update version 2 (change prices)
    # TODO: Insert version 3 (add 2 products)
    # TODO: Query version 0, 1, 2
    # TODO: Show history
    pass


# Exercise 4: Schema Management
def exercise_4():
    """Test schema enforcement and evolution"""
    # TODO: Create customers table (id, name, email)
    # TODO: Try to insert data with wrong schema (should fail)
    # TODO: Add new column 'phone' with mergeSchema=true
    # TODO: Verify new schema
    pass


# Exercise 5: Optimization
def exercise_5():
    """Optimize and vacuum Delta table"""
    # TODO: Create logs table
    # TODO: Insert 10 small batches (1 record each)
    # TODO: Count data files before optimization
    # TODO: Run OPTIMIZE
    # TODO: Count data files after optimization
    # TODO: Run VACUUM
    pass


if __name__ == "__main__":
    print("Day 11: Delta Lake\n")
    
    print("Exercise 1: Create Delta Table")
    exercise_1()
    
    print("\nExercise 2: CRUD Operations")
    exercise_2()
    
    print("\nExercise 3: Time Travel")
    exercise_3()
    
    print("\nExercise 4: Schema Management")
    exercise_4()
    
    print("\nExercise 5: Optimization")
    exercise_5()
    
    spark.stop()
