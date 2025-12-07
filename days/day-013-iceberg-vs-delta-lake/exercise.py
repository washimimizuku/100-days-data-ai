"""
Day 13: Iceberg vs Delta Lake - Exercises
"""

from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

# Initialize Spark with both Iceberg and Delta
builder = SparkSession.builder \
    .appName("Day13-IcebergVsDelta") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions,io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.local.type", "hadoop") \
    .config("spark.sql.catalog.local.warehouse", "warehouse")

spark = configure_spark_with_delta_pip(builder).getOrCreate()


# Exercise 1: Feature Comparison
def exercise_1():
    """Create same table in both formats and compare"""
    # TODO: Create Iceberg table 'local.db.products_iceberg'
    # TODO: Create Delta table 'products_delta' at /tmp/products_delta
    # TODO: Insert same data to both (5 products)
    # TODO: Compare metadata structure
    # TODO: Count files in each format
    pass


# Exercise 2: Time Travel
def exercise_2():
    """Compare time travel in both formats"""
    # TODO: Create Iceberg table with 3 versions
    # TODO: Create Delta table with 3 versions
    # TODO: Query version 0 in both
    # TODO: Query by timestamp in both
    # TODO: Compare syntax and results
    pass


# Exercise 3: Schema Evolution
def exercise_3():
    """Test schema evolution in both formats"""
    # TODO: Create Iceberg table (id, name)
    # TODO: Create Delta table (id, name)
    # TODO: Add 'email' column to Iceberg
    # TODO: Add 'email' column to Delta (with mergeSchema)
    # TODO: Compare ease and syntax
    pass


# Exercise 4: Partition Handling
def exercise_4():
    """Compare partitioning approaches"""
    # TODO: Create Iceberg table with hidden partitioning (days(date))
    # TODO: Create Delta table with explicit partitioning (DATE(date))
    # TODO: Query both without partition filters
    # TODO: Compare query patterns
    # TODO: Test Iceberg's automatic partition pruning
    pass


# Exercise 5: Maintenance Operations
def exercise_5():
    """Compare optimization commands"""
    # TODO: Create Iceberg table with 10 small files
    # TODO: Create Delta table with 10 small files
    # TODO: Run Iceberg rewrite_data_files
    # TODO: Run Delta OPTIMIZE
    # TODO: Compare file counts before/after
    # TODO: Compare command syntax
    pass


if __name__ == "__main__":
    print("Day 13: Iceberg vs Delta Lake\n")
    
    print("Exercise 1: Feature Comparison")
    exercise_1()
    
    print("\nExercise 2: Time Travel")
    exercise_2()
    
    print("\nExercise 3: Schema Evolution")
    exercise_3()
    
    print("\nExercise 4: Partition Handling")
    exercise_4()
    
    print("\nExercise 5: Maintenance Operations")
    exercise_5()
    
    spark.stop()
