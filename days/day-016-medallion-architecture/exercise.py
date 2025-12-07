"""
Day 16: Medallion Architecture - Exercises
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from delta import configure_spark_with_delta_pip

builder = SparkSession.builder.appName("Day16-Medallion")
spark = configure_spark_with_delta_pip(builder).getOrCreate()


# Exercise 1: Bronze Ingestion
def exercise_1():
    """Ingest raw data to bronze layer"""
    # TODO: Create sample raw data (orders)
    # TODO: Add ingestion metadata (_ingestion_time, _source)
    # TODO: Write to bronze layer (append mode)
    # TODO: Verify data written
    pass


# Exercise 2: Silver Transformation
def exercise_2():
    """Transform bronze to silver with data quality"""
    # TODO: Read bronze data
    # TODO: Remove duplicates
    # TODO: Filter null order_ids
    # TODO: Parse JSON fields
    # TODO: Add business date column
    # TODO: Write to silver layer (overwrite mode)
    pass


# Exercise 3: Gold Aggregation
def exercise_3():
    """Create gold layer aggregations"""
    # TODO: Read silver data
    # TODO: Aggregate by date (count, sum, avg)
    # TODO: Create customer metrics
    # TODO: Write to gold layer
    pass


# Exercise 4: Incremental Processing
def exercise_4():
    """Implement incremental pipeline"""
    # TODO: Read only new bronze records (since last run)
    # TODO: Transform to silver incrementally
    # TODO: Merge into silver table (upsert)
    # TODO: Aggregate to gold incrementally
    pass


# Exercise 5: Full Pipeline
def exercise_5():
    """Build complete medallion pipeline"""
    # TODO: Create MedallionPipeline class
    # TODO: Implement bronze ingestion
    # TODO: Implement silver transformation
    # TODO: Implement gold aggregation
    # TODO: Add error handling
    # TODO: Add logging
    # TODO: Run full pipeline
    pass


if __name__ == "__main__":
    print("Day 16: Medallion Architecture\n")
    
    print("Exercise 1: Bronze Ingestion")
    exercise_1()
    
    print("\nExercise 2: Silver Transformation")
    exercise_2()
    
    print("\nExercise 3: Gold Aggregation")
    exercise_3()
    
    print("\nExercise 4: Incremental Processing")
    exercise_4()
    
    print("\nExercise 5: Full Pipeline")
    exercise_5()
    
    spark.stop()
