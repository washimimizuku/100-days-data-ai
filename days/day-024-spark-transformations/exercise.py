"""Day 24: Spark Transformations - Exercises"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder.appName("Day24").getOrCreate()

def exercise_1():
    """Identify narrow vs wide transformations"""
    # TODO: List 5 narrow transformations
    # TODO: List 5 wide transformations
    # TODO: Explain why groupBy requires shuffle
    pass

def exercise_2():
    """Build and optimize transformation chain"""
    # TODO: Create DataFrame
    # TODO: Chain: filter -> select -> groupBy -> orderBy
    # TODO: Reorder to filter early
    # TODO: Compare execution plans
    pass

def exercise_3():
    """Join optimization"""
    # TODO: Create two DataFrames (one large, one small)
    # TODO: Regular join
    # TODO: Broadcast join
    # TODO: Compare execution times
    pass

def exercise_4():
    """Analyze shuffles"""
    # TODO: Build pipeline with multiple operations
    # TODO: Use explain() to identify shuffles
    # TODO: Count shuffle operations
    # TODO: Minimize shuffles
    pass

def exercise_5():
    """Performance tuning"""
    # TODO: Create pipeline used multiple times
    # TODO: Without cache - measure time
    # TODO: With cache - measure time
    # TODO: Compare results
    pass

if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    exercise_5()
    spark.stop()
