"""
Day 50: Checkpoint - Data Engineering Review Exercises

Complete these exercises to assess your understanding of Weeks 1-7.
Score yourself: 1 point per correct implementation.
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from kafka import KafkaProducer, KafkaConsumer
import json
import pandas as pd


# =============================================================================
# SECTION 1: Data Formats (10 points)
# =============================================================================

def exercise_1_1():
    """
    Choose the best format for each scenario:
    A) Real-time event streaming
    B) 100TB analytical dataset
    C) API response payload
    D) ML model training data
    
    Options: CSV, JSON, Parquet, Avro
    """
    # TODO: Return dict with scenario: format
    answers = {
        "A": "",  # Real-time streaming
        "B": "",  # 100TB analytics
        "C": "",  # API response
        "D": "",  # ML training
    }
    return answers


def exercise_1_2():
    """
    Convert CSV to Parquet with compression.
    Input: data.csv (100MB)
    Output: data.parquet with snappy compression
    """
    # TODO: Implement conversion
    # df = pd.read_csv("data.csv")
    # df.to_parquet("data.parquet", compression="snappy")
    pass


def exercise_1_3():
    """
    Explain when to use each compression algorithm:
    - Snappy
    - ZSTD
    - LZ4
    """
    # TODO: Return dict with algorithm: use_case
    use_cases = {
        "snappy": "",
        "zstd": "",
        "lz4": "",
    }
    return use_cases


# =============================================================================
# SECTION 2: Table Formats (10 points)
# =============================================================================

def exercise_2_1():
    """
    List 3 key differences between Iceberg and Delta Lake.
    """
    # TODO: Return list of differences
    differences = [
        "",
        "",
        "",
    ]
    return differences


def exercise_2_2():
    """
    Implement time travel query in Iceberg.
    Query data as of 2 hours ago.
    """
    # TODO: Write Spark SQL query
    query = """
    -- Your query here
    """
    return query


def exercise_2_3():
    """
    Explain ACID properties in Delta Lake.
    """
    # TODO: Return dict with property: explanation
    acid = {
        "Atomicity": "",
        "Consistency": "",
        "Isolation": "",
        "Durability": "",
    }
    return acid


# =============================================================================
# SECTION 3: Architecture (15 points)
# =============================================================================

def exercise_3_1():
    """
    Design medallion architecture for e-commerce data.
    Define what goes in Bronze, Silver, Gold layers.
    """
    # TODO: Return dict with layer: description
    medallion = {
        "bronze": "",
        "silver": "",
        "gold": "",
    }
    return medallion


def exercise_3_2():
    """
    Create star schema for sales analytics.
    Define fact and dimension tables.
    """
    # TODO: Return dict with table_name: columns
    schema = {
        "fact_sales": [],
        "dim_product": [],
        "dim_customer": [],
        "dim_date": [],
    }
    return schema


def exercise_3_3():
    """
    Implement SCD Type 2 logic.
    Handle updates to customer dimension.
    """
    def update_customer(existing_df, new_df):
        # TODO: Implement SCD Type 2
        # Add: effective_date, end_date, is_current
        pass
    
    return update_customer


# =============================================================================
# SECTION 4: Spark (20 points)
# =============================================================================

def exercise_4_1(spark):
    """
    Read Parquet, filter, aggregate, write result.
    
    Task:
    1. Read from "sales.parquet"
    2. Filter: amount > 100
    3. Aggregate: total sales by category
    4. Write to "results.parquet"
    """
    # TODO: Implement Spark pipeline
    pass


def exercise_4_2(spark):
    """
    Optimize this slow query:
    
    df1 = spark.read.parquet("large_table")  # 1TB
    df2 = spark.read.parquet("small_table")  # 1MB
    result = df1.join(df2, "id")
    
    What optimization would you apply?
    """
    # TODO: Return optimized code as string
    optimized = """
    # Your optimized code here
    """
    return optimized


def exercise_4_3(spark):
    """
    Explain what causes a shuffle in Spark.
    List 3 operations that trigger shuffles.
    """
    # TODO: Return list of operations
    shuffle_operations = [
        "",
        "",
        "",
    ]
    return shuffle_operations


def exercise_4_4(spark):
    """
    Implement partitioning strategy.
    Partition sales data by date and region.
    """
    # TODO: Write partitioned output
    # df.write.partitionBy(...).parquet("output")
    pass


# =============================================================================
# SECTION 5: Kafka (20 points)
# =============================================================================

def exercise_5_1():
    """
    Implement Kafka producer with proper configuration.
    
    Requirements:
    - Exactly-once semantics
    - Retries on failure
    - Proper serialization
    """
    # TODO: Create producer with config
    producer = None
    # producer = KafkaProducer(
    #     bootstrap_servers=['localhost:9092'],
    #     ...
    # )
    return producer


def exercise_5_2():
    """
    Implement Kafka consumer with consumer group.
    
    Requirements:
    - Consumer group "analytics"
    - Auto-commit disabled
    - Manual offset management
    """
    # TODO: Create consumer with config
    consumer = None
    return consumer


def exercise_5_3():
    """
    Explain consumer group rebalancing.
    What triggers it? What happens during rebalance?
    """
    # TODO: Return explanation
    explanation = {
        "triggers": [],
        "process": "",
        "impact": "",
    }
    return explanation


def exercise_5_4():
    """
    Design Kafka topic configuration for:
    - High throughput (1M msgs/sec)
    - 7-day retention
    - Exactly-once processing
    """
    # TODO: Return topic config
    config = {
        "partitions": 0,
        "replication_factor": 0,
        "retention_ms": 0,
        "min_insync_replicas": 0,
    }
    return config


# =============================================================================
# SECTION 6: Data Quality (10 points)
# =============================================================================

def exercise_6_1():
    """
    List the 6 dimensions of data quality.
    """
    # TODO: Return list of dimensions
    dimensions = [
        "",
        "",
        "",
        "",
        "",
        "",
    ]
    return dimensions


def exercise_6_2():
    """
    Implement data validation checks using Great Expectations.
    
    Checks:
    - Column 'age' between 0 and 120
    - Column 'email' matches email pattern
    - No nulls in 'user_id'
    """
    # TODO: Write expectation suite
    expectations = """
    # Your Great Expectations code here
    """
    return expectations


def exercise_6_3():
    """
    Design data lineage tracking for a pipeline:
    Raw Data → Cleaned → Aggregated → Report
    
    What metadata would you track?
    """
    # TODO: Return lineage metadata structure
    lineage = {
        "source": {},
        "transformations": [],
        "outputs": {},
    }
    return lineage


# =============================================================================
# SECTION 7: Streaming (15 points)
# =============================================================================

def exercise_7_1(spark):
    """
    Create Spark Structured Streaming query.
    
    Task:
    1. Read from Kafka topic "events"
    2. Parse JSON
    3. Add watermark (5 minutes)
    4. Window aggregation (1 minute)
    5. Write to console
    """
    # TODO: Implement streaming query
    pass


def exercise_7_2(spark):
    """
    Implement stream-to-stream join.
    
    Join clicks and purchases within 10-minute window.
    """
    # TODO: Implement join with time constraint
    pass


def exercise_7_3():
    """
    Explain watermarking in streaming.
    
    Questions:
    1. What is watermarking?
    2. How do you choose watermark delay?
    3. What happens to late data?
    """
    # TODO: Return explanations
    answers = {
        "definition": "",
        "choosing_delay": "",
        "late_data": "",
    }
    return answers


def exercise_7_4(spark):
    """
    Optimize streaming query performance.
    
    Current: Processing 100 events/sec, need 1000 events/sec
    What would you tune?
    """
    # TODO: Return list of optimizations
    optimizations = [
        "",
        "",
        "",
    ]
    return optimizations


# =============================================================================
# SCORING
# =============================================================================

def calculate_score():
    """
    Calculate your total score.
    
    Scoring:
    - Each exercise worth 1-3 points
    - Total: 100 points
    """
    scores = {
        "Data Formats": 0,      # /10
        "Table Formats": 0,     # /10
        "Architecture": 0,      # /15
        "Spark": 0,            # /20
        "Kafka": 0,            # /20
        "Data Quality": 0,     # /10
        "Streaming": 0,        # /15
    }
    
    total = sum(scores.values())
    
    print("=" * 60)
    print("CHECKPOINT ASSESSMENT RESULTS")
    print("=" * 60)
    for category, score in scores.items():
        print(f"{category:20s}: {score:3d} points")
    print("-" * 60)
    print(f"{'TOTAL':20s}: {total:3d} / 100")
    print("=" * 60)
    
    if total >= 90:
        print("🎉 Excellent! Ready for advanced topics.")
    elif total >= 75:
        print("👍 Good! Review weak areas.")
    elif total >= 60:
        print("📚 Fair. Revisit key concepts.")
    else:
        print("⚠️  Needs work. Review weeks thoroughly.")
    
    return total


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Day 50: Checkpoint - Data Engineering Review")
    print("=" * 60)
    print("\nComplete all exercises and score yourself.")
    print("See README.md for detailed instructions.\n")
    
    # Uncomment to run specific sections
    # exercise_1_1()
    # exercise_2_1()
    # etc.
    
    # Calculate final score
    # calculate_score()
