"""
Day 16: Medallion Architecture - Solutions
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from delta import configure_spark_with_delta_pip
from delta.tables import DeltaTable
import os
import shutil

builder = SparkSession.builder.appName("Day16-Medallion")
spark = configure_spark_with_delta_pip(builder).getOrCreate()

BRONZE_PATH = "/tmp/medallion/bronze"
SILVER_PATH = "/tmp/medallion/silver"
GOLD_PATH = "/tmp/medallion/gold"


def exercise_1():
    """Ingest raw data to bronze layer"""
    # Clean up
    if os.path.exists("/tmp/medallion"):
        shutil.rmtree("/tmp/medallion")
    
    # Sample raw data
    raw_data = [
        {"order_id": "1", "customer_id": "C1", "amount": 100.0, "status": "completed"},
        {"order_id": "2", "customer_id": "C2", "amount": 150.0, "status": "pending"},
        {"order_id": "1", "customer_id": "C1", "amount": 100.0, "status": "completed"},  # Duplicate
        {"order_id": None, "customer_id": "C3", "amount": 200.0, "status": "failed"},  # Bad data
    ]
    
    df = spark.createDataFrame(raw_data)
    
    # Add ingestion metadata
    bronze_df = df \
        .withColumn("_ingestion_time", current_timestamp()) \
        .withColumn("_source", lit("api"))
    
    # Write to bronze
    bronze_df.write \
        .format("delta") \
        .mode("append") \
        .save(f"{BRONZE_PATH}/orders")
    
    count = spark.read.format("delta").load(f"{BRONZE_PATH}/orders").count()
    print(f"Bronze records: {count} (includes duplicates and bad data)")


def exercise_2():
    """Transform bronze to silver with data quality"""
    # Read bronze
    bronze_df = spark.read.format("delta").load(f"{BRONZE_PATH}/orders")
    
    # Silver transformations
    silver_df = bronze_df \
        .dropDuplicates(["order_id"]) \
        .filter(col("order_id").isNotNull()) \
        .filter(col("customer_id").isNotNull()) \
        .withColumn("business_date", to_date(col("_ingestion_time"))) \
        .withColumn("amount_usd", col("amount").cast("decimal(10,2)")) \
        .select("order_id", "customer_id", "amount_usd", "status", "business_date")
    
    # Write to silver
    silver_df.write \
        .format("delta") \
        .mode("overwrite") \
        .partitionBy("business_date") \
        .save(f"{SILVER_PATH}/orders")
    
    count = spark.read.format("delta").load(f"{SILVER_PATH}/orders").count()
    print(f"Silver records: {count} (cleaned, deduplicated)")


def exercise_3():
    """Create gold layer aggregations"""
    # Read silver
    silver_df = spark.read.format("delta").load(f"{SILVER_PATH}/orders")
    
    # Gold aggregations
    gold_df = silver_df \
        .groupBy("business_date", "status") \
        .agg(
            count("order_id").alias("order_count"),
            sum("amount_usd").alias("total_amount"),
            avg("amount_usd").alias("avg_amount")
        ) \
        .orderBy("business_date", "status")
    
    # Write to gold
    gold_df.write \
        .format("delta") \
        .mode("overwrite") \
        .partitionBy("business_date") \
        .save(f"{GOLD_PATH}/daily_metrics")
    
    print("Gold metrics:")
    spark.read.format("delta").load(f"{GOLD_PATH}/daily_metrics").show()


def exercise_4():
    """Implement incremental processing"""
    # Simulate new bronze data
    new_data = [
        {"order_id": "3", "customer_id": "C1", "amount": 250.0, "status": "completed"},
        {"order_id": "4", "customer_id": "C2", "amount": 300.0, "status": "completed"},
    ]
    
    new_df = spark.createDataFrame(new_data) \
        .withColumn("_ingestion_time", current_timestamp()) \
        .withColumn("_source", lit("api"))
    
    # Append to bronze
    new_df.write \
        .format("delta") \
        .mode("append") \
        .save(f"{BRONZE_PATH}/orders")
    
    # Incremental silver transformation
    new_silver = new_df \
        .dropDuplicates(["order_id"]) \
        .filter(col("order_id").isNotNull()) \
        .withColumn("business_date", to_date(col("_ingestion_time"))) \
        .withColumn("amount_usd", col("amount").cast("decimal(10,2)")) \
        .select("order_id", "customer_id", "amount_usd", "status", "business_date")
    
    # Merge into silver
    silver_table = DeltaTable.forPath(spark, f"{SILVER_PATH}/orders")
    
    silver_table.alias("target").merge(
        new_silver.alias("source"),
        "target.order_id = source.order_id"
    ).whenMatchedUpdateAll() \
     .whenNotMatchedInsertAll() \
     .execute()
    
    print(f"Silver after incremental: {spark.read.format('delta').load(f'{SILVER_PATH}/orders').count()}")


def exercise_5():
    """Build complete medallion pipeline"""
    class MedallionPipeline:
        def __init__(self, spark, bronze_path, silver_path, gold_path):
            self.spark = spark
            self.bronze_path = bronze_path
            self.silver_path = silver_path
            self.gold_path = gold_path
        
        def ingest_to_bronze(self, data, table_name):
            """Ingest raw data to bronze"""
            try:
                df = self.spark.createDataFrame(data)
                df = df \
                    .withColumn("_ingestion_time", current_timestamp()) \
                    .withColumn("_source", lit("api"))
                
                df.write \
                    .format("delta") \
                    .mode("append") \
                    .save(f"{self.bronze_path}/{table_name}")
                
                print(f"✓ Ingested {df.count()} records to bronze/{table_name}")
            except Exception as e:
                print(f"✗ Bronze ingestion failed: {e}")
        
        def bronze_to_silver(self, table_name):
            """Transform bronze to silver"""
            try:
                bronze_df = self.spark.read \
                    .format("delta") \
                    .load(f"{self.bronze_path}/{table_name}")
                
                silver_df = bronze_df \
                    .dropDuplicates(["order_id"]) \
                    .filter(col("order_id").isNotNull()) \
                    .withColumn("business_date", to_date(col("_ingestion_time"))) \
                    .withColumn("amount_usd", col("amount").cast("decimal(10,2)")) \
                    .select("order_id", "customer_id", "amount_usd", "status", "business_date")
                
                silver_df.write \
                    .format("delta") \
                    .mode("overwrite") \
                    .partitionBy("business_date") \
                    .save(f"{self.silver_path}/{table_name}")
                
                print(f"✓ Transformed {silver_df.count()} records to silver/{table_name}")
            except Exception as e:
                print(f"✗ Silver transformation failed: {e}")
        
        def silver_to_gold(self, table_name, output_name):
            """Aggregate silver to gold"""
            try:
                silver_df = self.spark.read \
                    .format("delta") \
                    .load(f"{self.silver_path}/{table_name}")
                
                gold_df = silver_df \
                    .groupBy("business_date") \
                    .agg(
                        count("order_id").alias("order_count"),
                        sum("amount_usd").alias("total_amount")
                    )
                
                gold_df.write \
                    .format("delta") \
                    .mode("overwrite") \
                    .save(f"{self.gold_path}/{output_name}")
                
                print(f"✓ Created {gold_df.count()} aggregations in gold/{output_name}")
            except Exception as e:
                print(f"✗ Gold aggregation failed: {e}")
    
    # Run pipeline
    pipeline = MedallionPipeline(spark, BRONZE_PATH, SILVER_PATH, GOLD_PATH)
    
    data = [
        {"order_id": "5", "customer_id": "C3", "amount": 400.0, "status": "completed"},
        {"order_id": "6", "customer_id": "C1", "amount": 500.0, "status": "pending"},
    ]
    
    print("\n=== Running Full Pipeline ===")
    pipeline.ingest_to_bronze(data, "orders")
    pipeline.bronze_to_silver("orders")
    pipeline.silver_to_gold("orders", "metrics")


if __name__ == "__main__":
    print("Day 16: Medallion Architecture - Solutions\n")
    
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
