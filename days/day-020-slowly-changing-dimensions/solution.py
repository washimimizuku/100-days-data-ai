"""
Day 20: Slowly Changing Dimensions - Solutions
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from delta import configure_spark_with_delta_pip
from delta.tables import DeltaTable
import os
import shutil

builder = SparkSession.builder.appName("Day20-SCD")
spark = configure_spark_with_delta_pip(builder).getOrCreate()


def exercise_1():
    """Implement SCD Type 1 (overwrite)"""
    path = "/tmp/scd/type1_customer"
    if os.path.exists(path):
        shutil.rmtree(path)
    
    # Initial record
    df = spark.createDataFrame([
        (1, "C001", "Alice", "old@email.com")
    ], ["customer_key", "customer_id", "name", "email"])
    
    df.write.format("delta").save(path)
    print("Initial record:")
    spark.read.format("delta").load(path).show()
    
    # Update (overwrite)
    DeltaTable.forPath(spark, path).update(
        condition = "customer_id = 'C001'",
        set = {"email": "'new@email.com'"}
    )
    
    print("After Type 1 update (overwrite):")
    spark.read.format("delta").load(path).show()
    print("History lost - only new email exists")


def exercise_2():
    """Implement SCD Type 2 (new row)"""
    path = "/tmp/scd/type2_customer"
    if os.path.exists(path):
        shutil.rmtree(path)
    
    # Initial record
    df = spark.createDataFrame([
        (1, "C001", "Alice", "123 Old St", "2024-01-01", "9999-12-31", True)
    ], ["customer_key", "customer_id", "name", "address", "effective_date", "expiration_date", "is_current"])
    
    df.write.format("delta").save(path)
    print("Initial record:")
    spark.read.format("delta").load(path).show()
    
    # Update address - Expire old record
    DeltaTable.forPath(spark, path).update(
        condition = "customer_id = 'C001' AND is_current = TRUE",
        set = {
            "expiration_date": "'2024-06-15'",
            "is_current": "FALSE"
        }
    )
    
    # Insert new version
    new_version = spark.createDataFrame([
        (2, "C001", "Alice", "456 New Ave", "2024-06-15", "9999-12-31", True)
    ], ["customer_key", "customer_id", "name", "address", "effective_date", "expiration_date", "is_current"])
    
    new_version.write.format("delta").mode("append").save(path)
    
    print("\nAfter Type 2 update (new row):")
    spark.read.format("delta").load(path).orderBy("effective_date").show()
    
    print("\nCurrent records only:")
    spark.read.format("delta").load(path).where("is_current = TRUE").show()


def exercise_3():
    """Implement SCD Type 3 (previous column)"""
    path = "/tmp/scd/type3_customer"
    if os.path.exists(path):
        shutil.rmtree(path)
    
    # Initial record
    df = spark.createDataFrame([
        (1, "C001", "Alice", "123 Old St", None, None)
    ], ["customer_key", "customer_id", "name", "current_address", "previous_address", "address_change_date"])
    
    df.write.format("delta").save(path)
    print("Initial record:")
    spark.read.format("delta").load(path).show()
    
    # Update address
    DeltaTable.forPath(spark, path).update(
        condition = "customer_id = 'C001'",
        set = {
            "previous_address": "current_address",
            "current_address": "'456 New Ave'",
            "address_change_date": "'2024-06-15'"
        }
    )
    
    print("\nAfter Type 3 update (previous column):")
    spark.read.format("delta").load(path).show()
    print("Limited history - only 1 previous value tracked")


def exercise_4():
    """Query historical dimension state"""
    cust_path = "/tmp/scd/type2_customer"
    orders_path = "/tmp/scd/fact_orders"
    
    if os.path.exists(orders_path):
        shutil.rmtree(orders_path)
    
    # Create orders at different times
    orders = spark.createDataFrame([
        (1, 1, "2024-03-01", 100.0),  # Old address period
        (2, 2, "2024-08-01", 200.0)   # New address period
    ], ["order_id", "customer_key", "order_date", "amount"])
    
    orders.write.format("delta").save(orders_path)
    
    # Point-in-time join
    print("\n=== Point-in-Time Query ===")
    result = spark.read.format("delta").load(orders_path).alias("f") \
        .join(
            spark.read.format("delta").load(cust_path).alias("c"),
            (col("f.customer_key") == col("c.customer_key")) &
            (col("f.order_date") >= col("c.effective_date")) &
            (col("f.order_date") <= col("c.expiration_date"))
        ) \
        .select("f.order_id", "f.order_date", "c.name", "c.address", "f.amount")
    
    result.show()
    print("Each order shows customer address at time of order")


def exercise_5():
    """Implement generic SCD Type 2 merge"""
    def scd_type2_upsert(source_df, target_path, key_col, compare_cols):
        """Generic SCD Type 2 implementation"""
        
        # Add SCD columns to source
        source_df = source_df \
            .withColumn("effective_date", current_date()) \
            .withColumn("expiration_date", lit("9999-12-31").cast("date")) \
            .withColumn("is_current", lit(True))
        
        if not os.path.exists(target_path):
            # First load
            source_df.write.format("delta").save(target_path)
            return
        
        target = DeltaTable.forPath(spark, target_path)
        
        # Build change condition
        change_conditions = [f"target.{col} != source.{col}" for col in compare_cols]
        change_condition = " OR ".join(change_conditions)
        
        # Expire changed records
        target.alias("target").merge(
            source_df.alias("source"),
            f"target.{key_col} = source.{key_col} AND target.is_current = TRUE"
        ).whenMatchedUpdate(
            condition = change_condition,
            set = {
                "expiration_date": "current_date()",
                "is_current": "FALSE"
            }
        ).execute()
        
        # Insert new versions
        source_df.write.format("delta").mode("append").save(target_path)
    
    # Test
    path = "/tmp/scd/generic_customer"
    if os.path.exists(path):
        shutil.rmtree(path)
    
    # Initial load
    df1 = spark.createDataFrame([
        (1, "C001", "Alice", "alice@email.com", "New York")
    ], ["customer_key", "customer_id", "name", "email", "city"])
    
    scd_type2_upsert(df1, path, "customer_id", ["name", "email", "city"])
    print("\n=== Initial Load ===")
    spark.read.format("delta").load(path).show()
    
    # Update
    df2 = spark.createDataFrame([
        (2, "C001", "Alice", "newemail@email.com", "Boston")
    ], ["customer_key", "customer_id", "name", "email", "city"])
    
    scd_type2_upsert(df2, path, "customer_id", ["name", "email", "city"])
    print("\n=== After Update ===")
    spark.read.format("delta").load(path).orderBy("effective_date").show()


if __name__ == "__main__":
    print("Day 20: Slowly Changing Dimensions - Solutions\n")
    
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
