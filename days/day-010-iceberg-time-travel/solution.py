"""
Day 10: Iceberg Time Travel & Snapshots - Solutions
"""

from pyspark.sql import SparkSession
from datetime import datetime
import time

spark = SparkSession.builder \
    .appName("Day10-Iceberg-TimeTravel") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.local.type", "hadoop") \
    .config("spark.sql.catalog.local.warehouse", "warehouse") \
    .getOrCreate()


def exercise_1():
    """Create table with multiple versions and query historical data"""
    spark.sql("DROP TABLE IF EXISTS local.db.products")
    spark.sql("""
        CREATE TABLE local.db.products (
            id INT, name STRING, price DECIMAL(10,2)
        ) USING iceberg
    """)
    
    # Version 1
    df1 = spark.createDataFrame([(1, "Laptop", 999.99), (2, "Mouse", 29.99), (3, "Keyboard", 79.99)], 
                                 ["id", "name", "price"])
    df1.writeTo("local.db.products").append()
    time.sleep(1)
    
    # Version 2
    df2 = spark.createDataFrame([(4, "Monitor", 299.99), (5, "Webcam", 89.99)], 
                                 ["id", "name", "price"])
    df2.writeTo("local.db.products").append()
    time.sleep(1)
    
    # Version 3 - Update prices
    spark.sql("UPDATE local.db.products SET price = price * 1.1 WHERE id IN (1, 2)")
    
    # Query snapshots
    snapshots = spark.sql("SELECT snapshot_id, committed_at FROM local.db.products.snapshots ORDER BY committed_at")
    snapshot_ids = [row.snapshot_id for row in snapshots.collect()]
    
    print(f"Total snapshots: {len(snapshot_ids)}")
    
    # Query version 1
    df_v1 = spark.read.option("snapshot-id", snapshot_ids[0]).table("local.db.products")
    print(f"Version 1 count: {df_v1.count()}")
    
    # Query version 2
    df_v2 = spark.read.option("snapshot-id", snapshot_ids[1]).table("local.db.products")
    print(f"Version 2 count: {df_v2.count()}")


def exercise_2():
    """Analyze snapshot metadata and history"""
    snapshots = spark.sql("""
        SELECT snapshot_id, parent_id, committed_at, operation
        FROM local.db.products.snapshots
        ORDER BY committed_at
    """)
    print("Snapshots:")
    snapshots.show(truncate=False)
    
    history = spark.sql("SELECT * FROM local.db.products.history")
    print("History:")
    history.show(truncate=False)
    
    # Calculate growth
    snapshot_list = snapshots.collect()
    if len(snapshot_list) >= 2:
        first_count = spark.read.option("snapshot-id", snapshot_list[0].snapshot_id).table("local.db.products").count()
        last_count = spark.table("local.db.products").count()
        print(f"Data growth: {first_count} → {last_count} records ({last_count - first_count} added)")


def exercise_3():
    """Perform rollback and verify results"""
    spark.sql("DROP TABLE IF EXISTS local.db.orders")
    spark.sql("""
        CREATE TABLE local.db.orders (
            id INT, amount DECIMAL(10,2)
        ) USING iceberg
    """)
    
    # Version 1
    spark.createDataFrame([(1, 100.0), (2, 200.0)], ["id", "amount"]).writeTo("local.db.orders").append()
    time.sleep(1)
    
    # Version 2
    spark.createDataFrame([(3, 300.0)], ["id", "amount"]).writeTo("local.db.orders").append()
    time.sleep(1)
    
    # Version 3
    spark.createDataFrame([(4, 400.0)], ["id", "amount"]).writeTo("local.db.orders").append()
    
    print(f"Current count: {spark.table('local.db.orders').count()}")
    
    # Get first snapshot
    first_snapshot = spark.sql("SELECT snapshot_id FROM local.db.orders.snapshots ORDER BY committed_at LIMIT 1").first()[0]
    
    # Rollback
    spark.sql(f"CALL local.system.rollback_to_snapshot('db.orders', {first_snapshot})")
    print(f"After rollback: {spark.table('local.db.orders').count()}")


def exercise_4():
    """Manage snapshots and optimize storage"""
    spark.sql("DROP TABLE IF EXISTS local.db.logs")
    spark.sql("CREATE TABLE local.db.logs (id INT, message STRING) USING iceberg")
    
    # Create 5 snapshots
    for i in range(5):
        spark.createDataFrame([(i, f"Log {i}")], ["id", "message"]).writeTo("local.db.logs").append()
        time.sleep(0.5)
    
    snapshots_before = spark.sql("SELECT COUNT(*) as cnt FROM local.db.logs.snapshots").first()[0]
    print(f"Snapshots before expiration: {snapshots_before}")
    
    # Expire old snapshots, keep last 3
    spark.sql("""
        CALL local.system.expire_snapshots(
            table => 'db.logs',
            older_than => TIMESTAMP '2099-01-01 00:00:00',
            retain_last => 3
        )
    """)
    
    snapshots_after = spark.sql("SELECT COUNT(*) as cnt FROM local.db.logs.snapshots").first()[0]
    print(f"Snapshots after expiration: {snapshots_after}")


def exercise_5():
    """Read and process incremental changes"""
    spark.sql("DROP TABLE IF EXISTS local.db.events")
    spark.sql("CREATE TABLE local.db.events (id INT, event_type STRING) USING iceberg")
    
    # Batch 1
    df1 = spark.createDataFrame([(i, "click") for i in range(10)], ["id", "event_type"])
    df1.writeTo("local.db.events").append()
    
    snapshot_1 = spark.sql("SELECT snapshot_id FROM local.db.events.snapshots ORDER BY committed_at DESC LIMIT 1").first()[0]
    
    # Batch 2
    df2 = spark.createDataFrame([(i, "view") for i in range(10, 20)], ["id", "event_type"])
    df2.writeTo("local.db.events").append()
    
    snapshot_2 = spark.sql("SELECT snapshot_id FROM local.db.events.snapshots ORDER BY committed_at DESC LIMIT 1").first()[0]
    
    # Read incremental
    df_incremental = spark.read \
        .format("iceberg") \
        .option("start-snapshot-id", snapshot_1) \
        .option("end-snapshot-id", snapshot_2) \
        .load("local.db.events")
    
    print(f"Incremental records: {df_incremental.count()}")
    df_incremental.show()


if __name__ == "__main__":
    print("Day 10: Iceberg Time Travel & Snapshots - Solutions\n")
    
    print("Exercise 1: Time Travel Queries")
    exercise_1()
    
    print("\nExercise 2: Snapshot Analysis")
    exercise_2()
    
    print("\nExercise 3: Rollback Operations")
    exercise_3()
    
    print("\nExercise 4: Snapshot Management")
    exercise_4()
    
    print("\nExercise 5: Incremental Processing")
    exercise_5()
    
    spark.stop()
