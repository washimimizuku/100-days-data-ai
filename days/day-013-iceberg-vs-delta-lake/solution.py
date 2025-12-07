"""
Day 13: Iceberg vs Delta Lake - Solutions
"""

from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from delta.tables import DeltaTable
import os
import shutil
import time

builder = SparkSession.builder \
    .appName("Day13-IcebergVsDelta") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions,io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.local.type", "hadoop") \
    .config("spark.sql.catalog.local.warehouse", "warehouse")

spark = configure_spark_with_delta_pip(builder).getOrCreate()


def exercise_1():
    """Create same table in both formats and compare"""
    # Iceberg
    spark.sql("DROP TABLE IF EXISTS local.db.products_iceberg")
    spark.sql("""
        CREATE TABLE local.db.products_iceberg (
            id INT, name STRING, price DECIMAL(10,2)
        ) USING iceberg
    """)
    
    products = spark.createDataFrame([
        (i, f"Product {i}", 10.0 * i) for i in range(1, 6)
    ], ["id", "name", "price"])
    
    products.writeTo("local.db.products_iceberg").append()
    
    # Delta
    delta_path = "/tmp/products_delta"
    if os.path.exists(delta_path):
        shutil.rmtree(delta_path)
    
    products.write.format("delta").save(delta_path)
    
    # Compare
    iceberg_count = spark.table("local.db.products_iceberg").count()
    delta_count = spark.read.format("delta").load(delta_path).count()
    
    print(f"Iceberg records: {iceberg_count}")
    print(f"Delta records: {delta_count}")
    print(f"Iceberg metadata: warehouse/db/products_iceberg/metadata/")
    print(f"Delta metadata: {delta_path}/_delta_log/")


def exercise_2():
    """Compare time travel in both formats"""
    # Iceberg
    spark.sql("DROP TABLE IF EXISTS local.db.events_iceberg")
    spark.sql("CREATE TABLE local.db.events_iceberg (id INT, name STRING) USING iceberg")
    
    for i in range(3):
        df = spark.createDataFrame([(i, f"Event {i}")], ["id", "name"])
        df.writeTo("local.db.events_iceberg").append()
        time.sleep(0.5)
    
    # Delta
    delta_path = "/tmp/events_delta"
    if os.path.exists(delta_path):
        shutil.rmtree(delta_path)
    
    for i in range(3):
        df = spark.createDataFrame([(i, f"Event {i}")], ["id", "name"])
        df.write.format("delta").mode("append").save(delta_path)
        time.sleep(0.5)
    
    # Query version 0
    iceberg_snapshots = spark.sql("SELECT snapshot_id FROM local.db.events_iceberg.snapshots ORDER BY committed_at")
    first_snapshot = iceberg_snapshots.first()[0]
    
    iceberg_v0 = spark.read.option("snapshot-id", first_snapshot).table("local.db.events_iceberg")
    delta_v0 = spark.read.option("versionAsOf", 0).format("delta").load(delta_path)
    
    print(f"Iceberg v0 count: {iceberg_v0.count()}")
    print(f"Delta v0 count: {delta_v0.count()}")


def exercise_3():
    """Test schema evolution in both formats"""
    # Iceberg
    spark.sql("DROP TABLE IF EXISTS local.db.users_iceberg")
    spark.sql("CREATE TABLE local.db.users_iceberg (id INT, name STRING) USING iceberg")
    spark.createDataFrame([(1, "Alice")], ["id", "name"]).writeTo("local.db.users_iceberg").append()
    
    spark.sql("ALTER TABLE local.db.users_iceberg ADD COLUMN email STRING")
    print("Iceberg: Added email column")
    
    # Delta
    delta_path = "/tmp/users_delta"
    if os.path.exists(delta_path):
        shutil.rmtree(delta_path)
    
    spark.createDataFrame([(1, "Alice")], ["id", "name"]).write.format("delta").save(delta_path)
    
    new_data = spark.createDataFrame([(2, "Bob", "bob@example.com")], ["id", "name", "email"])
    new_data.write.format("delta").mode("append").option("mergeSchema", "true").save(delta_path)
    print("Delta: Added email column with mergeSchema")
    
    # Compare schemas
    iceberg_schema = spark.table("local.db.users_iceberg").columns
    delta_schema = spark.read.format("delta").load(delta_path).columns
    
    print(f"Iceberg schema: {iceberg_schema}")
    print(f"Delta schema: {delta_schema}")


def exercise_4():
    """Compare partitioning approaches"""
    # Iceberg with hidden partitioning
    spark.sql("DROP TABLE IF EXISTS local.db.orders_iceberg")
    spark.sql("""
        CREATE TABLE local.db.orders_iceberg (
            id INT, amount DECIMAL(10,2), order_date DATE
        ) USING iceberg
        PARTITIONED BY (days(order_date))
    """)
    
    orders = spark.createDataFrame([
        (1, 100.0, "2024-01-15"),
        (2, 200.0, "2024-01-16")
    ], ["id", "amount", "order_date"])
    orders.writeTo("local.db.orders_iceberg").append()
    
    # Query without partition filter (Iceberg handles it)
    result = spark.sql("SELECT * FROM local.db.orders_iceberg WHERE order_date = '2024-01-15'")
    print(f"Iceberg hidden partitioning result: {result.count()}")
    
    # Delta with explicit partitioning
    delta_path = "/tmp/orders_delta"
    if os.path.exists(delta_path):
        shutil.rmtree(delta_path)
    
    orders.write.format("delta").partitionBy("order_date").save(delta_path)
    
    result_delta = spark.read.format("delta").load(delta_path).where("order_date = '2024-01-15'")
    print(f"Delta explicit partitioning result: {result_delta.count()}")


def exercise_5():
    """Compare optimization commands"""
    # Iceberg
    spark.sql("DROP TABLE IF EXISTS local.db.logs_iceberg")
    spark.sql("CREATE TABLE local.db.logs_iceberg (id INT, message STRING) USING iceberg")
    
    for i in range(10):
        spark.createDataFrame([(i, f"Log {i}")], ["id", "message"]).writeTo("local.db.logs_iceberg").append()
    
    files_before = spark.sql("SELECT COUNT(*) as cnt FROM local.db.logs_iceberg.files").first()[0]
    print(f"Iceberg files before: {files_before}")
    
    spark.sql("CALL local.system.rewrite_data_files('db.logs_iceberg')")
    
    files_after = spark.sql("SELECT COUNT(*) as cnt FROM local.db.logs_iceberg.files").first()[0]
    print(f"Iceberg files after: {files_after}")
    
    # Delta
    delta_path = "/tmp/logs_delta"
    if os.path.exists(delta_path):
        shutil.rmtree(delta_path)
    
    for i in range(10):
        spark.createDataFrame([(i, f"Log {i}")], ["id", "message"]).write.format("delta").mode("append").save(delta_path)
    
    delta_files_before = len([f for f in os.listdir(delta_path) if f.endswith('.parquet')])
    print(f"Delta files before: {delta_files_before}")
    
    DeltaTable.forPath(spark, delta_path).optimize().executeCompaction()
    
    delta_files_after = len([f for f in os.listdir(delta_path) if f.endswith('.parquet')])
    print(f"Delta files after: {delta_files_after}")


if __name__ == "__main__":
    print("Day 13: Iceberg vs Delta Lake - Solutions\n")
    
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
