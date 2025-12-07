"""
Day 12: Delta Lake ACID Transactions - Solutions
"""

from pyspark.sql import SparkSession
from delta import *
from delta.tables import DeltaTable
import threading
import time
import os
import shutil

builder = SparkSession.builder \
    .appName("Day12-DeltaACID") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

spark = configure_spark_with_delta_pip(builder).getOrCreate()


def exercise_1():
    """Test atomic writes and failure handling"""
    path = "/tmp/orders"
    if os.path.exists(path):
        shutil.rmtree(path)
    
    # Atomic write
    orders = spark.createDataFrame([
        (i, f"Order {i}", 100.0 * i) for i in range(1, 6)
    ], ["id", "description", "amount"])
    
    orders.write.format("delta").save(path)
    count1 = spark.read.format("delta").load(path).count()
    print(f"After successful write: {count1} records")
    
    # Simulate failure with invalid schema
    try:
        bad_data = spark.createDataFrame([(6, "Bad")], ["id", "description"])
        bad_data.write.format("delta").mode("append").save(path)
    except Exception as e:
        print(f"Write failed as expected: {type(e).__name__}")
    
    # Verify no partial data
    count2 = spark.read.format("delta").load(path).count()
    print(f"After failed write: {count2} records (unchanged)")


def exercise_2():
    """Test concurrent readers with consistent snapshots"""
    path = "/tmp/products"
    if os.path.exists(path):
        shutil.rmtree(path)
    
    products = spark.createDataFrame([
        (i, f"Product {i}", 10.0 * i) for i in range(1, 11)
    ], ["id", "name", "price"])
    products.write.format("delta").save(path)
    
    results = []
    
    def reader(reader_id):
        count = spark.read.format("delta").load(path).count()
        results.append((reader_id, count))
        print(f"Reader {reader_id}: {count} records")
    
    threads = [threading.Thread(target=reader, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    counts = [r[1] for r in results]
    print(f"All readers saw same count: {len(set(counts)) == 1}")


def exercise_3():
    """Test concurrent writes and conflict handling"""
    path = "/tmp/inventory"
    if os.path.exists(path):
        shutil.rmtree(path)
    
    inventory = spark.createDataFrame([
        (i, f"Item {i}", 100) for i in range(1, 11)
    ], ["id", "name", "quantity"])
    inventory.write.format("delta").save(path)
    
    def writer1():
        delta_table = DeltaTable.forPath(spark, path)
        delta_table.update(
            condition = "id <= 5",
            set = {"quantity": "quantity + 10"}
        )
        print("Writer 1 completed")
    
    def writer2():
        delta_table = DeltaTable.forPath(spark, path)
        delta_table.update(
            condition = "id > 5",
            set = {"quantity": "quantity + 20"}
        )
        print("Writer 2 completed")
    
    t1 = threading.Thread(target=writer1)
    t2 = threading.Thread(target=writer2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    result = spark.read.format("delta").load(path)
    print(f"Final record count: {result.count()}")


def exercise_4():
    """Test isolation between readers and writers"""
    path = "/tmp/events"
    if os.path.exists(path):
        shutil.rmtree(path)
    
    events = spark.createDataFrame([
        (i, f"Event {i}") for i in range(1, 6)
    ], ["id", "name"])
    events.write.format("delta").save(path)
    
    reader_counts = []
    stop_reading = False
    
    def reader():
        while not stop_reading:
            count = spark.read.format("delta").load(path).count()
            reader_counts.append(count)
            time.sleep(0.1)
    
    def writer():
        time.sleep(0.2)
        new_events = spark.createDataFrame([
            (i, f"Event {i}") for i in range(6, 11)
        ], ["id", "name"])
        new_events.write.format("delta").mode("append").save(path)
        print("Writer added 5 records")
    
    reader_thread = threading.Thread(target=reader)
    writer_thread = threading.Thread(target=writer)
    
    reader_thread.start()
    writer_thread.start()
    writer_thread.join()
    time.sleep(0.5)
    stop_reading = True
    reader_thread.join()
    
    print(f"Reader saw counts: {set(reader_counts)}")
    print(f"Consistent snapshots: {all(c in [5, 10] for c in reader_counts)}")


def exercise_5():
    """Test recovery from transaction failures"""
    path = "/tmp/logs"
    if os.path.exists(path):
        shutil.rmtree(path)
    
    # Batch 1 - success
    batch1 = spark.createDataFrame([
        (i, f"Log {i}", "INFO") for i in range(1, 4)
    ], ["id", "message", "level"])
    batch1.write.format("delta").save(path)
    print(f"After batch 1: {spark.read.format('delta').load(path).count()} records")
    
    # Batch 2 - fail (wrong schema)
    try:
        batch2 = spark.createDataFrame([
            (i, f"Log {i}") for i in range(4, 7)
        ], ["id", "message"])
        batch2.write.format("delta").mode("append").save(path)
    except Exception as e:
        print(f"Batch 2 failed: {type(e).__name__}")
    
    print(f"After failed batch 2: {spark.read.format('delta').load(path).count()} records")
    
    # Batch 3 - success
    batch3 = spark.createDataFrame([
        (i, f"Log {i}", "WARN") for i in range(7, 10)
    ], ["id", "message", "level"])
    batch3.write.format("delta").mode("append").save(path)
    print(f"After batch 3: {spark.read.format('delta').load(path).count()} records")


if __name__ == "__main__":
    print("Day 12: Delta Lake ACID Transactions - Solutions\n")
    
    print("Exercise 1: Atomic Operations")
    exercise_1()
    
    print("\nExercise 2: Concurrent Reads")
    exercise_2()
    
    print("\nExercise 3: Concurrent Writes")
    exercise_3()
    
    print("\nExercise 4: Transaction Isolation")
    exercise_4()
    
    print("\nExercise 5: Failure Recovery")
    exercise_5()
    
    spark.stop()
