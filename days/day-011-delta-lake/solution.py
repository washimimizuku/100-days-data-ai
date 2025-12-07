"""
Day 11: Delta Lake - Solutions
"""

from pyspark.sql import SparkSession
from delta import *
from delta.tables import DeltaTable
import os
import shutil

builder = SparkSession.builder \
    .appName("Day11-DeltaLake") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

spark = configure_spark_with_delta_pip(builder).getOrCreate()


def exercise_1():
    """Create Delta table and verify"""
    path = "/tmp/employees"
    if os.path.exists(path):
        shutil.rmtree(path)
    
    df = spark.createDataFrame([
        (1, "Alice", 75000),
        (2, "Bob", 85000),
        (3, "Charlie", 65000)
    ], ["id", "name", "salary"])
    
    df.write.format("delta").save(path)
    
    df_read = spark.read.format("delta").load(path)
    print(f"Records: {df_read.count()}")
    df_read.show()


def exercise_2():
    """Perform INSERT, UPDATE, DELETE, MERGE"""
    path = "/tmp/employees"
    delta_table = DeltaTable.forPath(spark, path)
    
    # INSERT
    new_emp = spark.createDataFrame([
        (4, "David", 70000),
        (5, "Eve", 80000)
    ], ["id", "name", "salary"])
    new_emp.write.format("delta").mode("append").save(path)
    print(f"After INSERT: {spark.read.format('delta').load(path).count()}")
    
    # UPDATE
    delta_table.update(
        condition = "id = 1",
        set = {"salary": "salary * 1.1"}
    )
    print("Updated id=1 salary")
    
    # DELETE
    delta_table.delete("id = 3")
    print(f"After DELETE: {spark.read.format('delta').load(path).count()}")
    
    # MERGE
    merge_data = spark.createDataFrame([
        (2, "Bob Updated", 90000),
        (6, "Frank", 72000)
    ], ["id", "name", "salary"])
    
    delta_table.alias("target").merge(
        merge_data.alias("source"),
        "target.id = source.id"
    ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
    
    print(f"After MERGE: {spark.read.format('delta').load(path).count()}")


def exercise_3():
    """Query historical versions"""
    path = "/tmp/products"
    if os.path.exists(path):
        shutil.rmtree(path)
    
    # Version 0
    df1 = spark.createDataFrame([
        (1, "Laptop", 999.99),
        (2, "Mouse", 29.99),
        (3, "Keyboard", 79.99)
    ], ["id", "name", "price"])
    df1.write.format("delta").save(path)
    
    # Version 1
    delta_table = DeltaTable.forPath(spark, path)
    delta_table.update(set = {"price": "price * 1.1"})
    
    # Version 2
    df2 = spark.createDataFrame([
        (4, "Monitor", 299.99),
        (5, "Webcam", 89.99)
    ], ["id", "name", "price"])
    df2.write.format("delta").mode("append").save(path)
    
    # Query versions
    v0 = spark.read.format("delta").option("versionAsOf", 0).load(path)
    v1 = spark.read.format("delta").option("versionAsOf", 1).load(path)
    v2 = spark.read.format("delta").option("versionAsOf", 2).load(path)
    
    print(f"Version 0: {v0.count()} records")
    print(f"Version 1: {v1.count()} records")
    print(f"Version 2: {v2.count()} records")
    
    # History
    delta_table.history().select("version", "operation", "operationMetrics").show(truncate=False)


def exercise_4():
    """Test schema enforcement and evolution"""
    path = "/tmp/customers"
    if os.path.exists(path):
        shutil.rmtree(path)
    
    # Create table
    df = spark.createDataFrame([
        (1, "Alice", "alice@example.com"),
        (2, "Bob", "bob@example.com")
    ], ["id", "name", "email"])
    df.write.format("delta").save(path)
    
    # Try wrong schema (will fail)
    try:
        bad_df = spark.createDataFrame([(3, "Charlie")], ["id", "name"])
        bad_df.write.format("delta").mode("append").save(path)
    except Exception as e:
        print(f"Schema enforcement worked: {type(e).__name__}")
    
    # Schema evolution
    new_df = spark.createDataFrame([
        (3, "Charlie", "charlie@example.com", "555-1234")
    ], ["id", "name", "email", "phone"])
    
    new_df.write.format("delta").mode("append").option("mergeSchema", "true").save(path)
    
    result = spark.read.format("delta").load(path)
    print(f"Schema: {result.columns}")
    print(f"Records: {result.count()}")


def exercise_5():
    """Optimize and vacuum Delta table"""
    path = "/tmp/logs"
    if os.path.exists(path):
        shutil.rmtree(path)
    
    # Create many small files
    for i in range(10):
        df = spark.createDataFrame([(i, f"Log {i}")], ["id", "message"])
        df.write.format("delta").mode("append").save(path)
    
    # Count files before
    files_before = len([f for f in os.listdir(path) if f.endswith('.parquet')])
    print(f"Files before OPTIMIZE: {files_before}")
    
    # OPTIMIZE
    delta_table = DeltaTable.forPath(spark, path)
    delta_table.optimize().executeCompaction()
    
    files_after = len([f for f in os.listdir(path) if f.endswith('.parquet')])
    print(f"Files after OPTIMIZE: {files_after}")
    
    # VACUUM (set retention to 0 for demo)
    spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")
    delta_table.vacuum(0)
    
    files_vacuum = len([f for f in os.listdir(path) if f.endswith('.parquet')])
    print(f"Files after VACUUM: {files_vacuum}")


if __name__ == "__main__":
    print("Day 11: Delta Lake - Solutions\n")
    
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
