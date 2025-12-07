"""
Day 9: Apache Iceberg - Solutions
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_date
from datetime import datetime, timedelta

# Initialize Spark with Iceberg
spark = SparkSession.builder \
    .appName("IcebergDemo") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.local.type", "hadoop") \
    .config("spark.sql.catalog.local.warehouse", "warehouse") \
    .getOrCreate()

# Exercise 1: Create Iceberg Table
print("Exercise 1: Create Iceberg Table")
print("="*50)

# Create database
spark.sql("CREATE DATABASE IF NOT EXISTS local.db")

# Create table
spark.sql("""
    CREATE TABLE IF NOT EXISTS local.db.users (
        id INT,
        name STRING,
        age INT,
        city STRING
    ) USING iceberg
""")

# Insert data
users_data = [
    (1, "Alice", 25, "New York"),
    (2, "Bob", 30, "San Francisco"),
    (3, "Charlie", 28, "Austin"),
    (4, "Diana", 35, "Seattle"),
    (5, "Eve", 32, "Boston")
]

df = spark.createDataFrame(users_data, ["id", "name", "age", "city"])
df.writeTo("local.db.users").append()

print(f"Created table and inserted {df.count()} rows")
spark.sql("SELECT * FROM local.db.users").show()
print()

# Exercise 2: Hidden Partitioning
print("Exercise 2: Hidden Partitioning")
print("="*50)

# Create partitioned table
spark.sql("""
    CREATE TABLE IF NOT EXISTS local.db.events (
        event_id INT,
        user_id INT,
        event_type STRING,
        event_date DATE
    ) USING iceberg
    PARTITIONED BY (days(event_date))
""")

# Insert data with different dates
from datetime import date
events_data = [
    (1, 101, "click", date(2024, 1, 15)),
    (2, 102, "view", date(2024, 1, 16)),
    (3, 103, "purchase", date(2024, 1, 20)),
    (4, 104, "click", date(2024, 2, 5)),
    (5, 105, "view", date(2024, 2, 10))
]

df_events = spark.createDataFrame(events_data, ["event_id", "user_id", "event_type", "event_date"])
df_events.writeTo("local.db.events").append()

# Query without partition filter (Iceberg handles it)
result = spark.sql("""
    SELECT * FROM local.db.events
    WHERE event_date >= '2024-01-15' AND event_date < '2024-02-01'
""")

print("Query results (automatic partition pruning):")
result.show()
print(f"Returned {result.count()} rows from January")
print()

# Exercise 3: Schema Evolution
print("Exercise 3: Schema Evolution")
print("="*50)

# Add new column
spark.sql("""
    ALTER TABLE local.db.users
    ADD COLUMN email STRING
""")
print("Added 'email' column")

# Insert data with new column
new_users = [(6, "Frank", 40, "Chicago", "frank@example.com")]
df_new = spark.createDataFrame(new_users, ["id", "name", "age", "city", "email"])
df_new.writeTo("local.db.users").append()

# Query all data (old rows have null for email)
print("\nAll users (old rows have null email):")
spark.sql("SELECT * FROM local.db.users").show()

# Rename column
spark.sql("""
    ALTER TABLE local.db.users
    RENAME COLUMN age TO user_age
""")
print("Renamed 'age' to 'user_age'")
print()

# Exercise 4: Time Travel
print("Exercise 4: Time Travel")
print("="*50)

# Create new table for time travel demo
spark.sql("""
    CREATE TABLE IF NOT EXISTS local.db.products (
        product_id INT,
        name STRING,
        price DOUBLE
    ) USING iceberg
""")

# Snapshot 1: Initial data
df1 = spark.createDataFrame([
    (1, "Laptop", 999.99),
    (2, "Mouse", 29.99),
    (3, "Keyboard", 79.99)
], ["product_id", "name", "price"])
df1.writeTo("local.db.products").overwrite()
print("Snapshot 1: Initial data")

# Get snapshot ID
snapshots = spark.sql("SELECT snapshot_id FROM local.db.products.snapshots ORDER BY committed_at")
snapshot_1 = snapshots.collect()[0][0]

# Snapshot 2: Update prices
spark.sql("""
    UPDATE local.db.products
    SET price = price * 1.1
    WHERE product_id IN (1, 2)
""")
print("Snapshot 2: Updated prices")

snapshot_2 = spark.sql("SELECT snapshot_id FROM local.db.products.snapshots ORDER BY committed_at DESC").collect()[0][0]

# Snapshot 3: Delete a product
spark.sql("DELETE FROM local.db.products WHERE product_id = 3")
print("Snapshot 3: Deleted product")

# Query different snapshots
print("\nSnapshot 1 (original):")
spark.read.option("snapshot-id", snapshot_1).table("local.db.products").show()

print("Current snapshot:")
spark.sql("SELECT * FROM local.db.products").show()
print()

# Exercise 5: Metadata Exploration
print("Exercise 5: Metadata Exploration")
print("="*50)

# Snapshots
print("Snapshots:")
spark.sql("SELECT snapshot_id, committed_at, operation FROM local.db.products.snapshots").show()

# Files
print("\nData files:")
spark.sql("SELECT file_path, record_count, file_size_in_bytes FROM local.db.products.files").show(truncate=False)

# History
print("\nTable history:")
spark.sql("SELECT made_current_at, snapshot_id, is_current_ancestor FROM local.db.products.history").show()
print()

# Bonus Challenge
print("Bonus Challenge: Partition Evolution")
print("="*50)

# Create table with daily partitions
spark.sql("""
    CREATE TABLE IF NOT EXISTS local.db.logs (
        log_id INT,
        message STRING,
        log_date DATE
    ) USING iceberg
    PARTITIONED BY (days(log_date))
""")

# Insert data
logs_data = [
    (i, f"Log message {i}", date(2024, 1, 1) + timedelta(days=i))
    for i in range(90)
]
df_logs = spark.createDataFrame(logs_data, ["log_id", "message", "log_date"])
df_logs.writeTo("local.db.logs").append()
print("Inserted 90 days of logs with daily partitions")

# Evolve to monthly partitions
spark.sql("""
    ALTER TABLE local.db.logs
    REPLACE PARTITION FIELD days(log_date) WITH months(log_date)
""")
print("Evolved partitioning from daily to monthly")

# Query works across both partition schemes
result = spark.sql("SELECT COUNT(*) as count FROM local.db.logs WHERE log_date >= '2024-01-01'")
print(f"\nQuery result: {result.collect()[0][0]} rows")
print("(Works seamlessly across old daily and new monthly partitions)")

print("\n✅ All exercises completed!")

# Cleanup
spark.stop()
