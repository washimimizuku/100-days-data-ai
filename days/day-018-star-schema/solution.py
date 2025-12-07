"""
Day 18: Star Schema - Solutions
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from datetime import datetime, timedelta
import time

spark = SparkSession.builder.appName("Day18-StarSchema").getOrCreate()


def exercise_1():
    """Design star schema for retail business"""
    schema_design = {
        "fact_sales": {
            "grain": "One row per sale transaction",
            "columns": [
                "sale_id BIGINT (PK)",
                "date_key INT (FK)",
                "customer_key INT (FK)",
                "product_key INT (FK)",
                "store_key INT (FK)",
                "quantity INT (measure)",
                "unit_price DECIMAL(10,2) (measure)",
                "total_amount DECIMAL(10,2) (measure)",
                "discount_amount DECIMAL(10,2) (measure)"
            ]
        },
        "dim_date": [
            "date_key INT (PK, surrogate)",
            "date DATE",
            "year INT",
            "quarter INT",
            "month INT",
            "month_name VARCHAR",
            "week INT",
            "day_of_week INT",
            "day_name VARCHAR"
        ],
        "dim_customer": [
            "customer_key INT (PK, surrogate)",
            "customer_id VARCHAR (natural key)",
            "name VARCHAR",
            "email VARCHAR",
            "segment VARCHAR",
            "city VARCHAR",
            "country VARCHAR"
        ],
        "dim_product": [
            "product_key INT (PK, surrogate)",
            "product_id VARCHAR (natural key)",
            "name VARCHAR",
            "category VARCHAR",
            "brand VARCHAR",
            "unit_price DECIMAL(10,2)"
        ],
        "dim_store": [
            "store_key INT (PK, surrogate)",
            "store_id VARCHAR (natural key)",
            "name VARCHAR",
            "city VARCHAR",
            "region VARCHAR"
        ]
    }
    
    print("=== Star Schema Design ===\n")
    for table, columns in schema_design.items():
        print(f"{table.upper()}:")
        if isinstance(columns, dict):
            print(f"  Grain: {columns['grain']}")
            print("  Columns:")
            for col in columns['columns']:
                print(f"    - {col}")
        else:
            for col in columns:
                print(f"  - {col}")
        print()


def exercise_2():
    """Create dimension tables with sample data"""
    # Create dim_date
    start_date = datetime(2024, 1, 1)
    dates = [(start_date + timedelta(days=i),) for i in range(365)]
    
    dim_date = spark.createDataFrame(dates, ["date"]) \
        .withColumn("date_key", date_format("date", "yyyyMMdd").cast("int")) \
        .withColumn("year", year("date")) \
        .withColumn("quarter", quarter("date")) \
        .withColumn("month", month("date")) \
        .withColumn("month_name", date_format("date", "MMMM")) \
        .withColumn("week", weekofyear("date")) \
        .withColumn("day_of_week", dayofweek("date")) \
        .withColumn("day_name", date_format("date", "EEEE"))
    
    print(f"Created dim_date: {dim_date.count()} rows")
    dim_date.show(5)
    
    # Create dim_customer
    customers = [
        (1, "C001", "Alice Smith", "alice@email.com", "Premium", "New York", "USA"),
        (2, "C002", "Bob Jones", "bob@email.com", "Standard", "London", "UK"),
        (3, "C003", "Charlie Brown", "charlie@email.com", "Premium", "Paris", "France"),
    ]
    
    dim_customer = spark.createDataFrame(
        customers,
        ["customer_key", "customer_id", "name", "email", "segment", "city", "country"]
    )
    
    print(f"\nCreated dim_customer: {dim_customer.count()} rows")
    dim_customer.show()
    
    # Create dim_product
    products = [
        (1, "P001", "Laptop", "Electronics", "Dell", 999.99),
        (2, "P002", "Mouse", "Electronics", "Logitech", 29.99),
        (3, "P003", "Desk", "Furniture", "IKEA", 299.99),
    ]
    
    dim_product = spark.createDataFrame(
        products,
        ["product_key", "product_id", "name", "category", "brand", "unit_price"]
    )
    
    print(f"\nCreated dim_product: {dim_product.count()} rows")
    dim_product.show()
    
    return dim_date, dim_customer, dim_product


def exercise_3():
    """Create fact table and load data"""
    dim_date, dim_customer, dim_product = exercise_2()
    
    # Sample sales transactions
    sales = [
        (1, "2024-01-15", "C001", "P001", 2, 999.99, 1999.98, 100.00),
        (2, "2024-01-16", "C002", "P002", 5, 29.99, 149.95, 0.00),
        (3, "2024-01-17", "C003", "P003", 1, 299.99, 299.99, 30.00),
        (4, "2024-01-18", "C001", "P002", 3, 29.99, 89.97, 0.00),
    ]
    
    sales_df = spark.createDataFrame(
        sales,
        ["sale_id", "sale_date", "customer_id", "product_id", "quantity", "unit_price", "total_amount", "discount"]
    )
    
    # Join to get surrogate keys
    fact_sales = sales_df \
        .join(dim_date, sales_df.sale_date == dim_date.date) \
        .join(dim_customer, "customer_id") \
        .join(dim_product, "product_id") \
        .select(
            "sale_id",
            "date_key",
            "customer_key",
            "product_key",
            "quantity",
            sales_df.unit_price,
            "total_amount",
            "discount"
        )
    
    print("\n=== Fact Sales ===")
    fact_sales.show()
    print(f"Total sales: {fact_sales.count()}")
    
    return fact_sales


def exercise_4():
    """Write analytical queries"""
    fact_sales = exercise_3()
    dim_date, dim_customer, dim_product = exercise_2()
    
    # Query 1: Revenue by month
    print("\n=== Revenue by Month ===")
    revenue_by_month = fact_sales \
        .join(dim_date, "date_key") \
        .groupBy("month", "month_name") \
        .agg(sum("total_amount").alias("revenue")) \
        .orderBy("month")
    
    revenue_by_month.show()
    
    # Query 2: Top customers
    print("\n=== Top Customers ===")
    top_customers = fact_sales \
        .join(dim_customer, "customer_key") \
        .groupBy("name") \
        .agg(
            sum("total_amount").alias("total_revenue"),
            count("sale_id").alias("order_count")
        ) \
        .orderBy(desc("total_revenue"))
    
    top_customers.show()
    
    # Query 3: Product performance
    print("\n=== Product Performance by Category ===")
    product_perf = fact_sales \
        .join(dim_product, "product_key") \
        .groupBy("category") \
        .agg(
            count("sale_id").alias("sales_count"),
            sum("quantity").alias("units_sold"),
            sum("total_amount").alias("revenue")
        )
    
    product_perf.show()


def exercise_5():
    """Compare star schema vs normalized schema performance"""
    fact_sales = exercise_3()
    dim_date, dim_customer, dim_product = exercise_2()
    
    # Star schema query
    print("\n=== Star Schema Query ===")
    start = time.time()
    
    star_result = fact_sales \
        .join(dim_customer, "customer_key") \
        .join(dim_product, "product_key") \
        .join(dim_date, "date_key") \
        .groupBy("country", "category", "month") \
        .agg(sum("total_amount").alias("revenue"))
    
    star_count = star_result.count()
    star_time = time.time() - start
    
    print(f"Star schema: {star_count} rows in {star_time:.3f}s")
    print(f"Joins: 3 (fact → customer, product, date)")
    
    # Normalized schema simulation (more joins)
    print("\n=== Normalized Schema (Simulated) ===")
    print("Would require additional joins:")
    print("  - Customer → City → Country (2 extra joins)")
    print("  - Product → Category → Subcategory (2 extra joins)")
    print("  - Total: 7 joins vs 3 joins in star schema")
    print("  - Estimated 2-3x slower query time")


if __name__ == "__main__":
    print("Day 18: Star Schema - Solutions\n")
    
    print("Exercise 1: Design Star Schema")
    exercise_1()
    
    print("\nExercise 2: Create Dimensions")
    exercise_2()
    
    print("\nExercise 3: Create Fact Table")
    exercise_3()
    
    print("\nExercise 4: Query Star Schema")
    exercise_4()
    
    print("\nExercise 5: Performance Comparison")
    exercise_5()
    
    spark.stop()
