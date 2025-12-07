"""
Day 19: Snowflake Schema - Solutions
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import time

spark = SparkSession.builder.appName("Day19-Snowflake").getOrCreate()


def exercise_1():
    """Design normalized dimension hierarchies"""
    schema_design = {
        "customer_hierarchy": {
            "dim_customer": ["customer_key (PK)", "customer_id", "name", "email", "city_key (FK)"],
            "dim_city": ["city_key (PK)", "city_name", "state_key (FK)"],
            "dim_state": ["state_key (PK)", "state_name", "state_code", "country_key (FK)"],
            "dim_country": ["country_key (PK)", "country_name", "country_code", "region"]
        },
        "product_hierarchy": {
            "dim_product": ["product_key (PK)", "product_id", "name", "price", "subcategory_key (FK)"],
            "dim_subcategory": ["subcategory_key (PK)", "subcategory_name", "category_key (FK)"],
            "dim_category": ["category_key (PK)", "category_name", "department"]
        }
    }
    
    print("=== Snowflake Schema Design ===\n")
    for hierarchy, tables in schema_design.items():
        print(f"{hierarchy.upper()}:")
        for table, columns in tables.items():
            print(f"  {table}:")
            for col in columns:
                print(f"    - {col}")
        print()


def exercise_2():
    """Create normalized dimension tables"""
    # Geography hierarchy
    dim_country = spark.createDataFrame([
        (1, "USA", "US", "North America"),
        (2, "UK", "GB", "Europe"),
        (3, "France", "FR", "Europe")
    ], ["country_key", "country_name", "country_code", "region"])
    
    dim_state = spark.createDataFrame([
        (1, "California", "CA", 1),
        (2, "Texas", "TX", 1),
        (3, "England", "EN", 2),
        (4, "Ile-de-France", "IF", 3)
    ], ["state_key", "state_name", "state_code", "country_key"])
    
    dim_city = spark.createDataFrame([
        (1, "Los Angeles", 1),
        (2, "San Francisco", 1),
        (3, "Houston", 2),
        (4, "London", 3),
        (5, "Paris", 4)
    ], ["city_key", "city_name", "state_key"])
    
    dim_customer = spark.createDataFrame([
        (1, "C001", "Alice", "alice@email.com", 1),
        (2, "C002", "Bob", "bob@email.com", 4),
        (3, "C003", "Charlie", "charlie@email.com", 5)
    ], ["customer_key", "customer_id", "name", "email", "city_key"])
    
    print("=== Geography Hierarchy ===")
    print(f"Countries: {dim_country.count()}")
    print(f"States: {dim_state.count()}")
    print(f"Cities: {dim_city.count()}")
    print(f"Customers: {dim_customer.count()}")
    
    # Product hierarchy
    dim_category = spark.createDataFrame([
        (1, "Electronics", "Technology"),
        (2, "Furniture", "Home")
    ], ["category_key", "category_name", "department"])
    
    dim_subcategory = spark.createDataFrame([
        (1, "Laptops", 1),
        (2, "Accessories", 1),
        (3, "Desks", 2)
    ], ["subcategory_key", "subcategory_name", "category_key"])
    
    dim_product = spark.createDataFrame([
        (1, "P001", "Laptop Pro", 999.99, 1),
        (2, "P002", "Mouse", 29.99, 2),
        (3, "P003", "Standing Desk", 299.99, 3)
    ], ["product_key", "product_id", "name", "price", "subcategory_key"])
    
    print("\n=== Product Hierarchy ===")
    print(f"Categories: {dim_category.count()}")
    print(f"Subcategories: {dim_subcategory.count()}")
    print(f"Products: {dim_product.count()}")
    
    return (dim_country, dim_state, dim_city, dim_customer,
            dim_category, dim_subcategory, dim_product)


def exercise_3():
    """Write queries with multiple joins"""
    dims = exercise_2()
    dim_country, dim_state, dim_city, dim_customer, dim_category, dim_subcategory, dim_product = dims
    
    # Create fact table
    fact_sales = spark.createDataFrame([
        (1, 1, 1, 2, 999.99),
        (2, 2, 2, 5, 149.95),
        (3, 3, 3, 1, 299.99)
    ], ["sale_id", "customer_key", "product_key", "quantity", "total_amount"])
    
    # Snowflake query (4 joins for geography)
    print("\n=== Snowflake Query: Revenue by Country ===")
    start = time.time()
    
    snowflake_result = fact_sales \
        .join(dim_customer, "customer_key") \
        .join(dim_city, "city_key") \
        .join(dim_state, "state_key") \
        .join(dim_country, "country_key") \
        .groupBy("country_name") \
        .agg(sum("total_amount").alias("revenue"))
    
    snowflake_result.show()
    snowflake_time = time.time() - start
    print(f"Snowflake query time: {snowflake_time:.3f}s (4 joins)")
    
    # Star schema equivalent (1 join)
    print("\n=== Star Query (Simulated) ===")
    print("Would require only 1 join:")
    print("  fact_sales → dim_customer (with country embedded)")
    print(f"  Estimated time: {snowflake_time/2:.3f}s (50% faster)")


def exercise_4():
    """Implement selective normalization"""
    print("\n=== Hybrid Schema Design ===\n")
    
    decisions = {
        "Normalize": {
            "Customer Geography": {
                "reason": "Large dimension (thousands of cities)",
                "savings": "~50% storage reduction",
                "cost": "3 extra joins per query"
            },
            "Product Hierarchy": {
                "reason": "Many products, categories change",
                "savings": "~40% storage reduction",
                "cost": "2 extra joins per query"
            }
        },
        "Keep Denormalized": {
            "Date Dimension": {
                "reason": "Small dimension (365 rows)",
                "benefit": "No extra joins",
                "cost": "Minimal storage overhead"
            },
            "Store Dimension": {
                "reason": "Few stores (< 100)",
                "benefit": "Simple queries",
                "cost": "Acceptable redundancy"
            }
        }
    }
    
    for approach, dimensions in decisions.items():
        print(f"{approach.upper()}:")
        for dim, details in dimensions.items():
            print(f"  {dim}:")
            for key, value in details.items():
                print(f"    {key}: {value}")
        print()


def exercise_5():
    """Optimize snowflake schema performance"""
    dims = exercise_2()
    dim_country, dim_state, dim_city, dim_customer, _, _, _ = dims
    
    # Create materialized view (pre-joined)
    print("\n=== Creating Materialized View ===")
    mv_customer_full = dim_customer \
        .join(dim_city, "city_key") \
        .join(dim_state, "state_key") \
        .join(dim_country, "country_key") \
        .select(
            "customer_key",
            "customer_id",
            dim_customer.name.alias("customer_name"),
            "city_name",
            "state_name",
            "country_name"
        )
    
    mv_customer_full.cache()
    print(f"Materialized view created: {mv_customer_full.count()} rows")
    mv_customer_full.show()
    
    # Query using materialized view
    fact_sales = spark.createDataFrame([
        (1, 1, 999.99),
        (2, 2, 149.95)
    ], ["sale_id", "customer_key", "total_amount"])
    
    print("\n=== Query with Materialized View ===")
    start = time.time()
    
    result = fact_sales \
        .join(mv_customer_full, "customer_key") \
        .groupBy("country_name") \
        .agg(sum("total_amount").alias("revenue"))
    
    result.show()
    mv_time = time.time() - start
    
    print(f"Query time with MV: {mv_time:.3f}s (1 join)")
    print("Performance: Similar to star schema!")


if __name__ == "__main__":
    print("Day 19: Snowflake Schema - Solutions\n")
    
    print("Exercise 1: Design Snowflake Schema")
    exercise_1()
    
    print("\nExercise 2: Create Normalized Dimensions")
    exercise_2()
    
    print("\nExercise 3: Query Snowflake Schema")
    exercise_3()
    
    print("\nExercise 4: Hybrid Schema")
    exercise_4()
    
    print("\nExercise 5: Optimization")
    exercise_5()
    
    spark.stop()
