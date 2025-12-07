"""
Day 15: Data Warehouse vs Data Lake vs Lakehouse - Solutions
"""


def exercise_1():
    """Compare storage costs and performance for 1TB data"""
    data_size_tb = 1
    
    # Storage costs per TB per month
    warehouse_storage = 23 * data_size_tb  # $23/TB
    lake_storage = 0.023 * 1024 * data_size_tb  # $0.023/GB
    lakehouse_storage = lake_storage * 1.1  # 10% overhead for metadata
    
    # Query performance (relative to warehouse baseline)
    warehouse_perf = 1.0
    lake_perf = 10.0  # 10x slower
    lakehouse_perf = 2.0  # 2x slower
    
    print(f"{'='*60}")
    print(f"Storage Cost Comparison for {data_size_tb}TB")
    print(f"{'='*60}")
    print(f"Warehouse:  ${warehouse_storage:,.2f}/month")
    print(f"Lake:       ${lake_storage:,.2f}/month")
    print(f"Lakehouse:  ${lakehouse_storage:,.2f}/month")
    print(f"\nQuery Performance (relative to warehouse):")
    print(f"Warehouse:  {warehouse_perf}x (baseline)")
    print(f"Lake:       {lake_perf}x slower")
    print(f"Lakehouse:  {lakehouse_perf}x slower")
    print(f"{'='*60}\n")


def exercise_2():
    """Design data models for each architecture"""
    print("=== Warehouse Star Schema ===")
    warehouse_schema = """
    Fact Table: fact_orders
    - order_id (PK)
    - customer_key (FK)
    - product_key (FK)
    - date_key (FK)
    - quantity
    - amount
    
    Dimension Tables:
    - dim_customers (customer_key, customer_id, name, email, segment)
    - dim_products (product_key, product_id, name, category, price)
    - dim_dates (date_key, date, year, month, quarter, day_of_week)
    """
    print(warehouse_schema)
    
    print("\n=== Lake Folder Structure ===")
    lake_structure = """
    s3://datalake/
    ├── raw/
    │   ├── orders/
    │   │   ├── year=2024/
    │   │   │   ├── month=01/
    │   │   │   │   └── orders_20240101.parquet
    │   ├── customers/
    │   │   └── customers_20240101.parquet
    │   └── products/
    │       └── products_20240101.parquet
    ├── processed/
    └── analytics/
    """
    print(lake_structure)
    
    print("\n=== Lakehouse Table Structure ===")
    lakehouse_structure = """
    Delta Tables:
    - orders (partitioned by date, ACID enabled)
    - customers (SCD Type 2, time travel)
    - products (schema evolution enabled)
    
    Features:
    - Schema enforcement
    - ACID transactions
    - Time travel
    - Audit history
    """
    print(lakehouse_structure)


def exercise_3():
    """Write queries for each architecture"""
    print("=== Warehouse SQL Query ===")
    warehouse_query = """
    SELECT 
        c.customer_id,
        c.name,
        SUM(o.amount) as total_spent,
        COUNT(o.order_id) as order_count
    FROM fact_orders o
    JOIN dim_customers c ON o.customer_key = c.customer_key
    WHERE o.date_key >= 20240101
    GROUP BY c.customer_id, c.name
    ORDER BY total_spent DESC
    LIMIT 10;
    """
    print(warehouse_query)
    
    print("\n=== Lake Spark Query ===")
    lake_query = """
    from pyspark.sql.functions import sum, count
    
    # Read Parquet files
    orders = spark.read.parquet("s3://lake/raw/orders/")
    customers = spark.read.parquet("s3://lake/raw/customers/")
    
    # Join and aggregate
    result = orders.join(customers, "customer_id") \\
        .filter(orders.order_date >= "2024-01-01") \\
        .groupBy("customer_id", "name") \\
        .agg(
            sum("amount").alias("total_spent"),
            count("order_id").alias("order_count")
        ) \\
        .orderBy("total_spent", ascending=False) \\
        .limit(10)
    
    result.show()
    """
    print(lake_query)
    
    print("\n=== Lakehouse Delta Query ===")
    lakehouse_query = """
    from delta.tables import DeltaTable
    
    # Read Delta tables (with ACID)
    orders = spark.read.format("delta").load("s3://lakehouse/orders")
    customers = spark.read.format("delta").load("s3://lakehouse/customers")
    
    # Query with time travel
    orders_yesterday = spark.read \\
        .format("delta") \\
        .option("versionAsOf", 5) \\
        .load("s3://lakehouse/orders")
    
    # ACID update
    DeltaTable.forPath(spark, "s3://lakehouse/orders") \\
        .update(
            condition = "status = 'pending'",
            set = {"status": "'shipped'"}
        )
    """
    print(lakehouse_query)


def exercise_4():
    """Plan migration from warehouse to lakehouse"""
    print("=== Migration Plan: Warehouse → Lakehouse ===\n")
    
    steps = [
        "1. Assessment (2 weeks)",
        "   - Inventory warehouse tables",
        "   - Analyze query patterns",
        "   - Identify dependencies",
        "",
        "2. Setup Lakehouse (1 week)",
        "   - Configure S3/ADLS storage",
        "   - Setup Delta Lake/Iceberg",
        "   - Configure access controls",
        "",
        "3. Data Migration (4 weeks)",
        "   - Export warehouse to Parquet",
        "   - Convert to Delta tables",
        "   - Validate data integrity",
        "",
        "4. ETL Migration (3 weeks)",
        "   - Rewrite ETL pipelines",
        "   - Test transformations",
        "   - Parallel run validation",
        "",
        "5. BI Tool Migration (2 weeks)",
        "   - Update connections",
        "   - Migrate dashboards",
        "   - User training",
        "",
        "6. Cutover (1 week)",
        "   - Final sync",
        "   - Switch production",
        "   - Decommission warehouse"
    ]
    
    for step in steps:
        print(step)
    
    print("\n=== Cost Analysis ===")
    print(f"Current Warehouse: $15,000/month")
    print(f"Target Lakehouse:  $3,500/month")
    print(f"Monthly Savings:   $11,500")
    print(f"Migration Cost:    $50,000")
    print(f"ROI Period:        4.3 months")
    
    print("\n=== Risks ===")
    risks = [
        "- Data loss during migration",
        "- Performance degradation",
        "- Query compatibility issues",
        "- Team training requirements",
        "- Downtime during cutover"
    ]
    for risk in risks:
        print(risk)


def exercise_5():
    """Evaluate scenarios and recommend architecture"""
    scenarios = [
        {
            "name": "Financial Services",
            "data": "Structured transactions, 5TB",
            "workload": "BI reporting, compliance",
            "budget": "High",
            "team": "SQL analysts",
            "recommendation": "Data Warehouse",
            "reasoning": "Structured data, compliance needs, SQL team, budget allows",
            "cost": "$5,000/month"
        },
        {
            "name": "Tech Startup",
            "data": "Logs, events, user data, 50TB",
            "workload": "ML, analytics, exploration",
            "budget": "Limited",
            "team": "Data scientists",
            "recommendation": "Data Lake",
            "reasoning": "Cost-sensitive, unstructured data, ML focus, flexible exploration",
            "cost": "$1,200/month"
        },
        {
            "name": "E-commerce",
            "data": "Orders, clickstream, images, 20TB",
            "workload": "BI + ML + real-time",
            "budget": "Medium",
            "team": "Mixed (analysts + DS)",
            "recommendation": "Data Lakehouse",
            "reasoning": "Mixed workloads, ACID needed, cost optimization, future-proof",
            "cost": "$3,000/month"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario['name']}")
        print(f"{'='*60}")
        print(f"Data:        {scenario['data']}")
        print(f"Workload:    {scenario['workload']}")
        print(f"Budget:      {scenario['budget']}")
        print(f"Team:        {scenario['team']}")
        print(f"\n→ Recommendation: {scenario['recommendation']}")
        print(f"  Reasoning: {scenario['reasoning']}")
        print(f"  Est. Cost: {scenario['cost']}")


if __name__ == "__main__":
    print("Day 15: Data Warehouse vs Data Lake vs Lakehouse - Solutions\n")
    
    print("Exercise 1: Architecture Analysis")
    exercise_1()
    
    print("\nExercise 2: Data Modeling")
    exercise_2()
    
    print("\nExercise 3: Query Patterns")
    exercise_3()
    
    print("\nExercise 4: Migration Planning")
    exercise_4()
    
    print("\nExercise 5: Architecture Selection")
    exercise_5()
