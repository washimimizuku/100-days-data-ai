"""
Day 15: Data Warehouse vs Data Lake vs Lakehouse - Exercises
"""


# Exercise 1: Architecture Analysis
def exercise_1():
    """Compare storage costs and performance for 1TB data"""
    # TODO: Calculate warehouse storage cost (assume $23/TB/month)
    # TODO: Calculate lake storage cost (assume $0.023/GB/month = $23/TB/month for S3)
    # TODO: Calculate lakehouse storage cost (same as lake + metadata overhead)
    # TODO: Compare query performance (warehouse: 1x, lake: 10x slower, lakehouse: 2x slower)
    # TODO: Print comparison table
    pass


# Exercise 2: Data Modeling
def exercise_2():
    """Design data models for each architecture"""
    # TODO: Design warehouse star schema for e-commerce
    #       - Fact table: orders
    #       - Dimensions: customers, products, dates
    
    # TODO: Design lake folder structure
    #       - /raw/orders/year=2024/month=01/
    #       - /raw/customers/
    #       - /raw/products/
    
    # TODO: Design lakehouse table structure
    #       - Delta tables with partitioning
    #       - Schema enforcement
    pass


# Exercise 3: Query Patterns
def exercise_3():
    """Write queries for each architecture"""
    # TODO: Write warehouse SQL query
    #       SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id
    
    # TODO: Write lake Spark query (reading Parquet)
    #       df = spark.read.parquet("s3://lake/orders")
    #       df.groupBy("customer_id").sum("amount")
    
    # TODO: Write lakehouse Delta query
    #       df = spark.read.format("delta").load("s3://lakehouse/orders")
    #       With ACID update capability
    pass


# Exercise 4: Migration Planning
def exercise_4():
    """Plan migration from warehouse to lakehouse"""
    # TODO: List migration steps
    #       1. Export warehouse data
    #       2. Create Delta tables
    #       3. Migrate ETL
    #       4. Update BI tools
    #       5. Decommission warehouse
    
    # TODO: Estimate costs
    #       - Current warehouse: $15,000/month
    #       - Target lakehouse: $3,500/month
    #       - Migration cost: $50,000
    #       - ROI: 4.3 months
    
    # TODO: Identify risks
    #       - Data loss during migration
    #       - Performance degradation
    #       - Team training needed
    pass


# Exercise 5: Architecture Selection
def exercise_5():
    """Evaluate scenarios and recommend architecture"""
    scenarios = [
        {
            "name": "Financial Services",
            "data": "Structured transactions, 5TB",
            "workload": "BI reporting, compliance",
            "budget": "High",
            "team": "SQL analysts"
        },
        {
            "name": "Tech Startup",
            "data": "Logs, events, user data, 50TB",
            "workload": "ML, analytics, exploration",
            "budget": "Limited",
            "team": "Data scientists"
        },
        {
            "name": "E-commerce",
            "data": "Orders, clickstream, images, 20TB",
            "workload": "BI + ML + real-time",
            "budget": "Medium",
            "team": "Mixed (analysts + DS)"
        }
    ]
    
    # TODO: For each scenario:
    #       - Analyze requirements
    #       - Recommend architecture (warehouse/lake/lakehouse)
    #       - Justify decision
    #       - Estimate costs
    pass


if __name__ == "__main__":
    print("Day 15: Data Warehouse vs Data Lake vs Lakehouse\n")
    
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
