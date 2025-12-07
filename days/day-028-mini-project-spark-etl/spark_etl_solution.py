"""
Day 28: Spark ETL Pipeline - Complete Solution
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum, count, avg, when, lit, current_date, datediff

class SparkETL:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("ETL-Pipeline") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.shuffle.partitions", "50") \
            .getOrCreate()
    
    def extract(self):
        """Extract data from sources"""
        # Create sample data
        orders_data = [
            (1, 101, 501, 2, 29.99, "2024-01-15"),
            (2, 102, 502, 1, 49.99, "2024-01-15"),
            (3, 101, 503, 3, 19.99, "2024-01-16"),
            (4, 103, 501, 1, 29.99, "2024-01-16")
        ]
        orders = self.spark.createDataFrame(orders_data, 
            ["order_id", "customer_id", "product_id", "quantity", "price", "order_date"])
        
        customers_data = [
            (101, "Alice", "alice@example.com", "NYC", "USA"),
            (102, "Bob", "bob@example.com", "LA", "USA"),
            (103, "Charlie", "charlie@example.com", "Chicago", "USA")
        ]
        customers = self.spark.createDataFrame(customers_data,
            ["customer_id", "name", "email", "city", "country"])
        
        products_data = [
            (501, "Widget", "Electronics", 15.00),
            (502, "Gadget", "Electronics", 25.00),
            (503, "Tool", "Hardware", 10.00)
        ]
        products = self.spark.createDataFrame(products_data,
            ["product_id", "name", "category", "cost"])
        
        return orders, customers, products
    
    def transform_clean(self, df):
        """Clean data"""
        return df.dropDuplicates() \
            .filter(col("price") > 0) \
            .filter(col("quantity") > 0)
    
    def transform_enrich(self, orders, customers, products):
        """Enrich orders with customer and product data"""
        enriched = orders.join(customers, "customer_id") \
            .join(products, "product_id") \
            .withColumn("profit", (col("price") - col("cost")) * col("quantity")) \
            .withColumn("revenue", col("price") * col("quantity"))
        
        return enriched.select(
            "order_id", "customer_id", 
            customers["name"].alias("customer_name"),
            "product_id", products["name"].alias("product_name"),
            "category", "quantity", "price", "cost", "profit", "revenue", "order_date"
        )
    
    def transform_aggregate(self, enriched_orders):
        """Create aggregated datasets"""
        # Daily sales summary
        daily_sales = enriched_orders.groupBy("order_date").agg(
            count("order_id").alias("total_orders"),
            _sum("revenue").alias("total_revenue"),
            _sum("profit").alias("total_profit"),
            avg("revenue").alias("avg_order_value")
        )
        
        # Customer analytics
        customer_analytics = enriched_orders.groupBy("customer_id", "customer_name").agg(
            count("order_id").alias("total_orders"),
            _sum("revenue").alias("total_spent")
        ).withColumn("lifetime_value", col("total_spent")) \
         .withColumn("segment",
            when(col("lifetime_value") > 500, "High Value")
            .when(col("lifetime_value") > 100, "Medium Value")
            .otherwise("Low Value")
        )
        
        return daily_sales, customer_analytics
    
    def load(self, df, path, partition_by=None):
        """Load data to target"""
        writer = df.coalesce(1)
        if partition_by:
            writer = writer.repartition(partition_by)
        writer.write.mode("overwrite").parquet(path)
        print(f"Data written to {path}")
    
    def quality_check(self, df, name):
        """Validate data quality"""
        total = df.count()
        null_counts = df.select([
            count(when(col(c).isNull(), c)).alias(c) for c in df.columns
        ])
        
        print(f"\n{name} Quality Report:")
        print(f"Total records: {total}")
        print("Null counts:")
        null_counts.show()
        
        return {"dataset": name, "total_records": total, "status": "PASS"}
    
    def run(self):
        """Execute complete pipeline"""
        print("Starting ETL Pipeline...")
        
        # Extract
        print("\n1. Extracting data...")
        orders, customers, products = self.extract()
        
        # Transform - Clean
        print("\n2. Cleaning data...")
        orders_clean = self.transform_clean(orders)
        
        # Transform - Enrich
        print("\n3. Enriching data...")
        enriched_orders = self.transform_enrich(orders_clean, customers, products)
        enriched_orders.show()
        
        # Transform - Aggregate
        print("\n4. Creating aggregations...")
        daily_sales, customer_analytics = self.transform_aggregate(enriched_orders)
        
        print("\nDaily Sales:")
        daily_sales.show()
        
        print("\nCustomer Analytics:")
        customer_analytics.show()
        
        # Load
        print("\n5. Loading data...")
        self.load(enriched_orders, "/tmp/etl_output/enriched_orders", "order_date")
        self.load(daily_sales, "/tmp/etl_output/daily_sales")
        self.load(customer_analytics, "/tmp/etl_output/customer_analytics")
        
        # Quality checks
        print("\n6. Running quality checks...")
        self.quality_check(enriched_orders, "Enriched Orders")
        self.quality_check(daily_sales, "Daily Sales")
        self.quality_check(customer_analytics, "Customer Analytics")
        
        print("\n✅ ETL Pipeline completed successfully!")
        
        self.spark.stop()

if __name__ == "__main__":
    etl = SparkETL()
    etl.run()
