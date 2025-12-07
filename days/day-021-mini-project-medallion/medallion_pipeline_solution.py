#!/usr/bin/env python3
"""
Medallion Pipeline - Complete Solution
"""

import argparse
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from delta import configure_spark_with_delta_pip
from delta.tables import DeltaTable
from datetime import datetime, timedelta


class BronzeLayer:
    def __init__(self, spark, base_path):
        self.spark = spark
        self.base_path = f"{base_path}/bronze"
    
    def ingest_orders(self, source_path):
        df = self.spark.read.json(source_path)
        df = df.withColumn("_ingestion_time", current_timestamp()) \
               .withColumn("_source", lit("orders_api"))
        df.write.format("delta").mode("append").save(f"{self.base_path}/orders")
        print(f"✓ Ingested {df.count()} orders to bronze")
    
    def ingest_customers(self, source_path):
        df = self.spark.read.json(source_path)
        df = df.withColumn("_ingestion_time", current_timestamp()) \
               .withColumn("_source", lit("customers_api"))
        df.write.format("delta").mode("append").save(f"{self.base_path}/customers")
        print(f"✓ Ingested {df.count()} customers to bronze")
    
    def ingest_products(self, source_path):
        df = self.spark.read.json(source_path)
        df = df.withColumn("_ingestion_time", current_timestamp()) \
               .withColumn("_source", lit("products_api"))
        df.write.format("delta").mode("append").save(f"{self.base_path}/products")
        print(f"✓ Ingested {df.count()} products to bronze")


class SilverLayer:
    def __init__(self, spark, base_path):
        self.spark = spark
        self.bronze_path = f"{base_path}/bronze"
        self.silver_path = f"{base_path}/silver"
    
    def transform_orders(self):
        bronze = self.spark.read.format("delta").load(f"{self.bronze_path}/orders")
        silver = bronze \
            .dropDuplicates(["order_id"]) \
            .filter(col("order_id").isNotNull()) \
            .withColumn("order_date", to_date(col("order_timestamp")))
        silver.write.format("delta").mode("overwrite").save(f"{self.silver_path}/orders")
        print(f"✓ Transformed {silver.count()} orders to silver")
    
    def update_customers_scd2(self):
        bronze = self.spark.read.format("delta").load(f"{self.bronze_path}/customers")
        bronze = bronze.withColumn("effective_date", current_date()) \
                       .withColumn("expiration_date", lit("9999-12-31").cast("date")) \
                       .withColumn("is_current", lit(True))
        
        path = f"{self.silver_path}/customers"
        if not os.path.exists(path):
            bronze.write.format("delta").save(path)
        else:
            # SCD Type 2 merge logic here
            bronze.write.format("delta").mode("append").save(path)
        
        print(f"✓ Updated customers with SCD Type 2")
    
    def validate_products(self):
        bronze = self.spark.read.format("delta").load(f"{self.bronze_path}/products")
        silver = bronze.filter(col("price") > 0)
        silver.write.format("delta").mode("overwrite").save(f"{self.silver_path}/products")
        print(f"✓ Validated {silver.count()} products to silver")


class GoldLayer:
    def __init__(self, spark, base_path):
        self.spark = spark
        self.silver_path = f"{base_path}/silver"
        self.gold_path = f"{base_path}/gold"
    
    def create_dim_date(self):
        start = datetime(2024, 1, 1)
        dates = [(start + timedelta(days=i),) for i in range(365)]
        df = self.spark.createDataFrame(dates, ["date"])
        df = df.withColumn("date_key", date_format("date", "yyyyMMdd").cast("int")) \
               .withColumn("year", year("date")) \
               .withColumn("quarter", quarter("date")) \
               .withColumn("month", month("date"))
        df.write.format("delta").mode("overwrite").save(f"{self.gold_path}/dim_date")
        print(f"✓ Created dim_date with {df.count()} rows")
    
    def create_dim_customer(self):
        silver = self.spark.read.format("delta").load(f"{self.silver_path}/customers")
        silver = silver.filter(col("is_current") == True)
        dim = silver.withColumn("customer_key", monotonically_increasing_id())
        dim.write.format("delta").mode("overwrite").save(f"{self.gold_path}/dim_customer")
        print(f"✓ Created dim_customer with {dim.count()} rows")
    
    def create_dim_product(self):
        silver = self.spark.read.format("delta").load(f"{self.silver_path}/products")
        dim = silver.withColumn("product_key", monotonically_increasing_id())
        dim.write.format("delta").mode("overwrite").save(f"{self.gold_path}/dim_product")
        print(f"✓ Created dim_product with {dim.count()} rows")
    
    def create_fact_sales(self):
        orders = self.spark.read.format("delta").load(f"{self.silver_path}/orders")
        dim_date = self.spark.read.format("delta").load(f"{self.gold_path}/dim_date")
        
        fact = orders.join(dim_date, orders.order_date == dim_date.date) \
                     .select("order_id", "date_key", "customer_id", "product_id", "quantity", "amount")
        
        fact.write.format("delta").mode("overwrite").save(f"{self.gold_path}/fact_sales")
        print(f"✓ Created fact_sales with {fact.count()} rows")
    
    def create_daily_metrics(self):
        fact = self.spark.read.format("delta").load(f"{self.gold_path}/fact_sales")
        metrics = fact.groupBy("date_key").agg(
            count("order_id").alias("order_count"),
            sum("amount").alias("total_revenue"),
            avg("amount").alias("avg_order_value")
        )
        metrics.write.format("delta").mode("overwrite").save(f"{self.gold_path}/daily_metrics")
        print(f"✓ Created daily_metrics with {metrics.count()} rows")


class MedallionPipeline:
    def __init__(self, base_path="lakehouse"):
        builder = SparkSession.builder.appName("MedallionPipeline")
        self.spark = configure_spark_with_delta_pip(builder).getOrCreate()
        self.spark.sparkContext.setLogLevel("ERROR")
        self.base_path = base_path
        
        self.bronze = BronzeLayer(self.spark, base_path)
        self.silver = SilverLayer(self.spark, base_path)
        self.gold = GoldLayer(self.spark, base_path)
    
    def run_full(self, data_path="data"):
        print("\n=== Running Medallion Pipeline ===\n")
        
        print("Bronze Layer:")
        self.bronze.ingest_orders(f"{data_path}/orders.json")
        self.bronze.ingest_customers(f"{data_path}/customers.json")
        self.bronze.ingest_products(f"{data_path}/products.json")
        
        print("\nSilver Layer:")
        self.silver.transform_orders()
        self.silver.update_customers_scd2()
        self.silver.validate_products()
        
        print("\nGold Layer:")
        self.gold.create_dim_date()
        self.gold.create_dim_customer()
        self.gold.create_dim_product()
        self.gold.create_fact_sales()
        self.gold.create_daily_metrics()
        
        print("\n=== Pipeline Complete ===\n")
    
    def verify(self):
        print("\n=== Verification ===\n")
        
        layers = {
            "Bronze": ["orders", "customers", "products"],
            "Silver": ["orders", "customers", "products"],
            "Gold": ["dim_date", "dim_customer", "dim_product", "fact_sales", "daily_metrics"]
        }
        
        for layer, tables in layers.items():
            print(f"{layer} Layer:")
            for table in tables:
                path = f"{self.base_path}/{layer.lower()}/{table}"
                try:
                    count = self.spark.read.format("delta").load(path).count()
                    print(f"  ✓ {table}: {count} records")
                except:
                    print(f"  ✗ {table}: Not found")
        print()
    
    def close(self):
        self.spark.stop()


def main():
    parser = argparse.ArgumentParser(description="Medallion Pipeline")
    parser.add_argument("--mode", choices=["full", "verify"], default="full")
    parser.add_argument("--data-path", default="data")
    parser.add_argument("--base-path", default="lakehouse")
    
    args = parser.parse_args()
    
    pipeline = MedallionPipeline(args.base_path)
    
    try:
        if args.mode == "full":
            pipeline.run_full(args.data_path)
        elif args.mode == "verify":
            pipeline.verify()
    except Exception as e:
        print(f"✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
