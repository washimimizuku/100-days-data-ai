"""
Day 49: Real-Time Analytics - Streaming Analytics

Spark Structured Streaming application for real-time e-commerce analytics.
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import config


def create_spark_session():
    """Create Spark session with optimized configuration"""
    return SparkSession.builder \
        .appName("RealTimeAnalytics") \
        .config("spark.sql.shuffle.partitions", config.SHUFFLE_PARTITIONS) \
        .config("spark.streaming.stopGracefullyOnShutdown", "true") \
        .config("spark.sql.streaming.schemaInference", "true") \
        .getOrCreate()


def read_clickstream(spark):
    """Read clickstream from Kafka"""
    clickstream_schema = StructType([
        StructField("event_id", StringType()),
        StructField("user_id", StringType()),
        StructField("session_id", StringType()),
        StructField("product_id", StringType()),
        StructField("category", StringType()),
        StructField("action", StringType()),
        StructField("timestamp", StringType()),
        StructField("page_url", StringType()),
        StructField("referrer", StringType())
    ])
    
    return spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", config.KAFKA_BOOTSTRAP) \
        .option("subscribe", "clickstream") \
        .option("startingOffsets", "latest") \
        .option("maxOffsetsPerTrigger", config.MAX_OFFSETS_PER_TRIGGER) \
        .load() \
        .select(from_json(col("value").cast("string"), clickstream_schema).alias("data")) \
        .select("data.*") \
        .withColumn("timestamp", to_timestamp("timestamp")) \
        .withWatermark("timestamp", config.WATERMARK_DELAY)


def read_transactions(spark):
    """Read transactions from Kafka"""
    transaction_schema = StructType([
        StructField("transaction_id", StringType()),
        StructField("user_id", StringType()),
        StructField("session_id", StringType()),
        StructField("product_id", StringType()),
        StructField("category", StringType()),
        StructField("amount", DoubleType()),
        StructField("quantity", IntegerType()),
        StructField("timestamp", StringType()),
        StructField("payment_method", StringType())
    ])
    
    return spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", config.KAFKA_BOOTSTRAP) \
        .option("subscribe", "transactions") \
        .option("startingOffsets", "latest") \
        .option("maxOffsetsPerTrigger", config.MAX_OFFSETS_PER_TRIGGER) \
        .load() \
        .select(from_json(col("value").cast("string"), transaction_schema).alias("data")) \
        .select("data.*") \
        .withColumn("timestamp", to_timestamp("timestamp")) \
        .withWatermark("timestamp", config.WATERMARK_DELAY)


def calculate_page_views(clicks):
    """Calculate page views per product (1-minute windows)"""
    return clicks.groupBy(
        window("timestamp", "1 minute"),
        "product_id",
        "category"
    ).agg(
        count("*").alias("view_count"),
        countDistinct("user_id").alias("unique_users"),
        countDistinct("session_id").alias("unique_sessions")
    ).select(
        col("window.start").alias("window_start"),
        col("window.end").alias("window_end"),
        "product_id",
        "category",
        "view_count",
        "unique_users",
        "unique_sessions"
    )


def calculate_revenue(transactions):
    """Calculate revenue per category (5-minute windows)"""
    return transactions.groupBy(
        window("timestamp", "5 minutes"),
        "category"
    ).agg(
        count("*").alias("transaction_count"),
        sum("amount").alias("total_revenue"),
        avg("amount").alias("avg_transaction"),
        sum("quantity").alias("total_quantity")
    ).select(
        col("window.start").alias("window_start"),
        col("window.end").alias("window_end"),
        "category",
        "transaction_count",
        round("total_revenue", 2).alias("total_revenue"),
        round("avg_transaction", 2).alias("avg_transaction"),
        "total_quantity"
    )


def calculate_conversions(clicks, transactions):
    """Calculate conversion rate (clicks to purchases)"""
    # Join clicks with transactions within 10-minute window
    joined = clicks.alias("c").join(
        transactions.alias("t"),
        expr("""
            c.user_id = t.user_id AND
            c.product_id = t.product_id AND
            t.timestamp >= c.timestamp AND
            t.timestamp <= c.timestamp + interval 10 minutes
        """),
        "left_outer"
    )
    
    # Calculate metrics per product
    return joined.groupBy(
        window("c.timestamp", "5 minutes"),
        "c.product_id",
        "c.category"
    ).agg(
        count("c.event_id").alias("clicks"),
        count("t.transaction_id").alias("purchases"),
        sum(when(col("t.transaction_id").isNotNull(), col("t.amount")).otherwise(0)).alias("revenue")
    ).select(
        col("window.start").alias("window_start"),
        col("window.end").alias("window_end"),
        col("c.product_id").alias("product_id"),
        col("c.category").alias("category"),
        "clicks",
        "purchases",
        round((col("purchases") / col("clicks")) * 100, 2).alias("conversion_rate_pct"),
        round("revenue", 2).alias("revenue")
    )


def detect_anomalies(transactions):
    """Detect anomalous transactions using z-score"""
    # Calculate running statistics per category
    stats = transactions.groupBy("category").agg(
        avg("amount").alias("mean_amount"),
        stddev("amount").alias("stddev_amount")
    )
    
    # Join with transactions and calculate z-score
    with_stats = transactions.join(stats, "category")
    
    anomalies = with_stats.withColumn(
        "z_score",
        (col("amount") - col("mean_amount")) / col("stddev_amount")
    ).filter(
        abs(col("z_score")) > 3
    ).select(
        "transaction_id",
        "user_id",
        "product_id",
        "category",
        "amount",
        "quantity",
        "timestamp",
        round("z_score", 2).alias("z_score")
    )
    
    return anomalies


def write_to_console(df, query_name, output_mode="append"):
    """Write stream to console for monitoring"""
    return df.writeStream \
        .outputMode(output_mode) \
        .format("console") \
        .option("truncate", False) \
        .queryName(query_name) \
        .trigger(processingTime=config.TRIGGER_INTERVAL) \
        .start()


def write_to_kafka(df, topic, query_name):
    """Write stream to Kafka topic"""
    # Convert to JSON
    json_df = df.select(
        to_json(struct("*")).alias("value")
    )
    
    return json_df.writeStream \
        .outputMode("append") \
        .format("kafka") \
        .option("kafka.bootstrap.servers", config.KAFKA_BOOTSTRAP) \
        .option("topic", topic) \
        .option("checkpointLocation", f"{config.CHECKPOINT_DIR}/{query_name}") \
        .queryName(query_name) \
        .trigger(processingTime=config.TRIGGER_INTERVAL) \
        .start()


def write_to_parquet(df, path, query_name):
    """Write stream to Parquet files"""
    return df.writeStream \
        .outputMode("append") \
        .format("parquet") \
        .option("path", path) \
        .option("checkpointLocation", f"{config.CHECKPOINT_DIR}/{query_name}") \
        .queryName(query_name) \
        .trigger(processingTime=config.TRIGGER_INTERVAL) \
        .start()


def main():
    print("=" * 60)
    print("Real-Time Analytics with Spark + Kafka")
    print("=" * 60)
    print(f"Kafka: {config.KAFKA_BOOTSTRAP}")
    print(f"Trigger Interval: {config.TRIGGER_INTERVAL}")
    print(f"Watermark Delay: {config.WATERMARK_DELAY}")
    print("-" * 60)
    
    # Create Spark session
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    
    # Read streams
    print("Reading streams from Kafka...")
    clicks = read_clickstream(spark)
    transactions = read_transactions(spark)
    
    # Calculate metrics
    print("Setting up analytics queries...")
    
    # 1. Page views
    page_views = calculate_page_views(clicks)
    query1 = write_to_console(page_views, "page_views")
    
    # 2. Revenue
    revenue = calculate_revenue(transactions)
    query2 = write_to_parquet(revenue, f"{config.OUTPUT_DIR}/revenue", "revenue")
    
    # 3. Conversions
    conversions = calculate_conversions(clicks, transactions)
    query3 = write_to_kafka(conversions, "analytics-results", "conversions")
    
    # 4. Anomalies
    anomalies = detect_anomalies(transactions)
    query4 = write_to_console(anomalies, "anomalies")
    
    print("\nQueries started:")
    print("  1. Page Views -> Console")
    print("  2. Revenue -> Parquet")
    print("  3. Conversions -> Kafka (analytics-results)")
    print("  4. Anomalies -> Console")
    print("\nPress Ctrl+C to stop...")
    print("-" * 60)
    
    # Wait for termination
    try:
        spark.streams.awaitAnyTermination()
    except KeyboardInterrupt:
        print("\nStopping queries...")
        for query in spark.streams.active:
            query.stop()
        spark.stop()
        print("Stopped.")


if __name__ == "__main__":
    main()
