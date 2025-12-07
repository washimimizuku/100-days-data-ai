# Day 16: Medallion Architecture (Bronze/Silver/Gold)

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of today, you will:
- Understand the Bronze/Silver/Gold layering pattern
- Design multi-hop data pipelines
- Implement data quality progression
- Apply medallion architecture to real projects
- Understand incremental processing patterns

---

## Theory

### What is Medallion Architecture?

A **multi-layered data architecture** that progressively improves data quality through Bronze → Silver → Gold layers.

**Created by**: Databricks
**Purpose**: Organize data lakehouse with clear quality zones
**Pattern**: Raw → Cleaned → Aggregated

```
Sources → Bronze (Raw) → Silver (Cleaned) → Gold (Curated) → Analytics/ML
```

---

### The Three Layers

#### 🥉 Bronze Layer (Raw)

**Purpose**: Ingest raw data exactly as received

**Characteristics**:
- Exact copy of source data
- No transformations
- All records preserved (including bad data)
- Append-only
- Audit trail maintained

**Data Format**: Delta/Iceberg tables or Parquet files

**Example**:
```python
# Bronze ingestion
raw_df = spark.read.json("s3://source/events/")
raw_df.write \
    .format("delta") \
    .mode("append") \
    .option("mergeSchema", "true") \
    .save("s3://lakehouse/bronze/events")
```

**Schema**:
```
bronze_events:
- event_id
- event_type
- user_id
- timestamp
- payload (JSON)
- _ingestion_time
- _source_file
```

---

#### 🥈 Silver Layer (Cleaned)

**Purpose**: Clean, validate, and enrich data

**Characteristics**:
- Data quality rules applied
- Deduplication
- Schema enforcement
- Type conversions
- Business logic applied
- Bad records filtered/quarantined

**Transformations**:
- Parse JSON fields
- Remove duplicates
- Fix data types
- Validate business rules
- Enrich with lookups

**Example**:
```python
# Silver transformation
bronze_df = spark.read.format("delta").load("s3://lakehouse/bronze/events")

silver_df = bronze_df \
    .dropDuplicates(["event_id"]) \
    .filter(col("user_id").isNotNull()) \
    .withColumn("event_date", to_date("timestamp")) \
    .withColumn("payload_parsed", from_json("payload", schema)) \
    .select("event_id", "event_type", "user_id", "event_date", "payload_parsed.*")

silver_df.write \
    .format("delta") \
    .mode("overwrite") \
    .partitionBy("event_date") \
    .save("s3://lakehouse/silver/events")
```

**Schema**:
```
silver_events:
- event_id (deduplicated)
- event_type (validated)
- user_id (not null)
- event_date (partitioned)
- product_id
- amount
- _quality_score
```

---

#### 🥇 Gold Layer (Curated)

**Purpose**: Business-level aggregations and features

**Characteristics**:
- Aggregated metrics
- Business KPIs
- Feature tables for ML
- Optimized for consumption
- Denormalized for performance

**Transformations**:
- Aggregations (daily, weekly, monthly)
- Joins across domains
- Feature engineering
- Dimension modeling

**Example**:
```python
# Gold aggregation
silver_df = spark.read.format("delta").load("s3://lakehouse/silver/events")

gold_df = silver_df \
    .groupBy("user_id", "event_date") \
    .agg(
        count("event_id").alias("event_count"),
        sum("amount").alias("total_amount"),
        countDistinct("product_id").alias("unique_products")
    )

gold_df.write \
    .format("delta") \
    .mode("overwrite") \
    .partitionBy("event_date") \
    .save("s3://lakehouse/gold/user_daily_metrics")
```

**Schema**:
```
gold_user_daily_metrics:
- user_id
- event_date
- event_count
- total_amount
- unique_products
- avg_order_value
```

---

### Layer Comparison

| Aspect | Bronze | Silver | Gold |
|--------|--------|--------|------|
| **Purpose** | Raw ingestion | Cleaned data | Business metrics |
| **Quality** | Low (as-is) | Medium (validated) | High (curated) |
| **Schema** | Flexible | Enforced | Optimized |
| **Duplicates** | Possible | Removed | N/A |
| **Nulls** | Allowed | Handled | Minimal |
| **Partitioning** | Optional | Recommended | Required |
| **Updates** | Append | Merge/Overwrite | Overwrite |
| **Consumers** | Data engineers | Analysts | BI/ML |

---

### Data Flow Example

**E-commerce Order Pipeline**:

```
1. Bronze (Raw Orders)
   - Ingest from API/database
   - Store as-is with metadata
   - 1M records, 500MB

2. Silver (Cleaned Orders)
   - Remove duplicates (50K duplicates)
   - Validate order_id, customer_id
   - Parse JSON fields
   - Enrich with customer data
   - 950K records, 400MB

3. Gold (Order Metrics)
   - Daily order counts by category
   - Customer lifetime value
   - Product performance metrics
   - 10K aggregated records, 5MB
```

---

### Incremental Processing

**Bronze → Silver**:
```python
# Read new bronze records since last run
bronze_df = spark.read \
    .format("delta") \
    .option("readChangeFeed", "true") \
    .option("startingVersion", last_version) \
    .load("s3://lakehouse/bronze/events")

# Transform and merge into silver
silver_df = transform_bronze_to_silver(bronze_df)

silver_df.write \
    .format("delta") \
    .mode("append") \
    .save("s3://lakehouse/silver/events")
```

**Silver → Gold**:
```python
# Incremental aggregation
from delta.tables import DeltaTable

# Read new silver records
new_silver = spark.read \
    .format("delta") \
    .load("s3://lakehouse/silver/events") \
    .filter(col("event_date") == current_date)

# Aggregate
new_gold = new_silver.groupBy("user_id", "event_date").agg(...)

# Merge into gold
gold_table = DeltaTable.forPath(spark, "s3://lakehouse/gold/user_metrics")

gold_table.alias("target").merge(
    new_gold.alias("source"),
    "target.user_id = source.user_id AND target.event_date = source.event_date"
).whenMatchedUpdateAll() \
 .whenNotMatchedInsertAll() \
 .execute()
```

---

### Best Practices

1. **Bronze Layer**
   - Keep everything (audit trail)
   - Add ingestion metadata
   - Use schema evolution
   - Partition by ingestion date

2. **Silver Layer**
   - Enforce data quality
   - Document transformations
   - Quarantine bad records
   - Partition by business date

3. **Gold Layer**
   - Optimize for queries
   - Denormalize when needed
   - Cache frequently used
   - Document business logic

4. **General**
   - Use Delta/Iceberg for ACID
   - Implement idempotency
   - Monitor data quality
   - Version control pipelines

---

### Pipeline Implementation

```python
class MedallionPipeline:
    def ingest_to_bronze(self, source_path, table_name):
        df = spark.read.json(source_path)
        df.withColumn("_ingestion_time", current_timestamp()) \
          .write.format("delta").mode("append").save(f"bronze/{table_name}")
    
    def bronze_to_silver(self, table_name):
        bronze_df = spark.read.format("delta").load(f"bronze/{table_name}")
        bronze_df.dropDuplicates().filter(col("id").isNotNull()) \
          .write.format("delta").mode("overwrite").save(f"silver/{table_name}")
    
    def silver_to_gold(self, table_name):
        silver_df = spark.read.format("delta").load(f"silver/{table_name}")
        silver_df.groupBy("date").agg(count("*")) \
          .write.format("delta").mode("overwrite").save(f"gold/{table_name}_metrics")
```

---

### Variations

**4-Layer (Bronze/Silver/Gold/Platinum)**:
- Platinum: ML features, real-time aggregations

**5-Layer (Raw/Bronze/Silver/Gold/Presentation)**:
- Raw: Immutable source files
- Presentation: BI-specific views

**Domain-Specific**:
- Bronze: `bronze/sales/`, `bronze/marketing/`
- Silver: `silver/sales/`, `silver/marketing/`
- Gold: `gold/customer_360/`

---

## 💻 Exercises (40 min)

Open `exercise.py` and complete the tasks.

### Exercise 1: Bronze Ingestion
- Read raw JSON data
- Add ingestion metadata
- Write to bronze layer

### Exercise 2: Silver Transformation
- Read bronze data
- Apply data quality rules
- Write to silver layer

### Exercise 3: Gold Aggregation
- Read silver data
- Create business metrics
- Write to gold layer

### Exercise 4: Incremental Processing
- Implement incremental bronze→silver
- Implement incremental silver→gold
- Handle late-arriving data

### Exercise 5: Full Pipeline
- Build end-to-end medallion pipeline
- Add error handling
- Implement monitoring

---

## ✅ Quiz (5 min)

Answer these questions in `quiz.md`:

1. What are the three layers in medallion architecture?
2. What is stored in the bronze layer?
3. What transformations happen in silver?
4. What is the purpose of the gold layer?
5. Should bronze contain duplicates?
6. How do you handle bad data in silver?
7. What is incremental processing?
8. Why use Delta/Iceberg for medallion?

---

## 🎯 Key Takeaways

- **Bronze** - Raw data, no transformations, audit trail
- **Silver** - Cleaned, validated, deduplicated
- **Gold** - Aggregated, business metrics, optimized
- **Progressive quality** - Each layer improves data
- **Incremental** - Process only new/changed data
- **ACID** - Delta/Iceberg ensure consistency
- **Separation** - Clear boundaries between layers
- **Flexibility** - Can add more layers as needed

---

## 📚 Additional Resources

- [Databricks Medallion Architecture](https://www.databricks.com/glossary/medallion-architecture)
- [Delta Lake Best Practices](https://docs.delta.io/latest/best-practices.html)
- [Lakehouse Architecture Patterns](https://www.databricks.com/blog/2020/01/30/what-is-a-data-lakehouse.html)

---

## Tomorrow: Day 17 - Data Mesh

We'll explore decentralized data architecture and domain ownership.
