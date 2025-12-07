# Day 15: Data Warehouse vs Data Lake vs Lakehouse

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of today, you will:
- Understand the evolution from warehouses to lakehouses
- Compare architectures and use cases
- Identify strengths and limitations of each
- Choose the right architecture for your needs
- Understand modern hybrid approaches

---

## Theory

### Evolution of Data Architectures

```
1990s: Data Warehouse → 2010s: Data Lake → 2020s: Data Lakehouse
```

### Data Warehouse

**Definition**: Centralized repository of structured data optimized for analytics.

**Architecture**:
```
Source Systems → ETL → Data Warehouse → BI Tools
                         (Structured)
```

**Characteristics**:
- Structured data only (tables, schemas)
- Schema-on-write
- SQL-based queries
- ACID transactions
- High performance for BI
- Expensive storage

**Technologies**: Snowflake, Redshift, BigQuery, Teradata, Oracle

**Example**:
```sql
-- Traditional warehouse query
SELECT 
    customer_id,
    SUM(order_amount) as total_spent
FROM fact_orders
JOIN dim_customers USING (customer_id)
WHERE order_date >= '2024-01-01'
GROUP BY customer_id
```

**Pros**:
- ✅ Fast query performance
- ✅ ACID guarantees
- ✅ Mature tooling
- ✅ Strong governance
- ✅ Optimized for BI

**Cons**:
- ❌ Expensive storage
- ❌ Structured data only
- ❌ Limited ML support
- ❌ Rigid schemas
- ❌ Vendor lock-in

---

### Data Lake

**Definition**: Centralized repository storing raw data in native format at scale.

**Architecture**:
```
Source Systems → ELT → Data Lake → Processing → Analytics/ML
                        (Raw Files)
```

**Characteristics**:
- All data types (structured, semi-structured, unstructured)
- Schema-on-read
- Low-cost storage (S3, ADLS, GCS)
- No ACID (traditional lakes)
- Flexible but complex

**Technologies**: S3, ADLS, GCS, HDFS

**Example**:
```python
# Reading from data lake
df = spark.read.parquet("s3://lake/raw/orders/")
df = df.filter(col("order_date") >= "2024-01-01")
df.groupBy("customer_id").agg(sum("amount")).show()
```

**Pros**:
- ✅ Low storage cost
- ✅ All data types
- ✅ Flexible schemas
- ✅ ML-friendly
- ✅ Scalable

**Cons**:
- ❌ No ACID transactions
- ❌ Poor performance
- ❌ Data swamps (quality issues)
- ❌ Complex governance
- ❌ Inconsistent data

---

### Data Lakehouse

**Definition**: Hybrid combining warehouse performance with lake flexibility.

**Architecture**:
```
Source Systems → ELT → Lakehouse → Analytics/ML/BI
                       (Delta/Iceberg)
```

**Characteristics**:
- All data types
- ACID transactions
- Schema enforcement + evolution
- Time travel
- Unified platform
- Open formats

**Technologies**: Delta Lake, Apache Iceberg, Apache Hudi, Databricks, Snowflake

**Example**:
```python
# Lakehouse with Delta Lake
df = spark.read.format("delta").load("s3://lakehouse/orders")

# ACID update
spark.sql("""
    UPDATE delta.`s3://lakehouse/orders`
    SET status = 'shipped'
    WHERE order_id = 123
""")

# Time travel
df_yesterday = spark.read \
    .format("delta") \
    .option("versionAsOf", 5) \
    .load("s3://lakehouse/orders")
```

**Pros**:
- ✅ ACID transactions
- ✅ Low storage cost
- ✅ All data types
- ✅ BI + ML support
- ✅ Open formats
- ✅ Time travel

**Cons**:
- ⚠️ Newer technology
- ⚠️ Learning curve
- ⚠️ Requires maintenance

---

### Comparison Table

| Feature | Warehouse | Lake | Lakehouse |
|---------|-----------|------|-----------|
| **Data Types** | Structured | All | All |
| **Storage Cost** | High | Low | Low |
| **ACID** | ✅ | ❌ | ✅ |
| **Schema** | Rigid | Flexible | Both |
| **Performance** | Fast | Slow | Fast |
| **BI Support** | Excellent | Poor | Good |
| **ML Support** | Limited | Excellent | Excellent |
| **Governance** | Strong | Weak | Strong |
| **Time Travel** | Limited | ❌ | ✅ |
| **Cost** | $$$ | $ | $$ |

---

### Use Case Comparison

#### Data Warehouse
**Best for**:
- Traditional BI and reporting
- Structured transactional data
- Regulatory compliance
- Predictable workloads
- SQL-only teams

**Example**: Financial reporting, sales dashboards, compliance reports

#### Data Lake
**Best for**:
- Raw data storage
- Data science exploration
- Unstructured data (logs, images, videos)
- Cost-sensitive storage
- Flexible schemas

**Example**: Log storage, ML training data, IoT sensor data

#### Data Lakehouse
**Best for**:
- Unified analytics and ML
- Modern data platforms
- ACID requirements with flexibility
- BI + data science teams
- Cost optimization

**Example**: Customer 360, real-time analytics, ML pipelines

---

### Architecture Patterns

#### Lambda Architecture (Lake + Warehouse)
```
Sources → Batch Layer (Lake) → Warehouse → BI
       → Speed Layer (Streaming) ↗
```

**Pros**: Handles batch + streaming
**Cons**: Complex, duplicate logic

#### Kappa Architecture (Lake Only)
```
Sources → Streaming → Lake → Processing → Analytics
```

**Pros**: Simpler, single pipeline
**Cons**: No ACID, reprocessing hard

#### Lakehouse Architecture (Unified)
```
Sources → Lakehouse (Delta/Iceberg) → BI + ML + Streaming
```

**Pros**: Unified, ACID, simple
**Cons**: Newer, requires expertise

---

### Migration Paths

#### Warehouse → Lakehouse
1. Export warehouse data to Parquet
2. Create Delta/Iceberg tables
3. Migrate ETL pipelines
4. Update BI connections
5. Decommission warehouse

#### Lake → Lakehouse
1. Convert Parquet to Delta/Iceberg
2. Add ACID layer
3. Implement governance
4. Enable time travel
5. Optimize performance

---

### Real-World Example

**Scenario**: E-commerce company with 10TB data

**Warehouse Approach**:
```
Cost: $5,000/month storage + $10,000/month compute
Data: Orders, customers, products (structured only)
Limitations: Can't store clickstream, images
```

**Lake Approach**:
```
Cost: $230/month storage + $2,000/month compute
Data: All data types
Limitations: No ACID, data quality issues, slow queries
```

**Lakehouse Approach**:
```
Cost: $300/month storage + $3,000/month compute
Data: All data types with ACID
Benefits: Best of both worlds
```

---

### Decision Framework

**Choose Warehouse if**:
- Only structured data
- Strong SQL team
- Existing warehouse investment
- Regulatory requirements
- Budget allows

**Choose Lake if**:
- Cost is primary concern
- Mostly unstructured data
- No ACID needed
- Data science focus
- Flexible exploration

**Choose Lakehouse if**:
- Need ACID + flexibility
- BI + ML workloads
- Modern architecture
- Cost optimization
- Future-proof solution

---

## 💻 Exercises (40 min)

Open `exercise.py` and complete the tasks.

### Exercise 1: Architecture Analysis
- Compare storage costs for 1TB data
- Calculate query performance differences
- Analyze use case fit

### Exercise 2: Data Modeling
- Design warehouse star schema
- Design lake folder structure
- Design lakehouse table structure

### Exercise 3: Query Patterns
- Write warehouse SQL query
- Write lake Spark query
- Write lakehouse Delta query

### Exercise 4: Migration Planning
- Plan warehouse to lakehouse migration
- Estimate costs and timeline
- Identify risks

### Exercise 5: Architecture Selection
- Evaluate 3 scenarios
- Recommend architecture
- Justify decision

---

## ✅ Quiz (5 min)

Answer these questions in `quiz.md`:

1. What is the main difference between warehouse and lake?
2. What problem does lakehouse solve?
3. Which supports ACID transactions?
4. Which is cheapest for storage?
5. Which is best for ML workloads?
6. What is schema-on-write vs schema-on-read?
7. What technologies enable lakehouses?
8. When should you choose a warehouse?

---

## 🎯 Key Takeaways

- **Warehouse** - Structured, fast, expensive, ACID
- **Lake** - Flexible, cheap, no ACID, all data types
- **Lakehouse** - Best of both, ACID + flexibility
- **Evolution** - Industry moving toward lakehouses
- **Cost** - Lake/Lakehouse 10-20x cheaper storage
- **Use cases** - Different architectures for different needs
- **Modern trend** - Unified platforms (lakehouse)
- **Open formats** - Delta Lake, Iceberg enable lakehouses

---

## 📚 Additional Resources

- [Databricks Lakehouse Paper](https://www.cidrdb.org/cidr2021/papers/cidr2021_paper17.pdf)
- [Data Warehouse vs Data Lake](https://aws.amazon.com/compare/the-difference-between-a-data-warehouse-data-lake-and-data-mart/)
- [Lakehouse Architecture](https://www.databricks.com/glossary/data-lakehouse)

---

## Tomorrow: Day 16 - Medallion Architecture

We'll explore the Bronze/Silver/Gold layering pattern for data organization.
