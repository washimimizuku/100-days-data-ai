# Day 19: Snowflake Schema

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of today, you will:
- Understand snowflake schema structure
- Compare snowflake vs star schema
- Design normalized dimension hierarchies
- Implement snowflake schema in SQL
- Choose between star and snowflake patterns

---

## Theory

### What is Snowflake Schema?

A **normalized dimensional model** where dimension tables are normalized into multiple related tables, forming a snowflake shape.

**Pattern**: Star schema with normalized dimensions

```
        Dim_Date
            |
Dim_City ← Dim_Customer - Fact_Sales - Dim_Product → Dim_Category
    ↓                                        ↓
Dim_State                              Dim_Subcategory
    ↓
Dim_Country
```

---

### Star vs Snowflake

#### Star Schema (Denormalized)
```
Dim_Customer
- customer_key
- customer_id
- name
- city
- state
- country  ← All in one table
```

#### Snowflake Schema (Normalized)
```
Dim_Customer          Dim_City           Dim_State        Dim_Country
- customer_key    →   - city_key     →   - state_key  →   - country_key
- customer_id         - city_name        - state_name     - country_name
- name                - state_key        - country_key
- city_key
```

---

### Snowflake Schema Example

```sql
-- Fact Table (same as star)
CREATE TABLE fact_sales (
    sale_id BIGINT PRIMARY KEY,
    date_key INT,
    customer_key INT,
    product_key INT,
    quantity INT,
    total_amount DECIMAL(10,2)
);

-- Normalized Customer Dimension
CREATE TABLE dim_customer (
    customer_key INT PRIMARY KEY,
    customer_id VARCHAR(50),
    name VARCHAR(100),
    email VARCHAR(100),
    city_key INT  -- FK to dim_city
);

CREATE TABLE dim_city (
    city_key INT PRIMARY KEY,
    city_name VARCHAR(50),
    state_key INT  -- FK to dim_state
);

CREATE TABLE dim_state (
    state_key INT PRIMARY KEY,
    state_name VARCHAR(50),
    state_code VARCHAR(2),
    country_key INT  -- FK to dim_country
);

CREATE TABLE dim_country (
    country_key INT PRIMARY KEY,
    country_name VARCHAR(50),
    country_code VARCHAR(3),
    region VARCHAR(50)
);

-- Normalized Product Dimension
CREATE TABLE dim_product (
    product_key INT PRIMARY KEY,
    product_id VARCHAR(50),
    name VARCHAR(100),
    subcategory_key INT  -- FK to dim_subcategory
);

CREATE TABLE dim_subcategory (
    subcategory_key INT PRIMARY KEY,
    subcategory_name VARCHAR(50),
    category_key INT  -- FK to dim_category
);

CREATE TABLE dim_category (
    category_key INT PRIMARY KEY,
    category_name VARCHAR(50),
    department VARCHAR(50)
);
```

---

### Querying Snowflake Schema

**Simple Query** (more joins):
```sql
SELECT 
    co.country_name,
    SUM(f.total_amount) as revenue
FROM fact_sales f
JOIN dim_customer c ON f.customer_key = c.customer_key
JOIN dim_city ci ON c.city_key = ci.city_key
JOIN dim_state s ON ci.state_key = s.state_key
JOIN dim_country co ON s.country_key = co.country_key
WHERE co.country_name = 'USA'
GROUP BY co.country_name;
```

**Star Schema Equivalent** (fewer joins):
```sql
SELECT 
    c.country,
    SUM(f.total_amount) as revenue
FROM fact_sales f
JOIN dim_customer c ON f.customer_key = c.customer_key
WHERE c.country = 'USA'
GROUP BY c.country;
```

---

### Comparison Table

| Aspect | Star Schema | Snowflake Schema |
|--------|-------------|------------------|
| **Structure** | Denormalized | Normalized |
| **Joins** | Fewer (1 per dim) | More (multiple levels) |
| **Query Speed** | Faster | Slower |
| **Storage** | More (redundancy) | Less |
| **Maintenance** | Harder (updates) | Easier (updates) |
| **Complexity** | Simple | Complex |
| **Data Integrity** | Lower | Higher |
| **Use Case** | Read-heavy | Write-heavy |

---

### When to Use Each

#### Use Star Schema When:
- Query performance is critical
- Read-heavy workload (BI/analytics)
- Simple queries preferred
- Storage cost acceptable
- Dimension updates rare

#### Use Snowflake Schema When:
- Storage optimization needed
- Write-heavy workload
- Data integrity critical
- Dimension updates frequent
- Complex hierarchies exist

---

### Hybrid Approach

**Selective Normalization**: Normalize only large dimensions

```sql
-- Keep small dimensions denormalized (star)
CREATE TABLE dim_date (
    date_key INT PRIMARY KEY,
    date DATE,
    year INT,
    month INT,
    day INT
);

-- Normalize large dimensions (snowflake)
CREATE TABLE dim_customer (
    customer_key INT PRIMARY KEY,
    name VARCHAR(100),
    geography_key INT  -- Normalized
);

CREATE TABLE dim_geography (
    geography_key INT PRIMARY KEY,
    city VARCHAR(50),
    state VARCHAR(50),
    country VARCHAR(50)
);
```

---

### Storage Comparison

**Star Schema**:
```
dim_customer: 1M rows × 10 columns = 10M cells
Redundancy: city, state, country repeated
Storage: ~500MB
```

**Snowflake Schema**:
```
dim_customer: 1M rows × 5 columns = 5M cells
dim_city: 10K rows × 3 columns = 30K cells
dim_state: 100 rows × 3 columns = 300 cells
dim_country: 50 rows × 3 columns = 150 cells
Storage: ~250MB (50% savings)
```

---

### Implementation Example

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Snowflake").getOrCreate()

# Create normalized dimensions
dim_country = spark.createDataFrame([
    (1, "USA", "US", "North America"),
    (2, "UK", "GB", "Europe")
], ["country_key", "country_name", "country_code", "region"])

dim_state = spark.createDataFrame([
    (1, "California", "CA", 1),
    (2, "Texas", "TX", 1),
    (3, "England", "EN", 2)
], ["state_key", "state_name", "state_code", "country_key"])

dim_city = spark.createDataFrame([
    (1, "Los Angeles", 1),
    (2, "San Francisco", 1),
    (3, "Houston", 2),
    (4, "London", 3)
], ["city_key", "city_name", "state_key"])

dim_customer = spark.createDataFrame([
    (1, "C001", "Alice", "alice@email.com", 1),
    (2, "C002", "Bob", "bob@email.com", 4)
], ["customer_key", "customer_id", "name", "email", "city_key"])

# Query with multiple joins
result = fact_sales \
    .join(dim_customer, "customer_key") \
    .join(dim_city, "city_key") \
    .join(dim_state, "state_key") \
    .join(dim_country, "country_key") \
    .groupBy("country_name") \
    .agg(sum("total_amount").alias("revenue"))
```

---

### Performance Optimization

**Materialized Views**:
```sql
-- Pre-join normalized dimensions
CREATE MATERIALIZED VIEW mv_customer_geography AS
SELECT 
    c.customer_key,
    c.name,
    ci.city_name,
    s.state_name,
    co.country_name
FROM dim_customer c
JOIN dim_city ci ON c.city_key = ci.city_key
JOIN dim_state s ON ci.state_key = s.state_key
JOIN dim_country co ON s.country_key = co.country_key;

-- Query materialized view (fast like star)
SELECT country_name, COUNT(*)
FROM fact_sales f
JOIN mv_customer_geography c ON f.customer_key = c.customer_key
GROUP BY country_name;
```

---

### Best Practices

1. **Selective normalization** - Normalize only large dimensions
2. **Materialized views** - Pre-join for performance
3. **Indexing** - Index foreign keys in normalized tables
4. **Caching** - Cache frequently joined dimensions
5. **Partitioning** - Partition fact table by date
6. **Documentation** - Document hierarchy relationships
7. **Testing** - Compare query performance vs star

---

## 💻 Exercises (40 min)

Open `exercise.py` and complete the tasks.

### Exercise 1: Design Snowflake Schema
- Normalize customer dimension
- Normalize product dimension
- Define relationships

### Exercise 2: Create Normalized Dimensions
- Create geography hierarchy
- Create product hierarchy
- Load sample data

### Exercise 3: Query Snowflake Schema
- Write multi-join queries
- Compare with star schema
- Measure performance

### Exercise 4: Hybrid Schema
- Identify dimensions to normalize
- Keep small dimensions denormalized
- Implement hybrid approach

### Exercise 5: Optimization
- Create materialized views
- Add indexes
- Compare performance

---

## ✅ Quiz (5 min)

Answer these questions in `quiz.md`:

1. What is snowflake schema?
2. How is it different from star schema?
3. Why normalize dimensions?
4. What are the trade-offs?
5. When to use snowflake schema?
6. What are materialized views?
7. What is selective normalization?
8. Which is faster: star or snowflake?

---

## 🎯 Key Takeaways

- **Normalized** - Dimensions split into multiple tables
- **More joins** - Requires more joins than star schema
- **Less storage** - Reduces redundancy and storage
- **Slower queries** - More joins = slower performance
- **Better integrity** - Normalization improves data quality
- **Hybrid approach** - Normalize only large dimensions
- **Materialized views** - Pre-join for performance
- **Use case** - Best when storage matters more than speed

---

## 📚 Additional Resources

- [Kimball on Snowflake Schema](https://www.kimballgroup.com/2005/02/design-tip-62-snowflake-schemas/)
- [Star vs Snowflake](https://www.vertabelo.com/blog/star-schema-vs-snowflake-schema/)
- [Dimensional Modeling](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/)

---

## Tomorrow: Day 20 - Slowly Changing Dimensions

We'll explore handling dimension changes over time.
