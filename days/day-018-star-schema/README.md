# Day 18: Star Schema

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of today, you will:
- Understand star schema dimensional modeling
- Design fact and dimension tables
- Implement star schema in SQL
- Query star schema efficiently
- Compare star schema with normalized models

---

## Theory

### What is Star Schema?

A **dimensional modeling technique** where a central fact table is surrounded by dimension tables, forming a star shape.

**Created by**: Ralph Kimball (1990s)
**Purpose**: Optimize data warehouses for analytics and BI
**Pattern**: Denormalized for query performance

```
        Dim_Date
            |
Dim_Customer - Fact_Sales - Dim_Product
            |
        Dim_Store
```

---

### Components

#### Fact Table (Center)

**Purpose**: Store measurable business events

**Characteristics**:
- Contains metrics/measures (quantitative data)
- Contains foreign keys to dimensions
- Large number of rows (millions/billions)
- Narrow (few columns)
- Grain: Level of detail (e.g., one row per order line)

**Example**:
```sql
CREATE TABLE fact_sales (
    sale_id BIGINT PRIMARY KEY,
    date_key INT,           -- FK to dim_date
    customer_key INT,       -- FK to dim_customer
    product_key INT,        -- FK to dim_product
    store_key INT,          -- FK to dim_store
    quantity INT,           -- Measure
    unit_price DECIMAL(10,2), -- Measure
    total_amount DECIMAL(10,2), -- Measure
    discount_amount DECIMAL(10,2) -- Measure
);
```

---

#### Dimension Tables (Points)

**Purpose**: Provide context for facts

**Characteristics**:
- Contains descriptive attributes (qualitative data)
- Denormalized (not in 3NF)
- Fewer rows than facts (thousands)
- Wide (many columns)
- Slowly changing (SCD patterns)

**Example**:
```sql
CREATE TABLE dim_customer (
    customer_key INT PRIMARY KEY,  -- Surrogate key
    customer_id VARCHAR(50),        -- Natural key
    name VARCHAR(100),
    email VARCHAR(100),
    phone VARCHAR(20),
    address VARCHAR(200),
    city VARCHAR(50),
    state VARCHAR(50),
    country VARCHAR(50),
    segment VARCHAR(20),
    registration_date DATE,
    is_active BOOLEAN,
    effective_date DATE,            -- For SCD Type 2
    expiration_date DATE,
    is_current BOOLEAN
);
```

---

### Star Schema Example: E-commerce

```sql
-- Fact Table
CREATE TABLE fact_orders (
    order_key BIGINT PRIMARY KEY,
    date_key INT,
    customer_key INT,
    product_key INT,
    store_key INT,
    quantity INT,
    unit_price DECIMAL(10,2),
    total_amount DECIMAL(10,2),
    shipping_cost DECIMAL(10,2),
    tax_amount DECIMAL(10,2)
);

-- Dimension: Date
CREATE TABLE dim_date (
    date_key INT PRIMARY KEY,
    date DATE,
    year INT,
    quarter INT,
    month INT,
    month_name VARCHAR(20),
    week INT,
    day_of_week INT,
    day_name VARCHAR(20),
    is_weekend BOOLEAN,
    is_holiday BOOLEAN
);

-- Dimension: Customer
CREATE TABLE dim_customer (
    customer_key INT PRIMARY KEY,
    customer_id VARCHAR(50),
    name VARCHAR(100),
    email VARCHAR(100),
    segment VARCHAR(20),
    city VARCHAR(50),
    country VARCHAR(50)
);

-- Dimension: Product
CREATE TABLE dim_product (
    product_key INT PRIMARY KEY,
    product_id VARCHAR(50),
    name VARCHAR(100),
    category VARCHAR(50),
    subcategory VARCHAR(50),
    brand VARCHAR(50),
    unit_cost DECIMAL(10,2),
    unit_price DECIMAL(10,2)
);

-- Dimension: Store
CREATE TABLE dim_store (
    store_key INT PRIMARY KEY,
    store_id VARCHAR(50),
    name VARCHAR(100),
    city VARCHAR(50),
    state VARCHAR(50),
    country VARCHAR(50),
    region VARCHAR(50),
    manager VARCHAR(100)
);
```

---

### Surrogate vs Natural Keys

**Natural Key**: Business identifier (customer_id, product_sku)
**Surrogate Key**: System-generated integer (customer_key)

**Why Surrogate Keys?**
- Performance (integer joins faster)
- Handle changes (natural keys can change)
- Support SCD (multiple versions)
- Simplify relationships

```sql
-- Natural key might change
customer_id: "CUST-12345" → "CUST-67890"

-- Surrogate key stays stable
customer_key: 1001 (never changes)
```

---

### Querying Star Schema

**Simple Query**:
```sql
SELECT 
    d.year,
    d.month_name,
    SUM(f.total_amount) as revenue
FROM fact_orders f
JOIN dim_date d ON f.date_key = d.date_key
WHERE d.year = 2024
GROUP BY d.year, d.month_name
ORDER BY d.month_name;
```

**Multi-Dimension Query**:
```sql
SELECT 
    c.segment,
    p.category,
    d.quarter,
    SUM(f.quantity) as units_sold,
    SUM(f.total_amount) as revenue,
    AVG(f.unit_price) as avg_price
FROM fact_orders f
JOIN dim_customer c ON f.customer_key = c.customer_key
JOIN dim_product p ON f.product_key = p.product_key
JOIN dim_date d ON f.date_key = d.date_key
WHERE d.year = 2024
  AND c.country = 'USA'
GROUP BY c.segment, p.category, d.quarter;
```

---

### Star vs Normalized Schema

| Aspect | Star Schema | Normalized (3NF) |
|--------|-------------|------------------|
| **Structure** | Denormalized | Normalized |
| **Joins** | Few (1 level) | Many (multiple levels) |
| **Query Speed** | Fast | Slower |
| **Storage** | More (redundancy) | Less |
| **Updates** | Slower | Faster |
| **Use Case** | Analytics/BI | OLTP |
| **Complexity** | Simple | Complex |

**Normalized Example**:
```
Customer → Address → City → State → Country
(5 joins needed)
```

**Star Schema**:
```
Customer (includes city, state, country)
(1 join needed)
```

---

### Grain Definition

**Grain**: Level of detail in fact table

**Examples**:
- Order line level: One row per product in order
- Order level: One row per order
- Daily aggregate: One row per customer per day

**Choosing Grain**:
```sql
-- Atomic grain (most detailed)
fact_order_lines: order_id, line_number, product_id

-- Aggregated grain
fact_daily_sales: date, customer_id, SUM(amount)
```

**Rule**: Choose lowest grain possible, aggregate in queries

---

### Additive vs Non-Additive Measures

**Additive**: Can sum across all dimensions
```sql
-- Additive: quantity, amount
SUM(quantity) -- Valid across all dimensions
```

**Semi-Additive**: Can sum across some dimensions
```sql
-- Semi-additive: inventory balance
SUM(balance) -- Valid across products, NOT across time
```

**Non-Additive**: Cannot sum
```sql
-- Non-additive: ratios, percentages
AVG(unit_price) -- Must recalculate, not sum
```

---

### Implementing Star Schema

```python
# Create dimensions
dim_date = spark.sql("""
    SELECT CAST(date_format(date, 'yyyyMMdd') AS INT) as date_key,
           date, year(date) as year, month(date) as month
    FROM date_range
""")

# Create fact table
fact_sales = orders_df \
    .join(dim_date, orders_df.order_date == dim_date.date) \
    .join(dim_customer, "customer_id") \
    .select("order_id", "date_key", "customer_key", "quantity", "total_amount")
```

---

### Best Practices

1. **Use surrogate keys** - Integer keys for performance
2. **Denormalize dimensions** - Avoid dimension-to-dimension joins
3. **Choose atomic grain** - Lowest level of detail
4. **Date dimension** - Always include comprehensive date dimension
5. **Consistent naming** - fact_*, dim_* prefixes
6. **Document grain** - Clearly define fact table grain
7. **Handle NULLs** - Use "Unknown" dimension records
8. **Partition facts** - Partition by date for performance

---

## 💻 Exercises (40 min)

Open `exercise.py` and complete the tasks.

### Exercise 1: Design Star Schema
- Design fact and dimension tables for retail
- Define grain
- Choose measures and attributes

### Exercise 2: Create Dimensions
- Create dim_date with full calendar
- Create dim_customer
- Create dim_product

### Exercise 3: Create Fact Table
- Create fact_sales
- Load sample data
- Verify relationships

### Exercise 4: Query Star Schema
- Write revenue by month query
- Write top customers query
- Write product performance query

### Exercise 5: Performance Comparison
- Compare star vs normalized queries
- Measure execution time
- Analyze query plans

---

## ✅ Quiz (5 min)

Answer these questions in `quiz.md`:

1. What is a star schema?
2. What is a fact table?
3. What is a dimension table?
4. What is a surrogate key?
5. What is grain in a fact table?
6. Why denormalize dimensions?
7. What are additive measures?
8. When to use star schema?

---

## 🎯 Key Takeaways

- **Star shape** - Fact table surrounded by dimensions
- **Fact table** - Measures/metrics, many rows
- **Dimensions** - Context/attributes, fewer rows
- **Denormalized** - Optimized for read performance
- **Surrogate keys** - Integer keys for performance
- **Grain** - Level of detail in fact table
- **Simple queries** - Few joins, fast performance
- **BI optimized** - Designed for analytics

---

## 📚 Additional Resources

- [Kimball Dimensional Modeling](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/)
- [Star Schema Benchmark](http://www.cs.umb.edu/~poneil/StarSchemaB.PDF)
- [Data Warehouse Toolkit](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/books/data-warehouse-dw-toolkit/)

---

## Tomorrow: Day 19 - Snowflake Schema

We'll explore normalized dimensional modeling.
