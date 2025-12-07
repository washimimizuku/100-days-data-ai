# Day 20: Slowly Changing Dimensions (SCD)

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of today, you will:
- Understand SCD types (0, 1, 2, 3, 4, 6)
- Implement SCD Type 2 (most common)
- Handle dimension changes over time
- Choose appropriate SCD type for use cases
- Implement SCD in Delta Lake/Iceberg

---

## Theory

### What are Slowly Changing Dimensions?

**Dimensions that change over time** but at a slow, unpredictable rate.

**Examples**:
- Customer address changes
- Product price updates
- Employee department transfers
- Store location changes

**Challenge**: How to track historical changes?

---

### SCD Type 0: Retain Original

**Strategy**: Never change, keep original values

**Use Case**: Fixed attributes (birth date, SSN)

**Example**:
```sql
CREATE TABLE dim_customer (
    customer_key INT PRIMARY KEY,
    customer_id VARCHAR(50),
    birth_date DATE,  -- Never changes
    ssn VARCHAR(11)   -- Never changes
);
```

**Pros**: Simple, no history tracking needed
**Cons**: Can't track changes

---

### SCD Type 1: Overwrite

**Strategy**: Overwrite old value with new value (no history)

**Use Case**: Corrections, unimportant changes

**Example**:
```sql
-- Before
customer_key | customer_id | name  | email
1            | C001        | Alice | old@email.com

-- After update
customer_key | customer_id | name  | email
1            | C001        | Alice | new@email.com  -- Overwritten
```

**Implementation**:
```python
# Update in place
spark.sql("""
    UPDATE dim_customer
    SET email = 'new@email.com'
    WHERE customer_id = 'C001'
""")
```

**Pros**: Simple, saves storage
**Cons**: Loses history, can't analyze past

---

### SCD Type 2: Add New Row (Most Common)

**Strategy**: Add new row with new version, keep old row

**Use Case**: Track full history (addresses, prices, segments)

**Example**:
```sql
CREATE TABLE dim_customer (
    customer_key INT PRIMARY KEY,      -- Surrogate key
    customer_id VARCHAR(50),            -- Natural key
    name VARCHAR(100),
    address VARCHAR(200),
    effective_date DATE,                -- When this version became active
    expiration_date DATE,               -- When this version expired
    is_current BOOLEAN                  -- Is this the current version?
);

-- Before
customer_key | customer_id | address      | effective_date | expiration_date | is_current
1            | C001        | 123 Old St   | 2023-01-01     | 9999-12-31      | TRUE

-- After address change
customer_key | customer_id | address      | effective_date | expiration_date | is_current
1            | C001        | 123 Old St   | 2023-01-01     | 2024-06-15      | FALSE
2            | C001        | 456 New Ave  | 2024-06-15     | 9999-12-31      | TRUE
```

**Implementation**:
```python
from delta.tables import DeltaTable

# Expire old record
DeltaTable.forPath(spark, "dim_customer").update(
    condition = "customer_id = 'C001' AND is_current = TRUE",
    set = {
        "expiration_date": "2024-06-15",
        "is_current": "FALSE"
    }
)

# Insert new record
new_record = spark.createDataFrame([
    (2, "C001", "Alice", "456 New Ave", "2024-06-15", "9999-12-31", True)
], ["customer_key", "customer_id", "name", "address", "effective_date", "expiration_date", "is_current"])

new_record.write.format("delta").mode("append").save("dim_customer")
```

**Pros**: Full history, point-in-time analysis
**Cons**: More storage, complex queries

---

### SCD Type 3: Add New Column

**Strategy**: Add column for previous value (limited history)

**Use Case**: Track one previous value only

**Example**:
```sql
CREATE TABLE dim_customer (
    customer_key INT PRIMARY KEY,
    customer_id VARCHAR(50),
    name VARCHAR(100),
    current_address VARCHAR(200),
    previous_address VARCHAR(200),
    address_change_date DATE
);

-- Before
customer_key | customer_id | current_address | previous_address | address_change_date
1            | C001        | 123 Old St      | NULL             | NULL

-- After change
customer_key | customer_id | current_address | previous_address | address_change_date
1            | C001        | 456 New Ave     | 123 Old St       | 2024-06-15
```

**Pros**: Simple, tracks one change
**Cons**: Limited history (only 1 previous value)

---

### SCD Type 4: History Table

**Strategy**: Separate current and history tables

**Use Case**: Large dimensions, frequent changes

**Example**:
```sql
-- Current table
CREATE TABLE dim_customer_current (
    customer_key INT PRIMARY KEY,
    customer_id VARCHAR(50),
    name VARCHAR(100),
    address VARCHAR(200)
);

-- History table
CREATE TABLE dim_customer_history (
    history_key INT PRIMARY KEY,
    customer_key INT,
    customer_id VARCHAR(50),
    name VARCHAR(100),
    address VARCHAR(200),
    effective_date DATE,
    expiration_date DATE
);
```

**Pros**: Fast current queries, full history
**Cons**: Two tables to maintain

---

### SCD Type 6: Hybrid (1+2+3)

**Strategy**: Combine Type 1, 2, and 3

**Use Case**: Need current + previous + full history

**Example**:
```sql
CREATE TABLE dim_customer (
    customer_key INT PRIMARY KEY,
    customer_id VARCHAR(50),
    name VARCHAR(100),
    current_address VARCHAR(200),    -- Type 1: Always current
    historical_address VARCHAR(200), -- Type 3: Previous value
    effective_date DATE,             -- Type 2: Full history
    expiration_date DATE,
    is_current BOOLEAN
);
```

**Pros**: Flexible, multiple access patterns
**Cons**: Most complex, more storage

---

### SCD Type Comparison

| Type | Strategy | History | Storage | Complexity | Use Case |
|------|----------|---------|---------|------------|----------|
| **0** | No change | None | Low | Simple | Fixed attributes |
| **1** | Overwrite | None | Low | Simple | Corrections |
| **2** | New row | Full | High | Medium | Most common |
| **3** | New column | Limited | Medium | Simple | One previous |
| **4** | History table | Full | High | Medium | Large dims |
| **6** | Hybrid | Full | Highest | Complex | Flexible needs |

---

### Implementing SCD Type 2

```python
from delta.tables import DeltaTable

# Expire old record
DeltaTable.forPath(spark, path).update(
    condition = "customer_id = 'C001' AND is_current = TRUE",
    set = {"expiration_date": "2024-06-15", "is_current": "FALSE"}
)

# Insert new version
new_record.write.format("delta").mode("append").save(path)
```

---

### Querying SCD Type 2

**Current records only**:
```sql
SELECT * FROM dim_customer
WHERE is_current = TRUE;
```

**Point-in-time query**:
```sql
SELECT * FROM dim_customer
WHERE '2024-01-15' BETWEEN effective_date AND expiration_date;
```

**History for specific customer**:
```sql
SELECT * FROM dim_customer
WHERE customer_id = 'C001'
ORDER BY effective_date;
```

**Join fact with current dimension**:
```sql
SELECT f.*, c.name, c.address
FROM fact_orders f
JOIN dim_customer c ON f.customer_key = c.customer_key
WHERE c.is_current = TRUE;
```

**Join fact with historical dimension**:
```sql
SELECT f.*, c.name, c.address
FROM fact_orders f
JOIN dim_customer c ON f.customer_key = c.customer_key
  AND f.order_date BETWEEN c.effective_date AND c.expiration_date;
```

---

### Best Practices

1. **Use Type 2 by default** - Most flexible
2. **Surrogate keys** - Essential for Type 2
3. **Effective/expiration dates** - Track validity period
4. **is_current flag** - Fast current record queries
5. **9999-12-31** - Standard for "no expiration"
6. **Audit columns** - Add created_by, updated_at
7. **Idempotency** - Handle duplicate updates
8. **Testing** - Test with multiple changes

---

## 💻 Exercises (40 min)

Open `exercise.py` and complete the tasks.

### Exercise 1: SCD Type 1
- Create customer dimension
- Update email (overwrite)
- Verify no history

### Exercise 2: SCD Type 2
- Create customer dimension with SCD columns
- Insert initial records
- Update address (new row)
- Query current and historical

### Exercise 3: SCD Type 3
- Create dimension with previous column
- Update and track one previous value
- Compare with Type 2

### Exercise 4: Point-in-Time Query
- Create fact and dimension with Type 2
- Query as of specific date
- Verify historical accuracy

### Exercise 5: SCD Merge Function
- Implement generic SCD Type 2 merge
- Test with multiple updates
- Handle edge cases

---

## ✅ Quiz (5 min)

Answer these questions in `quiz.md`:

1. What are slowly changing dimensions?
2. What is SCD Type 1?
3. What is SCD Type 2?
4. When to use each SCD type?
5. What are effective/expiration dates?
6. What is is_current flag for?
7. How to query point-in-time?
8. Why use surrogate keys with SCD?

---

## 🎯 Key Takeaways

- **SCD** - Dimensions that change slowly over time
- **Type 1** - Overwrite (no history)
- **Type 2** - New row (full history) - Most common
- **Type 3** - New column (limited history)
- **Surrogate keys** - Essential for Type 2
- **Effective dates** - Track validity period
- **is_current** - Flag for current records
- **Point-in-time** - Query historical state

---

## 📚 Additional Resources

- [Kimball SCD](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/type-2/)
- [Delta Lake SCD](https://docs.delta.io/latest/delta-update.html#slowly-changing-data-scd-type-2-operation-into-delta-tables)
- [SCD Best Practices](https://www.sqlshack.com/implementing-slowly-changing-dimensions-scds-in-data-warehouses/)

---

## Tomorrow: Day 21 - Mini Project: Medallion Pipeline

We'll build a complete medallion architecture pipeline.
