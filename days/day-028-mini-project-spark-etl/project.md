# Spark ETL Pipeline - Project Specification

## Project Goal

Build a production-ready ETL pipeline to process e-commerce data using PySpark.

---

## Data Sources

### 1. Orders (CSV)
```
order_id,customer_id,product_id,quantity,price,order_date
1,101,501,2,29.99,2024-01-15
2,102,502,1,49.99,2024-01-15
```

### 2. Customers (JSON)
```json
{"customer_id": 101, "name": "Alice", "email": "alice@example.com", "city": "NYC", "country": "USA"}
{"customer_id": 102, "name": "Bob", "email": "bob@example.com", "city": "LA", "country": "USA"}
```

### 3. Products (CSV)
```
product_id,name,category,cost
501,Widget,Electronics,15.00
502,Gadget,Electronics,25.00
```

---

## Pipeline Stages

### Stage 1: Extract
- Read CSV and JSON files
- Infer schemas
- Handle missing files

### Stage 2: Transform
- Clean data (nulls, duplicates)
- Validate data types
- Enrich with joins
- Calculate metrics

### Stage 3: Load
- Write to Parquet
- Partition by date
- Optimize file sizes

### Stage 4: Quality
- Validate record counts
- Check for nulls
- Verify business rules

---

## Output Datasets

### 1. Enriched Orders
```
order_id, customer_id, customer_name, product_id, product_name, 
category, quantity, price, cost, profit, order_date
```

### 2. Daily Sales Summary
```
order_date, total_orders, total_revenue, total_profit, avg_order_value
```

### 3. Customer Analytics
```
customer_id, name, total_orders, total_spent, lifetime_value, segment
```

### 4. Quality Report
```
dataset, total_records, null_count, duplicate_count, status
```

---

## Business Rules

1. **Profit Calculation**: `profit = (price - cost) * quantity`
2. **Customer Segments**:
   - High Value: lifetime_value > $500
   - Medium Value: lifetime_value $100-$500
   - Low Value: lifetime_value < $100
3. **Data Quality**:
   - No nulls in key fields
   - Prices must be positive
   - Dates must be valid

---

## Performance Requirements

- Process 1M+ records
- Complete in < 5 minutes
- Use < 4GB memory
- Optimize file sizes (100-200 MB per file)

---

## Deliverables

1. **Code**: Complete ETL pipeline
2. **Data**: Generated sample data
3. **Tests**: Unit tests for transformations
4. **Docs**: Architecture and usage guide
5. **Report**: Performance metrics and insights

---

## Time Budget

- Setup: 15 min
- Extract: 20 min
- Transform: 40 min
- Load: 20 min
- Quality: 15 min
- Testing: 10 min
- **Total**: 2 hours
