# Day 21: Mini Project - Medallion Pipeline

## 📖 Project Overview (2 hours)

**Time**: 2 hours


Build a complete medallion architecture pipeline (Bronze → Silver → Gold) with data quality, SCD Type 2, and star schema.

**Time Allocation:**
- Planning & Setup: 20 min
- Bronze Layer: 30 min
- Silver Layer: 40 min
- Gold Layer: 30 min

---

## Project Goals

Create `medallion_pipeline.py` implementing:

1. **Bronze Layer**
   - Ingest raw JSON/CSV data
   - Add ingestion metadata
   - Preserve all records

2. **Silver Layer**
   - Data quality validation
   - Deduplication
   - SCD Type 2 for dimensions
   - Schema enforcement

3. **Gold Layer**
   - Star schema (fact + dimensions)
   - Business aggregations
   - Optimized for BI

4. **Orchestration**
   - End-to-end pipeline
   - Error handling
   - Logging

---

## Requirements

```bash
pip install pyspark delta-spark
```

---

## Data Model

### Bronze
```
bronze/
├── orders/          # Raw order data
├── customers/       # Raw customer data
└── products/        # Raw product data
```

### Silver
```
silver/
├── orders/          # Cleaned orders
├── customers/       # SCD Type 2 customers
└── products/        # Validated products
```

### Gold
```
gold/
├── fact_sales/      # Sales fact table
├── dim_customer/    # Customer dimension
├── dim_product/     # Product dimension
├── dim_date/        # Date dimension
└── daily_metrics/   # Aggregated metrics
```

---

## Implementation Guide

### 1. Bronze Layer

```python
class BronzeLayer:
    def ingest_orders(self, source_path):
        df = spark.read.json(source_path)
        df = df.withColumn("_ingestion_time", current_timestamp()) \
               .withColumn("_source", lit("orders_api"))
        df.write.format("delta").mode("append").save("bronze/orders")
```

### 2. Silver Layer

```python
class SilverLayer:
    def transform_orders(self):
        bronze = spark.read.format("delta").load("bronze/orders")
        silver = bronze \
            .dropDuplicates(["order_id"]) \
            .filter(col("order_id").isNotNull()) \
            .withColumn("order_date", to_date("order_timestamp"))
        silver.write.format("delta").mode("overwrite").save("silver/orders")
    
    def update_customers_scd2(self, new_customers):
        # Implement SCD Type 2
        pass
```

### 3. Gold Layer

```python
class GoldLayer:
    def create_fact_sales(self):
        orders = spark.read.format("delta").load("silver/orders")
        # Join with dimensions, create fact table
        pass
    
    def create_daily_metrics(self):
        fact = spark.read.format("delta").load("gold/fact_sales")
        metrics = fact.groupBy("date_key").agg(...)
        metrics.write.format("delta").mode("overwrite").save("gold/daily_metrics")
```

---

## Sample Data

**orders.json**:
```json
[
  {"order_id": "O001", "customer_id": "C001", "product_id": "P001", "quantity": 2, "amount": 199.98, "order_timestamp": "2024-01-15T10:30:00"},
  {"order_id": "O002", "customer_id": "C002", "product_id": "P002", "quantity": 1, "amount": 49.99, "order_timestamp": "2024-01-16T14:20:00"}
]
```

**customers.json**:
```json
[
  {"customer_id": "C001", "name": "Alice Smith", "email": "alice@email.com", "city": "New York", "segment": "Premium"},
  {"customer_id": "C002", "name": "Bob Jones", "email": "bob@email.com", "city": "London", "segment": "Standard"}
]
```

---

## Testing

Create `test_pipeline.sh`:

```bash
#!/bin/bash

echo "=== Testing Medallion Pipeline ==="

# Generate sample data
python generate_data.py

# Run pipeline
python medallion_pipeline.py --mode full

# Verify layers
python medallion_pipeline.py --verify

echo "=== Tests Complete ==="
```

---

## Deliverables

1. **medallion_pipeline.py** - Main pipeline implementation
2. **generate_data.py** - Sample data generator
3. **test_pipeline.sh** - Test script
4. **README.md** - Usage documentation
5. **requirements.txt** - Dependencies

---

## Success Criteria

✅ Bronze layer ingests raw data with metadata
✅ Silver layer applies data quality rules
✅ SCD Type 2 implemented for customers
✅ Gold layer creates star schema
✅ Daily metrics aggregated correctly
✅ Pipeline runs end-to-end
✅ Error handling implemented
✅ Tests pass

---

## Bonus Features

- Incremental processing (process only new data)
- Data quality dashboard
- Lineage tracking
- Performance monitoring
- Partition optimization
- Schema evolution handling

---

## Learning Outcomes

- Build production-ready medallion pipeline
- Implement SCD Type 2 in practice
- Create star schema from raw data
- Handle data quality issues
- Orchestrate multi-layer pipeline

---

## Resources

- [Medallion Architecture](https://www.databricks.com/glossary/medallion-architecture)
- [Delta Lake Guide](https://docs.delta.io/)
- [SCD Implementation](https://docs.delta.io/latest/delta-update.html)

---

## Next Steps

After completing this project:
1. Test all layers
2. Add monitoring
3. Document pipeline
4. Consider enhancements
5. Move to Week 4

---

## Tips

- Start with bronze (simplest)
- Test each layer independently
- Use small sample data first
- Add error handling incrementally
- Document as you build
