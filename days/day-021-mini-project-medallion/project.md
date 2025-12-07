# Mini Project: Medallion Pipeline

## Objective

Build a complete medallion architecture pipeline implementing Bronze/Silver/Gold layers with data quality, SCD Type 2, and star schema.

## Requirements

### Functional Requirements

1. **Bronze Layer**
   - Ingest JSON/CSV files
   - Add ingestion metadata (_ingestion_time, _source)
   - Preserve all records (no filtering)
   - Append-only writes

2. **Silver Layer**
   - Data quality validation
   - Deduplication
   - Schema enforcement
   - SCD Type 2 for customer dimension
   - Partitioning by date

3. **Gold Layer**
   - Star schema (fact_sales + dimensions)
   - Date dimension with full calendar
   - Daily aggregated metrics
   - Optimized for queries

4. **Pipeline Orchestration**
   - Run all layers in sequence
   - Error handling and logging
   - Verification checks
   - Idempotent execution

### Non-Functional Requirements

- Clean, modular code
- Comprehensive error handling
- Logging for debugging
- Test coverage
- Documentation

## Architecture

```
medallion_pipeline.py
├── BronzeLayer
│   ├── ingest_orders()
│   ├── ingest_customers()
│   └── ingest_products()
├── SilverLayer
│   ├── transform_orders()
│   ├── update_customers_scd2()
│   └── validate_products()
├── GoldLayer
│   ├── create_dim_date()
│   ├── create_dim_customer()
│   ├── create_dim_product()
│   ├── create_fact_sales()
│   └── create_daily_metrics()
└── Pipeline
    ├── run_full()
    ├── run_incremental()
    └── verify()
```

## Implementation Steps

1. **Setup** (15 min)
   - Initialize Spark with Delta
   - Create directory structure
   - Generate sample data

2. **Bronze Layer** (30 min)
   - Implement ingestion methods
   - Add metadata columns
   - Test with sample data

3. **Silver Layer** (40 min)
   - Implement data quality rules
   - Implement SCD Type 2 for customers
   - Test transformations

4. **Gold Layer** (30 min)
   - Create dimension tables
   - Create fact table
   - Create aggregations

5. **Testing & Polish** (15 min)
   - End-to-end test
   - Add error handling
   - Write documentation

## Deliverables

1. `medallion_pipeline.py` - Main implementation
2. `generate_data.py` - Data generator
3. `test_pipeline.sh` - Test script
4. `requirements.txt` - Dependencies
5. `README.md` - Usage guide

## Evaluation Criteria

- **Functionality** (40%) - All layers work correctly
- **Code Quality** (30%) - Clean, modular, documented
- **SCD Implementation** (15%) - Correct Type 2 handling
- **Error Handling** (15%) - Graceful failures

## Time Budget

- Total: 2 hours
- Core implementation: 90 minutes
- Testing/docs: 30 minutes
