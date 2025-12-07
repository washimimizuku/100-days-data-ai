# Day 41: dbt Basics

## 📖 Learning Objectives (15 min)

**Time**: 1 hour


By the end of this session, you will:
- Understand what dbt is and why it's used
- Learn dbt's core concepts: models, sources, tests, documentation
- Write SQL-based transformations with dbt
- Implement data quality tests
- Generate automatic documentation
- Understand dbt's built-in lineage tracking

---

## What is dbt?

dbt (data build tool) is a transformation framework that transforms data in the warehouse using SQL and software engineering practices.

**Key philosophy**: Transform data where it lives (in the warehouse) using SQL.

**What dbt does**: Transforms raw data, tests quality, documents models, tracks lineage, enables version control

**What dbt doesn't do**: Extract data, load data, orchestrate pipelines

---

## Core Concepts

### 1. Models
SQL files that define transformations:
```sql
-- models/staging/stg_customers.sql
SELECT
    id AS customer_id,
    LOWER(TRIM(email)) AS email,
    first_name,
    last_name
FROM {{ source('raw', 'customers') }}
WHERE email IS NOT NULL
```

### 2. Sources
Define raw data tables in YAML:
```yaml
sources:
  - name: raw
    tables:
      - name: customers
      - name: orders
```

### 3. Tests
Data quality checks:
```yaml
models:
  - name: stg_customers
    columns:
      - name: customer_id
        tests: [unique, not_null]
```

### 4. Documentation
Describe your data:
```yaml
models:
  - name: stg_customers
    description: "Cleaned customer data"
```

---

## dbt Project Structure

```
my_dbt_project/
├── dbt_project.yml
├── models/
│   ├── staging/
│   ├── intermediate/
│   └── marts/
├── tests/
├── macros/
└── seeds/
```

---

## Writing dbt Models

### Staging Model
```sql
WITH source AS (
    SELECT * FROM {{ source('raw', 'customers') }}
),
cleaned AS (
    SELECT
        id AS customer_id,
        LOWER(TRIM(email)) AS email,
        CONCAT(first_name, ' ', last_name) AS full_name
    FROM source
    WHERE email IS NOT NULL
)
SELECT * FROM cleaned
```

### Intermediate Model
```sql
WITH customers AS (
    SELECT * FROM {{ ref('stg_customers') }}
),
orders AS (
    SELECT * FROM {{ ref('stg_orders') }}
),
customer_orders AS (
    SELECT
        c.customer_id,
        COUNT(o.order_id) AS total_orders,
        SUM(o.amount) AS total_spent
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY 1
)
SELECT * FROM customer_orders
```

---

## dbt Jinja Functions

### ref() - Reference Models
```sql
SELECT * FROM {{ ref('stg_customers') }}
-- Compiles to: SELECT * FROM analytics.staging.stg_customers
```

### source() - Reference Raw Tables
```sql
SELECT * FROM {{ source('raw', 'customers') }}
-- Compiles to: SELECT * FROM raw_data.customers
```

### Custom Macros
```sql
-- macros/cents_to_dollars.sql
{% macro cents_to_dollars(column_name) %}
    ({{ column_name }} / 100.0)::decimal(10,2)
{% endmacro %}

-- Usage:
SELECT {{ cents_to_dollars('amount_cents') }} AS amount_dollars
```

---

## dbt Tests

### Built-in Tests
```yaml
models:
  - name: stg_customers
    columns:
      - name: customer_id
        tests: [unique, not_null]
      - name: status
        tests:
          - accepted_values:
              values: ['active', 'inactive']
```

### Custom Tests
```sql
-- tests/assert_positive_amounts.sql
SELECT * FROM {{ ref('stg_orders') }}
WHERE amount <= 0
```

### Relationships Test
```yaml
- name: customer_id
  tests:
    - relationships:
        to: ref('stg_customers')
        field: customer_id
```

---

## dbt Commands

```bash
# Run models
dbt run
dbt run --select stg_customers
dbt run --select stg_customers+  # downstream
dbt run --select +stg_customers  # upstream

# Test data
dbt test
dbt test --select stg_customers

# Generate docs
dbt docs generate
dbt docs serve
```

---

## Materializations

### View (default)
```sql
{{ config(materialized='view') }}
SELECT * FROM {{ ref('stg_customers') }}
```

### Table
```sql
{{ config(materialized='table') }}
SELECT * FROM {{ ref('stg_customers') }}
```

### Incremental
```sql
{{ config(materialized='incremental', unique_key='order_id') }}
SELECT * FROM {{ ref('stg_orders') }}
{% if is_incremental() %}
    WHERE order_date > (SELECT MAX(order_date) FROM {{ this }})
{% endif %}
```

### Ephemeral
```sql
{{ config(materialized='ephemeral') }}
SELECT * FROM {{ ref('stg_customers') }}
```

---

## dbt Project Configuration

```yaml
# dbt_project.yml
name: 'my_analytics'
version: '1.0.0'
config-version: 2

models:
  my_analytics:
    staging:
      +materialized: view
    intermediate:
      +materialized: ephemeral
    marts:
      +materialized: table
```

---

## Benefits of dbt

**Version Control**: All transformations in Git with code review and rollback

**Testing**: Automated data quality checks catch issues early

**Documentation**: Auto-generated docs with lineage graphs

**Modularity**: Reusable components with DRY principle

**Lineage**: Automatic dependency tracking and impact analysis

---

## Best Practices

1. **Staging layer**: Clean and standardize raw data
2. **Intermediate layer**: Apply business logic
3. **Marts layer**: Create analytics-ready tables
4. **Test everything**: Add tests to all models
5. **Document models**: Describe purpose and columns
6. **Use CTEs**: Make SQL readable
7. **Incremental for large tables**: Optimize performance

---

## 💻 Exercises (40 min)

### Exercise 1: Create Staging Model
Write a staging model to clean raw customer data.

### Exercise 2: Create Intermediate Model
Join customers and orders with aggregations.

### Exercise 3: Add Tests
Add data quality tests to your models.

### Exercise 4: Write Documentation
Document your models with descriptions.

### Exercise 5: Create Macro
Write a reusable macro for common logic.

### Exercise 6: Incremental Model
Implement an incremental model for orders.

### Exercise 7: Custom Test
Write a custom test for business rules.

### Exercise 8: Generate Lineage
Understand model dependencies and lineage.

---

## ✅ Quiz (5 min)

Test your understanding of dbt concepts and usage.

---

## 🎯 Key Takeaways

- **Transform in warehouse**: dbt transforms data where it lives using SQL
- **Software engineering**: Version control, testing, documentation
- **Modular**: Break transformations into reusable models
- **Automatic lineage**: dbt tracks dependencies automatically
- **Testing built-in**: Data quality tests are first-class citizens
- **Documentation**: Auto-generated docs with lineage graphs

---

## 📚 Resources

- [dbt Documentation](https://docs.getdbt.com/)
- [dbt Learn](https://courses.getdbt.com/)
- [dbt Discourse](https://discourse.getdbt.com/)
- [dbt Best Practices](https://docs.getdbt.com/guides/best-practices)

---

## Tomorrow: Day 42 - Mini Project: Data Quality Framework

Build a comprehensive data quality framework combining Great Expectations, validation rules, profiling, and dbt tests into an integrated quality monitoring system.
