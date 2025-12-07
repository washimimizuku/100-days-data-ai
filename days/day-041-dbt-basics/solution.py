"""
Day 41: dbt Basics - Solutions

Note: These solutions simulate dbt concepts in Python.
In practice, dbt uses SQL files and YAML configuration.
"""


def ref(model_name):
    """Simulate dbt ref() function"""
    return f"{{{{ ref('{model_name}') }}}}"


def source(source_name, table_name):
    """Simulate dbt source() function"""
    return f"{{{{ source('{source_name}', '{table_name}') }}}}"


def exercise_1_create_staging_model():
    """Create staging model"""
    sql = f"""
-- models/staging/stg_customers.sql
WITH source AS (
    SELECT * FROM {source('raw', 'customers')}
),

cleaned AS (
    SELECT
        id AS customer_id,
        LOWER(TRIM(email)) AS email,
        CONCAT(first_name, ' ', last_name) AS full_name,
        status,
        created_at,
        updated_at
    FROM source
    WHERE email IS NOT NULL
)

SELECT * FROM cleaned
"""
    return sql.strip()


def exercise_2_create_intermediate_model():
    """Create intermediate model"""
    sql = f"""
-- models/intermediate/int_customer_orders.sql
WITH customers AS (
    SELECT * FROM {ref('stg_customers')}
),

orders AS (
    SELECT * FROM {ref('stg_orders')}
),

customer_orders AS (
    SELECT
        c.customer_id,
        c.email,
        c.full_name,
        COUNT(o.order_id) AS total_orders,
        SUM(o.amount) AS total_spent,
        MAX(o.order_date) AS last_order_date
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY 1, 2, 3
)

SELECT * FROM customer_orders
"""
    return sql.strip()


def exercise_3_add_tests():
    """Add data quality tests"""
    tests = {
        'version': 2,
        'models': [
            {
                'name': 'stg_customers',
                'columns': [
                    {
                        'name': 'customer_id',
                        'tests': ['unique', 'not_null']
                    },
                    {
                        'name': 'email',
                        'tests': ['not_null']
                    },
                    {
                        'name': 'status',
                        'tests': [
                            {
                                'accepted_values': {
                                    'values': ['active', 'inactive']
                                }
                            }
                        ]
                    }
                ]
            }
        ]
    }
    return tests


def exercise_4_write_documentation():
    """Write model documentation"""
    docs = {
        'version': 2,
        'models': [
            {
                'name': 'stg_customers',
                'description': 'Cleaned and standardized customer data from raw source',
                'columns': [
                    {
                        'name': 'customer_id',
                        'description': 'Unique identifier for each customer'
                    },
                    {
                        'name': 'email',
                        'description': 'Customer email address (normalized to lowercase and trimmed)'
                    },
                    {
                        'name': 'full_name',
                        'description': 'Customer full name (concatenated from first and last name)'
                    }
                ]
            }
        ]
    }
    return docs


def exercise_5_create_macro():
    """Create reusable macro"""
    def cents_to_dollars(column_name):
        return f"({column_name} / 100.0)::decimal(10,2)"
    
    return cents_to_dollars


def exercise_6_incremental_model():
    """Create incremental model"""
    sql = f"""
-- models/marts/fct_orders.sql
{{{{ config(materialized='incremental', unique_key='order_id') }}}}

SELECT
    order_id,
    customer_id,
    order_date,
    amount,
    status
FROM {ref('stg_orders')}

{{% if is_incremental() %}}
    WHERE order_date > (SELECT MAX(order_date) FROM {{{{ this }}}})
{{% endif %}}
"""
    return sql.strip()


def exercise_7_custom_test():
    """Write custom test"""
    sql = f"""
-- tests/assert_positive_order_amounts.sql
SELECT
    order_id,
    amount
FROM {ref('stg_orders')}
WHERE amount <= 0
"""
    return sql.strip()


def exercise_8_generate_lineage():
    """Generate model lineage"""
    lineage = {
        'stg_customers': {
            'depends_on': ['raw.customers'],
            'used_by': ['int_customer_orders']
        },
        'stg_orders': {
            'depends_on': ['raw.orders'],
            'used_by': ['int_customer_orders', 'fct_orders']
        },
        'int_customer_orders': {
            'depends_on': ['stg_customers', 'stg_orders'],
            'used_by': []
        },
        'fct_orders': {
            'depends_on': ['stg_orders'],
            'used_by': []
        }
    }
    return lineage


def visualize_lineage(lineage):
    """Visualize lineage graph"""
    viz = "dbt Model Lineage:\n" + "=" * 50 + "\n\n"
    
    for model, deps in lineage.items():
        viz += f"{model}\n"
        if deps['depends_on']:
            viz += f"  ← Depends on: {', '.join(deps['depends_on'])}\n"
        if deps['used_by']:
            viz += f"  → Used by: {', '.join(deps['used_by'])}\n"
        viz += "\n"
    
    return viz


if __name__ == "__main__":
    print("Day 41: dbt Basics - Solutions\n")
    print("=" * 70)
    
    print("\n📋 Exercise 1: Create Staging Model")
    print("-" * 70)
    sql = exercise_1_create_staging_model()
    print(sql)
    
    print("\n📋 Exercise 2: Create Intermediate Model")
    print("-" * 70)
    sql = exercise_2_create_intermediate_model()
    print(sql)
    
    print("\n📋 Exercise 3: Add Tests")
    print("-" * 70)
    tests = exercise_3_add_tests()
    print(f"Added tests for {len(tests['models'][0]['columns'])} columns")
    for col in tests['models'][0]['columns']:
        print(f"  {col['name']}: {col['tests']}")
    
    print("\n📋 Exercise 4: Write Documentation")
    print("-" * 70)
    docs = exercise_4_write_documentation()
    print(f"Model: {docs['models'][0]['name']}")
    print(f"Description: {docs['models'][0]['description']}")
    print(f"Documented columns: {len(docs['models'][0]['columns'])}")
    
    print("\n📋 Exercise 5: Create Macro")
    print("-" * 70)
    macro = exercise_5_create_macro()
    print(f"Macro usage: {macro('amount_cents')}")
    
    print("\n📋 Exercise 6: Incremental Model")
    print("-" * 70)
    sql = exercise_6_incremental_model()
    print(sql)
    
    print("\n📋 Exercise 7: Custom Test")
    print("-" * 70)
    sql = exercise_7_custom_test()
    print(sql)
    
    print("\n📋 Exercise 8: Generate Lineage")
    print("-" * 70)
    lineage = exercise_8_generate_lineage()
    viz = visualize_lineage(lineage)
    print(viz)
    
    print("=" * 70)
    print("✅ All exercises completed!")
    print("=" * 70)
