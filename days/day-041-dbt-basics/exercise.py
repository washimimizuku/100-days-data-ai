"""
Day 41: dbt Basics - Exercises

Note: These exercises simulate dbt concepts in Python.
In practice, dbt uses SQL files and YAML configuration.
"""


def exercise_1_create_staging_model():
    """
    Exercise 1: Create staging model
    
    Write SQL for a staging model that:
    1. Selects from raw.customers
    2. Normalizes email (lowercase, trim)
    3. Creates full_name from first_name and last_name
    4. Filters out null emails
    
    Return the SQL string
    """
    # TODO: Write SQL for staging model
    # TODO: Use {{ source('raw', 'customers') }}
    # TODO: Apply transformations
    # TODO: Return SQL string
    pass


def exercise_2_create_intermediate_model():
    """
    Exercise 2: Create intermediate model
    
    Write SQL that:
    1. Joins stg_customers and stg_orders
    2. Aggregates: count orders, sum amount, max date
    3. Groups by customer
    
    Return the SQL string
    """
    # TODO: Write SQL with CTEs
    # TODO: Use {{ ref('stg_customers') }}
    # TODO: Use {{ ref('stg_orders') }}
    # TODO: Add aggregations
    # TODO: Return SQL string
    pass


def exercise_3_add_tests():
    """
    Exercise 3: Add data quality tests
    
    Create test configuration for stg_customers:
    1. customer_id: unique, not_null
    2. email: not_null
    3. status: accepted_values ['active', 'inactive']
    
    Return YAML configuration as dict
    """
    # TODO: Create test configuration dict
    # TODO: Add unique test for customer_id
    # TODO: Add not_null tests
    # TODO: Add accepted_values test
    # TODO: Return configuration dict
    pass


def exercise_4_write_documentation():
    """
    Exercise 4: Write model documentation
    
    Create documentation for stg_customers:
    - Model description
    - Column descriptions for: customer_id, email, full_name
    
    Return documentation as dict
    """
    # TODO: Create documentation dict
    # TODO: Add model description
    # TODO: Add column descriptions
    # TODO: Return documentation dict
    pass


def exercise_5_create_macro():
    """
    Exercise 5: Create reusable macro
    
    Write a macro that converts cents to dollars:
    - Input: column name
    - Output: SQL expression (column / 100.0)::decimal(10,2)
    
    Return macro function
    """
    # TODO: Define macro function
    # TODO: Take column_name as parameter
    # TODO: Return SQL expression
    pass


def exercise_6_incremental_model():
    """
    Exercise 6: Create incremental model
    
    Write SQL for incremental orders model:
    1. Select from stg_orders
    2. If incremental, filter for new records only
    3. Use order_date > MAX(order_date)
    
    Return SQL string with incremental logic
    """
    # TODO: Write base SELECT
    # TODO: Add incremental filter logic
    # TODO: Use {% if is_incremental() %}
    # TODO: Return SQL string
    pass


def exercise_7_custom_test():
    """
    Exercise 7: Write custom test
    
    Create test that checks:
    - All order amounts are positive
    - Returns rows that fail the test
    
    Return SQL for custom test
    """
    # TODO: Write SQL that selects failing rows
    # TODO: Check amount > 0
    # TODO: Return SQL string
    pass


def exercise_8_generate_lineage():
    """
    Exercise 8: Generate model lineage
    
    Given these models:
    - stg_customers (from raw.customers)
    - stg_orders (from raw.orders)
    - int_customer_orders (from stg_customers, stg_orders)
    - fct_orders (from stg_orders)
    
    Return lineage graph as dict
    """
    # TODO: Create lineage graph
    # TODO: Map each model to its dependencies
    # TODO: Return lineage dict
    pass


# Helper function to simulate dbt ref()
def ref(model_name):
    """Simulate dbt ref() function"""
    return f"analytics.{model_name}"


# Helper function to simulate dbt source()
def source(source_name, table_name):
    """Simulate dbt source() function"""
    return f"{source_name}.{table_name}"


if __name__ == "__main__":
    print("Day 41: dbt Basics - Exercises\n")
    print("Note: These exercises simulate dbt concepts in Python.")
    print("In practice, dbt uses SQL files and YAML configuration.\n")
    
    # Uncomment to run exercises
    # print("Exercise 1: Create Staging Model")
    # sql = exercise_1_create_staging_model()
    # print(sql)
    
    # print("\nExercise 2: Create Intermediate Model")
    # sql = exercise_2_create_intermediate_model()
    # print(sql)
    
    # print("\nExercise 3: Add Tests")
    # tests = exercise_3_add_tests()
    # print(tests)
    
    # print("\nExercise 4: Write Documentation")
    # docs = exercise_4_write_documentation()
    # print(docs)
    
    # print("\nExercise 5: Create Macro")
    # macro = exercise_5_create_macro()
    # print(macro('amount_cents'))
    
    # print("\nExercise 6: Incremental Model")
    # sql = exercise_6_incremental_model()
    # print(sql)
    
    # print("\nExercise 7: Custom Test")
    # sql = exercise_7_custom_test()
    # print(sql)
    
    # print("\nExercise 8: Generate Lineage")
    # lineage = exercise_8_generate_lineage()
    # print(lineage)
