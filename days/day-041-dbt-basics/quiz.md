# Day 41 Quiz: dbt Basics

## Questions

1. **What is dbt (data build tool)?**
   - A) A database management system
   - B) A transformation framework that transforms data in the warehouse using SQL
   - C) An ETL tool for data extraction
   - D) A data visualization tool

2. **What does dbt NOT do?**
   - A) Transform data
   - B) Test data quality
   - C) Extract and load data (it's the T in ELT)
   - D) Generate documentation

3. **What is a dbt model?**
   - A) A machine learning model
   - B) A SQL file that defines a transformation
   - C) A database schema
   - D) A data visualization

4. **What does the ref() function do in dbt?**
   - A) Deletes references
   - B) References other dbt models and creates dependencies
   - C) References external APIs
   - D) References documentation

5. **What does the source() function do in dbt?**
   - A) Creates new data sources
   - B) References raw data tables defined in sources.yml
   - C) Downloads data from sources
   - D) Validates data sources


6. **What are the four built-in dbt tests?**
   - A) select, insert, update, delete
   - B) unique, not_null, accepted_values, relationships
   - C) count, sum, avg, max
   - D) create, read, update, delete

7. **What is a dbt materialization?**
   - A) A data type
   - B) How a model is persisted in the warehouse (view, table, incremental, ephemeral)
   - C) A test type
   - D) A documentation format

8. **What does 'dbt run' do?**
   - A) Runs tests
   - B) Executes models and creates tables/views in the warehouse
   - C) Generates documentation
   - D) Deletes old data

9. **What is the purpose of dbt's automatic lineage tracking?**
   - A) To slow down queries
   - B) To understand dependencies between models and enable impact analysis
   - C) To create backups
   - D) To encrypt data

10. **What is an incremental model in dbt?**
    - A) A model that runs faster
    - B) A model that only processes new/changed records instead of full refresh
    - C) A model with more columns
    - D) A temporary model

---

## Answers

1. **B** - dbt (data build tool) is a transformation framework that enables analytics engineers to transform data in their warehouse using SQL and software engineering best practices.

2. **C** - dbt does NOT extract and load data. It's the "T" (Transform) in ELT. Use tools like Fivetran or Airbyte for extraction and loading.

3. **B** - A dbt model is a SQL file that defines a transformation. Each model typically creates one table or view in the warehouse.

4. **B** - The ref() function references other dbt models and automatically creates dependencies in the DAG, enabling proper execution order and lineage tracking.

5. **B** - The source() function references raw data tables that are defined in sources.yml files, providing a layer of abstraction over raw table names.

6. **B** - The four built-in dbt tests are: unique (no duplicates), not_null (no nulls), accepted_values (value in allowed list), and relationships (foreign key exists).

7. **B** - Materialization defines how a model is persisted in the warehouse: view (virtual), table (physical), incremental (append new records), or ephemeral (CTE only).

8. **B** - 'dbt run' executes all models (or selected models) and creates the corresponding tables/views in the data warehouse based on the SQL transformations.

9. **B** - dbt automatically tracks lineage by analyzing ref() and source() functions, enabling understanding of dependencies and impact analysis when models change.

10. **B** - An incremental model only processes new or changed records instead of doing a full refresh, making it efficient for large datasets that grow over time.

---

## Scoring

- **9-10 correct**: Excellent! You understand dbt fundamentals.
- **7-8 correct**: Good! Review materializations and tests.
- **5-6 correct**: Fair. Revisit core concepts and functions.
- **Below 5**: Review the README and dbt documentation.
