"""
Day 8: Table Formats Introduction - Solutions
"""

# Exercise 1: Concept Understanding
print("Exercise 1: Concept Understanding")
print("="*50)

print("\nQ1: Difference between file format and table format:")
print("""
File Format (Parquet, Avro):
- How individual files are stored
- Single file operations
- No transaction support

Table Format (Iceberg, Delta, Hudi):
- How collections of files work together as a table
- Multi-file ACID transactions
- Schema evolution, time travel, metadata management
""")

print("\nQ2: Problems table formats solve:")
print("""
1. ACID transactions - Atomic, consistent, isolated, durable operations
2. Concurrent reads/writes - Multiple users can work simultaneously
3. Schema evolution - Change schemas without breaking existing data
4. Time travel - Query historical versions of data
5. Metadata management - Track files, partitions, statistics
""")

print("\nQ3: Main table formats:")
print("""
1. Apache Iceberg - Created by Netflix (2017)
2. Delta Lake - Created by Databricks (2019)
3. Apache Hudi - Created by Uber (2016)
""")

# Exercise 2: Feature Comparison
print("\n" + "="*50)
print("Exercise 2: Feature Comparison")
print("="*50)

comparison = {
    'Iceberg': {
        'creator': 'Netflix',
        'year': 2017,
        'hidden_partitioning': 'Yes',
        'partition_evolution': 'Yes',
        'best_for': 'Multi-engine environments, complex schema evolution'
    },
    'Delta Lake': {
        'creator': 'Databricks',
        'year': 2019,
        'hidden_partitioning': 'No',
        'partition_evolution': 'No',
        'best_for': 'Databricks, Spark-heavy workloads, streaming + batch'
    },
    'Hudi': {
        'creator': 'Uber',
        'year': 2016,
        'hidden_partitioning': 'No',
        'partition_evolution': 'No',
        'best_for': 'CDC workloads, frequent updates/deletes'
    }
}

for format_name, features in comparison.items():
    print(f"\n{format_name}:")
    for key, value in features.items():
        print(f"  {key}: {value}")

# Exercise 3: ACID Transactions
print("\n" + "="*50)
print("Exercise 3: ACID Transactions")
print("="*50)

acid_properties = {
    'Atomicity': {
        'definition': 'All or nothing - operations either complete fully or not at all',
        'example': 'Adding 100 files to a table - either all 100 are added or none',
        'why_matters': 'Prevents partial writes that corrupt data'
    },
    'Consistency': {
        'definition': 'Data always in valid state, constraints maintained',
        'example': 'Schema validation ensures all rows match table schema',
        'why_matters': 'Prevents invalid data from entering the table'
    },
    'Isolation': {
        'definition': 'Concurrent operations don\'t interfere with each other',
        'example': 'Two writers can write simultaneously without conflicts',
        'why_matters': 'Enables multiple users to work concurrently'
    },
    'Durability': {
        'definition': 'Once committed, changes are permanent',
        'example': 'After commit(), data survives system crashes',
        'why_matters': 'Guarantees data won\'t be lost'
    }
}

for property_name, details in acid_properties.items():
    print(f"\n{property_name}:")
    for key, value in details.items():
        print(f"  {key}: {value}")

# Exercise 4: Use Case Analysis
print("\n" + "="*50)
print("Exercise 4: Use Case Analysis")
print("="*50)

scenarios = [
    {
        'scenario': 'Multi-cloud data lake with Spark, Trino, and Flink',
        'choice': 'Iceberg',
        'reason': 'Best multi-engine support, works across all three engines seamlessly'
    },
    {
        'scenario': 'Databricks-based data platform',
        'choice': 'Delta Lake',
        'reason': 'Native Databricks integration, optimized for Spark, best performance'
    },
    {
        'scenario': 'Real-time CDC from MySQL to data lake',
        'choice': 'Hudi',
        'reason': 'Designed for CDC, excellent upsert/delete performance, incremental processing'
    },
    {
        'scenario': 'Large-scale analytics with partition evolution',
        'choice': 'Iceberg',
        'reason': 'Only format with partition evolution, hidden partitioning simplifies queries'
    },
    {
        'scenario': 'Streaming + batch processing on Spark',
        'choice': 'Delta Lake',
        'reason': 'Excellent streaming support, unified batch + streaming, Spark-optimized'
    }
]

for item in scenarios:
    print(f"\n{item['scenario']}")
    print(f"  Choice: {item['choice']}")
    print(f"  Reason: {item['reason']}")

# Exercise 5: Time Travel
print("\n" + "="*50)
print("Exercise 5: Time Travel")
print("="*50)

print("\nQ1: What is time travel?")
print("""
Time travel is the ability to query historical versions of a table.
Each write creates a new snapshot/version, and you can read any previous version.
""")

print("\nQ2: Use cases for time travel:")
use_cases = [
    "Audit and compliance - See what data looked like at specific time",
    "Debugging - Investigate when bad data was introduced",
    "Rollback - Revert to previous version if something goes wrong",
    "Reproducibility - Recreate ML training data from specific point in time",
    "A/B testing - Compare results from different data versions"
]
for i, use_case in enumerate(use_cases, 1):
    print(f"{i}. {use_case}")

print("\nQ3: How time travel works:")
print("""
Technical implementation:
1. Each write creates a new snapshot/version with unique ID
2. Metadata tracks which files belong to each snapshot
3. Old data files are kept (not deleted immediately)
4. Query specifies snapshot ID or timestamp
5. Table format reads metadata for that snapshot
6. Returns data as it existed at that point in time

Example:
- Version 1: files [A, B]
- Version 2: files [A, B, C] (added C)
- Version 3: files [A, C] (removed B)
- Query version 1: reads [A, B]
- Query version 3: reads [A, C]
""")

# Bonus Challenge
print("\n" + "="*50)
print("Bonus Challenge: Selection Flowchart")
print("="*50)

def recommend_table_format(use_case):
    """
    Recommend table format based on use case
    """
    engines = use_case.get('engines', [])
    updates_freq = use_case.get('updates_frequency', 'low')
    platform = use_case.get('platform', 'multi-cloud')
    partition_changes = use_case.get('partition_changes', False)
    
    # Databricks platform
    if platform == 'databricks':
        return {
            'format': 'Delta Lake',
            'reason': 'Native Databricks integration, best performance on platform'
        }
    
    # High update frequency
    if updates_freq == 'high':
        return {
            'format': 'Hudi',
            'reason': 'Optimized for frequent updates/deletes, excellent CDC support'
        }
    
    # Partition evolution needed
    if partition_changes:
        return {
            'format': 'Iceberg',
            'reason': 'Only format supporting partition evolution'
        }
    
    # Multiple engines
    if len(engines) > 1 or 'trino' in engines or 'flink' in engines:
        return {
            'format': 'Iceberg',
            'reason': 'Best multi-engine support across Spark, Trino, Flink, Presto'
        }
    
    # Default: Spark-only, low updates
    return {
        'format': 'Delta Lake',
        'reason': 'Good default choice, mature, well-supported'
    }

# Test cases
test_cases = [
    {
        'name': 'Multi-engine data lake',
        'use_case': {
            'engines': ['spark', 'trino', 'flink'],
            'updates_frequency': 'low',
            'platform': 'multi-cloud',
            'partition_changes': True
        }
    },
    {
        'name': 'Databricks platform',
        'use_case': {
            'engines': ['spark'],
            'updates_frequency': 'medium',
            'platform': 'databricks',
            'partition_changes': False
        }
    },
    {
        'name': 'CDC workload',
        'use_case': {
            'engines': ['spark'],
            'updates_frequency': 'high',
            'platform': 'aws',
            'partition_changes': False
        }
    }
]

for test in test_cases:
    result = recommend_table_format(test['use_case'])
    print(f"\n{test['name']}:")
    print(f"  Recommended: {result['format']}")
    print(f"  Reason: {result['reason']}")

print("\n✅ All exercises completed!")
