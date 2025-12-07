"""
Day 8: Table Formats Introduction - Exercises
Complete each exercise below
"""

# Exercise 1: Concept Understanding
# TODO: Write your answers as comments

print("Exercise 1: Concept Understanding")
print("="*50)

# Q1: What is the difference between file format and table format?
# Your answer:


# Q2: List 3 problems that table formats solve
# Your answer:
# 1.
# 2.
# 3.

# Q3: Name the 3 main table formats and their creators
# Your answer:


# Exercise 2: Feature Comparison
# TODO: Fill in the comparison matrix

print("\nExercise 2: Feature Comparison")
print("="*50)

comparison = {
    'Iceberg': {
        'creator': '?',
        'hidden_partitioning': '?',
        'best_for': '?'
    },
    'Delta Lake': {
        'creator': '?',
        'hidden_partitioning': '?',
        'best_for': '?'
    },
    'Hudi': {
        'creator': '?',
        'hidden_partitioning': '?',
        'best_for': '?'
    }
}

# Print your comparison
for format_name, features in comparison.items():
    print(f"\n{format_name}:")
    for key, value in features.items():
        print(f"  {key}: {value}")


# Exercise 3: ACID Transactions
# TODO: Explain each ACID property

print("\nExercise 3: ACID Transactions")
print("="*50)

acid_properties = {
    'Atomicity': {
        'definition': '?',
        'example': '?'
    },
    'Consistency': {
        'definition': '?',
        'example': '?'
    },
    'Isolation': {
        'definition': '?',
        'example': '?'
    },
    'Durability': {
        'definition': '?',
        'example': '?'
    }
}

# Print your explanations
for property_name, details in acid_properties.items():
    print(f"\n{property_name}:")
    for key, value in details.items():
        print(f"  {key}: {value}")


# Exercise 4: Use Case Analysis
# TODO: Choose appropriate table format for each scenario

print("\nExercise 4: Use Case Analysis")
print("="*50)

scenarios = {
    'Scenario 1: Multi-cloud data lake with Spark, Trino, and Flink': '?',
    'Scenario 2: Databricks-based data platform': '?',
    'Scenario 3: Real-time CDC from MySQL to data lake': '?',
    'Scenario 4: Large-scale analytics with partition evolution': '?',
    'Scenario 5: Streaming + batch processing on Spark': '?'
}

# Fill in your choices and print
for scenario, choice in scenarios.items():
    print(f"\n{scenario}")
    print(f"  Choice: {choice}")
    print(f"  Reason: ?")


# Exercise 5: Time Travel
# TODO: Answer time travel questions

print("\nExercise 5: Time Travel")
print("="*50)

# Q1: What is time travel in table formats?
# Your answer:


# Q2: Give 3 use cases for time travel
# Your answer:
# 1.
# 2.
# 3.

# Q3: How does time travel work technically?
# Your answer:


# Bonus Challenge
# TODO: Design a table format selection flowchart

print("\nBonus Challenge: Selection Flowchart")
print("="*50)

def recommend_table_format(use_case):
    """
    Recommend table format based on use case
    
    Args:
        use_case: dict with keys:
            - engines: list of query engines
            - updates_frequency: 'low', 'medium', 'high'
            - platform: 'databricks', 'aws', 'multi-cloud'
            - partition_changes: bool
    
    Returns:
        str: recommended format
    """
    # Your implementation here
    pass

# Test your function
test_cases = [
    {
        'engines': ['spark', 'trino', 'flink'],
        'updates_frequency': 'low',
        'platform': 'multi-cloud',
        'partition_changes': True
    },
    {
        'engines': ['spark'],
        'updates_frequency': 'medium',
        'platform': 'databricks',
        'partition_changes': False
    },
    {
        'engines': ['spark'],
        'updates_frequency': 'high',
        'platform': 'aws',
        'partition_changes': False
    }
]

for i, test in enumerate(test_cases, 1):
    result = recommend_table_format(test)
    print(f"\nTest {i}: {result}")
