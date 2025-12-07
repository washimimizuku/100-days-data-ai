"""
Day 3: Apache Arrow - Solutions
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd
import time
import sys

# Exercise 1: Create Arrow Table
print("Exercise 1: Create Arrow Table")

data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 28],
    'salary': [75000, 85000, 70000]
}

# Create Arrow table
table = pa.table(data)

print("Schema:")
print(table.schema)
print("\nData:")
print(table.to_pandas())

# Convert to Pandas
df = table.to_pandas()
print(f"\nConverted to Pandas: {type(df)}")
print()

# Exercise 2: Arrow vs Pandas Memory
print("Exercise 2: Arrow vs Pandas Memory")

# Create large dataset
data_large = {'col': list(range(1000000))}

# Pandas
df = pd.DataFrame(data_large)
pandas_memory = df.memory_usage(deep=True).sum()

# Arrow
table = pa.table(data_large)
arrow_memory = table.nbytes

print(f"Pandas memory: {pandas_memory / 1024 / 1024:.2f} MB")
print(f"Arrow memory: {arrow_memory / 1024 / 1024:.2f} MB")
print(f"Memory savings: {(1 - arrow_memory/pandas_memory) * 100:.1f}%")
print()

# Exercise 3: Parquet with Arrow
print("Exercise 3: Parquet with Arrow")

# Create sample data first
df_sample = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'age': [25, 30, 28, 35, 32],
    'salary': [75000, 95000, 70000, 90000, 82000]
})
df_sample.to_parquet('sample_data.parquet')

# Read with Arrow
table = pq.read_table('sample_data.parquet')
print(f"Read {table.num_rows} rows")

# Select columns
subset = table.select(['name', 'salary'])
print(f"\nSelected columns: {subset.column_names}")

# Filter rows
filtered = table.filter(pc.greater(table['salary'], 80000))
print(f"Filtered to {filtered.num_rows} rows where salary > 80000")

# Write back
pq.write_table(filtered, 'filtered_data.parquet')
print("Written filtered data")
print()

# Exercise 4: Arrow Compute Functions
print("Exercise 4: Arrow Compute Functions")

table = pa.table({
    'id': range(1000),
    'value': [i * 1.5 for i in range(1000)],
    'category': ['A', 'B', 'C'] * 333 + ['A']
})

# Compute statistics
mean_val = pc.mean(table['value'])
max_val = pc.max(table['value'])
min_val = pc.min(table['value'])

print(f"Mean: {mean_val.as_py():.2f}")
print(f"Max: {max_val.as_py():.2f}")
print(f"Min: {min_val.as_py():.2f}")

# Sort
sorted_table = table.sort_by([('value', 'descending')])
print(f"\nSorted by value (descending)")
print(f"Top value: {sorted_table['value'][0].as_py():.2f}")

# Compare with Pandas
df = table.to_pandas()

start = time.time()
df['value'].mean()
pandas_time = time.time() - start

start = time.time()
pc.mean(table['value'])
arrow_time = time.time() - start

print(f"\nPandas mean: {pandas_time:.6f}s")
print(f"Arrow mean: {arrow_time:.6f}s")
print()

# Exercise 5: Zero-Copy Demo
print("Exercise 5: Zero-Copy Demo")

# Create large DataFrame
df_large = pd.DataFrame({
    'id': range(100000),
    'value': range(100000)
})

# Convert to Arrow
table = pa.Table.from_pandas(df_large)

# Access column (zero-copy)
start = time.time()
arrow_col = table['id']
arrow_time = time.time() - start

# Pandas column access
start = time.time()
pandas_col = df_large['id']
pandas_time = time.time() - start

print(f"Arrow column access: {arrow_time:.6f}s")
print(f"Pandas column access: {pandas_time:.6f}s")
print(f"Arrow is {pandas_time/arrow_time:.1f}x faster (zero-copy)")
print()

# Bonus Challenge
print("Bonus Challenge: Arrow ETL Pipeline")

def arrow_etl_pipeline(input_file, output_file):
    """
    ETL pipeline using Arrow
    """
    start = time.time()
    
    # Extract: Read from Parquet
    table = pq.read_table(input_file)
    
    # Transform: Filter and compute
    table = table.filter(pc.greater(table['value'], 500))
    
    # Add computed column
    doubled = pc.multiply(table['value'], 2)
    table = table.append_column('value_doubled', doubled)
    
    # Load: Write to Parquet
    pq.write_table(table, output_file)
    
    elapsed = time.time() - start
    return elapsed

# Create test data
test_df = pd.DataFrame({
    'id': range(10000),
    'value': range(10000)
})
test_df.to_parquet('test_input.parquet')

# Run Arrow pipeline
arrow_time = arrow_etl_pipeline('test_input.parquet', 'test_output_arrow.parquet')
print(f"Arrow ETL: {arrow_time:.3f}s")

# Compare with Pandas
start = time.time()
df = pd.read_parquet('test_input.parquet')
df = df[df['value'] > 500]
df['value_doubled'] = df['value'] * 2
df.to_parquet('test_output_pandas.parquet')
pandas_time = time.time() - start

print(f"Pandas ETL: {pandas_time:.3f}s")
print(f"Arrow is {pandas_time/arrow_time:.2f}x faster")

print("\n✅ All exercises completed!")
