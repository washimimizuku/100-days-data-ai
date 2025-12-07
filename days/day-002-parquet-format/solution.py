"""
Day 2: Parquet Format - Solutions
"""

import pandas as pd
import pyarrow.parquet as pq
import time
import os

# Exercise 1: Basic Parquet Operations
print("Exercise 1: Basic Parquet Operations")

df = pd.DataFrame({
    'id': range(1000),
    'name': [f'User_{i}' for i in range(1000)],
    'value': [i * 1.5 for i in range(1000)],
    'category': [f'Cat_{i % 5}' for i in range(1000)]
})

# Write to Parquet
df.to_parquet('data.parquet', compression='snappy')
print(f"Written {len(df)} rows to data.parquet")

# Read back
df_read = pd.read_parquet('data.parquet')
print(f"Read {len(df_read)} rows")
print(f"Data verified: {df.equals(df_read)}")
print()

# Exercise 2: Compression Comparison
print("Exercise 2: Compression Comparison")

df_large = pd.DataFrame({
    'id': range(10000),
    'value': range(10000),
    'text': [f'Text_{i}' * 10 for i in range(10000)]
})

compressions = ['snappy', 'gzip', None]
results = {}

for comp in compressions:
    comp_name = comp if comp else 'none'
    filename = f'data_{comp_name}.parquet'
    
    # Write
    start = time.time()
    df_large.to_parquet(filename, compression=comp)
    write_time = time.time() - start
    
    # Get size
    size = os.path.getsize(filename)
    
    # Read
    start = time.time()
    pd.read_parquet(filename)
    read_time = time.time() - start
    
    results[comp_name] = {
        'size_kb': size / 1024,
        'write_time': write_time,
        'read_time': read_time
    }

for comp, stats in results.items():
    print(f"{comp}:")
    print(f"  Size: {stats['size_kb']:.2f} KB")
    print(f"  Write: {stats['write_time']:.3f}s")
    print(f"  Read: {stats['read_time']:.3f}s")
print()

# Exercise 3: Column Selection
print("Exercise 3: Column Selection")

# Create wide DataFrame
df_wide = pd.DataFrame({
    f'col_{i}': range(5000) for i in range(20)
})

df_wide.to_parquet('wide_data.parquet')

# Read all columns
start = time.time()
df_all = pd.read_parquet('wide_data.parquet')
time_all = time.time() - start

# Read 3 columns
start = time.time()
df_subset = pd.read_parquet('wide_data.parquet', columns=['col_0', 'col_5', 'col_10'])
time_subset = time.time() - start

print(f"Read all 20 columns: {time_all:.3f}s")
print(f"Read 3 columns: {time_subset:.3f}s")
print(f"Speedup: {time_all/time_subset:.2f}x faster")
print()

# Exercise 4: CSV to Parquet Conversion
print("Exercise 4: CSV to Parquet Conversion")

# Read CSV
df_csv = pd.read_csv('../../data/day-001/employees.csv')

# Write to Parquet
df_csv.to_parquet('employees.parquet', compression='snappy')

# Compare sizes
csv_size = os.path.getsize('../../data/day-001/employees.csv')
parquet_size = os.path.getsize('employees.parquet')

print(f"CSV size: {csv_size / 1024:.2f} KB")
print(f"Parquet size: {parquet_size / 1024:.2f} KB")
print(f"Compression ratio: {csv_size / parquet_size:.2f}x smaller")

# Compare read performance
start = time.time()
pd.read_csv('../../data/day-001/employees.csv')
csv_time = time.time() - start

start = time.time()
pd.read_parquet('employees.parquet')
parquet_time = time.time() - start

print(f"CSV read: {csv_time:.4f}s")
print(f"Parquet read: {parquet_time:.4f}s")
print()

# Exercise 5: Metadata Inspection
print("Exercise 5: Metadata Inspection")

parquet_file = pq.ParquetFile('employees.parquet')

# Schema
print("Schema:")
print(parquet_file.schema)

# Metadata
print(f"\nRows: {parquet_file.metadata.num_rows}")
print(f"Columns: {parquet_file.metadata.num_columns}")
print(f"Row groups: {parquet_file.metadata.num_row_groups}")

# Column statistics
print("\nColumn Statistics:")
for i in range(parquet_file.metadata.num_row_groups):
    row_group = parquet_file.metadata.row_group(i)
    for j in range(row_group.num_columns):
        column = row_group.column(j)
        if column.statistics:
            print(f"  {column.path_in_schema}: min={column.statistics.min}, max={column.statistics.max}")
print()

# Exercise 6: Partitioned Data
print("Exercise 6: Partitioned Data")

# Create sales data with dates
from datetime import datetime, timedelta

dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(365)]
df_sales = pd.DataFrame({
    'date': dates,
    'amount': [100 + i * 10 for i in range(365)],
    'product': ['Product_A', 'Product_B'] * 182 + ['Product_A']
})

# Add year and month columns
df_sales['year'] = df_sales['date'].dt.year
df_sales['month'] = df_sales['date'].dt.month

# Write partitioned
df_sales.to_parquet(
    'sales_partitioned',
    partition_cols=['year', 'month'],
    compression='snappy'
)
print("Written partitioned data to sales_partitioned/")

# Read specific partition
df_jan = pd.read_parquet('sales_partitioned/year=2024/month=1')
print(f"January data: {len(df_jan)} rows")
print()

# Bonus Challenge
print("Bonus Challenge: CSV to Parquet Converter")

def csv_to_parquet(csv_file, parquet_file, compression='snappy'):
    """
    Convert CSV file to Parquet format
    
    Args:
        csv_file: Path to input CSV
        parquet_file: Path to output Parquet
        compression: Compression algorithm
    
    Returns:
        dict with size comparison
    """
    try:
        # Read CSV
        df = pd.read_csv(csv_file)
        csv_size = os.path.getsize(csv_file)
        
        # Write Parquet
        df.to_parquet(parquet_file, compression=compression)
        parquet_size = os.path.getsize(parquet_file)
        
        return {
            'success': True,
            'rows': len(df),
            'csv_size_kb': csv_size / 1024,
            'parquet_size_kb': parquet_size / 1024,
            'compression_ratio': csv_size / parquet_size,
            'space_saved_percent': ((csv_size - parquet_size) / csv_size) * 100
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

# Test
result = csv_to_parquet('../../data/day-001/employees.csv', 'test_output.parquet')
if result['success']:
    print(f"Converted {result['rows']} rows")
    print(f"CSV: {result['csv_size_kb']:.2f} KB")
    print(f"Parquet: {result['parquet_size_kb']:.2f} KB")
    print(f"Compression: {result['compression_ratio']:.2f}x")
    print(f"Space saved: {result['space_saved_percent']:.1f}%")

print("\n✅ All exercises completed!")
