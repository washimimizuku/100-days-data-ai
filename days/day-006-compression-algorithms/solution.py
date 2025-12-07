"""
Day 6: Compression Algorithms - Solutions
"""

import pandas as pd
import time
import os

# Exercise 1: Compression Benchmark
print("Exercise 1: Compression Benchmark")

df = pd.DataFrame({
    'id': range(50000),
    'name': [f'User_{i}' for i in range(50000)],
    'text': ['Sample text data ' * 20 for _ in range(50000)]
})

compressions = ['snappy', 'gzip', 'zstd', None]
results = {}

for comp in compressions:
    comp_name = comp if comp else 'none'
    filename = f'benchmark_{comp_name}.parquet'
    
    # Write
    start = time.time()
    df.to_parquet(filename, compression=comp)
    write_time = time.time() - start
    
    # Size
    size = os.path.getsize(filename)
    
    results[comp_name] = {
        'write_time': write_time,
        'size_mb': size / 1024 / 1024
    }

print("Write Performance:")
for comp, stats in results.items():
    print(f"{comp}:")
    print(f"  Write time: {stats['write_time']:.3f}s")
    print(f"  File size: {stats['size_mb']:.2f} MB")
print()

# Exercise 2: Read Performance
print("Exercise 2: Read Performance")

for comp in compressions:
    comp_name = comp if comp else 'none'
    filename = f'benchmark_{comp_name}.parquet'
    
    # Read
    start = time.time()
    pd.read_parquet(filename)
    read_time = time.time() - start
    
    results[comp_name]['read_time'] = read_time
    
    # Compression ratio
    uncompressed_size = results['none']['size_mb']
    compressed_size = results[comp_name]['size_mb']
    ratio = uncompressed_size / compressed_size if compressed_size > 0 else 1
    results[comp_name]['ratio'] = ratio

print("Read Performance & Compression Ratios:")
for comp, stats in results.items():
    print(f"{comp}:")
    print(f"  Read time: {stats['read_time']:.3f}s")
    print(f"  Compression ratio: {stats['ratio']:.2f}x")

# Best overall
best_balanced = min(
    [(k, v) for k, v in results.items() if k != 'none'],
    key=lambda x: x[1]['write_time'] + x[1]['read_time'] + (1/x[1]['ratio'])
)
print(f"\nBest balanced: {best_balanced[0]}")
print()

# Exercise 3: ZSTD Levels
print("Exercise 3: ZSTD Levels")

df_zstd = pd.DataFrame({
    'id': range(30000),
    'text': ['Text data ' * 30 for _ in range(30000)]
})

levels = [1, 3, 10, 22]
zstd_results = {}

for level in levels:
    filename = f'zstd_level_{level}.parquet'
    
    # Write
    start = time.time()
    df_zstd.to_parquet(filename, compression='zstd', compression_level=level)
    write_time = time.time() - start
    
    # Size
    size = os.path.getsize(filename)
    
    # Read
    start = time.time()
    pd.read_parquet(filename)
    read_time = time.time() - start
    
    zstd_results[level] = {
        'write_time': write_time,
        'read_time': read_time,
        'size_mb': size / 1024 / 1024
    }

print("ZSTD Compression Levels:")
for level, stats in zstd_results.items():
    print(f"Level {level}:")
    print(f"  Write: {stats['write_time']:.3f}s")
    print(f"  Read: {stats['read_time']:.3f}s")
    print(f"  Size: {stats['size_mb']:.2f} MB")

# Optimal level (balance of speed and size)
optimal = min(zstd_results.items(), key=lambda x: x[1]['write_time'] + x[1]['size_mb'])
print(f"\nOptimal level: {optimal[0]}")
print()

# Exercise 4: Data Type Impact
print("Exercise 4: Data Type Impact")

# Numeric data
df_numeric = pd.DataFrame({
    'col1': range(20000),
    'col2': [i * 1.5 for i in range(20000)],
    'col3': [i ** 2 for i in range(20000)]
})

# Text data
df_text = pd.DataFrame({
    'col1': [f'Text_{i}' * 10 for i in range(20000)],
    'col2': [f'Data_{i}' * 10 for i in range(20000)],
    'col3': [f'Value_{i}' * 10 for i in range(20000)]
})

for data_type, df_test in [('numeric', df_numeric), ('text', df_text)]:
    print(f"\n{data_type.capitalize()} data:")
    
    for comp in ['snappy', 'gzip']:
        filename = f'{data_type}_{comp}.parquet'
        df_test.to_parquet(filename, compression=comp)
        size = os.path.getsize(filename)
        print(f"  {comp}: {size / 1024:.2f} KB")
    
    # Uncompressed
    filename = f'{data_type}_none.parquet'
    df_test.to_parquet(filename, compression=None)
    uncompressed = os.path.getsize(filename)
    print(f"  none: {uncompressed / 1024:.2f} KB")
print()

# Exercise 5: Cost Calculation
print("Exercise 5: Cost Calculation")

data_size_tb = 1  # 1 TB
data_size_gb = data_size_tb * 1024
cost_per_gb_month = 0.023

compression_ratios = {
    'none': 1.0,
    'snappy': 2.5,
    'zstd': 3.5,
    'gzip': 4.5
}

print(f"Storage costs for {data_size_tb}TB of data:")
for comp, ratio in compression_ratios.items():
    compressed_gb = data_size_gb / ratio
    monthly_cost = compressed_gb * cost_per_gb_month
    annual_cost = monthly_cost * 12
    
    print(f"\n{comp} ({ratio}x compression):")
    print(f"  Compressed size: {compressed_gb:.0f} GB")
    print(f"  Monthly cost: ${monthly_cost:.2f}")
    print(f"  Annual cost: ${annual_cost:.2f}")

# Savings
no_compression_annual = data_size_gb * cost_per_gb_month * 12
zstd_annual = (data_size_gb / 3.5) * cost_per_gb_month * 12
savings = no_compression_annual - zstd_annual
print(f"\nAnnual savings with ZSTD: ${savings:.2f}")
print()

# Exercise 6: Real-World Scenario
print("Exercise 6: Real-World Scenario")

# Simulate different data tiers
scenarios = {
    'hot': {'size_gb': 100, 'compression': 'snappy', 'ratio': 2.5},
    'warm': {'size_gb': 500, 'compression': 'zstd', 'ratio': 3.5},
    'cold': {'size_gb': 1000, 'compression': 'gzip', 'ratio': 4.5}
}

total_uncompressed = 0
total_compressed = 0

print("Data Lake Storage Strategy:")
for tier, config in scenarios.items():
    uncompressed = config['size_gb']
    compressed = uncompressed / config['ratio']
    monthly_cost = compressed * cost_per_gb_month
    
    total_uncompressed += uncompressed
    total_compressed += compressed
    
    print(f"\n{tier.capitalize()} tier:")
    print(f"  Size: {uncompressed} GB")
    print(f"  Compression: {config['compression']}")
    print(f"  Compressed: {compressed:.0f} GB")
    print(f"  Monthly cost: ${monthly_cost:.2f}")

print(f"\nTotal:")
print(f"  Uncompressed: {total_uncompressed} GB")
print(f"  Compressed: {total_compressed:.0f} GB")
print(f"  Compression ratio: {total_uncompressed/total_compressed:.2f}x")
print(f"  Monthly savings: ${(total_uncompressed - total_compressed) * cost_per_gb_month:.2f}")
print()

# Bonus Challenge
print("Bonus Challenge: Compression Recommender")

def recommend_compression(access_pattern, data_size_gb, cpu_available):
    """
    Recommend compression algorithm
    """
    # Hot data (frequently accessed)
    if access_pattern == 'hot':
        if cpu_available == 'low':
            return {
                'compression': 'snappy',
                'reason': 'Fast compression/decompression, low CPU usage, good for frequent access'
            }
        else:
            return {
                'compression': 'zstd',
                'level': 3,
                'reason': 'Better compression than snappy, still fast with available CPU'
            }
    
    # Warm data (occasional access)
    elif access_pattern == 'warm':
        return {
            'compression': 'zstd',
            'level': 5,
            'reason': 'Good balance of compression and speed for occasional access'
        }
    
    # Cold data (rare access)
    elif access_pattern == 'cold':
        if data_size_gb > 1000:
            return {
                'compression': 'gzip',
                'reason': 'Maximum compression for large cold storage, cost savings outweigh slow access'
            }
        else:
            return {
                'compression': 'zstd',
                'level': 10,
                'reason': 'Better compression than snappy, faster than gzip for smaller datasets'
            }
    
    return {
        'compression': 'snappy',
        'reason': 'Default safe choice'
    }

# Test cases
test_cases = [
    ('hot', 100, 'low'),
    ('hot', 100, 'high'),
    ('warm', 500, 'medium'),
    ('cold', 2000, 'high'),
    ('cold', 100, 'medium')
]

print("Compression recommendations:")
for pattern, size, cpu in test_cases:
    result = recommend_compression(pattern, size, cpu)
    print(f"\n{pattern}, {size}GB, CPU: {cpu}")
    print(f"  → {result['compression']}")
    if 'level' in result:
        print(f"    Level: {result['level']}")
    print(f"    Reason: {result['reason']}")

print("\n✅ All exercises completed!")
