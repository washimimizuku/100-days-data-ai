#!/bin/bash

echo "Testing Spark ETL Pipeline..."
echo "=============================="

# Generate sample data
echo -e "\n1. Generating sample data..."
python generate_data.py

# Run ETL pipeline
echo -e "\n2. Running ETL pipeline..."
python spark_etl_solution.py

# Verify outputs
echo -e "\n3. Verifying outputs..."
if [ -d "/tmp/etl_output" ]; then
    echo "✅ Output directory created"
    ls -lh /tmp/etl_output/
else
    echo "❌ Output directory not found"
    exit 1
fi

echo -e "\n✅ All tests passed!"
