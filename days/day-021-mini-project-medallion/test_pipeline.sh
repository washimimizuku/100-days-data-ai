#!/bin/bash

echo "=========================================="
echo "Testing Medallion Pipeline"
echo "=========================================="

# Generate sample data
echo -e "\n[1/3] Generating sample data..."
python generate_data.py

# Run full pipeline
echo -e "\n[2/3] Running full pipeline..."
python medallion_pipeline_solution.py --mode full

# Verify results
echo -e "\n[3/3] Verifying results..."
python medallion_pipeline_solution.py --mode verify

echo -e "\n=========================================="
echo "All tests completed!"
echo "=========================================="
