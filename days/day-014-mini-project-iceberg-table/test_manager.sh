#!/bin/bash

echo "=========================================="
echo "Testing Iceberg Table Manager"
echo "=========================================="

SCRIPT="iceberg_manager_solution.py"

echo -e "\n[1/7] Creating table..."
python $SCRIPT create --table test_users --schema "id:int,name:string,age:int,email:string"

echo -e "\n[2/7] Inserting data (batch 1)..."
python $SCRIPT insert --table test_users --data '[{"id":1,"name":"Alice","age":25,"email":"alice@example.com"}]'

echo -e "\n[3/7] Inserting data (batch 2)..."
python $SCRIPT insert --table test_users --data '[{"id":2,"name":"Bob","age":30,"email":"bob@example.com"},{"id":3,"name":"Charlie","age":35,"email":"charlie@example.com"}]'

echo -e "\n[4/7] Querying current data..."
python $SCRIPT query --table test_users

echo -e "\n[5/7] Listing snapshots..."
python $SCRIPT snapshots --table test_users

echo -e "\n[6/7] Showing statistics..."
python $SCRIPT stats --table test_users

echo -e "\n[7/7] Optimizing table..."
python $SCRIPT optimize --table test_users

echo -e "\n=========================================="
echo "All tests completed!"
echo "=========================================="
