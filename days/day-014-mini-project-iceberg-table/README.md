# Day 14: Mini Project - Iceberg Table Manager

## 📖 Project Overview (2 hours)

**Time**: 2 hours


Build a CLI tool to manage Apache Iceberg tables with operations like create, query, optimize, and time travel.

**Time Allocation:**
- Planning & Setup: 20 min
- Core Implementation: 60 min
- Testing & Documentation: 40 min

---

## Project Goals

Create `iceberg_manager.py` with these features:

1. **Table Operations**
   - Create table with schema
   - Insert data
   - Query data
   - Update/Delete records

2. **Time Travel**
   - List snapshots
   - Query by snapshot ID
   - Query by timestamp
   - Rollback to snapshot

3. **Maintenance**
   - Expire old snapshots
   - Remove orphan files
   - Compact small files
   - Show table statistics

4. **CLI Interface**
   - Argparse-based commands
   - Verbose mode
   - Error handling

---

## Requirements

```bash
pip install pyspark pyiceberg
```

---

## Implementation Guide

### 1. CLI Structure

```python
# Commands
iceberg_manager.py create --table users --schema id:int,name:string,age:int
iceberg_manager.py insert --table users --data data.json
iceberg_manager.py query --table users
iceberg_manager.py snapshots --table users
iceberg_manager.py rollback --table users --snapshot-id 123456
iceberg_manager.py optimize --table users
iceberg_manager.py stats --table users
```

### 2. Core Functions

- `create_table(table_name, schema)` - Create Iceberg table
- `insert_data(table_name, data)` - Insert records
- `query_table(table_name, snapshot_id=None)` - Query with time travel
- `list_snapshots(table_name)` - Show snapshot history
- `rollback_snapshot(table_name, snapshot_id)` - Rollback
- `optimize_table(table_name)` - Compact files
- `show_stats(table_name)` - Display table info

### 3. Features to Implement

**Basic (Required):**
- Create table
- Insert data
- Query current data
- List snapshots

**Intermediate:**
- Time travel queries
- Rollback operations
- Schema display

**Advanced:**
- Optimize/compact
- Expire snapshots
- Statistics dashboard

---

## Example Usage

```bash
# Create table
python iceberg_manager.py create \
  --table orders \
  --schema "order_id:int,customer_id:int,amount:decimal,status:string"

# Insert data
python iceberg_manager.py insert \
  --table orders \
  --data '[{"order_id":1,"customer_id":101,"amount":99.99,"status":"pending"}]'

# Query
python iceberg_manager.py query --table orders

# List snapshots
python iceberg_manager.py snapshots --table orders

# Time travel
python iceberg_manager.py query --table orders --snapshot-id 123456

# Rollback
python iceberg_manager.py rollback --table orders --snapshot-id 123456

# Optimize
python iceberg_manager.py optimize --table orders

# Stats
python iceberg_manager.py stats --table orders
```

---

## Testing

Create `test_manager.sh`:

```bash
#!/bin/bash

echo "=== Testing Iceberg Manager ==="

# Create table
python iceberg_manager.py create --table test_users --schema "id:int,name:string,age:int"

# Insert data
python iceberg_manager.py insert --table test_users --data '[{"id":1,"name":"Alice","age":25}]'
python iceberg_manager.py insert --table test_users --data '[{"id":2,"name":"Bob","age":30}]'

# Query
python iceberg_manager.py query --table test_users

# Snapshots
python iceberg_manager.py snapshots --table test_users

# Stats
python iceberg_manager.py stats --table test_users

echo "=== Tests Complete ==="
```

---

## Deliverables

1. **iceberg_manager.py** - Main CLI tool
2. **test_manager.sh** - Test script
3. **README.md** - Usage documentation
4. **requirements.txt** - Dependencies

---

## Bonus Features

- Export table to Parquet
- Import from CSV/JSON
- Schema evolution commands
- Partition management
- Metadata inspection
- Performance benchmarking

---

## Success Criteria

✅ CLI accepts all basic commands
✅ Can create and query tables
✅ Time travel works correctly
✅ Snapshots listed properly
✅ Rollback functions
✅ Error handling implemented
✅ Code is documented
✅ Tests pass

---

## Learning Outcomes

- Practical Iceberg table management
- CLI tool development
- Snapshot lifecycle handling
- Production-ready error handling
- Testing and documentation

---

## Resources

- [PyIceberg Documentation](https://py.iceberg.apache.org/)
- [Iceberg Spark Integration](https://iceberg.apache.org/docs/latest/spark-getting-started/)
- [Argparse Tutorial](https://docs.python.org/3/howto/argparse.html)

---

## Next Steps

After completing this project:
1. Test all commands
2. Add error handling
3. Document usage
4. Consider additional features
5. Move to Day 15

---

## Tips

- Start with basic create/query
- Add time travel next
- Implement maintenance last
- Test incrementally
- Handle errors gracefully
- Use verbose mode for debugging
