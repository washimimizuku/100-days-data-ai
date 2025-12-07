# Production Data API - Project Specification

## Data Models

### Dataset Metadata
```python
{
  "id": "sales",
  "name": "Sales Data",
  "description": "Monthly sales transactions",
  "rows": 1000,
  "columns": 5,
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

### Column Schema
```python
{
  "name": "amount",
  "type": "float",
  "nullable": false,
  "description": "Transaction amount in USD"
}
```

### Query Request
```python
{
  "columns": ["date", "amount"],
  "filters": {
    "amount": {"gt": 100, "lt": 1000},
    "status": {"eq": "completed"}
  },
  "sort_by": "date",
  "sort_order": "desc",
  "limit": 50,
  "offset": 0
}
```

### Query Response
```python
{
  "data": [...],
  "count": 50,
  "total": 234,
  "execution_time_ms": 45.2,
  "has_more": true
}
```

---

## API Endpoints

### 1. List Datasets
```
GET /datasets
Response: List of dataset metadata
```

### 2. Get Dataset
```
GET /datasets/{id}
Response: Dataset metadata
```

### 3. Get Schema
```
GET /datasets/{id}/schema
Response: Column definitions
```

### 4. Query Dataset
```
POST /datasets/{id}/query
Body: QueryRequest
Response: QueryResponse
```

### 5. Export Dataset
```
POST /datasets/{id}/export
Body: {format: "csv|json|parquet", filters: {...}}
Response: File download
```

---

## Sample Data

### Sales Dataset
```csv
id,date,product,amount,status
1,2024-01-01,laptop,999.99,completed
2,2024-01-02,mouse,29.99,completed
3,2024-01-03,keyboard,79.99,pending
```

### Users Dataset
```csv
id,name,email,created_at,status
1,Alice,alice@example.com,2024-01-01,active
2,Bob,bob@example.com,2024-01-02,active
```

### Products Dataset
```csv
id,name,category,price,stock
1,Laptop,electronics,999.99,50
2,Mouse,electronics,29.99,200
```

---

## Implementation Details

### Filter Operators
- `eq`: Equal
- `ne`: Not equal
- `gt`: Greater than
- `lt`: Less than
- `gte`: Greater than or equal
- `lte`: Less than or equal
- `in`: In list
- `like`: Pattern match

### Validation Rules
- `limit`: 1-1000, default 100
- `offset`: >= 0, default 0
- `columns`: Must exist in dataset
- `filters`: Valid operators only
- `sort_by`: Must be valid column

### Error Responses
```python
{
  "detail": "Dataset not found",
  "error_code": "DATASET_NOT_FOUND",
  "status_code": 404
}
```

---

## Testing Strategy

### Unit Tests
- Model validation
- Filter logic
- Data transformations

### API Tests
- All endpoints
- Validation errors
- Edge cases

### Integration Tests
- End-to-end workflows
- Multiple datasets
- Complex queries

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Response time | < 200ms |
| Throughput | > 100 req/s |
| Test coverage | > 80% |
| Error rate | < 1% |
