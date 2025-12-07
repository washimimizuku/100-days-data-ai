# Day 56: Mini Project - Production Data API

## 🎯 Project Overview

Build a **production-ready data API** that serves datasets with filtering, pagination, and validation. Apply everything learned in Week 8: REST principles, FastAPI, async patterns, Pydantic validation, and comprehensive testing.

**Time**: 2 hours  
**Difficulty**: Intermediate  
**Topics**: FastAPI, Testing, Data APIs, Production Patterns

---

## 📋 Project Requirements

### Scenario

Build a **Dataset API** that allows users to:
1. List available datasets
2. Query datasets with filters and pagination
3. Get dataset metadata and schema
4. Export data in multiple formats
5. Track API usage and metrics

### Business Requirements

**Features**:
- RESTful API design
- Async operations for performance
- Pydantic validation
- Comprehensive test coverage (>80%)
- API documentation
- Error handling
- Rate limiting (bonus)

**SLAs**:
- Response time: < 200ms for queries
- Availability: 99.9%
- Test coverage: > 80%

---

## 🏗️ Architecture

```
API Endpoints
├── GET /datasets - List all datasets
├── GET /datasets/{id} - Get dataset metadata
├── GET /datasets/{id}/schema - Get dataset schema
├── POST /datasets/{id}/query - Query dataset
└── POST /datasets/{id}/export - Export dataset

Data Layer
├── In-memory storage (for demo)
├── Sample datasets (sales, users, products)
└── Query engine

Testing
├── Unit tests (models, utils)
├── API tests (endpoints)
└── Integration tests (end-to-end)
```

---

## 💻 Implementation

### Part 1: Data Models (20 min)

Create Pydantic models for:
- Dataset metadata
- Column schema
- Query request/response
- Export request

**File**: `models.py`

### Part 2: API Endpoints (40 min)

Implement FastAPI endpoints:
- List datasets
- Get metadata
- Query with filters
- Export data

**File**: `main.py`

### Part 3: Data Layer (20 min)

Implement data operations:
- Load sample data
- Filter and paginate
- Format output

**File**: `data.py`

### Part 4: Tests (30 min)

Write comprehensive tests:
- Model validation tests
- API endpoint tests
- Integration tests

**File**: `test_api.py`

### Part 5: Documentation (10 min)

- API documentation (automatic)
- README with examples
- Deployment guide

---

## 📁 Project Structure

```
day-056-mini-project-data-api/
├── README.md
├── project.md
├── main.py              # FastAPI application
├── models.py            # Pydantic models
├── data.py              # Data layer
├── test_api.py          # Tests
├── requirements.txt     # Dependencies
└── sample_data/         # Sample datasets
    ├── sales.csv
    ├── users.csv
    └── products.csv
```

---

## 🚀 Getting Started

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run application
uvicorn main:app --reload

# Run tests
pytest test_api.py -v --cov
```

### 2. Test API

```bash
# List datasets
curl http://localhost:8000/datasets

# Query dataset
curl -X POST http://localhost:8000/datasets/sales/query \
  -H "Content-Type: application/json" \
  -d '{"filters": {"amount": {"gt": 100}}, "limit": 10}'

# View docs
open http://localhost:8000/docs
```

---

## 🎯 Success Criteria

### Functional Requirements
- [ ] All endpoints working
- [ ] Filtering and pagination
- [ ] Data validation
- [ ] Error handling
- [ ] API documentation

### Non-Functional Requirements
- [ ] Test coverage > 80%
- [ ] Response time < 200ms
- [ ] Async operations
- [ ] Clean code structure
- [ ] Comprehensive documentation

### Testing
- [ ] Unit tests pass
- [ ] API tests pass
- [ ] Integration tests pass
- [ ] Edge cases covered

---

## 📊 API Examples

### List Datasets

```bash
GET /datasets

Response:
{
  "datasets": [
    {
      "id": "sales",
      "name": "Sales Data",
      "rows": 1000,
      "columns": 5
    }
  ]
}
```

### Query Dataset

```bash
POST /datasets/sales/query
{
  "columns": ["date", "amount", "product"],
  "filters": {
    "amount": {"gt": 100, "lt": 1000},
    "product": {"in": ["laptop", "phone"]}
  },
  "sort_by": "date",
  "sort_order": "desc",
  "limit": 50,
  "offset": 0
}

Response:
{
  "data": [...],
  "count": 50,
  "total": 234,
  "execution_time_ms": 45.2
}
```

### Get Schema

```bash
GET /datasets/sales/schema

Response:
{
  "columns": [
    {"name": "id", "type": "integer", "nullable": false},
    {"name": "date", "type": "date", "nullable": false},
    {"name": "amount", "type": "float", "nullable": false}
  ]
}
```

---

## 🎁 Bonus Challenges (Optional)

### Bonus 1: Rate Limiting (15 min)
Implement rate limiting (100 requests/minute per IP).

### Bonus 2: Caching (15 min)
Cache query results for 5 minutes.

### Bonus 3: Authentication (20 min)
Add API key authentication.

### Bonus 4: Metrics (15 min)
Track API usage metrics (requests, latency, errors).

---

## 🐛 Troubleshooting

### Issue: Tests failing
**Solution**: Check test database setup, verify fixtures

### Issue: Slow queries
**Solution**: Add pagination, optimize filters

### Issue: Validation errors
**Solution**: Check Pydantic models, verify input data

---

## 📚 Key Concepts Applied

This project demonstrates:
- ✅ REST API design (Day 51)
- ✅ FastAPI basics (Day 52)
- ✅ Async/await patterns (Day 53)
- ✅ Pydantic validation (Day 54)
- ✅ Testing with pytest (Day 55)

---

## 🎓 Learning Outcomes

After completing this project, you will:
- Build production-ready APIs
- Write comprehensive tests
- Apply async patterns
- Validate data effectively
- Handle errors gracefully
- Document APIs automatically
- Deploy data services

---

## 📝 Deliverables

1. `main.py` - Complete FastAPI application
2. `models.py` - Pydantic models
3. `data.py` - Data layer implementation
4. `test_api.py` - Comprehensive test suite
5. `requirements.txt` - Dependencies
6. Documentation of design decisions

---

## 🔗 Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [pytest Documentation](https://docs.pytest.org/)

---

## ✅ Completion Checklist

- [ ] Setup complete (dependencies installed)
- [ ] Models defined and validated
- [ ] All endpoints implemented
- [ ] Data layer working
- [ ] Tests written (>80% coverage)
- [ ] Tests passing
- [ ] API documentation generated
- [ ] Error handling implemented
- [ ] Performance acceptable
- [ ] Code documented

---

## 🎉 Next Steps

After completing this project:
1. Review your code and identify improvements
2. Compare with reference implementation
3. Try bonus challenges
4. Deploy to production (optional)
5. Move on to Week 9: Machine Learning

---

## Tomorrow: Day 57 - ML Workflow Overview

Begin Week 9 with machine learning fundamentals.
