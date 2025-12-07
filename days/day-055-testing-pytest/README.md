# Day 55: Testing Data Pipelines with pytest

## 📖 Learning Objectives

By the end of this session, you will:
- Understand testing fundamentals with pytest
- Test FastAPI endpoints
- Test data transformations and pipelines
- Use fixtures for test setup
- Mock external dependencies
- Measure test coverage

**Time**: 1 hour  
**Level**: Intermediate

---

## Why Test?

Catch bugs early, refactor safely, document behavior, prevent regressions, build confidence

---

## pytest Basics

```bash
pip install pytest pytest-cov
```

```python
# test_math.py
def add(a, b):
    return a + b

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0

# Run: pytest test_math.py
```

**Test Discovery**: Files `test_*.py` or `*_test.py`, Functions `test_*`, Classes `Test*`

---

## Testing FastAPI

```bash
pip install httpx pytest
```

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello"}

client = TestClient(app)

# Test with Arrange-Act-Assert (AAA) pattern
def test_read_root():
    response = client.get("/")  # Act
    assert response.status_code == 200  # Assert
    assert response.json() == {"message": "Hello"}

def test_create_user():
    user_data = {"name": "Alice", "email": "alice@example.com"}  # Arrange
    response = client.post("/users", json=user_data)  # Act
    assert response.status_code == 201  # Assert
    assert response.json()["name"] == "Alice"
```

---

## Fixtures

```python
import pytest

# Basic fixture
@pytest.fixture
def sample_user():
    return {"name": "Alice", "email": "alice@example.com"}

def test_user_creation(sample_user):
    assert sample_user["name"] == "Alice"

# Setup/Teardown fixture
@pytest.fixture
def database():
    db = create_database()  # Setup
    yield db
    db.close()  # Teardown

def test_query(database):
    result = database.query("SELECT * FROM users")
    assert len(result) > 0
```

---

## Testing CRUD & Validation

```python
# CRUD operations
def test_create_item():
    response = client.post("/items", json={"name": "Test", "price": 9.99})
    assert response.status_code == 201 and "id" in response.json()

def test_read_item():
    response = client.get("/items/1")
    assert response.status_code == 200 and response.json()["name"] == "Test"

def test_update_item():
    response = client.put("/items/1", json={"name": "Updated", "price": 19.99})
    assert response.status_code == 200 and response.json()["name"] == "Updated"

def test_delete_item():
    response = client.delete("/items/1")
    assert response.status_code == 204

# Validation errors
def test_invalid_email():
    response = client.post("/users", json={"name": "Alice", "email": "invalid-email"})
    assert response.status_code == 422
    assert "email" in response.json()["detail"][0]["loc"]

def test_negative_price():
    response = client.post("/items", json={"name": "Test", "price": -10})
    assert response.status_code == 422
```

---

## Mocking

```python
from unittest.mock import patch, Mock

# Mock external API
@patch('requests.get')
def test_fetch_external_data(mock_get):
    mock_response = Mock()
    mock_response.json.return_value = {"data": "test"}
    mock_get.return_value = mock_response
    
    result = fetch_external_data()
    assert result == {"data": "test"}
    mock_get.assert_called_once()

# Mock database
@pytest.fixture
def mock_db():
    with patch('app.database.get_db') as mock:
        mock.return_value = {"users": []}
        yield mock

def test_list_users(mock_db):
    response = client.get("/users")
    assert response.status_code == 200
```

---

## Testing Data Transformations & Pipelines

```python
import pandas as pd

# Data transformation test
def test_clean_data():
    df = pd.DataFrame({'name': ['A', 'B', None], 'price': ['10.5', '20.0', '30.5']})
    result = clean_data(df)
    assert len(result) == 2 and result['price'].dtype == float

# Parametrized tests
@pytest.mark.parametrize("input,expected", [(2, 4), (3, 9), (4, 16), (5, 25)])
def test_square(input, expected):
    assert input ** 2 == expected

# ETL pipeline test
def test_etl_pipeline():
    raw_data = extract_data("source.csv")
    assert len(raw_data) > 0
    transformed = transform_data(raw_data)
    assert 'cleaned_column' in transformed.columns
    assert load_data(transformed, "output.csv") == True

# Error testing
def test_division_by_zero():
    with pytest.raises(ZeroDivisionError):
        result = 10 / 0

def test_invalid_input():
    with pytest.raises(ValueError, match="Invalid input"):
        process_data("invalid")
```

---

## Test Coverage & Async

```bash
# Run with coverage
pytest --cov=app --cov-report=html
open htmlcov/index.html
```

```python
# Async testing
@pytest.mark.asyncio
async def test_async_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/async-endpoint")
    assert response.status_code == 200
```

---

## Test Organization & Best Practices

```
tests/
├── conftest.py          # Shared fixtures
├── test_api.py          # API tests
├── test_models.py       # Model tests
└── test_transforms.py   # Data transformation tests
```

**Best Practices**:
1. One assertion per test, descriptive names
2. Independent tests, no dependencies
3. Fast tests, use fixtures for reuse
4. Mock external services
5. Test edge cases (empty, nulls, boundaries)
6. Maintain tests with code changes

---

## Example: Complete Test Suite

```python
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

@pytest.fixture
def sample_product():
    return {"name": "Laptop", "price": 999.99, "quantity": 10}

class TestProducts:
    def test_create_product(self, sample_product):
        response = client.post("/products", json=sample_product)
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == sample_product["name"]
        assert "id" in data
    
    def test_list_products(self):
        response = client.get("/products")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
    
    def test_get_product(self):
        response = client.get("/products/1")
        assert response.status_code == 200
        assert "name" in response.json()
    
    def test_update_product(self):
        update_data = {"name": "Updated Laptop", "price": 899.99}
        response = client.put("/products/1", json=update_data)
        assert response.status_code == 200
        assert response.json()["name"] == "Updated Laptop"
    
    def test_delete_product(self):
        response = client.delete("/products/1")
        assert response.status_code == 204
    
    def test_product_not_found(self):
        response = client.get("/products/9999")
        assert response.status_code == 404
    
    def test_invalid_price(self):
        invalid_product = {"name": "Test", "price": -10}
        response = client.post("/products", json=invalid_product)
        assert response.status_code == 422
```

---

## 💻 Exercises

Complete the exercises in `exercise.py`:

### Exercise 1: Basic Tests
Write tests for simple functions.

### Exercise 2: API Tests
Test FastAPI endpoints.

### Exercise 3: Fixtures
Create reusable test fixtures.

### Exercise 4: Mocking
Mock external dependencies.

### Exercise 5: Data Pipeline Tests
Test data transformations.

---

## ✅ Quiz

Test your understanding in `quiz.md`.

---

## 🎯 Key Takeaways

- pytest automatically discovers and runs tests
- Use TestClient for FastAPI testing
- Fixtures provide reusable test setup
- Mock external dependencies to isolate tests
- Parametrize tests to reduce duplication
- Test edge cases and error conditions
- Measure coverage to find untested code
- Keep tests fast, independent, and focused

---

## 📚 Resources

- [pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)

---

## Tomorrow: Day 56 - Mini Project: Production Data API

Build a complete production-ready data API with tests.
