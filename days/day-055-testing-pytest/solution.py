"""
Day 55: Testing Data Pipelines with pytest - Solutions
"""
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
import pandas as pd


# Exercise 1: Basic Tests
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0

def test_multiply():
    assert multiply(2, 3) == 6
    assert multiply(-2, 3) == -6
    assert multiply(0, 5) == 0

def test_divide():
    assert divide(10, 2) == 5
    assert divide(9, 3) == 3
    
def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        divide(10, 0)


# Exercise 2: API Tests
app = FastAPI()

items_db = {}
next_id = 1

@app.get("/")
def read_root():
    return {"message": "Hello"}

@app.get("/items/{item_id}")
def get_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return items_db[item_id]

@app.post("/items")
def create_item(item: dict):
    global next_id
    item_id = next_id
    next_id += 1
    items_db[item_id] = {"id": item_id, **item}
    return items_db[item_id]

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello"}

def test_create_item():
    response = client.post("/items", json={"name": "Test", "price": 9.99})
    assert response.status_code == 200
    assert "id" in response.json()
    assert response.json()["name"] == "Test"

def test_get_item():
    # Create item first
    create_response = client.post("/items", json={"name": "Test2", "price": 19.99})
    item_id = create_response.json()["id"]
    
    # Get item
    response = client.get(f"/items/{item_id}")
    assert response.status_code == 200
    assert response.json()["name"] == "Test2"

def test_get_nonexistent_item():
    response = client.get("/items/9999")
    assert response.status_code == 404


# Exercise 3: Fixtures
@pytest.fixture
def sample_user():
    return {"name": "Alice", "email": "alice@example.com", "age": 30}

@pytest.fixture
def sample_products():
    return [
        {"name": "Laptop", "price": 999.99},
        {"name": "Mouse", "price": 29.99},
        {"name": "Keyboard", "price": 79.99}
    ]

@pytest.fixture
def test_client():
    return TestClient(app)

def test_with_user_fixture(sample_user):
    assert sample_user["name"] == "Alice"
    assert sample_user["age"] == 30

def test_with_products_fixture(sample_products):
    assert len(sample_products) == 3
    assert sample_products[0]["name"] == "Laptop"


# Exercise 4: Mocking
def fetch_external_data(url):
    import requests
    response = requests.get(url)
    return response.json()

@patch('requests.get')
def test_fetch_external_data(mock_get):
    # Setup mock
    mock_response = Mock()
    mock_response.json.return_value = {"data": "test"}
    mock_get.return_value = mock_response
    
    # Test
    result = fetch_external_data("https://api.example.com/data")
    
    # Assert
    assert result == {"data": "test"}
    mock_get.assert_called_once_with("https://api.example.com/data")


# Exercise 5: Data Pipeline Tests
def clean_data(df):
    df = df.dropna()
    df['price'] = df['price'].astype(float)
    return df

def test_clean_data_removes_nulls():
    df = pd.DataFrame({
        'name': ['A', 'B', None, 'D'],
        'price': ['10.5', '20.0', '30.5', '40.0']
    })
    
    result = clean_data(df)
    
    assert len(result) == 3
    assert result['name'].isna().sum() == 0

def test_clean_data_converts_types():
    df = pd.DataFrame({
        'name': ['A', 'B'],
        'price': ['10.5', '20.0']
    })
    
    result = clean_data(df)
    
    assert result['price'].dtype == float
    assert result['price'].iloc[0] == 10.5

def test_clean_data_empty_dataframe():
    df = pd.DataFrame({'name': [], 'price': []})
    result = clean_data(df)
    assert len(result) == 0


# Parametrized Tests
@pytest.mark.parametrize("a,b,expected", [
    (2, 3, 5),
    (0, 0, 0),
    (-1, 1, 0),
    (10, -5, 5)
])
def test_add_parametrized(a, b, expected):
    assert add(a, b) == expected


if __name__ == "__main__":
    print("Day 55: Testing with pytest - Solutions\n")
    print("Run tests with: pytest solution.py -v")
    print("Run with coverage: pytest solution.py --cov")
