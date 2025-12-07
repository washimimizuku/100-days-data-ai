"""
Day 55: Testing Data Pipelines with pytest - Exercises

Practice writing tests for APIs and data pipelines.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# Exercise 1: Basic Tests
def exercise_1():
    """
    Exercise 1: Test Simple Functions
    
    Write tests for these functions:
    - add(a, b): Returns sum
    - multiply(a, b): Returns product
    - divide(a, b): Returns quotient, raises ZeroDivisionError
    
    TODO: Implement tests
    """
    def add(a, b):
        return a + b
    
    def multiply(a, b):
        return a * b
    
    def divide(a, b):
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return a / b
    
    # TODO: Write tests
    pass


# Exercise 2: API Tests
def exercise_2():
    """
    Exercise 2: Test FastAPI Endpoints
    
    Create tests for a simple API:
    - GET / - Returns {"message": "Hello"}
    - GET /items/{item_id} - Returns item
    - POST /items - Creates item
    
    TODO: Implement API tests
    """
    app = FastAPI()
    
    @app.get("/")
    def read_root():
        return {"message": "Hello"}
    
    # TODO: Add more endpoints
    
    # TODO: Write tests using TestClient
    pass


# Exercise 3: Fixtures
def exercise_3():
    """
    Exercise 3: Create Test Fixtures
    
    Create fixtures for:
    - sample_user: User data dict
    - sample_products: List of products
    - test_client: FastAPI TestClient
    
    TODO: Implement fixtures
    """
    # TODO: Create fixtures with @pytest.fixture
    pass


# Exercise 4: Mocking
def exercise_4():
    """
    Exercise 4: Mock External Dependencies
    
    Test a function that calls external API:
    - Mock the API call
    - Verify function behavior
    - Check mock was called correctly
    
    TODO: Implement mocking tests
    """
    # TODO: Use unittest.mock or pytest-mock
    pass


# Exercise 5: Data Pipeline Tests
def exercise_5():
    """
    Exercise 5: Test Data Transformations
    
    Test a data cleaning pipeline:
    - Remove null values
    - Convert types
    - Validate output schema
    
    TODO: Implement data pipeline tests
    """
    import pandas as pd
    
    def clean_data(df):
        df = df.dropna()
        df['price'] = df['price'].astype(float)
        return df
    
    # TODO: Write tests for clean_data
    pass


if __name__ == "__main__":
    print("Day 55: Testing with pytest - Exercises\n")
    print("Run tests with: pytest exercise.py -v")
