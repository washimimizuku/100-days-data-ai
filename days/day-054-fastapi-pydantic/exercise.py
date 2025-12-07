"""
Day 54: FastAPI Pydantic Validation - Exercises

Practice advanced Pydantic validation patterns.
"""
from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Optional, Union
from datetime import date, datetime


# Exercise 1: Field Constraints
def exercise_1():
    """
    Exercise 1: Product Model with Constraints
    
    Create a Product model with:
    - name: 1-100 characters
    - sku: Pattern ABC-1234 (3 uppercase letters, dash, 4 digits)
    - price: Greater than 0, less than or equal to 1000000
    - quantity: Non-negative integer
    - tags: List of strings, max 10 items
    - description: Optional string, max 500 characters
    
    TODO: Implement Product model
    """
    # TODO: Define Product model with constraints
    pass


# Exercise 2: Custom Validators
def exercise_2():
    """
    Exercise 2: User Registration with Validation
    
    Create a UserRegistration model with:
    - username: Alphanumeric only, 3-20 characters
    - email: Valid email format, lowercase
    - password: Min 8 chars, must contain uppercase, lowercase, digit
    - age: 18-100
    
    Use custom validators for username and password.
    
    TODO: Implement with custom validators
    """
    # TODO: Define UserRegistration model
    pass


# Exercise 3: Root Validators
def exercise_3():
    """
    Exercise 3: Date Range Validation
    
    Create a DateRange model with:
    - start_date: date
    - end_date: date
    - description: str
    
    Use root validator to ensure:
    - end_date is after start_date
    - date range is not more than 1 year
    
    TODO: Implement with root validator
    """
    # TODO: Define DateRange model
    pass


# Exercise 4: Nested Models
def exercise_4():
    """
    Exercise 4: Order with Nested Models
    
    Create models:
    
    Address:
    - street: str
    - city: str
    - country: str
    - zip_code: str
    
    OrderItem:
    - product_id: str
    - quantity: int (positive)
    - price: float (positive)
    
    Order:
    - order_id: str
    - customer_name: str
    - shipping_address: Address
    - items: List[OrderItem] (at least 1 item)
    - total: float (calculated from items)
    
    Use root validator to calculate total.
    
    TODO: Implement nested models
    """
    # TODO: Define Address, OrderItem, Order models
    pass


# Exercise 5: Field Dependencies
def exercise_5():
    """
    Exercise 5: Conditional Validation
    
    Create a ShippingInfo model with:
    - country: str
    - state: Optional[str]
    - zip_code: Optional[str]
    
    Validation rules:
    - If country is "US", state is required
    - If country is "US", zip_code must be 5 digits
    - If country is "CA", zip_code must match pattern A1A 1A1
    
    TODO: Implement conditional validation
    """
    # TODO: Define ShippingInfo model
    pass


# Exercise 6: Data Transformation
def exercise_6():
    """
    Exercise 6: Data Cleaning
    
    Create a Contact model that:
    - Capitalizes name (title case)
    - Lowercases email
    - Formats phone (removes non-digits, adds dashes)
    - Strips whitespace from all strings
    
    TODO: Implement with transformation validators
    """
    # TODO: Define Contact model with transformations
    pass


# Exercise 7: Dataset Schema Validation
def exercise_7():
    """
    Exercise 7: Data Engineering Schema
    
    Create models for dataset schema:
    
    Column:
    - name: Valid identifier (letters, numbers, underscore)
    - type: One of: string, integer, float, boolean, date
    - nullable: bool
    - description: Optional[str]
    
    DatasetSchema:
    - name: str
    - columns: List[Column] (at least 1)
    - primary_key: Optional[str]
    
    Validate:
    - Column names are unique
    - Primary key exists in columns
    
    TODO: Implement dataset schema validation
    """
    # TODO: Define Column and DatasetSchema models
    pass


# Exercise 8: Query Validation
def exercise_8():
    """
    Exercise 8: Data Query Model
    
    Create a DataQuery model with:
    - dataset: str
    - columns: Optional[List[str]]
    - filters: Optional[dict]
    - sort_by: Optional[str]
    - sort_order: "asc" or "desc"
    - limit: 1-10000, default 100
    - offset: Non-negative, default 0
    
    Validate filters have valid operators: eq, ne, gt, lt, gte, lte, in
    
    TODO: Implement query validation
    """
    # TODO: Define DataQuery model
    pass


if __name__ == "__main__":
    print("Day 54: FastAPI Pydantic Validation - Exercises\n")
    print("Run exercises and test validation:")
    print("  python exercise.py")
