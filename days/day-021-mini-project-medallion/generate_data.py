#!/usr/bin/env python3
"""Generate sample data for medallion pipeline"""

import json
import os
from datetime import datetime, timedelta

def generate_orders():
    orders = [
        {"order_id": "O001", "customer_id": "C001", "product_id": "P001", "quantity": 2, "amount": 199.98, "order_timestamp": "2024-01-15T10:30:00"},
        {"order_id": "O002", "customer_id": "C002", "product_id": "P002", "quantity": 1, "amount": 49.99, "order_timestamp": "2024-01-16T14:20:00"},
        {"order_id": "O003", "customer_id": "C001", "product_id": "P003", "quantity": 3, "amount": 299.97, "order_timestamp": "2024-01-17T09:15:00"},
        {"order_id": "O001", "customer_id": "C001", "product_id": "P001", "quantity": 2, "amount": 199.98, "order_timestamp": "2024-01-15T10:30:00"},  # Duplicate
    ]
    return orders

def generate_customers():
    customers = [
        {"customer_id": "C001", "name": "Alice Smith", "email": "alice@email.com", "city": "New York", "segment": "Premium"},
        {"customer_id": "C002", "name": "Bob Jones", "email": "bob@email.com", "city": "London", "segment": "Standard"},
        {"customer_id": "C003", "name": "Charlie Brown", "email": "charlie@email.com", "city": "Paris", "segment": "Premium"},
    ]
    return customers

def generate_products():
    products = [
        {"product_id": "P001", "name": "Laptop", "category": "Electronics", "price": 99.99},
        {"product_id": "P002", "name": "Mouse", "category": "Electronics", "price": 49.99},
        {"product_id": "P003", "name": "Desk", "category": "Furniture", "price": 99.99},
    ]
    return products

def main():
    os.makedirs("data", exist_ok=True)
    
    with open("data/orders.json", "w") as f:
        json.dump(generate_orders(), f, indent=2)
    
    with open("data/customers.json", "w") as f:
        json.dump(generate_customers(), f, indent=2)
    
    with open("data/products.json", "w") as f:
        json.dump(generate_products(), f, indent=2)
    
    print("✓ Generated sample data in data/ directory")

if __name__ == "__main__":
    main()
