"""
Generate sample data for Spark ETL pipeline
"""
import csv
import json
import random
from datetime import datetime, timedelta

def generate_orders(num_orders=1000):
    """Generate sample orders"""
    with open('data/orders.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['order_id', 'customer_id', 'product_id', 'quantity', 'price', 'order_date'])
        
        for i in range(1, num_orders + 1):
            customer_id = random.randint(1, 100)
            product_id = random.randint(1, 50)
            quantity = random.randint(1, 5)
            price = round(random.uniform(10, 100), 2)
            order_date = (datetime.now() - timedelta(days=random.randint(0, 30))).strftime('%Y-%m-%d')
            writer.writerow([i, customer_id, product_id, quantity, price, order_date])
    
    print(f"Generated {num_orders} orders")

def generate_customers(num_customers=100):
    """Generate sample customers"""
    cities = ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix']
    
    with open('data/customers.json', 'w') as f:
        for i in range(1, num_customers + 1):
            customer = {
                'customer_id': i,
                'name': f'Customer_{i}',
                'email': f'customer{i}@example.com',
                'city': random.choice(cities),
                'country': 'USA'
            }
            f.write(json.dumps(customer) + '\n')
    
    print(f"Generated {num_customers} customers")

def generate_products(num_products=50):
    """Generate sample products"""
    categories = ['Electronics', 'Hardware', 'Software', 'Accessories']
    
    with open('data/products.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['product_id', 'name', 'category', 'cost'])
        
        for i in range(1, num_products + 1):
            name = f'Product_{i}'
            category = random.choice(categories)
            cost = round(random.uniform(5, 50), 2)
            writer.writerow([i, name, category, cost])
    
    print(f"Generated {num_products} products")

if __name__ == "__main__":
    import os
    os.makedirs('data', exist_ok=True)
    
    generate_orders(1000)
    generate_customers(100)
    generate_products(50)
    
    print("\n✅ Sample data generated successfully!")
