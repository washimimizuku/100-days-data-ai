"""
Day 4: Avro - Solutions
"""

from avro.datafile import DataFileWriter, DataFileReader
from avro.io import DatumWriter, DatumReader
import avro.schema
import json
import os

# Exercise 1: Basic Avro
print("Exercise 1: Basic Avro")

# Define schema
schema = avro.schema.parse('''
{
    "type": "record",
    "name": "Employee",
    "fields": [
        {"name": "name", "type": "string"},
        {"name": "age", "type": "int"},
        {"name": "department", "type": "string"}
    ]
}
''')

# Write records
with DataFileWriter(open("employees.avro", "wb"), DatumWriter(), schema) as writer:
    writer.append({"name": "Alice", "age": 25, "department": "Engineering"})
    writer.append({"name": "Bob", "age": 30, "department": "Sales"})
    writer.append({"name": "Charlie", "age": 28, "department": "Marketing"})
    writer.append({"name": "Diana", "age": 35, "department": "Engineering"})
    writer.append({"name": "Eve", "age": 32, "department": "Sales"})

print("Written 5 employees to employees.avro")

# Read records
with DataFileReader(open("employees.avro", "rb"), DatumReader()) as reader:
    for employee in reader:
        print(f"  {employee['name']}, {employee['age']}, {employee['department']}")
print()

# Exercise 2: Schema Evolution
print("Exercise 2: Schema Evolution")

# V1 Schema
schema_v1 = avro.schema.parse('''
{
    "type": "record",
    "name": "User",
    "fields": [
        {"name": "id", "type": "int"},
        {"name": "name", "type": "string"},
        {"name": "email", "type": "string"}
    ]
}
''')

# Write with V1
with DataFileWriter(open("users_v1.avro", "wb"), DatumWriter(), schema_v1) as writer:
    writer.append({"id": 1, "name": "Alice", "email": "alice@example.com"})
    writer.append({"id": 2, "name": "Bob", "email": "bob@example.com"})

print("Written data with V1 schema")

# V2 Schema (added optional phone field)
schema_v2 = avro.schema.parse('''
{
    "type": "record",
    "name": "User",
    "fields": [
        {"name": "id", "type": "int"},
        {"name": "name", "type": "string"},
        {"name": "email", "type": "string"},
        {"name": "phone", "type": ["null", "string"], "default": null}
    ]
}
''')

# Read V1 data with V2 schema
print("Reading V1 data with V2 schema:")
with DataFileReader(open("users_v1.avro", "rb"), DatumReader()) as reader:
    for user in reader:
        phone = user.get('phone', 'N/A')
        print(f"  {user['name']}: {user['email']}, phone: {phone}")
print()

# Exercise 3: Avro vs JSON Size
print("Exercise 3: Avro vs JSON Size")

# Create dataset
data = [
    {"id": i, "name": f"User_{i}", "value": i * 1.5}
    for i in range(100)
]

# Write to Avro
schema = avro.schema.parse('''
{
    "type": "record",
    "name": "Data",
    "fields": [
        {"name": "id", "type": "int"},
        {"name": "name", "type": "string"},
        {"name": "value", "type": "double"}
    ]
}
''')

with DataFileWriter(open("data.avro", "wb"), DatumWriter(), schema) as writer:
    for record in data:
        writer.append(record)

# Write to JSON
with open("data.json", "w") as f:
    json.dump(data, f)

# Compare sizes
avro_size = os.path.getsize("data.avro")
json_size = os.path.getsize("data.json")

print(f"Avro size: {avro_size / 1024:.2f} KB")
print(f"JSON size: {json_size / 1024:.2f} KB")
print(f"Avro is {json_size / avro_size:.2f}x smaller")
print(f"Space saved: {(1 - avro_size/json_size) * 100:.1f}%")
print()

# Exercise 4: Kafka Simulation
print("Exercise 4: Kafka Simulation")

# Event schema
event_schema = avro.schema.parse('''
{
    "type": "record",
    "name": "Event",
    "fields": [
        {"name": "user_id", "type": "int"},
        {"name": "action", "type": "string"},
        {"name": "timestamp", "type": "long"}
    ]
}
''')

# Simulate producer
import time
events = [
    {"user_id": 123, "action": "click", "timestamp": int(time.time() * 1000)},
    {"user_id": 456, "action": "view", "timestamp": int(time.time() * 1000)},
    {"user_id": 789, "action": "purchase", "timestamp": int(time.time() * 1000)}
]

with DataFileWriter(open("events.avro", "wb"), DatumWriter(), event_schema) as writer:
    for event in events:
        writer.append(event)
        print(f"Producer: Sent {event['action']} event from user {event['user_id']}")

# Simulate consumer
print("\nConsumer: Reading events...")
with DataFileReader(open("events.avro", "rb"), DatumReader()) as reader:
    for event in reader:
        print(f"  Received: User {event['user_id']} did {event['action']}")
print()

# Exercise 5: Complex Schema
print("Exercise 5: Complex Schema")

# Schema with nested record and array
complex_schema = avro.schema.parse('''
{
    "type": "record",
    "name": "Person",
    "fields": [
        {"name": "name", "type": "string"},
        {
            "name": "address",
            "type": {
                "type": "record",
                "name": "Address",
                "fields": [
                    {"name": "street", "type": "string"},
                    {"name": "city", "type": "string"}
                ]
            }
        },
        {
            "name": "skills",
            "type": {"type": "array", "items": "string"}
        }
    ]
}
''')

# Write complex data
with DataFileWriter(open("complex.avro", "wb"), DatumWriter(), complex_schema) as writer:
    writer.append({
        "name": "Alice",
        "address": {"street": "123 Main St", "city": "New York"},
        "skills": ["Python", "SQL", "Spark"]
    })
    writer.append({
        "name": "Bob",
        "address": {"street": "456 Oak Ave", "city": "San Francisco"},
        "skills": ["Java", "Kafka", "Kubernetes"]
    })

print("Written complex data")

# Read complex data
with DataFileReader(open("complex.avro", "rb"), DatumReader()) as reader:
    for person in reader:
        print(f"  {person['name']} lives in {person['address']['city']}")
        print(f"    Skills: {', '.join(person['skills'])}")
print()

# Bonus Challenge
print("Bonus Challenge: Avro to JSON Converter")

def avro_to_json(avro_file, json_file):
    """
    Convert Avro file to JSON
    """
    records = []
    with DataFileReader(open(avro_file, "rb"), DatumReader()) as reader:
        for record in reader:
            records.append(record)
    
    with open(json_file, "w") as f:
        json.dump(records, f, indent=2)
    
    print(f"Converted {len(records)} records from {avro_file} to {json_file}")
    
    # Compare sizes
    avro_size = os.path.getsize(avro_file)
    json_size = os.path.getsize(json_file)
    print(f"Avro: {avro_size / 1024:.2f} KB")
    print(f"JSON: {json_size / 1024:.2f} KB")

# Test converter
avro_to_json("employees.avro", "employees.json")

print("\n✅ All exercises completed!")
