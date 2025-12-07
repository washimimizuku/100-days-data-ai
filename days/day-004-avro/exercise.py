"""
Day 4: Avro - Exercises
Complete each exercise below
"""

from avro.datafile import DataFileWriter, DataFileReader
from avro.io import DatumWriter, DatumReader
import avro.schema
import json

# Exercise 1: Basic Avro
# TODO: Define schema for employee data (name, age, department)
# TODO: Write 5 records to Avro file
# TODO: Read and print all records

print("Exercise 1: Basic Avro")
# Your code here


# Exercise 2: Schema Evolution
# TODO: Create V1 schema with 3 fields (id, name, email)
# TODO: Write data with V1 schema
# TODO: Create V2 schema adding optional field (phone)
# TODO: Read V1 data with V2 schema

print("\nExercise 2: Schema Evolution")
# Your code here


# Exercise 3: Avro vs JSON Size
# TODO: Create dataset with 100 records
# TODO: Write to Avro
# TODO: Write to JSON
# TODO: Compare file sizes

print("\nExercise 3: Avro vs JSON Size")
# Your code here


# Exercise 4: Kafka Simulation
# TODO: Simulate Kafka producer (write events to Avro file)
# TODO: Simulate Kafka consumer (read events from Avro file)
# TODO: Use event schema (user_id, action, timestamp)

print("\nExercise 4: Kafka Simulation")
# Your code here


# Exercise 5: Complex Schema
# TODO: Create schema with nested record (address: street, city)
# TODO: Create schema with array field (skills)
# TODO: Write and read complex data

print("\nExercise 5: Complex Schema")
# Your code here


# Bonus Challenge
# TODO: Create Avro to JSON converter
# TODO: Read Avro file and convert to JSON
# TODO: Preserve data types

print("\nBonus Challenge: Avro to JSON Converter")

def avro_to_json(avro_file, json_file):
    """
    Convert Avro file to JSON
    
    Args:
        avro_file: Input Avro file
        json_file: Output JSON file
    """
    # Your code here
    pass

# Test your converter
# avro_to_json('data.avro', 'data.json')
