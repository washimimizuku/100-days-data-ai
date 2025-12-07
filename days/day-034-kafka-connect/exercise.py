"""
Day 34: Kafka Connect - Exercises

Prerequisites:
1. Start Kafka Connect:
   docker run -d --name kafka-connect -p 8083:8083 \
     confluentinc/cp-kafka-connect:latest
2. Install: pip install requests kafka-python
"""

import requests
import json
import time


CONNECT_URL = "http://localhost:8083"


def exercise_1_file_source_connector():
    """
    Exercise 1: File Source Connector
    
    Configure file source connector to read from file.
    
    TODO: Create connector configuration for FileStreamSource
    TODO: Set file path to /tmp/input.txt
    TODO: Set topic to 'file-input'
    TODO: POST configuration to Connect REST API
    TODO: Verify connector is running
    """
    pass


def exercise_2_file_sink_connector():
    """
    Exercise 2: File Sink Connector
    
    Configure file sink connector to write to file.
    
    TODO: Create connector configuration for FileStreamSink
    TODO: Set file path to /tmp/output.txt
    TODO: Set topics to 'file-input'
    TODO: POST configuration to Connect REST API
    TODO: Verify data flows from source to sink
    """
    pass


def exercise_3_jdbc_source_connector():
    """
    Exercise 3: JDBC Source Connector
    
    Pull data from PostgreSQL into Kafka.
    
    TODO: Create JDBC source connector configuration
    TODO: Set connection URL, user, password
    TODO: Configure incrementing mode with id column
    TODO: Set table whitelist
    TODO: POST configuration to Connect REST API
    TODO: Verify data is flowing to Kafka
    """
    pass


def exercise_4_jdbc_sink_connector():
    """
    Exercise 4: JDBC Sink Connector
    
    Push data from Kafka to PostgreSQL.
    
    TODO: Create JDBC sink connector configuration
    TODO: Set connection URL, user, password
    TODO: Configure upsert mode with primary key
    TODO: Set topics to consume from
    TODO: POST configuration to Connect REST API
    TODO: Verify data is written to database
    """
    pass


def exercise_5_connector_management():
    """
    Exercise 5: Connector Management
    
    Use REST API to manage connectors.
    
    TODO: List all connectors
    TODO: Get status of specific connector
    TODO: Pause a connector
    TODO: Resume a connector
    TODO: Restart a connector
    TODO: Delete a connector
    """
    pass


def exercise_6_single_message_transforms():
    """
    Exercise 6: Single Message Transforms
    
    Apply SMTs to add timestamps and mask fields.
    
    TODO: Create connector with InsertField transform
    TODO: Add timestamp field to messages
    TODO: Add MaskField transform for sensitive data
    TODO: Configure field masking for 'ssn' and 'credit_card'
    TODO: Verify transforms are applied
    """
    pass


def exercise_7_custom_source_connector():
    """
    Exercise 7: Custom Source Connector
    
    Implement custom source connector in Python.
    
    TODO: Create CustomSourceConnector class
    TODO: Implement poll() method to fetch data
    TODO: Send data to Kafka topic
    TODO: Handle errors and retries
    TODO: Implement graceful shutdown
    """
    pass


def exercise_8_monitoring_error_handling():
    """
    Exercise 8: Monitoring and Error Handling
    
    Monitor connector health and handle errors.
    
    TODO: Get connector status and metrics
    TODO: Check task status
    TODO: Configure error handling (tolerance, DLQ)
    TODO: Monitor connector lag
    TODO: Set up alerts for failures
    """
    pass


if __name__ == "__main__":
    print("Day 34: Kafka Connect - Exercises\n")
    print("=" * 60)
    print("\nMake sure Kafka Connect is running:")
    print("  docker run -d --name kafka-connect -p 8083:8083 \\")
    print("    confluentinc/cp-kafka-connect:latest\n")
    
    # Uncomment to run exercises
    # print("\nExercise 1: File Source Connector")
    # exercise_1_file_source_connector()
    
    # print("\nExercise 2: File Sink Connector")
    # exercise_2_file_sink_connector()
    
    # print("\nExercise 3: JDBC Source Connector")
    # exercise_3_jdbc_source_connector()
    
    # print("\nExercise 4: JDBC Sink Connector")
    # exercise_4_jdbc_sink_connector()
    
    # print("\nExercise 5: Connector Management")
    # exercise_5_connector_management()
    
    # print("\nExercise 6: Single Message Transforms")
    # exercise_6_single_message_transforms()
    
    # print("\nExercise 7: Custom Source Connector")
    # exercise_7_custom_source_connector()
    
    # print("\nExercise 8: Monitoring and Error Handling")
    # exercise_8_monitoring_error_handling()
