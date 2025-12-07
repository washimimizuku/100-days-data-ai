"""Day 90: AWS Lambda - Exercises

NOTE: These exercises use mock implementations for learning.
For real Lambda deployment, use AWS Console, SAM, or Serverless Framework.
"""

import json
from typing import Dict, Any, List
from datetime import datetime


# Exercise 1: Basic Lambda Handler
def basic_lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Create a basic Lambda handler.
    
    Args:
        event: Input data
        context: Runtime context
    
    Returns:
        Response with statusCode and body
    
    TODO: Extract 'name' from event
    TODO: Return greeting message
    TODO: Handle missing name with default
    """
    pass


# Exercise 2: API Gateway Handler
def api_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Handle HTTP requests from API Gateway.
    
    Event structure:
    {
        'httpMethod': 'GET',
        'path': '/items',
        'queryStringParameters': {'id': '123'},
        'body': '{"data": "value"}'
    }
    
    TODO: Extract HTTP method and path
    TODO: Handle GET, POST, PUT, DELETE
    TODO: Return appropriate status codes
    TODO: Parse body for POST/PUT requests
    """
    pass


# Exercise 3: S3 Event Processor
def s3_event_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Process S3 file upload events.
    
    Event structure:
    {
        'Records': [
            {
                's3': {
                    'bucket': {'name': 'my-bucket'},
                    'object': {'key': 'file.txt', 'size': 1024}
                }
            }
        ]
    }
    
    TODO: Extract bucket name and object key
    TODO: Process each record
    TODO: Return summary of processed files
    """
    pass


# Exercise 4: ML Inference Handler
class MockModel:
    """Mock ML model for inference."""
    
    def predict(self, features: List[float]) -> List[float]:
        """Mock prediction."""
        return [sum(features) / len(features)]


# Global model (loaded once)
model = MockModel()


def ml_inference_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Run ML model inference.
    
    Event structure:
    {
        'body': '{"features": [1.0, 2.0, 3.0]}'
    }
    
    TODO: Parse features from body
    TODO: Run model prediction
    TODO: Return prediction with confidence
    TODO: Handle invalid input
    """
    pass


# Exercise 5: Async Lambda Pipeline
class MockLambdaClient:
    """Mock Lambda client for async invocation."""
    
    def __init__(self):
        self.invocations = []
    
    def invoke(self, FunctionName: str, InvocationType: str, 
               Payload: str) -> Dict[str, Any]:
        """Mock Lambda invocation."""
        self.invocations.append({
            'function': FunctionName,
            'type': InvocationType,
            'payload': Payload
        })
        return {'StatusCode': 202}


def pipeline_orchestrator(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Orchestrate multi-Lambda pipeline.
    
    Pipeline steps:
    1. Validate data
    2. Process data (async)
    3. Store results (async)
    
    TODO: Create Lambda client
    TODO: Invoke validation function
    TODO: Invoke processing function asynchronously
    TODO: Invoke storage function asynchronously
    TODO: Return pipeline status
    """
    pass


# Bonus: DynamoDB Integration
class MockDynamoDBTable:
    """Mock DynamoDB table."""
    
    def __init__(self, table_name: str):
        self.table_name = table_name
        self.items = {}
    
    def put_item(self, Item: Dict[str, Any]) -> Dict[str, Any]:
        """Store item."""
        item_id = Item.get('id')
        self.items[item_id] = Item
        return {'ResponseMetadata': {'HTTPStatusCode': 200}}
    
    def get_item(self, Key: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve item."""
        item_id = Key.get('id')
        if item_id in self.items:
            return {'Item': self.items[item_id]}
        return {}


def dynamodb_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Store data in DynamoDB.
    
    TODO: Create DynamoDB table
    TODO: Extract data from event
    TODO: Store item with timestamp
    TODO: Return success response
    """
    pass


# Mock Context Class
class MockContext:
    """Mock Lambda context."""
    
    def __init__(self):
        self.request_id = "mock-request-id-123"
        self.function_name = "mock-function"
        self.memory_limit_in_mb = 512
        self.invoked_function_arn = "arn:aws:lambda:us-east-1:123456789012:function:mock"


if __name__ == "__main__":
    print("Day 90: AWS Lambda - Exercises")
    print("=" * 50)
    
    context = MockContext()
    
    # Test Exercise 1
    print("\nExercise 1: Basic Lambda Handler")
    event1 = {"name": "Alice"}
    print(f"Event: {event1}")
    # result1 = basic_lambda_handler(event1, context)
    # print(f"Result: {result1}")
    
    # Test Exercise 2
    print("\nExercise 2: API Gateway Handler")
    event2 = {
        "httpMethod": "GET",
        "path": "/items",
        "queryStringParameters": {"id": "123"}
    }
    print(f"Event: {event2}")
    # result2 = api_handler(event2, context)
    # print(f"Result: {result2}")
    
    # Test Exercise 3
    print("\nExercise 3: S3 Event Processor")
    event3 = {
        "Records": [
            {
                "s3": {
                    "bucket": {"name": "my-bucket"},
                    "object": {"key": "data.csv", "size": 1024}
                }
            }
        ]
    }
    print(f"Event: {event3}")
    # result3 = s3_event_handler(event3, context)
    # print(f"Result: {result3}")
    
    # Test Exercise 4
    print("\nExercise 4: ML Inference Handler")
    event4 = {
        "body": json.dumps({"features": [1.0, 2.0, 3.0]})
    }
    print(f"Event: {event4}")
    # result4 = ml_inference_handler(event4, context)
    # print(f"Result: {result4}")
    
    # Test Exercise 5
    print("\nExercise 5: Async Lambda Pipeline")
    event5 = {"data": "sample data"}
    print(f"Event: {event5}")
    # result5 = pipeline_orchestrator(event5, context)
    # print(f"Result: {result5}")
    
    print("\n" + "=" * 50)
    print("Complete the TODOs to finish the exercises!")
    print("\nNote: These are mock implementations for learning.")
    print("For real Lambda deployment, use AWS Console or SAM.")
