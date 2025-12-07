"""Day 90: AWS Lambda - Solutions

NOTE: Mock implementations for learning without AWS account.
"""

import json
from typing import Dict, Any, List
from datetime import datetime


# Exercise 1: Basic Lambda Handler
def basic_lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Basic Lambda handler."""
    name = event.get('name', 'World')
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': f'Hello, {name}!',
            'request_id': context.request_id
        })
    }


# Exercise 2: API Gateway Handler
def api_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Handle HTTP requests from API Gateway."""
    method = event.get('httpMethod', 'GET')
    path = event.get('path', '/')
    query_params = event.get('queryStringParameters', {})
    body = event.get('body', '{}')
    
    if method == 'GET':
        item_id = query_params.get('id', 'all')
        return {
            'statusCode': 200,
            'body': json.dumps({
                'method': 'GET',
                'path': path,
                'item_id': item_id
            })
        }
    
    elif method == 'POST':
        data = json.loads(body) if body else {}
        return {
            'statusCode': 201,
            'body': json.dumps({
                'method': 'POST',
                'created': data
            })
        }
    
    elif method == 'PUT':
        data = json.loads(body) if body else {}
        item_id = query_params.get('id')
        return {
            'statusCode': 200,
            'body': json.dumps({
                'method': 'PUT',
                'updated': item_id,
                'data': data
            })
        }
    
    elif method == 'DELETE':
        item_id = query_params.get('id')
        return {
            'statusCode': 200,
            'body': json.dumps({
                'method': 'DELETE',
                'deleted': item_id
            })
        }
    
    return {
        'statusCode': 405,
        'body': json.dumps({'error': 'Method not allowed'})
    }


# Exercise 3: S3 Event Processor
def s3_event_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Process S3 file upload events."""
    processed_files = []
    
    for record in event.get('Records', []):
        s3_info = record.get('s3', {})
        bucket = s3_info.get('bucket', {}).get('name')
        obj = s3_info.get('object', {})
        key = obj.get('key')
        size = obj.get('size', 0)
        
        processed_files.append({
            'bucket': bucket,
            'key': key,
            'size': size,
            'processed_at': datetime.now().isoformat()
        })
        
        print(f"Processed: s3://{bucket}/{key} ({size} bytes)")
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'processed_count': len(processed_files),
            'files': processed_files
        })
    }


# Exercise 4: ML Inference Handler
class MockModel:
    """Mock ML model."""
    
    def predict(self, features: List[float]) -> List[float]:
        """Mock prediction."""
        return [sum(features) / len(features)]


model = MockModel()


def ml_inference_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Run ML model inference."""
    try:
        body = event.get('body', '{}')
        data = json.loads(body) if isinstance(body, str) else body
        features = data.get('features', [])
        
        if not features:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing features'})
            }
        
        prediction = model.predict(features)
        confidence = min(prediction[0] / 10.0, 1.0)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'prediction': prediction,
                'confidence': confidence,
                'model_version': '1.0',
                'request_id': context.request_id
            })
        }
    
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


# Exercise 5: Async Lambda Pipeline
class MockLambdaClient:
    """Mock Lambda client."""
    
    def __init__(self):
        self.invocations = []
    
    def invoke(self, FunctionName: str, InvocationType: str, 
               Payload: str) -> Dict[str, Any]:
        """Mock Lambda invocation."""
        self.invocations.append({
            'function': FunctionName,
            'type': InvocationType,
            'payload': Payload,
            'timestamp': datetime.now().isoformat()
        })
        return {'StatusCode': 202}


def pipeline_orchestrator(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Orchestrate multi-Lambda pipeline."""
    lambda_client = MockLambdaClient()
    
    # Step 1: Validate data
    validation_payload = json.dumps({
        'action': 'validate',
        'data': event.get('data')
    })
    
    lambda_client.invoke(
        FunctionName='data-validator',
        InvocationType='RequestResponse',
        Payload=validation_payload
    )
    
    # Step 2: Process data (async)
    processing_payload = json.dumps({
        'action': 'process',
        'data': event.get('data')
    })
    
    lambda_client.invoke(
        FunctionName='data-processor',
        InvocationType='Event',
        Payload=processing_payload
    )
    
    # Step 3: Store results (async)
    storage_payload = json.dumps({
        'action': 'store',
        'data': event.get('data')
    })
    
    lambda_client.invoke(
        FunctionName='data-storage',
        InvocationType='Event',
        Payload=storage_payload
    )
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'pipeline_status': 'initiated',
            'steps': len(lambda_client.invocations),
            'invocations': lambda_client.invocations
        })
    }


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
    """Store data in DynamoDB."""
    table = MockDynamoDBTable('predictions')
    
    item_id = event.get('id', context.request_id)
    data = event.get('data', {})
    
    item = {
        'id': item_id,
        'data': data,
        'timestamp': datetime.now().isoformat(),
        'request_id': context.request_id
    }
    
    response = table.put_item(Item=item)
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'stored': item_id,
            'table': table.table_name
        })
    }


# Mock Context Class
class MockContext:
    """Mock Lambda context."""
    
    def __init__(self):
        self.request_id = f"req-{datetime.now().timestamp()}"
        self.function_name = "mock-function"
        self.memory_limit_in_mb = 512
        self.invoked_function_arn = "arn:aws:lambda:us-east-1:123456789012:function:mock"


def demo_lambda_functions():
    """Demonstrate Lambda functions."""
    print("Day 90: AWS Lambda - Solutions Demo\n" + "=" * 60)
    
    context = MockContext()
    
    print("\n1. Basic Lambda Handler")
    event1 = {"name": "Alice"}
    result1 = basic_lambda_handler(event1, context)
    print(f"   Input: {event1}")
    print(f"   Output: {json.loads(result1['body'])['message']}")
    
    print("\n2. API Gateway Handler")
    event2 = {
        "httpMethod": "POST",
        "path": "/items",
        "body": json.dumps({"name": "Item 1", "price": 29.99})
    }
    result2 = api_handler(event2, context)
    print(f"   Method: POST")
    print(f"   Response: {result2['statusCode']}")
    
    print("\n3. S3 Event Processor")
    event3 = {
        "Records": [
            {
                "s3": {
                    "bucket": {"name": "data-bucket"},
                    "object": {"key": "uploads/data.csv", "size": 2048}
                }
            },
            {
                "s3": {
                    "bucket": {"name": "data-bucket"},
                    "object": {"key": "uploads/model.pkl", "size": 4096}
                }
            }
        ]
    }
    result3 = s3_event_handler(event3, context)
    body3 = json.loads(result3['body'])
    print(f"   Processed {body3['processed_count']} files")
    
    print("\n4. ML Inference Handler")
    event4 = {
        "body": json.dumps({"features": [1.5, 2.5, 3.5, 4.5]})
    }
    result4 = ml_inference_handler(event4, context)
    body4 = json.loads(result4['body'])
    print(f"   Prediction: {body4['prediction']}")
    print(f"   Confidence: {body4['confidence']:.2f}")
    
    print("\n5. Async Lambda Pipeline")
    event5 = {"data": "sample dataset"}
    result5 = pipeline_orchestrator(event5, context)
    body5 = json.loads(result5['body'])
    print(f"   Pipeline status: {body5['pipeline_status']}")
    print(f"   Steps invoked: {body5['steps']}")
    
    print("\n6. DynamoDB Handler")
    event6 = {
        "id": "pred-001",
        "data": {"model": "v1", "score": 0.95}
    }
    result6 = dynamodb_handler(event6, context)
    body6 = json.loads(result6['body'])
    print(f"   Stored item: {body6['stored']}")
    print(f"   Table: {body6['table']}")
    
    print("\n" + "=" * 60)
    print("All Lambda patterns demonstrated!")


if __name__ == "__main__":
    demo_lambda_functions()
