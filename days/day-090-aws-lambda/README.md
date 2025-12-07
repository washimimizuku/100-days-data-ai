# Day 90: AWS Lambda

## Learning Objectives

**Time**: 1 hour

- Understand serverless computing with AWS Lambda
- Learn Lambda function structure and lifecycle
- Implement event-driven architectures
- Deploy AI/ML models with Lambda

## Theory (15 minutes)

### What is AWS Lambda?

AWS Lambda is a serverless compute service that runs code in response to events without managing servers.

**Key Benefits**:
- No server management
- Automatic scaling
- Pay per execution
- Built-in fault tolerance

### Lambda Basics

**Function Structure**:
```python
def lambda_handler(event, context):
    """
    Lambda function entry point.
    
    Args:
        event: Input data (dict)
        context: Runtime information
    
    Returns:
        Response dict
    """
    return {
        'statusCode': 200,
        'body': 'Hello from Lambda!'
    }
```

**Event Object**: Contains input data from trigger
**Context Object**: Provides runtime information (request ID, memory limit, etc.)

### Lambda Triggers

**Common Event Sources**:
- API Gateway (HTTP requests)
- S3 (file uploads)
- DynamoDB (table changes)
- EventBridge (scheduled events)
- SQS (message queues)
- SNS (notifications)

### Lambda Lifecycle

1. **Cold Start**: First invocation, initialize runtime
2. **Warm Execution**: Reuse existing container
3. **Timeout**: Max execution time (15 minutes)
4. **Cleanup**: Container recycled after idle period

### API Gateway Integration

```python
def lambda_handler(event, context):
    """Handle HTTP request from API Gateway."""
    method = event['httpMethod']
    path = event['path']
    body = event.get('body', '')
    
    if method == 'GET':
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'GET request'})
        }
    
    return {
        'statusCode': 405,
        'body': json.dumps({'error': 'Method not allowed'})
    }
```

### S3 Event Processing

```python
import json

def lambda_handler(event, context):
    """Process S3 file upload."""
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        
        print(f"Processing {key} from {bucket}")
        
        # Download and process file
        # s3.download_file(bucket, key, '/tmp/file')
        
    return {'statusCode': 200}
```

### Environment Variables

```python
import os

def lambda_handler(event, context):
    """Use environment variables."""
    api_key = os.environ.get('API_KEY')
    model_path = os.environ.get('MODEL_PATH', 's3://bucket/model.pkl')
    
    return {'config': model_path}
```

### Lambda Layers

Layers package dependencies separately from function code.

**Benefits**:
- Reduce deployment package size
- Share code across functions
- Faster deployments

**Example**:
```
my-layer/
└── python/
    └── lib/
        └── python3.9/
            └── site-packages/
                ├── numpy/
                └── pandas/
```

### Error Handling

```python
def lambda_handler(event, context):
    """Handle errors gracefully."""
    try:
        result = process_data(event)
        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }
    except ValueError as e:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': str(e)})
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Internal error'})
        }
```

### ML Model Inference

```python
import json
import boto3
import joblib

# Load model once (outside handler)
s3 = boto3.client('s3')
s3.download_file('ml-bucket', 'model.pkl', '/tmp/model.pkl')
model = joblib.load('/tmp/model.pkl')

def lambda_handler(event, context):
    """Run model inference."""
    data = json.loads(event['body'])
    features = data['features']
    
    prediction = model.predict([features])
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'prediction': prediction.tolist()
        })
    }
```

### Async Invocation

```python
import boto3

lambda_client = boto3.client('lambda')

# Invoke another Lambda asynchronously
response = lambda_client.invoke(
    FunctionName='data-processor',
    InvocationType='Event',  # Async
    Payload=json.dumps({'data': 'value'})
)
```

### Lambda with DynamoDB

```python
import boto3

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('predictions')

def lambda_handler(event, context):
    """Store prediction in DynamoDB."""
    prediction_id = event['id']
    result = event['result']
    
    table.put_item(
        Item={
            'id': prediction_id,
            'result': result,
            'timestamp': context.request_id
        }
    )
    
    return {'statusCode': 200}
```

### Performance Optimization

**Cold Start Reduction**:
- Minimize package size
- Use Lambda layers
- Keep functions warm with scheduled pings
- Increase memory (faster CPU)

**Memory vs Cost**:
- More memory = faster execution
- Find optimal balance
- Use Lambda Power Tuning

### Monitoring with CloudWatch

```python
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """Log to CloudWatch."""
    logger.info(f"Processing request: {context.request_id}")
    
    # Custom metrics
    logger.info(f"METRIC|ProcessingTime|{duration}|Milliseconds")
    
    return {'statusCode': 200}
```

### Lambda Pricing

**Cost Factors**:
- Number of requests
- Duration (GB-seconds)
- Memory allocation

**Example**:
- 1M requests/month
- 512 MB memory
- 200ms average duration
- Cost: ~$5/month

**Free Tier**:
- 1M requests/month
- 400,000 GB-seconds/month

### Deployment Methods

**AWS Console**: Manual upload
**AWS CLI**: Command-line deployment
**SAM**: Serverless Application Model
**Serverless Framework**: Multi-cloud tool
**Terraform**: Infrastructure as code

### Lambda Limitations

**Constraints**:
- Max execution time: 15 minutes
- Max memory: 10 GB
- Max deployment package: 250 MB (unzipped)
- /tmp storage: 10 GB
- Concurrent executions: 1000 (default)

### Use Cases

**Data Processing**:
- ETL pipelines
- Image/video processing
- Log analysis
- Real-time transformations

**API Backends**:
- REST APIs
- GraphQL resolvers
- Webhooks
- Microservices

**ML Inference**:
- Real-time predictions
- Batch scoring
- Model serving
- Feature extraction

**Automation**:
- Scheduled tasks
- Event-driven workflows
- Infrastructure automation
- Notifications

### Best Practices

1. **Keep functions small**: Single responsibility
2. **Reuse connections**: Database, S3 clients
3. **Use environment variables**: Configuration
4. **Handle errors gracefully**: Return proper status codes
5. **Monitor and log**: CloudWatch integration
6. **Optimize cold starts**: Minimize dependencies
7. **Use layers**: Share common code
8. **Set appropriate timeouts**: Avoid hanging functions

### Why This Matters

Lambda enables serverless AI/ML deployments without infrastructure management. It's ideal for event-driven architectures, API backends, and scalable inference endpoints. Understanding Lambda is essential for building cost-effective, scalable AI applications.

## Exercise (40 minutes)

Complete the exercises in `exercise.py`:

1. **Basic Lambda**: Create simple Lambda handler
2. **API Handler**: Process HTTP requests
3. **S3 Processor**: Handle S3 events
4. **ML Inference**: Deploy model inference
5. **Async Pipeline**: Build multi-Lambda workflow

## Resources

- [AWS Lambda Documentation](https://docs.aws.amazon.com/lambda/)
- [Lambda Best Practices](https://docs.aws.amazon.com/lambda/latest/dg/best-practices.html)
- [Serverless Framework](https://www.serverless.com/)
- [AWS SAM](https://aws.amazon.com/serverless/sam/)

## Next Steps

- Complete the exercises
- Review the solution
- Take the quiz
- Move to Day 91: Mini Project - AI Agent

Tomorrow you'll build a complete AI agent that combines all the concepts from Week 13: agents, tools, LangGraph, and AWS deployment.
