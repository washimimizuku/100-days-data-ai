# Day 89: AWS S3 & EC2

## Learning Objectives

**Time**: 1 hour

- Understand AWS S3 for data storage
- Learn EC2 for compute resources
- Implement data pipelines with S3
- Deploy AI applications on EC2

## Theory (15 minutes)

### AWS Overview

Amazon Web Services (AWS) provides cloud infrastructure for deploying and scaling AI applications. Two fundamental services are S3 (storage) and EC2 (compute).

### Amazon S3 (Simple Storage Service)

**What is S3?**: Object storage service for storing and retrieving any amount of data.

**Key Concepts**:
- **Buckets**: Containers for objects (like folders)
- **Objects**: Files with metadata
- **Keys**: Unique identifiers for objects
- **Regions**: Geographic locations

**Use Cases**:
- Data lakes for ML training data
- Model storage and versioning
- Application backups
- Static website hosting

### S3 Storage Classes

**Standard**: Frequent access, high durability
**Intelligent-Tiering**: Automatic cost optimization
**Glacier**: Long-term archival, low cost
**One Zone-IA**: Infrequent access, single AZ

### S3 Operations

**Upload**:
```python
import boto3

s3 = boto3.client('s3')
s3.upload_file('local.txt', 'my-bucket', 'remote.txt')
```

**Download**:
```python
s3.download_file('my-bucket', 'remote.txt', 'local.txt')
```

**List Objects**:
```python
response = s3.list_objects_v2(Bucket='my-bucket')
for obj in response['Contents']:
    print(obj['Key'])
```

**Delete**:
```python
s3.delete_object(Bucket='my-bucket', Key='file.txt')
```

### S3 for ML Workflows

**Training Data**:
```python
# Store datasets
s3.upload_file('train.csv', 'ml-bucket', 'data/train.csv')

# Load for training
import pandas as pd
obj = s3.get_object(Bucket='ml-bucket', Key='data/train.csv')
df = pd.read_csv(obj['Body'])
```

**Model Storage**:
```python
# Save model
import joblib
joblib.dump(model, 'model.pkl')
s3.upload_file('model.pkl', 'ml-bucket', 'models/v1/model.pkl')

# Load model
s3.download_file('ml-bucket', 'models/v1/model.pkl', 'model.pkl')
model = joblib.load('model.pkl')
```

### Amazon EC2 (Elastic Compute Cloud)

**What is EC2?**: Virtual servers in the cloud for running applications.

**Key Concepts**:
- **Instances**: Virtual machines
- **AMI**: Amazon Machine Image (OS template)
- **Instance Types**: CPU, memory, storage configurations
- **Security Groups**: Firewall rules

**Instance Types**:
- **t2/t3**: General purpose, burstable
- **c5**: Compute optimized
- **r5**: Memory optimized
- **p3/g4**: GPU instances for ML

### EC2 Operations

**Launch Instance**:
```python
ec2 = boto3.client('ec2')

response = ec2.run_instances(
    ImageId='ami-12345678',
    InstanceType='t2.micro',
    MinCount=1,
    MaxCount=1,
    KeyName='my-key'
)
```

**List Instances**:
```python
response = ec2.describe_instances()
for reservation in response['Reservations']:
    for instance in reservation['Instances']:
        print(f"{instance['InstanceId']}: {instance['State']['Name']}")
```

**Stop/Start**:
```python
ec2.stop_instances(InstanceIds=['i-1234567890abcdef0'])
ec2.start_instances(InstanceIds=['i-1234567890abcdef0'])
```

**Terminate**:
```python
ec2.terminate_instances(InstanceIds=['i-1234567890abcdef0'])
```

### Deploying AI on EC2

**Setup**:
```bash
# SSH into instance
ssh -i key.pem ec2-user@ec2-ip-address

# Install dependencies
sudo yum update -y
sudo yum install python3 -y
pip3 install torch transformers boto3

# Download model from S3
aws s3 cp s3://ml-bucket/models/model.pkl .
```

**Run Application**:
```python
# app.py
from flask import Flask, request
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict([data['features']])
    return {'prediction': prediction.tolist()}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### S3 + EC2 Integration

**Data Pipeline**:
```python
# 1. Upload data to S3
s3.upload_file('data.csv', 'pipeline-bucket', 'input/data.csv')

# 2. EC2 processes data
# (Running on EC2 instance)
s3.download_file('pipeline-bucket', 'input/data.csv', 'data.csv')
processed = process_data('data.csv')
processed.to_csv('output.csv')
s3.upload_file('output.csv', 'pipeline-bucket', 'output/data.csv')

# 3. Download results
s3.download_file('pipeline-bucket', 'output/data.csv', 'results.csv')
```

### Security Best Practices

**IAM Roles**: Use roles instead of access keys
**Encryption**: Enable S3 encryption at rest
**Security Groups**: Restrict EC2 access
**VPC**: Use Virtual Private Cloud for isolation
**Least Privilege**: Grant minimum required permissions

### Cost Optimization

**S3**:
- Use appropriate storage class
- Enable lifecycle policies
- Delete unused objects

**EC2**:
- Use spot instances for batch jobs
- Stop instances when not in use
- Right-size instance types
- Use auto-scaling

### Boto3 Basics

```python
import boto3

# Create clients
s3 = boto3.client('s3')
ec2 = boto3.client('ec2')

# Or use resources (higher-level)
s3_resource = boto3.resource('s3')
bucket = s3_resource.Bucket('my-bucket')
```

### Error Handling

```python
from botocore.exceptions import ClientError

try:
    s3.upload_file('file.txt', 'bucket', 'key')
except ClientError as e:
    if e.response['Error']['Code'] == 'NoSuchBucket':
        print("Bucket doesn't exist")
    else:
        raise
```

### Monitoring

**CloudWatch**: Monitor metrics and logs
**S3 Metrics**: Request metrics, storage metrics
**EC2 Metrics**: CPU, network, disk usage

### Use Cases

**ML Training Pipeline**:
1. Store training data in S3
2. Launch EC2 GPU instance
3. Download data from S3
4. Train model
5. Upload model to S3
6. Terminate instance

**Inference API**:
1. Deploy model on EC2
2. Load model from S3
3. Serve predictions via API
4. Log results to S3

**Data Processing**:
1. Upload raw data to S3
2. EC2 processes data
3. Store results in S3
4. Trigger downstream workflows

### Why This Matters

AWS S3 and EC2 are fundamental for deploying production AI systems. S3 provides scalable storage for data and models, while EC2 provides compute for training and inference. Understanding these services is essential for cloud-based AI applications.

## Exercise (40 minutes)

Complete the exercises in `exercise.py`:

1. **S3 Operations**: Upload, download, list files
2. **Bucket Management**: Create and configure buckets
3. **EC2 Management**: Launch and manage instances
4. **Data Pipeline**: Build S3 + EC2 pipeline
5. **ML Deployment**: Deploy model on EC2 with S3

## Resources

- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)
- [AWS EC2 Documentation](https://docs.aws.amazon.com/ec2/)
- [Boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [AWS Free Tier](https://aws.amazon.com/free/)

## Next Steps

- Complete the exercises
- Review the solution
- Take the quiz
- Move to Day 90: AWS Lambda

Tomorrow you'll learn about serverless computing with AWS Lambda for event-driven AI applications.
