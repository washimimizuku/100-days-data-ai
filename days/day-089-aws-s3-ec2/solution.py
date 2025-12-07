"""Day 89: AWS S3 & EC2 - Solutions

NOTE: Mock implementations for learning without AWS account.
"""

from typing import List, Dict, Any
from datetime import datetime
import json


# Exercise 1: S3 Operations
class MockS3Client:
    """Mock S3 client."""
    
    def __init__(self):
        self.buckets = {}
    
    def upload_file(self, filename: str, bucket: str, key: str) -> bool:
        """Upload file to S3."""
        if bucket not in self.buckets:
            self.buckets[bucket] = {}
        
        self.buckets[bucket][key] = {
            "filename": filename,
            "size": len(filename) * 100,  # Mock size
            "last_modified": datetime.now().isoformat()
        }
        return True
    
    def download_file(self, bucket: str, key: str, filename: str) -> bool:
        """Download file from S3."""
        if bucket in self.buckets and key in self.buckets[bucket]:
            return True
        return False
    
    def list_objects(self, bucket: str, prefix: str = "") -> List[Dict]:
        """List objects in bucket."""
        if bucket not in self.buckets:
            return []
        
        objects = []
        for key, info in self.buckets[bucket].items():
            if key.startswith(prefix):
                objects.append({"Key": key, **info})
        return objects
    
    def delete_object(self, bucket: str, key: str) -> bool:
        """Delete object from S3."""
        if bucket in self.buckets and key in self.buckets[bucket]:
            del self.buckets[bucket][key]
            return True
        return False


# Exercise 2: Bucket Management
class S3BucketManager:
    """Manage S3 buckets."""
    
    def __init__(self, s3_client: MockS3Client):
        self.s3 = s3_client
    
    def create_bucket(self, bucket_name: str, region: str = "us-east-1") -> bool:
        """Create S3 bucket."""
        if bucket_name not in self.s3.buckets:
            self.s3.buckets[bucket_name] = {}
            return True
        return False
    
    def list_buckets(self) -> List[str]:
        """List all buckets."""
        return list(self.s3.buckets.keys())
    
    def delete_bucket(self, bucket_name: str) -> bool:
        """Delete bucket."""
        if bucket_name in self.s3.buckets and not self.s3.buckets[bucket_name]:
            del self.s3.buckets[bucket_name]
            return True
        return False
    
    def configure_lifecycle(self, bucket_name: str, rules: List[Dict]) -> bool:
        """Configure lifecycle rules."""
        if bucket_name in self.s3.buckets:
            self.s3.buckets[bucket_name]["_lifecycle"] = rules
            return True
        return False


# Exercise 3: EC2 Management
class MockEC2Client:
    """Mock EC2 client."""
    
    def __init__(self):
        self.instances = {}
        self.instance_counter = 0
    
    def run_instances(self, image_id: str, instance_type: str, count: int = 1) -> List[str]:
        """Launch EC2 instances."""
        instance_ids = []
        for _ in range(count):
            self.instance_counter += 1
            instance_id = f"i-{self.instance_counter:016x}"
            self.instances[instance_id] = {
                "InstanceId": instance_id,
                "ImageId": image_id,
                "InstanceType": instance_type,
                "State": "running",
                "LaunchTime": datetime.now().isoformat()
            }
            instance_ids.append(instance_id)
        return instance_ids
    
    def describe_instances(self, instance_ids: List[str] = None) -> List[Dict]:
        """Describe instances."""
        if instance_ids:
            return [self.instances[iid] for iid in instance_ids if iid in self.instances]
        return list(self.instances.values())
    
    def stop_instances(self, instance_ids: List[str]) -> bool:
        """Stop instances."""
        for iid in instance_ids:
            if iid in self.instances:
                self.instances[iid]["State"] = "stopped"
        return True
    
    def start_instances(self, instance_ids: List[str]) -> bool:
        """Start instances."""
        for iid in instance_ids:
            if iid in self.instances:
                self.instances[iid]["State"] = "running"
        return True
    
    def terminate_instances(self, instance_ids: List[str]) -> bool:
        """Terminate instances."""
        for iid in instance_ids:
            if iid in self.instances:
                del self.instances[iid]
        return True


# Exercise 4: Data Pipeline
class DataPipeline:
    """S3 + EC2 data pipeline."""
    
    def __init__(self, s3_client: MockS3Client, ec2_client: MockEC2Client):
        self.s3 = s3_client
        self.ec2 = ec2_client
    
    def upload_data(self, local_file: str, bucket: str, key: str) -> bool:
        """Upload data to S3."""
        return self.s3.upload_file(local_file, bucket, key)
    
    def process_on_ec2(self, bucket: str, input_key: str, output_key: str) -> Dict:
        """Process data on EC2."""
        # Launch instance
        instance_ids = self.ec2.run_instances("ami-12345", "t2.micro", 1)
        instance_id = instance_ids[0]
        
        # Simulate processing
        input_data = self.s3.list_objects(bucket, input_key)
        
        # Upload results
        self.s3.upload_file("processed_data", bucket, output_key)
        
        # Terminate instance
        self.ec2.terminate_instances([instance_id])
        
        return {
            "instance_id": instance_id,
            "input_key": input_key,
            "output_key": output_key,
            "status": "completed"
        }
    
    def download_results(self, bucket: str, key: str, local_file: str) -> bool:
        """Download results from S3."""
        return self.s3.download_file(bucket, key, local_file)


# Exercise 5: ML Deployment
class MLDeployment:
    """Deploy ML model on EC2 with S3."""
    
    def __init__(self, s3_client: MockS3Client, ec2_client: MockEC2Client):
        self.s3 = s3_client
        self.ec2 = ec2_client
        self.model_instance = None
    
    def upload_model(self, model_file: str, bucket: str, key: str) -> bool:
        """Upload model to S3."""
        return self.s3.upload_file(model_file, bucket, key)
    
    def deploy_model(self, bucket: str, model_key: str, instance_type: str = "t2.micro") -> str:
        """Deploy model on EC2."""
        # Launch instance
        instance_ids = self.ec2.run_instances("ami-ml", instance_type, 1)
        instance_id = instance_ids[0]
        
        # Store deployment info
        self.model_instance = {
            "instance_id": instance_id,
            "bucket": bucket,
            "model_key": model_key,
            "status": "deployed"
        }
        
        return instance_id
    
    def predict(self, instance_id: str, data: Dict) -> Dict:
        """Make prediction."""
        if self.model_instance and self.model_instance["instance_id"] == instance_id:
            return {
                "prediction": [0.8, 0.2],
                "instance_id": instance_id,
                "timestamp": datetime.now().isoformat()
            }
        return {"error": "Instance not found"}
    
    def shutdown(self, instance_id: str) -> bool:
        """Shutdown model instance."""
        return self.ec2.terminate_instances([instance_id])


# Bonus: Cost Calculator
class AWSCostCalculator:
    """Calculate AWS costs."""
    
    def __init__(self):
        self.s3_storage_cost = 0.023
        self.ec2_costs = {
            "t2.micro": 0.0116,
            "t2.small": 0.023,
            "c5.large": 0.085
        }
    
    def calculate_s3_cost(self, storage_gb: float, months: int = 1) -> float:
        """Calculate S3 storage cost."""
        return storage_gb * self.s3_storage_cost * months
    
    def calculate_ec2_cost(self, instance_type: str, hours: float) -> float:
        """Calculate EC2 compute cost."""
        hourly_rate = self.ec2_costs.get(instance_type, 0.0116)
        return hourly_rate * hours
    
    def estimate_pipeline_cost(self, storage_gb: float, instance_type: str, hours: float) -> Dict:
        """Estimate total pipeline cost."""
        s3_cost = self.calculate_s3_cost(storage_gb)
        ec2_cost = self.calculate_ec2_cost(instance_type, hours)
        
        return {
            "s3_cost": round(s3_cost, 2),
            "ec2_cost": round(ec2_cost, 2),
            "total_cost": round(s3_cost + ec2_cost, 2)
        }


def demo_aws_services():
    """Demonstrate AWS S3 & EC2."""
    print("Day 89: AWS S3 & EC2 - Solutions Demo\n" + "=" * 60)
    
    print("\n1. S3 Operations")
    s3 = MockS3Client()
    s3.upload_file("data.csv", "my-bucket", "data/data.csv")
    s3.upload_file("model.pkl", "my-bucket", "models/model.pkl")
    objects = s3.list_objects("my-bucket", "data/")
    print(f"   Uploaded files, found {len(objects)} objects with prefix 'data/'")
    
    print("\n2. Bucket Management")
    bucket_mgr = S3BucketManager(s3)
    bucket_mgr.create_bucket("ml-bucket")
    bucket_mgr.create_bucket("data-bucket")
    buckets = bucket_mgr.list_buckets()
    print(f"   Created buckets: {buckets}")
    
    print("\n3. EC2 Management")
    ec2 = MockEC2Client()
    instance_ids = ec2.run_instances("ami-12345", "t2.micro", 2)
    print(f"   Launched instances: {instance_ids}")
    instances = ec2.describe_instances()
    print(f"   Running instances: {len(instances)}")
    ec2.stop_instances([instance_ids[0]])
    print(f"   Stopped instance: {instance_ids[0]}")
    
    print("\n4. Data Pipeline")
    pipeline = DataPipeline(s3, ec2)
    pipeline.upload_data("input.csv", "pipeline-bucket", "input/data.csv")
    result = pipeline.process_on_ec2("pipeline-bucket", "input/", "output/")
    print(f"   Pipeline status: {result['status']}")
    
    print("\n5. ML Deployment")
    deployment = MLDeployment(s3, ec2)
    deployment.upload_model("model.pkl", "ml-bucket", "models/v1/model.pkl")
    instance_id = deployment.deploy_model("ml-bucket", "models/v1/model.pkl")
    print(f"   Model deployed on: {instance_id}")
    prediction = deployment.predict(instance_id, {"features": [1, 2, 3]})
    print(f"   Prediction: {prediction['prediction']}")
    
    print("\n6. Cost Calculator")
    calculator = AWSCostCalculator()
    costs = calculator.estimate_pipeline_cost(storage_gb=10, instance_type="t2.micro", hours=24)
    print(f"   Estimated costs: ${costs['total_cost']}")
    print(f"   S3: ${costs['s3_cost']}, EC2: ${costs['ec2_cost']}")
    
    print("\n" + "=" * 60)
    print("All AWS concepts demonstrated!")


if __name__ == "__main__":
    demo_aws_services()
