"""Day 89: AWS S3 & EC2 - Exercises

NOTE: These exercises use mock implementations for learning.
For real AWS operations, install boto3 and configure credentials.
"""

from typing import List, Dict, Any
from datetime import datetime


# Exercise 1: S3 Operations
class MockS3Client:
    """Mock S3 client for exercises."""
    
    def __init__(self):
        self.buckets = {}
    
    def upload_file(self, filename: str, bucket: str, key: str) -> bool:
        """
        Upload file to S3.
        
        Args:
            filename: Local file path
            bucket: S3 bucket name
            key: S3 object key
        
        Returns:
            Success status
        """
        # TODO: Implement mock upload
        # TODO: Store file info in self.buckets
        pass
    
    def download_file(self, bucket: str, key: str, filename: str) -> bool:
        """Download file from S3."""
        # TODO: Implement mock download
        pass
    
    def list_objects(self, bucket: str, prefix: str = "") -> List[Dict]:
        """List objects in bucket."""
        # TODO: Return list of objects matching prefix
        pass
    
    def delete_object(self, bucket: str, key: str) -> bool:
        """Delete object from S3."""
        # TODO: Implement mock delete
        pass


# Exercise 2: Bucket Management
class S3BucketManager:
    """Manage S3 buckets."""
    
    def __init__(self, s3_client: MockS3Client):
        self.s3 = s3_client
    
    def create_bucket(self, bucket_name: str, region: str = "us-east-1") -> bool:
        """
        Create S3 bucket.
        
        Args:
            bucket_name: Name of bucket
            region: AWS region
        
        Returns:
            Success status
        """
        # TODO: Create bucket in s3 client
        pass
    
    def list_buckets(self) -> List[str]:
        """List all buckets."""
        # TODO: Return list of bucket names
        pass
    
    def delete_bucket(self, bucket_name: str) -> bool:
        """Delete bucket."""
        # TODO: Delete bucket if empty
        pass
    
    def configure_lifecycle(self, bucket_name: str, rules: List[Dict]) -> bool:
        """Configure lifecycle rules."""
        # TODO: Set lifecycle rules for bucket
        pass


# Exercise 3: EC2 Management
class MockEC2Client:
    """Mock EC2 client for exercises."""
    
    def __init__(self):
        self.instances = {}
        self.instance_counter = 0
    
    def run_instances(self, image_id: str, instance_type: str, 
                     count: int = 1) -> List[str]:
        """
        Launch EC2 instances.
        
        Args:
            image_id: AMI ID
            instance_type: Instance type (t2.micro, etc.)
            count: Number of instances
        
        Returns:
            List of instance IDs
        """
        # TODO: Create mock instances
        # TODO: Return instance IDs
        pass
    
    def describe_instances(self, instance_ids: List[str] = None) -> List[Dict]:
        """Describe instances."""
        # TODO: Return instance information
        pass
    
    def stop_instances(self, instance_ids: List[str]) -> bool:
        """Stop instances."""
        # TODO: Change instance state to stopped
        pass
    
    def start_instances(self, instance_ids: List[str]) -> bool:
        """Start instances."""
        # TODO: Change instance state to running
        pass
    
    def terminate_instances(self, instance_ids: List[str]) -> bool:
        """Terminate instances."""
        # TODO: Remove instances
        pass


# Exercise 4: Data Pipeline
class DataPipeline:
    """S3 + EC2 data pipeline."""
    
    def __init__(self, s3_client: MockS3Client, ec2_client: MockEC2Client):
        self.s3 = s3_client
        self.ec2 = ec2_client
    
    def upload_data(self, local_file: str, bucket: str, key: str) -> bool:
        """Upload data to S3."""
        # TODO: Upload file to S3
        pass
    
    def process_on_ec2(self, bucket: str, input_key: str, 
                      output_key: str) -> Dict:
        """
        Process data on EC2.
        
        Steps:
        1. Launch EC2 instance
        2. Download data from S3
        3. Process data
        4. Upload results to S3
        5. Terminate instance
        """
        # TODO: Implement processing pipeline
        pass
    
    def download_results(self, bucket: str, key: str, local_file: str) -> bool:
        """Download results from S3."""
        # TODO: Download processed data
        pass


# Exercise 5: ML Deployment
class MLDeployment:
    """Deploy ML model on EC2 with S3."""
    
    def __init__(self, s3_client: MockS3Client, ec2_client: MockEC2Client):
        self.s3 = s3_client
        self.ec2 = ec2_client
        self.model_instance = None
    
    def upload_model(self, model_file: str, bucket: str, key: str) -> bool:
        """Upload model to S3."""
        # TODO: Upload model file
        pass
    
    def deploy_model(self, bucket: str, model_key: str, 
                    instance_type: str = "t2.micro") -> str:
        """
        Deploy model on EC2.
        
        Steps:
        1. Launch EC2 instance
        2. Download model from S3
        3. Start inference server
        
        Returns:
            Instance ID
        """
        # TODO: Launch instance
        # TODO: Configure for model serving
        pass
    
    def predict(self, instance_id: str, data: Dict) -> Dict:
        """Make prediction using deployed model."""
        # TODO: Send data to instance
        # TODO: Return prediction
        pass
    
    def shutdown(self, instance_id: str) -> bool:
        """Shutdown model instance."""
        # TODO: Terminate instance
        pass


# Bonus: Cost Calculator
class AWSCostCalculator:
    """Calculate AWS costs."""
    
    def __init__(self):
        self.s3_storage_cost = 0.023  # per GB/month
        self.ec2_costs = {
            "t2.micro": 0.0116,  # per hour
            "t2.small": 0.023,
            "c5.large": 0.085
        }
    
    def calculate_s3_cost(self, storage_gb: float, months: int = 1) -> float:
        """Calculate S3 storage cost."""
        # TODO: Calculate cost
        pass
    
    def calculate_ec2_cost(self, instance_type: str, hours: float) -> float:
        """Calculate EC2 compute cost."""
        # TODO: Calculate cost
        pass
    
    def estimate_pipeline_cost(self, storage_gb: float, 
                              instance_type: str, hours: float) -> Dict:
        """Estimate total pipeline cost."""
        # TODO: Calculate S3 + EC2 costs
        pass


if __name__ == "__main__":
    print("Day 89: AWS S3 & EC2 - Exercises")
    print("=" * 50)
    
    # Test Exercise 1
    print("\nExercise 1: S3 Operations")
    s3 = MockS3Client()
    print(f"S3 client created: {s3 is not None}")
    
    # Test Exercise 2
    print("\nExercise 2: Bucket Management")
    bucket_mgr = S3BucketManager(s3)
    print(f"Bucket manager created: {bucket_mgr is not None}")
    
    # Test Exercise 3
    print("\nExercise 3: EC2 Management")
    ec2 = MockEC2Client()
    print(f"EC2 client created: {ec2 is not None}")
    
    # Test Exercise 4
    print("\nExercise 4: Data Pipeline")
    pipeline = DataPipeline(s3, ec2)
    print(f"Pipeline created: {pipeline is not None}")
    
    # Test Exercise 5
    print("\nExercise 5: ML Deployment")
    deployment = MLDeployment(s3, ec2)
    print(f"Deployment created: {deployment is not None}")
    
    print("\n" + "=" * 50)
    print("Complete the TODOs to finish the exercises!")
    print("\nNote: These are mock implementations for learning.")
    print("For real AWS operations, use boto3 with proper credentials.")
