# Day 89 Quiz: AWS S3 & EC2

## Questions

### Question 1
What is Amazon S3 primarily used for?

A) Running virtual machines in the cloud
B) Object storage for storing and retrieving data
C) Managing database instances
D) Executing serverless functions

### Question 2
Which S3 storage class is best for long-term archival with lowest cost?

A) S3 Standard
B) S3 Intelligent-Tiering
C) S3 Glacier
D) S3 One Zone-IA

### Question 3
What is an EC2 instance?

A) A storage bucket for files
B) A virtual server in the cloud
C) A database service
D) A content delivery network

### Question 4
Which EC2 instance type is optimized for machine learning workloads?

A) t2.micro
B) c5.large
C) r5.xlarge
D) p3.2xlarge

### Question 5
What is the correct boto3 code to upload a file to S3?

A) s3.put_file('local.txt', 'bucket', 'key')
B) s3.upload_file('local.txt', 'bucket', 'key')
C) s3.send_file('local.txt', 'bucket', 'key')
D) s3.store_file('local.txt', 'bucket', 'key')

### Question 6
What is an AMI in EC2?

A) Amazon Machine Image - an OS template
B) Amazon Memory Instance - a RAM configuration
C) Amazon Managed Infrastructure - a service tier
D) Amazon Monitoring Interface - a logging tool

### Question 7
Which AWS service should you use instead of hardcoding access keys?

A) S3 buckets
B) EC2 security groups
C) IAM roles
D) CloudWatch

### Question 8
What happens when you terminate an EC2 instance?

A) It stops temporarily and can be restarted
B) It is permanently deleted
C) It enters hibernation mode
D) It switches to a smaller instance type

### Question 9
In a typical ML pipeline, what is the correct order of operations?

A) Train on EC2 → Store data in S3 → Deploy model
B) Store data in S3 → Train on EC2 → Store model in S3
C) Deploy model → Store data in S3 → Train on EC2
D) Store model in S3 → Store data in S3 → Train on EC2

### Question 10
Which is a cost optimization strategy for EC2?

A) Always use the largest instance type
B) Keep all instances running 24/7
C) Use spot instances for batch jobs
D) Never stop instances once started

## Answer Key

1. B) Object storage for storing and retrieving data
2. C) S3 Glacier
3. B) A virtual server in the cloud
4. D) p3.2xlarge
5. B) s3.upload_file('local.txt', 'bucket', 'key')
6. A) Amazon Machine Image - an OS template
7. C) IAM roles
8. B) It is permanently deleted
9. B) Store data in S3 → Train on EC2 → Store model in S3
10. C) Use spot instances for batch jobs

## Scoring Guide

- 10/10: AWS Expert! Ready for cloud deployment
- 8-9/10: Strong understanding, review missed concepts
- 6-7/10: Good foundation, practice more with AWS
- Below 6: Review the material and exercises again

