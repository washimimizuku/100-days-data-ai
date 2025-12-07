# Day 90 Quiz: AWS Lambda

## Questions

### Question 1
What is AWS Lambda?

A) A database service for storing data
B) A serverless compute service that runs code in response to events
C) A container orchestration platform
D) A content delivery network

### Question 2
What are the two parameters every Lambda function receives?

A) request and response
B) input and output
C) event and context
D) data and metadata

### Question 3
What is a "cold start" in Lambda?

A) When a function fails to execute
B) When a function is invoked for the first time and must initialize
C) When a function runs in a cold region
D) When a function is paused by AWS

### Question 4
What is the maximum execution time for a Lambda function?

A) 5 minutes
B) 10 minutes
C) 15 minutes
D) 30 minutes

### Question 5
Which AWS service is commonly used with Lambda to create REST APIs?

A) S3
B) DynamoDB
C) API Gateway
D) CloudFront

### Question 6
What is the purpose of Lambda Layers?

A) To add security to functions
B) To package dependencies separately from function code
C) To create multiple versions of a function
D) To monitor function performance

### Question 7
How does Lambda pricing work?

A) Fixed monthly fee per function
B) Based on number of requests and duration (GB-seconds)
C) Based on code size only
D) Free for all users

### Question 8
Where should you initialize resources like database connections in Lambda?

A) Inside the handler function
B) Outside the handler function (global scope)
C) In a separate initialization function
D) In the event object

### Question 9
What is the correct way to invoke another Lambda function asynchronously?

A) InvocationType='Sync'
B) InvocationType='Event'
C) InvocationType='Async'
D) InvocationType='Background'

### Question 10
What is the maximum deployment package size for Lambda (unzipped)?

A) 50 MB
B) 100 MB
C) 250 MB
D) 500 MB

## Answer Key

1. B) A serverless compute service that runs code in response to events
2. C) event and context
3. B) When a function is invoked for the first time and must initialize
4. C) 15 minutes
5. C) API Gateway
6. B) To package dependencies separately from function code
7. B) Based on number of requests and duration (GB-seconds)
8. B) Outside the handler function (global scope)
9. B) InvocationType='Event'
10. C) 250 MB

## Scoring Guide

- 10/10: Lambda Expert! Ready for serverless deployment
- 8-9/10: Strong understanding, review missed concepts
- 6-7/10: Good foundation, practice more with Lambda
- Below 6: Review the material and exercises again
