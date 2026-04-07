# AI_POC

s3://ai-poc-sagemaker-endpoint/medgemma/model.tar.gz
Role: SageMakerExecutionRole
 - AWS managed policy: AmazonSageMakerFullAccess:
    - Covers:
        - s3:GetObject / s3:ListBucket — pull model.tar.gz from your bucket
        - ecr:GetDownloadUrlForLayer — pull the HuggingFace DLC container
        image
        - sagemaker:* — create/manage endpoints
        - logs:* — write to CloudWatch for debugging