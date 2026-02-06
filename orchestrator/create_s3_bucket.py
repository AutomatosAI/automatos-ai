#!/usr/bin/env python3
"""Create S3 bucket for document storage"""
import boto3
import os
from botocore.exceptions import ClientError

s3_client = boto3.client(
    's3',
    region_name=os.getenv('AWS_REGION', 'us-east-1'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

bucket_name = 'automatos-ai'
region = os.getenv('AWS_REGION', 'us-east-1')

try:
    if region == 'us-east-1':
        s3_client.create_bucket(Bucket=bucket_name)
    else:
        s3_client.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={'LocationConstraint': region}
        )
    print(f'✅ Created S3 bucket: {bucket_name}')
except ClientError as e:
    error_code = e.response['Error']['Code']
    if error_code == 'BucketAlreadyOwnedByYou':
        print(f'✅ Bucket {bucket_name} already exists and is owned by you')
    elif error_code == 'BucketAlreadyExists':
        print(f'⚠️  Bucket {bucket_name} already exists (owned by someone else)')
    else:
        print(f'❌ Error creating bucket: {e}')
        raise
except Exception as e:
    print(f'❌ Error: {e}')
    raise
