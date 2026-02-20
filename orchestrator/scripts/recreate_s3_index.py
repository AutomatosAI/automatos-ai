"""
Recreate S3 Vectors Index at New Dimension
============================================

Deletes the existing S3 Vectors index and lets the app recreate it
at the dimension specified by S3_VECTORS_DIMENSION in config.

Usage:
    python -m scripts.recreate_s3_index

    # Dry run (show what would happen):
    python -m scripts.recreate_s3_index --dry-run
"""

import sys
import os
import argparse

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config


def main():
    parser = argparse.ArgumentParser(description="Recreate S3 Vectors index at new dimension")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without making changes")
    args = parser.parse_args()

    bucket_name = config.S3_VECTORS_BUCKET
    index_name = config.S3_VECTORS_INDEX_NAME
    new_dimension = config.S3_VECTORS_DIMENSION
    region = config.AWS_REGION

    if not bucket_name:
        print("ERROR: S3_VECTORS_BUCKET not configured")
        sys.exit(1)

    print(f"S3 Vectors Index Recreation")
    print(f"=" * 50)
    print(f"Bucket:        {bucket_name}")
    print(f"Index:         {index_name}")
    print(f"New Dimension: {new_dimension}")
    print(f"Metric:        {config.S3_VECTORS_METRIC}")
    print(f"Region:        {region}")
    print()

    if args.dry_run:
        print("[DRY RUN] Would delete index and recreate at new dimension")
        print("[DRY RUN] All existing vectors would be lost")
        return

    try:
        import boto3
        client = boto3.client(
            "s3vectors",
            region_name=region,
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
        )
    except Exception as e:
        print(f"ERROR: Failed to create S3 Vectors client: {e}")
        sys.exit(1)

    # Delete existing index
    print(f"Deleting index '{index_name}' from bucket '{bucket_name}'...")
    try:
        client.delete_index(
            vectorBucketName=bucket_name,
            indexName=index_name,
        )
        print(f"  Deleted successfully")
    except client.exceptions.ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("NotFoundException", "ResourceNotFoundException"):
            print(f"  Index doesn't exist (nothing to delete)")
        else:
            print(f"  ERROR: {e}")
            sys.exit(1)
    except Exception as e:
        error_str = str(e)
        if "NotFound" in error_str or "does not exist" in error_str:
            print(f"  Index doesn't exist (nothing to delete)")
        else:
            print(f"  ERROR: {e}")
            sys.exit(1)

    # Create new index at target dimension
    print(f"Creating index '{index_name}' with dimension={new_dimension}...")
    try:
        client.create_index(
            vectorBucketName=bucket_name,
            indexName=index_name,
            dimension=new_dimension,
            dataType="float32",
            distanceMetric=config.S3_VECTORS_METRIC,
        )
        print(f"  Created successfully")
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    print()
    print("Done! Next steps:")
    print("  1. Update Railway env: S3_VECTORS_DIMENSION=4096")
    print("  2. Update Railway env: EMBEDDING_MODEL=qwen/qwen3-embedding-8b")
    print("  3. Update Railway env: EMBEDDING_PROVIDER=openrouter")
    print("  4. Re-sync all documents from Google Drive")


if __name__ == "__main__":
    main()
