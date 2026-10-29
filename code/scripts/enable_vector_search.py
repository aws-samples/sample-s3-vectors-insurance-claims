# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Post-deployment script: Verify S3 Vectors setup.
The vector bucket and index are created by CDK. This script verifies
they exist and are ready to use.

Usage: python enable_vector_search.py <vector-bucket-name> [index-name]
"""

import boto3
import json
import sys


def verify_vector_setup(vector_bucket_name: str, index_name: str, region: str = None):
    """Verify the S3 Vectors bucket and index are configured."""
    kwargs = {}
    if region:
        kwargs["region_name"] = region
    s3vectors = boto3.client("s3vectors", **kwargs)

    # Verify vector bucket exists
    try:
        bucket_resp = s3vectors.get_vector_bucket(
            vectorBucketName=vector_bucket_name,
        )
        print(f"Vector bucket found: {vector_bucket_name}")
        print(f"  ARN: {bucket_resp.get('vectorBucketArn', 'N/A')}")
    except Exception as e:
        print(f"ERROR: Vector bucket '{vector_bucket_name}' not found: {e}")
        sys.exit(1)

    # Verify vector index exists
    try:
        index_resp = s3vectors.get_index(
            vectorBucketName=vector_bucket_name,
            indexName=index_name,
        )
        print(f"Vector index found: {index_name}")
        print(f"  ARN: {index_resp.get('indexArn', 'N/A')}")
        print(f"  Dimension: {index_resp.get('dimension', 'N/A')}")
        print(f"  Distance metric: {index_resp.get('distanceMetric', 'N/A')}")
        print(f"  Data type: {index_resp.get('dataType', 'N/A')}")
    except Exception as e:
        print(f"ERROR: Vector index '{index_name}' not found: {e}")
        sys.exit(1)

    print("\nS3 Vectors setup verified. Ready for document indexing.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python enable_vector_search.py "
            "<vector-bucket-name> [index-name] [region]"
        )
        sys.exit(1)

    bucket = sys.argv[1]
    index = sys.argv[2] if len(sys.argv) > 2 else "document-embeddings"
    region = sys.argv[3] if len(sys.argv) > 3 else None
    verify_vector_setup(bucket, index, region)
