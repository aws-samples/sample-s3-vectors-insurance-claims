# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Check the index mode of the S3 Vectors indexes used by this stack.

Metadata pre-filtering (higher recall on filtered searches) requires the
index to be in ENHANCED mode. Each index takes its default mode from the
vector bucket: buckets created on or after 2026-09-30 default to ENHANCED,
so a fresh deployment of this stack already produces pre-filtering indexes.
Existing CLASSIC indexes can be switched in place with UpdateIndexMode.
This helper verifies that, printing the mode for each index.

  CLASSIC  - metadata filters are evaluated in tandem with the vector search
  ENHANCED - metadata filters are applied before the vector search (pre-filtering)

Requires boto3 >= 1.43.108 (first release with the s3vectors index-mode APIs).

Usage:
  python check_index_mode.py <vector-bucket-name> [index-name ...]

If no index names are given, defaults to the two indexes this stack creates:
  document-embeddings, image-embeddings
"""

import sys
import boto3

DEFAULT_INDEXES = ["document-embeddings", "image-embeddings"]
MIN_BOTO3 = "1.43.108"


def check_index_mode(vector_bucket_name: str, index_names: list[str]) -> bool:
    """Print the index mode for each index. Return True if all are ENHANCED."""
    s3vectors_client = boto3.client("s3vectors")

    all_enhanced = True
    for index_name in index_names:
        try:
            response = s3vectors_client.get_index(
                vectorBucketName=vector_bucket_name,
                indexName=index_name,
            )
            mode = response["index"].get("indexMode")
            if mode is None:
                # Older boto3 releases drop the indexMode field from the response,
                # so its absence doesn't mean CLASSIC.
                print(
                    f"  {index_name}: UNKNOWN (indexMode not in response; "
                    f"boto3 {boto3.__version__} is too old, upgrade to >= {MIN_BOTO3})"
                )
                all_enhanced = False
                continue
            marker = "pre-filtering ON" if mode == "ENHANCED" else "pre-filtering OFF"
            print(f"  {index_name}: {mode} ({marker})")
            if mode != "ENHANCED":
                all_enhanced = False
        except s3vectors_client.exceptions.NotFoundException:
            print(f"  {index_name}: NOT FOUND")
            all_enhanced = False

    return all_enhanced


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_index_mode.py <vector-bucket-name> [index-name ...]")
        sys.exit(1)

    bucket = sys.argv[1]
    indexes = sys.argv[2:] if len(sys.argv) > 2 else DEFAULT_INDEXES

    print(f"Vector bucket: {bucket}")
    ok = check_index_mode(bucket, indexes)

    if ok:
        print("\nAll indexes are in ENHANCED mode. Metadata pre-filtering is active.")
        sys.exit(0)
    else:
        print(
            "\nOne or more indexes are not confirmed as ENHANCED. "
            f"If any show UNKNOWN, upgrade boto3 (pip install \"boto3>={MIN_BOTO3}\") and rerun. "
            "If any show CLASSIC, see the pre-filtering note in the repository README "
            "for how to switch them with UpdateIndexMode."
        )
        sys.exit(1)
