# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Delete Handler Lambda
Removes vectors from the S3 Vectors index when documents are deleted
from S3. Handles chunked documents by deleting all chunk vectors
associated with the deleted document.

Triggered by S3 OBJECT_REMOVED events via SQS.

Versioning-safe: in a versioned bucket, an ObjectRemoved event doesn't always
mean the object is gone. Permanently deleting a noncurrent version, or
removing a delete marker to "undelete" an object, both emit
s3:ObjectRemoved:Delete while the object is still (or again) current. Before
deleting vectors, the handler checks whether the key still exists:

- Object gone (delete or delete marker created): delete its vectors.
- Object exists and is indexed (a noncurrent version was removed): keep them.
- Object exists but isn't indexed (delete marker removed): queue it for
  re-indexing on the processing queue.
"""

import json
import os
import logging
from urllib.parse import unquote_plus

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

s3_client = boto3.client("s3")
s3vectors_client = boto3.client("s3vectors")
sqs_client = boto3.client("sqs")

VECTOR_BUCKET_NAME = os.environ["VECTOR_BUCKET_NAME"]
VECTOR_INDEX_NAME = os.environ["VECTOR_INDEX_NAME"]
IMAGE_INDEX_NAME = os.environ.get("IMAGE_INDEX_NAME", "")
PROCESSING_QUEUE_URL = os.environ.get("PROCESSING_QUEUE_URL", "")

IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "webp"}


def handler(event, context):
    """Process SQS messages containing S3 delete event notifications."""
    deleted = 0
    skipped = 0

    for record in event["Records"]:
        try:
            body = json.loads(record["body"])

            if body.get("Event") == "s3:TestEvent":
                logger.info("Skipping S3 test event")
                continue

            s3_records = body.get("Records", [])
            for s3_record in s3_records:
                if "s3" not in s3_record:
                    skipped += 1
                    continue

                bucket = s3_record["s3"]["bucket"]["name"]
                key = s3_record["s3"]["object"]["key"]
                extension = key.rsplit(".", 1)[-1].lower() if "." in key else ""
                is_image = extension in IMAGE_EXTENSIONS

                try:
                    if _object_exists(bucket, key):
                        # Versioned bucket: a noncurrent version or a delete
                        # marker was removed, so the object is still current.
                        if _vectors_exist(key, is_image):
                            logger.info(
                                f"{key} still exists and is indexed "
                                "(noncurrent version removed); keeping vectors"
                            )
                        else:
                            _queue_reindex(bucket, s3_record)
                            logger.info(f"{key} restored; queued for re-indexing")
                        skipped += 1
                        continue

                    logger.info(f"Deleting vectors for: {key}")
                    if is_image:
                        # Delete from image index
                        count = _delete_image_vector(key)
                    else:
                        # Delete all chunk vectors for this document
                        count = _delete_document_vectors(key)
                    deleted += count
                    logger.info(f"Deleted {count} vector(s) for {key}")
                except Exception as e:
                    logger.warning(f"Could not delete vectors for {key}: {e}")
                    skipped += 1

        except Exception as e:
            logger.error(f"Error processing record: {e}", exc_info=True)

    return {"deleted": deleted, "skipped": skipped}


def _object_exists(bucket: str, key: str) -> bool:
    """Return True if the key has a current (non-deleted) version.

    Event keys are URL-encoded, so decode before calling S3. Raises on
    unexpected errors (the caller logs and skips the key) rather than
    deleting vectors on a guess.
    """
    try:
        s3_client.head_object(Bucket=bucket, Key=unquote_plus(key))
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in ("404", "NoSuchKey", "NotFound") or status == 404:
            return False
        # HeadObject on a key whose current version is a delete marker
        # returns 405 Method Not Allowed; treat it as deleted.
        if status == 405:
            return False
        raise


def _vectors_exist(key: str, is_image: bool) -> bool:
    """Return True if the document (first chunk) or image is in its index."""
    if is_image:
        if not IMAGE_INDEX_NAME:
            return False
        index_name, vector_key = IMAGE_INDEX_NAME, key
    else:
        index_name, vector_key = VECTOR_INDEX_NAME, f"{key}#chunk-0"

    response = s3vectors_client.get_vectors(
        vectorBucketName=VECTOR_BUCKET_NAME,
        indexName=index_name,
        keys=[vector_key],
    )
    return bool(response.get("vectors"))


def _queue_reindex(bucket: str, s3_record: dict):
    """Send the object to the processing queue in S3 event format."""
    if not PROCESSING_QUEUE_URL:
        logger.warning("PROCESSING_QUEUE_URL not set; cannot re-index restored object")
        return
    message = {
        "Records": [{
            "eventSource": "aws:s3",
            "eventName": "ObjectCreated:Restore",
            "s3": {
                "bucket": {"name": bucket},
                "object": {"key": s3_record["s3"]["object"]["key"]},
            },
        }]
    }
    sqs_client.send_message(QueueUrl=PROCESSING_QUEUE_URL, MessageBody=json.dumps(message))


def _delete_document_vectors(key: str) -> int:
    """Delete all chunk vectors associated with a document.

    Handles both legacy single-vector entries and chunked documents.
    """
    keys_to_delete = []

    # Always try to delete the base key (legacy non-chunked format)
    keys_to_delete.append(key)

    # Find all chunk vectors by checking known chunk pattern
    # We check up to 200 chunks (covers documents up to ~200 pages)
    chunk_keys = [f"{key}#chunk-{i}" for i in range(200)]

    # Use GetVectors to check which chunks actually exist (batch of 50 max)
    existing_chunks = []
    for i in range(0, len(chunk_keys), 50):
        batch = chunk_keys[i:i + 50]
        try:
            response = s3vectors_client.get_vectors(
                vectorBucketName=VECTOR_BUCKET_NAME,
                indexName=VECTOR_INDEX_NAME,
                keys=batch,
            )
            existing_chunks.extend(
                v["key"] for v in response.get("vectors", [])
            )
        except Exception as e:
            logger.warning(f"Error checking chunks for {key}: {e}")
            break

        # If we got no results in this batch, no more chunks exist
        if not response.get("vectors"):
            break

    keys_to_delete.extend(existing_chunks)

    # Remove duplicates
    keys_to_delete = list(set(keys_to_delete))

    if keys_to_delete:
        # DeleteVectors is idempotent - deleting non-existent keys is a no-op
        # Batch delete in groups of 500 (API limit)
        for i in range(0, len(keys_to_delete), 500):
            batch = keys_to_delete[i:i + 500]
            s3vectors_client.delete_vectors(
                vectorBucketName=VECTOR_BUCKET_NAME,
                indexName=VECTOR_INDEX_NAME,
                keys=batch,
            )

    return len(keys_to_delete)


def _delete_image_vector(key: str) -> int:
    """Delete a vector from the image index."""
    if not IMAGE_INDEX_NAME:
        logger.warning(f"Image index not configured, cannot delete: {key}")
        return 0

    try:
        s3vectors_client.delete_vectors(
            vectorBucketName=VECTOR_BUCKET_NAME,
            indexName=IMAGE_INDEX_NAME,
            keys=[key],
        )
        return 1
    except Exception as e:
        logger.warning(f"Failed to delete image vector for {key}: {e}")
        return 0
