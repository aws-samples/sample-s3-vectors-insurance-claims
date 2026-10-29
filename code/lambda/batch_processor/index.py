# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Batch Processor Lambda
Processes existing S3 documents in batches for vector indexing.
Designed to be orchestrated by Step Functions for large backlogs.

Supports chunking and multimodal image embeddings.
"""

import json
import os
import logging
import time
import base64
import boto3
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

s3_client = boto3.client("s3")
s3vectors_client = boto3.client("s3vectors")
bedrock_client = boto3.client("bedrock-runtime")
textract_client = boto3.client("textract")
cloudwatch_client = boto3.client("cloudwatch")

DOCUMENT_BUCKET_NAME = os.environ["DOCUMENT_BUCKET_NAME"]
VECTOR_BUCKET_NAME = os.environ["VECTOR_BUCKET_NAME"]
VECTOR_INDEX_NAME = os.environ["VECTOR_INDEX_NAME"]
EMBEDDING_MODEL_ID = os.environ.get(
    "EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"
)
EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS", "1024"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "100"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "10"))
MAX_TEXT_LENGTH = 25000
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".json"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def handler(event, context):
    """
    Process a batch of existing documents.

    Input: {"prefix": "documents/legal/", "continuation_token": null}
    """
    prefix = event.get("prefix", "documents/")
    continuation_token = event.get("continuation_token")
    max_items = event.get("max_items", BATCH_SIZE)

    list_params = {
        "Bucket": DOCUMENT_BUCKET_NAME,
        "Prefix": prefix,
        "MaxKeys": max_items,
    }
    if continuation_token:
        list_params["ContinuationToken"] = continuation_token

    response = s3_client.list_objects_v2(**list_params)
    objects = response.get("Contents", [])

    # Filter to supported, unindexed documents
    to_process = []
    for obj in objects:
        key = obj["Key"]
        ext = "." + key.rsplit(".", 1)[-1].lower() if "." in key else ""
        if (ext in SUPPORTED_EXTENSIONS or ext in IMAGE_EXTENSIONS) and not _is_indexed(key):
            to_process.append(key)

    logger.info(
        f"Found {len(to_process)} unindexed out of {len(objects)} objects"
    )

    results = {"success": 0, "failed": 0, "skipped": 0, "errors": []}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_process_single, key): key
            for key in to_process
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                status = future.result()
                results[status] = results.get(status, 0) + 1
            except Exception as e:
                results["failed"] += 1
                results["errors"].append({"key": key, "error": str(e)})
                logger.error(f"Failed {key}: {e}")

    output = {
        "processed": results["success"],
        "failed": results["failed"],
        "skipped": results["skipped"],
        "total_objects": len(objects),
        "has_more": response.get("IsTruncated", False),
    }
    if response.get("IsTruncated"):
        output["continuation_token"] = response["NextContinuationToken"]

    logger.info(f"Batch complete: {json.dumps(output)}")
    return output


def _is_indexed(key: str) -> bool:
    """Check if a document already has a vector in the index (checks first chunk)."""
    try:
        # Check for chunked format first
        chunk_key = f"{key}#chunk-0"
        response = s3vectors_client.get_vectors(
            vectorBucketName=VECTOR_BUCKET_NAME,
            indexName=VECTOR_INDEX_NAME,
            keys=[chunk_key, key],  # Check both chunked and legacy format
        )
        return len(response.get("vectors", [])) > 0
    except Exception:
        return False


def _process_single(key: str) -> str:
    """Process a single document end-to-end with chunking."""
    start_time = time.time()
    extension = key.rsplit(".", 1)[-1].lower() if "." in key else ""

    # Handle images separately
    if f".{extension}" in IMAGE_EXTENSIONS:
        return _process_image(key)

    text = _extract_text(DOCUMENT_BUCKET_NAME, key)
    if not text or len(text.strip()) < 10:
        return "skipped"

    # Chunk the text
    chunks = _chunk_text(text[:MAX_TEXT_LENGTH])
    total_chunks = len(chunks)

    # Generate embeddings for all chunks
    embeddings = []
    for chunk in chunks:
        embedding = _generate_embedding(chunk["text"])
        embeddings.append(embedding)

    # Store all chunk vectors
    _store_chunked_vectors(key, chunks, embeddings, total_chunks)

    processing_time_ms = (time.time() - start_time) * 1000
    _emit_metric("BatchIndexingLatencyMs", processing_time_ms, extension)

    return "success"


def _process_image(key: str) -> str:
    """Process an image file through multimodal embedding."""
    image_index = os.environ.get("IMAGE_INDEX_NAME", "")
    if not image_index:
        return "skipped"

    response = s3_client.get_object(Bucket=DOCUMENT_BUCKET_NAME, Key=key)
    image_bytes = response["Body"].read()

    body = json.dumps({
        "inputImage": base64.b64encode(image_bytes).decode("utf-8"),
    })
    resp = bedrock_client.invoke_model(
        modelId="amazon.titan-embed-image-v1",
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    embedding = json.loads(resp["body"].read())["embedding"]

    s3vectors_client.put_vectors(
        vectorBucketName=VECTOR_BUCKET_NAME,
        indexName=image_index,
        vectors=[{
            "key": key,
            "data": {"float32": embedding},
            "metadata": {
                "file_type": key.rsplit(".", 1)[-1].lower(),
                "document_key": key,
            },
        }],
    )
    return "success"


def _overlap_tail(text: str, overlap: int) -> str:
    """Return up to `overlap` trailing characters of `text`, starting on a word boundary.

    Kept in sync with embedding_generator._overlap_tail (each Lambda is
    packaged separately, so the helper is duplicated).
    """
    if overlap <= 0 or not text:
        return ""
    if len(text) <= overlap:
        return text.strip()
    tail = text[-overlap:]
    if not text[-overlap - 1].isspace():
        cut = next((i for i, c in enumerate(tail) if c.isspace()), len(tail))
        tail = tail[cut:]
    return tail.strip()


def _chunk_text(text: str) -> list:
    """Split text into overlapping chunks using paragraph boundaries."""
    chunks = []
    paragraphs = text.split("\n\n")
    current_chunk = ""
    chunk_index = 0

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) > CHUNK_SIZE and current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "chunk_index": chunk_index,
            })
            # Keep overlap on a word boundary, separated from the next paragraph
            overlap_text = _overlap_tail(current_chunk, CHUNK_OVERLAP)
            current_chunk = f"{overlap_text}\n\n{paragraph}" if overlap_text else paragraph
            chunk_index += 1
        else:
            current_chunk += ("\n\n" if current_chunk else "") + paragraph

    if current_chunk.strip():
        chunks.append({
            "text": current_chunk.strip(),
            "chunk_index": chunk_index,
        })

    return chunks


def _extract_text(bucket: str, key: str) -> Optional[str]:
    """Extract text based on file type."""
    ext = key.rsplit(".", 1)[-1].lower()
    if ext in ("txt", "md"):
        resp = s3_client.get_object(Bucket=bucket, Key=key)
        return resp["Body"].read().decode("utf-8")
    elif ext == "pdf":
        return _extract_pdf_text(bucket, key)
    elif ext == "json":
        resp = s3_client.get_object(Bucket=bucket, Key=key)
        data = json.loads(resp["Body"].read().decode("utf-8"))
        return json.dumps(data, indent=2, default=str)
    return None


def _extract_pdf_text(bucket: str, key: str) -> Optional[str]:
    """Extract text from PDF. Uses async Textract for files > 10MB."""
    head = s3_client.head_object(Bucket=bucket, Key=key)
    size = head["ContentLength"]

    if size <= 10 * 1024 * 1024:
        resp = textract_client.detect_document_text(
            Document={"S3Object": {"Bucket": bucket, "Name": key}}
        )
        return "\n".join(
            b["Text"] for b in resp.get("Blocks", []) if b["BlockType"] == "LINE"
        )

    logger.info(f"Using async Textract for large PDF ({size} bytes): {key}")
    start_resp = textract_client.start_document_text_detection(
        DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}}
    )
    job_id = start_resp["JobId"]

    while True:
        result = textract_client.get_document_text_detection(JobId=job_id)
        status = result["JobStatus"]
        if status == "SUCCEEDED":
            break
        elif status == "FAILED":
            logger.error(f"Textract failed for {key}: {result.get('StatusMessage')}")
            return None
        time.sleep(2)

    lines = [b["Text"] for b in result.get("Blocks", []) if b["BlockType"] == "LINE"]
    next_token = result.get("NextToken")
    while next_token:
        result = textract_client.get_document_text_detection(
            JobId=job_id, NextToken=next_token
        )
        lines.extend(b["Text"] for b in result.get("Blocks", []) if b["BlockType"] == "LINE")
        next_token = result.get("NextToken")

    return "\n".join(lines)


def _generate_embedding(text: str) -> list:
    """Generate embedding via Bedrock Titan V2."""
    resp = bedrock_client.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "inputText": text,
            "dimensions": EMBEDDING_DIMENSIONS,
            "normalize": True,
        }),
    )
    return json.loads(resp["body"].read())["embedding"]


def _store_chunked_vectors(key: str, chunks: list, embeddings: list, total_chunks: int):
    """Store chunked embeddings in the S3 Vectors index."""
    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        summary = chunk["text"][:500].replace("\n", " ").strip()
        vectors.append({
            "key": f"{key}#chunk-{chunk['chunk_index']}",
            "data": {"float32": embedding},
            "metadata": {
                "source_text": summary,
                "file_type": key.rsplit(".", 1)[-1].lower(),
                "document_key": key,
                "chunk_index": chunk["chunk_index"],
                "total_chunks": total_chunks,
            },
        })

    # Batch in groups of 500
    for i in range(0, len(vectors), 500):
        batch = vectors[i:i + 500]
        s3vectors_client.put_vectors(
            vectorBucketName=VECTOR_BUCKET_NAME,
            indexName=VECTOR_INDEX_NAME,
            vectors=batch,
        )


def _emit_metric(metric_name: str, value: float, file_type: str):
    """Emit custom CloudWatch metric."""
    try:
        cloudwatch_client.put_metric_data(
            Namespace="S3VectorsSearch",
            MetricData=[{
                "MetricName": metric_name,
                "Value": value,
                "Unit": "Milliseconds",
                "Dimensions": [
                    {"Name": "FileType", "Value": file_type},
                ],
            }],
        )
    except Exception as e:
        logger.warning(f"Failed to emit CloudWatch metric: {e}")
