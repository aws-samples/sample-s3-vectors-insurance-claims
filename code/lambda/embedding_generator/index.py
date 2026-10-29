# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Embedding Generator Lambda
Processes S3 event notifications via SQS, extracts text from documents,
chunks it into semantically meaningful segments, generates vector embeddings
using Amazon Bedrock Titan V2, and stores them in an S3 Vectors index.

Supports text, PDF, JSON, and image files (multimodal embeddings).
"""

import json
import os
import logging
import time
import base64
import boto3
from typing import Optional

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
IMAGE_INDEX_NAME = os.environ.get("IMAGE_INDEX_NAME", "")
EMBEDDING_MODEL_ID = os.environ["EMBEDDING_MODEL_ID"]
MULTIMODAL_MODEL_ID = os.environ.get("MULTIMODAL_MODEL_ID", "amazon.titan-embed-image-v1")
EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS", "1024"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))
MAX_TEXT_LENGTH = 25000  # Titan V2 supports up to ~8,192 tokens

IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "webp"}
TEXT_EXTENSIONS = {"txt", "md"}
SUPPORTED_EXTENSIONS = {"pdf", "txt", "md", "json"} | IMAGE_EXTENSIONS


def handler(event, context):
    """Process SQS messages containing S3 event notifications."""
    results = []

    for record in event["Records"]:
        try:
            body = json.loads(record["body"])

            # Skip S3 test events (sent when notification is first configured)
            if body.get("Event") == "s3:TestEvent":
                logger.info("Skipping S3 test event")
                continue

            # S3 event notifications wrap records in a "Records" array
            s3_records = body.get("Records", [])
            if not s3_records:
                logger.warning(f"No S3 records found in message: {json.dumps(body)[:200]}")
                continue

            for s3_record in s3_records:
                if "s3" not in s3_record:
                    logger.warning(f"Skipping non-S3 record: {s3_record.get('eventSource', 'unknown')}")
                    continue

                bucket = s3_record["s3"]["bucket"]["name"]
                key = s3_record["s3"]["object"]["key"]
                logger.info(f"Processing: s3://{bucket}/{key}")
                result = process_document(bucket, key)
                results.append(result)
        except Exception as e:
            logger.error(f"Error processing record: {e}", exc_info=True)
            raise  # Let SQS retry / route to DLQ

    return {"processed": len(results), "results": results}


def process_document(bucket: str, key: str) -> dict:
    """Extract text, chunk, generate embeddings, and store in S3 Vectors index."""
    start_time = time.time()
    extension = key.rsplit(".", 1)[-1].lower() if "." in key else ""

    # Route images to multimodal embedding pipeline
    if extension in IMAGE_EXTENSIONS:
        return _process_image(bucket, key, start_time)

    # Text-based document pipeline
    text = extract_text(bucket, key)
    if not text or len(text.strip()) < 10:
        logger.warning(f"Insufficient text extracted from {key}")
        return {"key": key, "status": "skipped", "reason": "insufficient_text"}

    # Chunk the text
    chunks = chunk_text(text[:MAX_TEXT_LENGTH])
    total_chunks = len(chunks)
    logger.info(f"Chunked {key} into {total_chunks} segments")

    # Generate embeddings for all chunks
    embeddings = []
    for chunk in chunks:
        embedding = generate_embedding(chunk["text"])
        embeddings.append(embedding)

    # Store all chunk vectors
    store_chunked_vectors(key, chunks, embeddings, total_chunks)

    # Clean up orphaned chunks if document was re-uploaded with fewer chunks
    _cleanup_orphan_chunks(key, total_chunks)

    # Emit custom CloudWatch metric
    processing_time_ms = (time.time() - start_time) * 1000
    _emit_metric("IndexingLatencyMs", processing_time_ms, extension)

    logger.info(f"Indexed {key} - {total_chunks} chunks, {EMBEDDING_DIMENSIONS} dimensions, {processing_time_ms:.0f}ms")
    return {
        "key": key,
        "status": "success",
        "chunks": total_chunks,
        "dimensions": EMBEDDING_DIMENSIONS,
        "latency_ms": round(processing_time_ms, 1),
    }


def _process_image(bucket: str, key: str, start_time: float) -> dict:
    """Process an image file through multimodal embedding pipeline."""
    if not IMAGE_INDEX_NAME:
        logger.warning(f"Image index not configured, skipping image: {key}")
        return {"key": key, "status": "skipped", "reason": "no_image_index"}

    # Read image bytes
    response = s3_client.get_object(Bucket=bucket, Key=key)
    image_bytes = response["Body"].read()

    # Generate multimodal embedding
    embedding = generate_image_embedding(image_bytes)

    # Store in image-specific index
    s3vectors_client.put_vectors(
        vectorBucketName=VECTOR_BUCKET_NAME,
        indexName=IMAGE_INDEX_NAME,
        vectors=[{
            "key": key,
            "data": {"float32": embedding},
            "metadata": {
                "file_type": key.rsplit(".", 1)[-1].lower(),
                "document_key": key,
                "description": f"Image: {key.split('/')[-1]}",
            },
        }],
    )

    processing_time_ms = (time.time() - start_time) * 1000
    _emit_metric("IndexingLatencyMs", processing_time_ms, "image")

    logger.info(f"Indexed image {key} - {processing_time_ms:.0f}ms")
    return {"key": key, "status": "success", "type": "image", "latency_ms": round(processing_time_ms, 1)}


def _overlap_tail(text: str, overlap: int) -> str:
    """Return up to `overlap` trailing characters of `text`, starting on a word boundary.

    A plain text[-overlap:] slice can start mid-word (e.g. "l: 2024 Toyota"),
    so drop any partial leading word.
    """
    if overlap <= 0 or not text:
        return ""
    if len(text) <= overlap:
        return text.strip()
    tail = text[-overlap:]
    if not text[-overlap - 1].isspace():
        # Slice started inside a word; skip to the first whitespace
        cut = next((i for i, c in enumerate(tail) if c.isspace()), len(tail))
        tail = tail[cut:]
    return tail.strip()


def chunk_text(text: str, chunk_size: int = None, chunk_overlap: int = None) -> list:
    """Split text into overlapping chunks for embedding.

    Uses paragraph boundaries when possible, falling back to
    sentence boundaries, then character splits with overlap.
    """
    chunk_size = chunk_size or CHUNK_SIZE
    chunk_overlap = chunk_overlap or CHUNK_OVERLAP
    chunks = []

    # Split on paragraph boundaries (double newlines)
    paragraphs = text.split("\n\n")

    current_chunk = ""
    chunk_index = 0

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "chunk_index": chunk_index,
            })
            # Keep overlap from the end of the current chunk, starting on a
            # word boundary and separated from the next paragraph
            overlap_text = _overlap_tail(current_chunk, chunk_overlap)
            current_chunk = f"{overlap_text}\n\n{paragraph}" if overlap_text else paragraph
            chunk_index += 1
        else:
            current_chunk += ("\n\n" if current_chunk else "") + paragraph

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append({
            "text": current_chunk.strip(),
            "chunk_index": chunk_index,
        })

    return chunks


def extract_text(bucket: str, key: str) -> Optional[str]:
    """Extract text content based on file type."""
    extension = key.rsplit(".", 1)[-1].lower()

    if extension in TEXT_EXTENSIONS:
        return _read_text_file(bucket, key)
    elif extension == "pdf":
        return _extract_pdf_text(bucket, key)
    elif extension == "json":
        return _extract_json_text(bucket, key)
    else:
        logger.warning(f"Unsupported file type for text extraction: {extension}")
        return None


def _read_text_file(bucket: str, key: str) -> str:
    """Read plain text or markdown files directly."""
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return response["Body"].read().decode("utf-8")


def _extract_pdf_text(bucket: str, key: str) -> str:
    """Extract text from PDF using Amazon Textract.

    Uses synchronous API for files <= 10MB, async API for larger files.
    """
    # Check file size
    head = s3_client.head_object(Bucket=bucket, Key=key)
    size = head["ContentLength"]

    if size <= 10 * 1024 * 1024:  # 10MB limit for sync API
        response = textract_client.detect_document_text(
            Document={"S3Object": {"Bucket": bucket, "Name": key}}
        )
        lines = [
            block["Text"]
            for block in response.get("Blocks", [])
            if block["BlockType"] == "LINE"
        ]
        return "\n".join(lines)

    # Async API for large PDFs
    logger.info(f"Using async Textract for large PDF ({size} bytes): {key}")

    start_response = textract_client.start_document_text_detection(
        DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}}
    )
    job_id = start_response["JobId"]

    # Poll for completion (Lambda has up to 10 min timeout)
    while True:
        result = textract_client.get_document_text_detection(JobId=job_id)
        status = result["JobStatus"]

        if status == "SUCCEEDED":
            break
        elif status == "FAILED":
            logger.error(f"Textract job failed for {key}: {result.get('StatusMessage')}")
            return None
        time.sleep(2)

    # Collect all pages of results
    lines = []
    lines.extend(
        block["Text"]
        for block in result.get("Blocks", [])
        if block["BlockType"] == "LINE"
    )

    # Handle pagination
    next_token = result.get("NextToken")
    while next_token:
        result = textract_client.get_document_text_detection(
            JobId=job_id, NextToken=next_token
        )
        lines.extend(
            block["Text"]
            for block in result.get("Blocks", [])
            if block["BlockType"] == "LINE"
        )
        next_token = result.get("NextToken")

    return "\n".join(lines)


def _extract_json_text(bucket: str, key: str) -> str:
    """Flatten JSON to a text representation."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        data = json.loads(response["Body"].read().decode("utf-8", errors="replace"))
        return json.dumps(data, indent=2, default=str)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning(f"Failed to parse JSON from {key}: {e}")
        return None


def generate_embedding(text: str) -> list:
    """Generate vector embedding using Bedrock Titan Embeddings V2."""
    response = bedrock_client.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "inputText": text,
            "dimensions": EMBEDDING_DIMENSIONS,
            "normalize": True,
        }),
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


def generate_image_embedding(image_bytes: bytes) -> list:
    """Generate embedding for an image using Titan Multimodal Embeddings."""
    body = json.dumps({
        "inputImage": base64.b64encode(image_bytes).decode("utf-8"),
    })

    response = bedrock_client.invoke_model(
        modelId=MULTIMODAL_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=body,
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


def store_chunked_vectors(key: str, chunks: list, embeddings: list, total_chunks: int):
    """Store chunked vector embeddings in the S3 Vectors index."""
    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        summary = chunk["text"][:500].replace("\n", " ").strip()
        vector_key = f"{key}#chunk-{chunk['chunk_index']}"

        vectors.append({
            "key": vector_key,
            "data": {"float32": embedding},
            "metadata": {
                "source_text": summary,
                "file_type": key.rsplit(".", 1)[-1].lower(),
                "document_key": key,
                "chunk_index": chunk["chunk_index"],
                "total_chunks": total_chunks,
            },
        })

    # PutVectors supports batch of up to 500 vectors
    batch_size = 500
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        s3vectors_client.put_vectors(
            vectorBucketName=VECTOR_BUCKET_NAME,
            indexName=VECTOR_INDEX_NAME,
            vectors=batch,
        )

    logger.info(f"Stored {len(vectors)} chunk vectors for {key}")


def _cleanup_orphan_chunks(key: str, new_total_chunks: int):
    """Remove orphaned chunk vectors when a document is re-uploaded with fewer chunks."""
    try:
        # Check if there are chunks beyond the new total
        orphan_index = new_total_chunks
        orphan_keys = []
        # Check up to 100 potential orphans (reasonable upper bound)
        for i in range(orphan_index, orphan_index + 100):
            orphan_keys.append(f"{key}#chunk-{i}")

        if orphan_keys:
            # GetVectors to check which exist
            response = s3vectors_client.get_vectors(
                vectorBucketName=VECTOR_BUCKET_NAME,
                indexName=VECTOR_INDEX_NAME,
                keys=orphan_keys[:50],  # Check first 50
            )
            existing_orphans = [v["key"] for v in response.get("vectors", [])]

            if existing_orphans:
                s3vectors_client.delete_vectors(
                    vectorBucketName=VECTOR_BUCKET_NAME,
                    indexName=VECTOR_INDEX_NAME,
                    keys=existing_orphans,
                )
                logger.info(f"Cleaned up {len(existing_orphans)} orphaned chunks for {key}")
    except Exception as e:
        logger.warning(f"Orphan cleanup failed for {key}: {e}")


def _emit_metric(metric_name: str, value: float, file_type: str):
    """Emit custom CloudWatch metric for observability."""
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
