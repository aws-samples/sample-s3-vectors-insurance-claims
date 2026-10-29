# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Search Handler Lambda
Exposes a semantic search API via API Gateway. Converts natural language
queries into embeddings and performs vector similarity search using
the S3 Vectors API.

Features:
- Confidence threshold filtering (distance-based)
- Deduplication by source document (returns best chunk per document)
- Metadata filtering for hybrid search
- Multi-index support (text and image indexes)
- Image-as-query support (base64 encoded image input)
- Custom CloudWatch metrics for observability
"""

import json
import os
import logging
import time
import boto3
from botocore.exceptions import ClientError
from typing import Any

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

s3_client = boto3.client("s3")
s3vectors_client = boto3.client("s3vectors")
bedrock_client = boto3.client("bedrock-runtime")
cloudwatch_client = boto3.client("cloudwatch")

DOCUMENT_BUCKET_NAME = os.environ["DOCUMENT_BUCKET_NAME"]
VECTOR_BUCKET_NAME = os.environ["VECTOR_BUCKET_NAME"]
VECTOR_INDEX_NAME = os.environ["VECTOR_INDEX_NAME"]
IMAGE_INDEX_NAME = os.environ.get("IMAGE_INDEX_NAME", "")
EMBEDDING_MODEL_ID = os.environ["EMBEDDING_MODEL_ID"]
MULTIMODAL_MODEL_ID = os.environ.get("MULTIMODAL_MODEL_ID", "amazon.titan-embed-image-v1")
EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS", "1024"))
DEFAULT_TOP_K = int(os.environ.get("DEFAULT_TOP_K", "10"))
DISTANCE_THRESHOLD = float(os.environ.get("DISTANCE_THRESHOLD", "0.75"))


def handler(event, context):
    """
    Handle semantic search requests.

    Request body:
    {
        "query": "IP assignment clauses in M&A transactions",
        "top_k": 10,
        "filters": {"file_type": "pdf"},
        "distance_threshold": 0.75,
        "index": "document-embeddings",
        "query_image": "<base64-encoded-image>"
    }

    The "filters" object is passed through to the S3 Vectors QueryVectors API.
    On an ENHANCED-mode index, filters are applied *before* the similarity
    search (metadata pre-filtering), so narrowly scoped queries return more of
    the relevant matches. Example: scope a search to a single claim folder with
    the $startsWith operator:
        "filters": {"document_key": {"$startsWith": "documents/claims/claim-003"}}
    Each index takes its default mode from the vector bucket and can be
    switched to ENHANCED with UpdateIndexMode; see the repository README.

    For text queries against the text index, uses Titan Text Embeddings V2.
    For text queries against the image index, uses Titan Multimodal Embeddings.
    For image queries (base64), uses Titan Multimodal Embeddings.
    """
    start = time.time()
    try:
        body = json.loads(event.get("body", "{}"))
        query = body.get("query")
        query_image = body.get("query_image")  # base64-encoded image
        top_k = body.get("top_k", DEFAULT_TOP_K)
        filters = body.get("filters", {})
        threshold = body.get("distance_threshold", DISTANCE_THRESHOLD)
        index_name = body.get("index", VECTOR_INDEX_NAME)

        # Input validation
        if not query and not query_image:
            return _response(400, {"error": "Missing 'query' or 'query_image' parameter"})

        if query and len(query) > 10000:
            return _response(400, {"error": "Query exceeds maximum length of 10,000 characters"})

        if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
            return _response(400, {"error": "top_k must be an integer between 1 and 100"})

        if not isinstance(threshold, (int, float)) or threshold < 0.0 or threshold > 2.0:
            return _response(400, {"error": "distance_threshold must be a number between 0.0 and 2.0"})

        if index_name not in (VECTOR_INDEX_NAME, IMAGE_INDEX_NAME):
            return _response(400, {"error": f"index must be '{VECTOR_INDEX_NAME}' or '{IMAGE_INDEX_NAME}'"})

        logger.info(f"Search: query='{query[:100] if query else '(image)'}', top_k={top_k}, threshold={threshold}, index={index_name}")

        # Generate embedding based on query type and target index
        if query_image:
            # Image-as-query: always use multimodal model
            query_embedding = _generate_image_query_embedding(query_image)
        elif index_name == IMAGE_INDEX_NAME:
            # Text query against image index: use multimodal model
            query_embedding = _generate_multimodal_text_embedding(query)
        else:
            # Text query against text index: use text embedding model
            query_embedding = _generate_query_embedding(query)

        raw_results = _vector_search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
            index_name=index_name,
        )

        # Apply confidence threshold
        filtered_results = [
            r for r in raw_results
            if r["distance"] <= threshold
        ]

        # Deduplicate by source document (keep best chunk per document)
        deduplicated = _deduplicate_by_document(filtered_results)

        # Enrich with S3 metadata
        enriched = _enrich_results(deduplicated)

        latency_ms = (time.time() - start) * 1000

        # Emit search latency metric
        _emit_metric("SearchLatencyMs", latency_ms)

        logger.info(f"Returned {len(enriched)} results in {latency_ms:.0f}ms (raw={len(raw_results)}, filtered={len(filtered_results)})")

        response_body = {
            "query": query or "(image query)",
            "result_count": len(enriched),
            "latency_ms": round(latency_ms, 1),
            "distance_threshold": threshold,
            "index": index_name,
            "raw_matches": len(raw_results),
            "results": enriched,
        }

        if not enriched and raw_results:
            response_body["message"] = (
                "No sufficiently relevant documents found above the confidence threshold. "
                "Try rephrasing your query or increasing the distance_threshold parameter."
            )

        return _response(200, response_body)

    except json.JSONDecodeError:
        return _response(400, {"error": "Request body must be valid JSON"})

    except ClientError as e:
        # Client-side problems (e.g. a malformed metadata filter) come back from
        # S3 Vectors / Bedrock as ValidationException; surface them as 400.
        error = e.response.get("Error", {})
        if error.get("Code") == "ValidationException":
            logger.warning(f"Search validation error: {error.get('Message')}")
            return _response(400, {
                "error": f"Invalid request: {error.get('Message', 'validation failed')}",
                "hint": "Filters use the form {\"field\": {\"$operator\": value}}, "
                        "e.g. {\"document_key\": {\"$startsWith\": \"documents/claims/\"}}",
            })
        logger.error(f"Search error: {e}", exc_info=True)
        return _response(500, {"error": "Internal search error"})

    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        return _response(500, {"error": "Internal search error"})


def _generate_query_embedding(query: str) -> list:
    """Generate embedding for the search query using Titan Text Embeddings V2."""
    response = bedrock_client.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "inputText": query,
            "dimensions": EMBEDDING_DIMENSIONS,
            "normalize": True,
        }),
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


def _generate_multimodal_text_embedding(query: str) -> list:
    """Generate embedding for text using Titan Multimodal model.

    Used when querying the image index with a text query (cross-modal search).
    """
    response = bedrock_client.invoke_model(
        modelId=MULTIMODAL_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "inputText": query,
        }),
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


def _generate_image_query_embedding(image_base64: str) -> list:
    """Generate embedding for a base64-encoded image using Titan Multimodal.

    Used for image-as-query search (find similar images).
    """
    response = bedrock_client.invoke_model(
        modelId=MULTIMODAL_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "inputImage": image_base64,
        }),
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


def _vector_search(
    query_embedding: list,
    top_k: int,
    filters: dict,
    index_name: str = None,
) -> list:
    """Perform vector similarity search using the S3 Vectors API."""
    query_params = {
        "vectorBucketName": VECTOR_BUCKET_NAME,
        "indexName": index_name or VECTOR_INDEX_NAME,
        "queryVector": {"float32": query_embedding},
        "topK": top_k,
        "returnDistance": True,
        "returnMetadata": True,
    }

    if filters:
        query_params["filter"] = filters

    response = s3vectors_client.query_vectors(**query_params)

    return [
        {
            "key": match["key"],
            "distance": match.get("distance", 0),
            "metadata": match.get("metadata", {}),
        }
        for match in response.get("vectors", [])
    ]


def _deduplicate_by_document(results: list) -> list:
    """Group results by source document and keep the best-matching chunk per document.

    This prevents multiple chunks from the same document from dominating results.
    """
    seen_documents = {}

    for result in results:
        metadata = result.get("metadata", {})
        # Use document_key from metadata if available (chunked docs), otherwise use the vector key
        doc_key = metadata.get("document_key", result["key"].split("#chunk-")[0])

        if doc_key not in seen_documents or result["distance"] < seen_documents[doc_key]["distance"]:
            seen_documents[doc_key] = result

    # Return sorted by distance (most relevant first)
    return sorted(seen_documents.values(), key=lambda x: x["distance"])


def _enrich_results(results: list) -> list:
    """Add document metadata from S3 to search results."""
    enriched = []
    for result in results:
        metadata = result.get("metadata", {})
        # Resolve the actual document key (strip chunk suffix for S3 lookup)
        doc_key = metadata.get("document_key", result["key"].split("#chunk-")[0])

        entry = {
            "document_key": doc_key,
            "distance": round(result["distance"], 4),
            "summary": metadata.get("source_text", ""),
            "chunk_index": metadata.get("chunk_index"),
            "total_chunks": metadata.get("total_chunks"),
        }

        try:
            head = s3_client.head_object(
                Bucket=DOCUMENT_BUCKET_NAME, Key=doc_key
            )
            entry.update({
                "content_type": head.get("ContentType", ""),
                "last_modified": head["LastModified"].isoformat(),
                "size_bytes": head["ContentLength"],
            })
        except Exception as e:
            logger.warning(f"Could not enrich {doc_key}: {e}")

        # Include any additional filterable metadata
        vector_metadata = {
            k: v for k, v in metadata.items()
            if k not in ("source_text", "document_key", "chunk_index", "total_chunks")
        }
        if vector_metadata:
            entry["vector_metadata"] = vector_metadata

        enriched.append(entry)

    return enriched


def _emit_metric(metric_name: str, value: float):
    """Emit custom CloudWatch metric."""
    try:
        cloudwatch_client.put_metric_data(
            Namespace="S3VectorsSearch",
            MetricData=[{
                "MetricName": metric_name,
                "Value": value,
                "Unit": "Milliseconds",
            }],
        )
    except Exception as e:
        logger.warning(f"Failed to emit CloudWatch metric: {e}")


def _response(status_code: int, body: Any) -> dict:
    """Format API Gateway proxy response."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body, default=str),
    }
