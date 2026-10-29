# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Unit tests for the Search Handler Lambda.
"""

import json
import os
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

os.environ["DOCUMENT_BUCKET_NAME"] = "test-bucket"
os.environ["VECTOR_BUCKET_NAME"] = "test-vector-bucket"
os.environ["VECTOR_INDEX_NAME"] = "test-index"
os.environ["EMBEDDING_MODEL_ID"] = "amazon.titan-embed-text-v2:0"
os.environ["EMBEDDING_DIMENSIONS"] = "1024"
os.environ["DEFAULT_TOP_K"] = "10"
os.environ["LOG_LEVEL"] = "DEBUG"

from .conftest import load_lambda  # noqa: E402  (env vars must be set first)

search_index = load_lambda("search_handler")
handler = search_index.handler
_vector_search = search_index._vector_search
_enrich_results = search_index._enrich_results


@pytest.fixture
def mock_embedding():
    return [0.02] * 1024


@pytest.fixture
def api_event():
    """Sample API Gateway proxy event."""
    return {
        "body": json.dumps({
            "query": "IP assignment clauses in M&A transactions",
            "top_k": 5,
        })
    }


@patch("search_handler_index.s3_client")
@patch("search_handler_index.s3vectors_client")
@patch("search_handler_index.bedrock_client")
def test_search_returns_results(
    mock_bedrock, mock_s3vectors, mock_s3, api_event, mock_embedding
):
    """Test a successful search request."""
    # Mock embedding generation
    bedrock_body = MagicMock()
    bedrock_body.read.return_value = json.dumps(
        {"embedding": mock_embedding}
    ).encode()
    mock_bedrock.invoke_model.return_value = {"body": bedrock_body}

    # Mock S3 Vectors query
    mock_s3vectors.query_vectors.return_value = {
        "vectors": [
            {
                "key": "documents/legal/contract-a.pdf",
                "distance": 0.05,
                "metadata": {"source_text": "A legal contract..."},
            },
            {
                "key": "documents/legal/contract-b.pdf",
                "distance": 0.13,
                "metadata": {"source_text": "Another contract..."},
            },
        ]
    }

    # Mock head_object for enrichment
    mock_s3.head_object.return_value = {
        "Metadata": {},
        "ContentType": "application/pdf",
        "LastModified": datetime(2024, 11, 15, tzinfo=timezone.utc),
        "ContentLength": 245780,
    }

    result = handler(api_event, None)

    assert result["statusCode"] == 200
    body = json.loads(result["body"])
    assert body["result_count"] == 2
    assert body["results"][0]["document_key"].endswith("contract-a.pdf")
    assert body["results"][0]["distance"] == 0.05


def test_search_missing_query():
    """Test that missing query returns 400."""
    event = {"body": json.dumps({})}
    result = handler(event, None)

    assert result["statusCode"] == 400
    body = json.loads(result["body"])
    assert "error" in body


@patch("search_handler_index.s3vectors_client")
def test_search_with_filters(mock_s3vectors, mock_embedding):
    """Test search with metadata filters."""
    mock_s3vectors.query_vectors.return_value = {"vectors": []}

    _vector_search(
        query_embedding=mock_embedding,
        top_k=5,
        filters={"file_type": "pdf"},
    )

    call_args = mock_s3vectors.query_vectors.call_args[1]
    assert call_args["topK"] == 5
    assert call_args["filter"] == {"file_type": "pdf"}
    assert call_args["vectorBucketName"] == "test-vector-bucket"
    assert call_args["indexName"] == "test-index"


@patch("search_handler_index.s3vectors_client")
@patch("search_handler_index.bedrock_client")
def test_search_empty_results(mock_bedrock, mock_s3vectors, mock_embedding):
    """Test search that returns no matches."""
    bedrock_body = MagicMock()
    bedrock_body.read.return_value = json.dumps(
        {"embedding": mock_embedding}
    ).encode()
    mock_bedrock.invoke_model.return_value = {"body": bedrock_body}

    mock_s3vectors.query_vectors.return_value = {"vectors": []}

    event = {
        "body": json.dumps({
            "query": "something very obscure",
            "top_k": 5,
        })
    }
    result = handler(event, None)

    assert result["statusCode"] == 200
    body = json.loads(result["body"])
    assert body["result_count"] == 0
    assert body["results"] == []


@patch("search_handler_index.s3_client")
def test_enrich_results_handles_errors(mock_s3):
    """Test that enrichment gracefully handles missing objects."""
    mock_s3.head_object.side_effect = Exception("NoSuchKey")

    results = [{"key": "missing/doc.pdf", "distance": 0.1, "metadata": {}}]
    enriched = _enrich_results(results)

    assert len(enriched) == 1
    assert enriched[0]["document_key"] == "missing/doc.pdf"
    assert enriched[0]["distance"] == 0.1
    # summary comes from vector metadata (source_text), so it's still present
    assert enriched[0]["summary"] == ""
    # S3 enrichment fields are omitted because head_object failed
    for field in ("content_type", "last_modified", "size_bytes"):
        assert field not in enriched[0]


@patch("search_handler_index.s3vectors_client")
@patch("search_handler_index.bedrock_client")
def test_invalid_filter_returns_400(mock_bedrock, mock_s3vectors, mock_embedding):
    """A malformed metadata filter is a client error, not a server error."""
    from botocore.exceptions import ClientError

    bedrock_body = MagicMock()
    bedrock_body.read.return_value = json.dumps({"embedding": mock_embedding}).encode()
    mock_bedrock.invoke_model.return_value = {"body": bedrock_body}
    mock_s3vectors.query_vectors.side_effect = ClientError(
        {"Error": {"Code": "ValidationException", "Message": "Invalid filter"}},
        "QueryVectors",
    )

    event = {"body": json.dumps({
        "query": "repair cost",
        # Wrong shape: operator outside, field inside
        "filters": {"$startsWith": {"document_key": "documents/claims/"}},
    })}
    result = handler(event, None)

    assert result["statusCode"] == 400
    body = json.loads(result["body"])
    assert "Invalid filter" in body["error"]
    assert "$startsWith" in body["hint"]


@patch("search_handler_index.s3vectors_client")
@patch("search_handler_index.bedrock_client")
def test_other_aws_errors_return_500(mock_bedrock, mock_s3vectors, mock_embedding):
    """Non-validation AWS errors stay 500 and don't leak details."""
    from botocore.exceptions import ClientError

    bedrock_body = MagicMock()
    bedrock_body.read.return_value = json.dumps({"embedding": mock_embedding}).encode()
    mock_bedrock.invoke_model.return_value = {"body": bedrock_body}
    mock_s3vectors.query_vectors.side_effect = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "secret detail"}},
        "QueryVectors",
    )

    result = handler({"body": json.dumps({"query": "x"})}, None)

    assert result["statusCode"] == 500
    assert "secret detail" not in result["body"]


def test_malformed_json_body_returns_400():
    """A request body that isn't JSON returns 400."""
    result = handler({"body": "{not json"}, None)

    assert result["statusCode"] == 400
    assert "valid JSON" in json.loads(result["body"])["error"]


@patch("search_handler_index.s3vectors_client")
def test_startswith_filter_passed_through(mock_s3vectors, mock_embedding):
    """The documented $startsWith filter shape reaches QueryVectors unchanged."""
    mock_s3vectors.query_vectors.return_value = {"vectors": []}
    prefix_filter = {"document_key": {"$startsWith": "documents/claims/claim-003"}}

    _vector_search(query_embedding=mock_embedding, top_k=5, filters=prefix_filter)

    assert mock_s3vectors.query_vectors.call_args[1]["filter"] == prefix_filter
