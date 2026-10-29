# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Unit tests for the Embedding Generator Lambda.
Uses mocked AWS services to test text extraction, embedding generation,
and vector storage logic.
"""

import json
import os
import pytest
from unittest.mock import patch, MagicMock

# Set environment variables before importing the handler
os.environ["DOCUMENT_BUCKET_NAME"] = "test-bucket"
os.environ["VECTOR_BUCKET_NAME"] = "test-vector-bucket"
os.environ["VECTOR_INDEX_NAME"] = "test-index"
os.environ["EMBEDDING_MODEL_ID"] = "amazon.titan-embed-text-v2:0"
os.environ["EMBEDDING_DIMENSIONS"] = "1024"
os.environ["LOG_LEVEL"] = "DEBUG"

from .conftest import load_lambda  # noqa: E402  (env vars must be set first)

embedding_index = load_lambda("embedding_generator")
handler = embedding_index.handler
process_document = embedding_index.process_document
extract_text = embedding_index.extract_text
_extract_json_text = embedding_index._extract_json_text
generate_embedding = embedding_index.generate_embedding


@pytest.fixture
def mock_embedding():
    """Return a fake 1024-dimension embedding."""
    return [0.01] * 1024


@pytest.fixture
def sqs_event():
    """Sample SQS event wrapping an S3 notification."""
    return {
        "Records": [
            {
                "body": json.dumps({
                    "Records": [
                        {
                            "s3": {
                                "bucket": {"name": "test-bucket"},
                                "object": {
                                    "key": "documents/legal/contract.txt"
                                },
                            }
                        }
                    ]
                })
            }
        ]
    }


@patch("embedding_generator_index.s3_client")
@patch("embedding_generator_index.s3vectors_client")
@patch("embedding_generator_index.bedrock_client")
def test_process_document_text_file(
    mock_bedrock, mock_s3vectors, mock_s3, mock_embedding
):
    """Test processing a plain text document."""
    # Mock S3 get_object
    mock_body = MagicMock()
    mock_body.read.return_value = b"This is a sample legal contract."
    mock_s3.get_object.return_value = {"Body": mock_body}

    # Mock Bedrock invoke_model
    bedrock_body = MagicMock()
    bedrock_body.read.return_value = json.dumps(
        {"embedding": mock_embedding}
    ).encode()
    mock_bedrock.invoke_model.return_value = {"body": bedrock_body}

    # Mock s3vectors put_vectors
    mock_s3vectors.put_vectors.return_value = {}

    result = process_document("test-bucket", "documents/legal/contract.txt")

    assert result["status"] == "success"
    assert result["dimensions"] == 1024
    mock_bedrock.invoke_model.assert_called_once()
    mock_s3vectors.put_vectors.assert_called_once()

    # Verify put_vectors was called with correct structure
    call_args = mock_s3vectors.put_vectors.call_args[1]
    assert call_args["vectorBucketName"] == "test-vector-bucket"
    assert call_args["indexName"] == "test-index"
    assert len(call_args["vectors"]) == 1
    vector = call_args["vectors"][0]
    # Vectors are keyed per chunk: {document_key}#chunk-{index}
    assert vector["key"] == "documents/legal/contract.txt#chunk-0"
    assert "float32" in vector["data"]
    assert vector["metadata"]["document_key"] == "documents/legal/contract.txt"
    assert vector["metadata"]["file_type"] == "txt"


@patch("embedding_generator_index.s3_client")
def test_process_document_insufficient_text(mock_s3):
    """Test that short documents are skipped."""
    mock_body = MagicMock()
    mock_body.read.return_value = b"Hi"
    mock_s3.get_object.return_value = {"Body": mock_body}

    result = process_document("test-bucket", "documents/short.txt")

    assert result["status"] == "skipped"
    assert result["reason"] == "insufficient_text"


def test_extract_text_unsupported_type():
    """Test that unsupported file types return None."""
    result = extract_text("test-bucket", "documents/file.xlsx")
    assert result is None


@patch("embedding_generator_index.s3_client")
def test_extract_json_text(mock_s3):
    """Test JSON text extraction."""
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps(
        {"title": "Test", "content": "Sample content"}
    ).encode()
    mock_s3.get_object.return_value = {"Body": mock_body}

    result = _extract_json_text("test-bucket", "documents/data.json")

    assert "Test" in result
    assert "Sample content" in result


@patch("embedding_generator_index.bedrock_client")
def test_generate_embedding(mock_bedrock, mock_embedding):
    """Test embedding generation via Bedrock."""
    bedrock_body = MagicMock()
    bedrock_body.read.return_value = json.dumps(
        {"embedding": mock_embedding}
    ).encode()
    mock_bedrock.invoke_model.return_value = {"body": bedrock_body}

    result = generate_embedding("Sample text for embedding")

    assert len(result) == 1024
    mock_bedrock.invoke_model.assert_called_once()

    # Verify the request payload
    call_args = mock_bedrock.invoke_model.call_args
    body = json.loads(call_args[1]["body"])
    assert body["inputText"] == "Sample text for embedding"
    assert body["dimensions"] == 1024
    assert body["normalize"] is True


@patch("embedding_generator_index.s3_client")
@patch("embedding_generator_index.s3vectors_client")
@patch("embedding_generator_index.bedrock_client")
def test_handler_processes_sqs_event(
    mock_bedrock, mock_s3vectors, mock_s3, sqs_event, mock_embedding
):
    """Test the full handler with an SQS event."""
    mock_body = MagicMock()
    mock_body.read.return_value = b"A substantial legal document text."
    mock_s3.get_object.return_value = {"Body": mock_body}
    mock_s3vectors.put_vectors.return_value = {}

    bedrock_body = MagicMock()
    bedrock_body.read.return_value = json.dumps(
        {"embedding": mock_embedding}
    ).encode()
    mock_bedrock.invoke_model.return_value = {"body": bedrock_body}

    result = handler(sqs_event, None)

    assert result["processed"] == 1
    assert result["results"][0]["status"] == "success"


def test_overlap_tail_starts_on_word_boundary():
    """The chunk overlap never starts mid-word."""
    tail = embedding_index._overlap_tail

    assert tail("", 200) == ""
    assert tail("some text", 0) == ""
    assert tail("short text", 200) == "short text"
    assert tail("aaaa bbbb cccc", 7) == "cccc"        # "bb cccc" -> partial word dropped
    assert tail("aaaa bbbb cccc", 9) == "bbbb cccc"   # slice already on a boundary
    assert tail("x" * 50, 10) == ""                    # no whitespace -> no overlap


def test_chunk_text_overlap_is_clean():
    """Chunks start on a word and the overlap is separated from the next paragraph."""
    paragraphs = [
        "Mileage at time of theft: 18,400 miles " + "detail " * 120,
        "PERSONAL PROPERTY IN VEHICLE - Laptop computer: $1,200",
    ]
    text = "\n\n".join(paragraphs)

    chunks = embedding_index.chunk_text(text, chunk_size=500, chunk_overlap=200)

    assert len(chunks) == 2
    second = chunks[1]["text"]
    # Starts with a whole word from the source text
    first_word = second.split()[0]
    assert first_word in text.split()
    # Overlap and next paragraph are not glued together
    assert "\n\nPERSONAL PROPERTY" in second
    assert "milesPERSONAL" not in second and "detailPERSONAL" not in second
