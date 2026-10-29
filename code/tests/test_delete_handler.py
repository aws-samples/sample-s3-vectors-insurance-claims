# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Unit tests for the Delete Handler Lambda, including versioned-bucket behavior.
"""

import json
import os
from unittest.mock import patch

import pytest
from botocore.exceptions import ClientError

os.environ["VECTOR_BUCKET_NAME"] = "test-vector-bucket"
os.environ["VECTOR_INDEX_NAME"] = "test-index"
os.environ["IMAGE_INDEX_NAME"] = "test-image-index"
os.environ["PROCESSING_QUEUE_URL"] = "https://sqs.us-east-1.amazonaws.com/123456789012/processing"
os.environ["LOG_LEVEL"] = "DEBUG"

from .conftest import load_lambda  # noqa: E402  (env vars must be set first)

delete_index = load_lambda("delete_handler")
handler = delete_index.handler

MOD = "delete_handler_index"


def _event(key: str, event_name: str = "ObjectRemoved:Delete") -> dict:
    """SQS event wrapping a single S3 ObjectRemoved notification."""
    return {"Records": [{"body": json.dumps({"Records": [{
        "eventSource": "aws:s3",
        "eventName": event_name,
        "s3": {"bucket": {"name": "doc-bucket"}, "object": {"key": key}},
    }]})}]}


def _client_error(status: int, code: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": code},
         "ResponseMetadata": {"HTTPStatusCode": status}},
        "HeadObject",
    )


@pytest.fixture
def clients():
    with patch(f"{MOD}.s3_client") as s3, \
         patch(f"{MOD}.s3vectors_client") as s3v, \
         patch(f"{MOD}.sqs_client") as sqs:
        yield s3, s3v, sqs


@pytest.mark.parametrize("status,code", [(404, "404"), (404, "NoSuchKey"), (405, "MethodNotAllowed")])
def test_object_gone_deletes_vectors(clients, status, code):
    """Plain delete (404) or delete marker now current (405): vectors are removed."""
    s3, s3v, sqs = clients
    s3.head_object.side_effect = _client_error(status, code)
    s3v.get_vectors.side_effect = [
        {"vectors": [{"key": "documents/a.txt#chunk-0"}, {"key": "documents/a.txt#chunk-1"}]},
        {"vectors": []},
    ]

    result = handler(_event("documents/a.txt", "ObjectRemoved:DeleteMarkerCreated"), None)

    deleted_keys = s3v.delete_vectors.call_args[1]["keys"]
    assert set(deleted_keys) == {"documents/a.txt", "documents/a.txt#chunk-0", "documents/a.txt#chunk-1"}
    assert result["deleted"] == 3
    sqs.send_message.assert_not_called()


def test_noncurrent_version_removed_keeps_vectors(clients):
    """Object still current and indexed: deleting an old version leaves vectors alone."""
    s3, s3v, sqs = clients
    s3.head_object.return_value = {"ContentLength": 10}
    s3v.get_vectors.return_value = {"vectors": [{"key": "documents/a.txt#chunk-0"}]}

    result = handler(_event("documents/a.txt"), None)

    s3v.delete_vectors.assert_not_called()
    sqs.send_message.assert_not_called()
    assert result == {"deleted": 0, "skipped": 1}


def test_delete_marker_removed_requeues_for_indexing(clients):
    """Undelete (delete marker removed): object exists, no vectors, so re-index."""
    s3, s3v, sqs = clients
    s3.head_object.return_value = {"ContentLength": 10}
    s3v.get_vectors.return_value = {"vectors": []}

    handler(_event("documents/a.txt"), None)

    s3v.delete_vectors.assert_not_called()
    sqs.send_message.assert_called_once()
    kwargs = sqs.send_message.call_args[1]
    assert kwargs["QueueUrl"] == os.environ["PROCESSING_QUEUE_URL"]
    record = json.loads(kwargs["MessageBody"])["Records"][0]
    # Same shape the embedding generator reads from S3 notifications
    assert record["s3"]["bucket"]["name"] == "doc-bucket"
    assert record["s3"]["object"]["key"] == "documents/a.txt"


def test_restored_image_checks_image_index(clients):
    """Images are looked up by plain key in the image index."""
    s3, s3v, sqs = clients
    s3.head_object.return_value = {"ContentLength": 10}
    s3v.get_vectors.return_value = {"vectors": []}

    handler(_event("documents/images/dent.png"), None)

    call = s3v.get_vectors.call_args[1]
    assert call["indexName"] == "test-image-index"
    assert call["keys"] == ["documents/images/dent.png"]
    sqs.send_message.assert_called_once()


def test_url_encoded_key_is_decoded_for_head_object(clients):
    """Event keys are URL-encoded; HeadObject must use the decoded key."""
    s3, s3v, sqs = clients
    s3.head_object.side_effect = _client_error(404, "404")
    s3v.get_vectors.return_value = {"vectors": []}

    handler(_event("documents/claims/my+claim%281%29.txt"), None)

    assert s3.head_object.call_args[1]["Key"] == "documents/claims/my claim(1).txt"


def test_unexpected_head_error_does_not_delete(clients):
    """If existence can't be determined, don't delete vectors on a guess."""
    s3, s3v, sqs = clients
    s3.head_object.side_effect = _client_error(500, "InternalError")

    result = handler(_event("documents/a.txt"), None)

    s3v.delete_vectors.assert_not_called()
    sqs.send_message.assert_not_called()
    assert result == {"deleted": 0, "skipped": 1}


def test_s3_test_event_is_ignored(clients):
    s3, s3v, sqs = clients
    event = {"Records": [{"body": json.dumps({"Event": "s3:TestEvent"})}]}

    assert handler(event, None) == {"deleted": 0, "skipped": 0}
    s3.head_object.assert_not_called()
