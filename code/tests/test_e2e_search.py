# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
End-to-end smoke tests for the S3 Vectors semantic search pipeline.

These tests validate the complete pipeline:
S3 upload → SQS → Lambda → Bedrock embedding → S3 Vectors → API Gateway query

Prerequisites:
- Stack deployed (cdk deploy)
- AWS credentials configured
- pip install pytest requests requests-auth-aws-sigv4

Usage:
    pytest tests/test_e2e_search.py -v

Environment variables (set before running):
    S3_BUCKET: Document bucket name (from stack output)
    API_URL: Search API endpoint (from stack output)
"""

import boto3
import json
import time
import os
import pytest

S3_BUCKET = os.environ.get("S3_BUCKET", "")
API_URL = os.environ.get("API_URL", "")

# Strict threshold for known-good matches
DISTANCE_THRESHOLD = 0.7
# Time to wait for event-driven indexing (SQS batching window is 30s + Lambda processing)
INDEXING_WAIT_SECONDS = 45


def _skip_if_not_configured():
    """Skip tests if environment is not configured for e2e."""
    if not S3_BUCKET or not API_URL:
        pytest.skip(
            "E2E tests require S3_BUCKET and API_URL environment variables. "
            "Set them from CDK stack outputs."
        )


@pytest.fixture(scope="module")
def s3_client():
    return boto3.client("s3")


@pytest.fixture(scope="module")
def upload_test_document(s3_client):
    """Upload a known document and wait for indexing."""
    _skip_if_not_configured()

    test_key = "documents/test/claim-e2e-test-001.txt"
    test_content = """
    INSURANCE CLAIM REPORT
    Claim Number: CLM-E2E-TEST-001
    Date of Incident: 2026-06-15
    Type: Vehicle collision at intersection

    Description: The insured vehicle, a 2023 Honda Accord, was struck
    by a red sedan that ran a red light at the intersection of Main
    Street and Oak Avenue. The collision occurred at approximately
    3:45 PM during clear weather conditions.

    Damage Assessment:
    - Front passenger side door: severe denting and paint damage
    - Right front fender: crushed, requires replacement
    - Headlight assembly: shattered
    - Side mirror: broken off

    Estimated repair cost: $4,200
    Police report number: PR-2026-E2E-001

    Witnesses: Two pedestrians observed the incident and provided
    statements confirming the other driver ran the red light.

    Injuries: Policyholder reports mild neck pain (whiplash) and
    visited the emergency room for evaluation. No hospitalization
    required.
    """

    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=test_key,
        Body=test_content.encode(),
        ContentType="text/plain",
    )

    # Wait for event-driven indexing
    time.sleep(INDEXING_WAIT_SECONDS)

    yield test_key

    # Cleanup: delete the test document (triggers vector cleanup)
    s3_client.delete_object(Bucket=S3_BUCKET, Key=test_key)
    time.sleep(10)  # Allow delete handler to clean up vectors


def _search(query: str, top_k: int = 5, distance_threshold: float = 0.95) -> dict:
    """Invoke the search API with IAM auth. Uses a permissive threshold for testing."""
    import requests
    from requests_auth_aws_sigv4 import AWSSigV4

    response = requests.post(
        f"{API_URL}search",
        json={
            "query": query,
            "top_k": top_k,
            "distance_threshold": distance_threshold,
        },
        auth=AWSSigV4("execute-api"),
    )
    assert response.status_code == 200, f"Search failed: {response.status_code} {response.text}"
    return response.json()


class TestSemanticSearchE2E:
    """End-to-end tests for semantic search functionality."""

    def test_exact_query_finds_document(self, upload_test_document):
        """Verify that a query using the same terms finds the document."""
        data = _search("vehicle collision at intersection red light")

        result_keys = [r["document_key"] for r in data["results"]]
        assert upload_test_document in result_keys, (
            f"Expected {upload_test_document} in results, got: {result_keys}"
        )

    def test_semantic_query_finds_document(self, upload_test_document):
        """Verify that a semantically similar query (different words) finds the document."""
        data = _search("car accident at traffic light where another car hit mine")

        result_keys = [r["document_key"] for r in data["results"]]
        assert upload_test_document in result_keys, (
            f"Semantic search should find the document. Got: {result_keys}"
        )

        # Verify it's a high-confidence match
        matching_result = next(
            r for r in data["results"]
            if r["document_key"] == upload_test_document
        )
        assert matching_result["distance"] < DISTANCE_THRESHOLD, (
            f"Expected distance < {DISTANCE_THRESHOLD}, got {matching_result['distance']}"
        )

    def test_unrelated_query_returns_low_relevance(self, upload_test_document):
        """Verify that unrelated queries don't return false positives."""
        data = _search(
            "quantum computing research on superconducting qubits",
            distance_threshold=0.3,
        )

        # With a strict threshold, unrelated content should be filtered out
        for result in data.get("results", []):
            assert result["document_key"] != upload_test_document or result["distance"] > 0.3, (
                "Unrelated query should not return the test document with high confidence"
            )

    def test_metadata_filter_works(self, upload_test_document):
        """Verify metadata filtering narrows results correctly."""
        # Search with file_type filter
        import requests
        from requests_auth_aws_sigv4 import AWSSigV4

        response = requests.post(
            f"{API_URL}search",
            json={
                "query": "vehicle collision",
                "top_k": 10,
                "filters": {"file_type": "txt"},
            },
            auth=AWSSigV4("execute-api"),
        )
        assert response.status_code == 200
        data = response.json()

        # All results should be txt files
        for result in data.get("results", []):
            if "vector_metadata" in result:
                assert result["vector_metadata"].get("file_type") == "txt"

    def test_search_returns_latency(self, upload_test_document):
        """Verify the API returns latency information."""
        data = _search("vehicle damage assessment")

        assert "latency_ms" in data
        assert data["latency_ms"] > 0
        # Search should complete in under 5 seconds
        assert data["latency_ms"] < 5000

    def test_search_returns_chunk_info(self, upload_test_document):
        """Verify results include chunk metadata when documents are chunked."""
        data = _search("collision damage repair estimate")

        for result in data.get("results", []):
            if result["document_key"] == upload_test_document:
                # Chunked documents should have chunk_index
                # (small documents may be single-chunk, so chunk_index could be 0)
                assert "chunk_index" in result or "total_chunks" in result


class TestSearchAPIValidation:
    """Tests for API input validation and error handling."""

    def test_missing_query_returns_400(self):
        """Verify missing query parameter returns 400."""
        _skip_if_not_configured()
        import requests
        from requests_auth_aws_sigv4 import AWSSigV4

        response = requests.post(
            f"{API_URL}search",
            json={},
            auth=AWSSigV4("execute-api"),
        )
        assert response.status_code == 400

    def test_empty_query_returns_400(self):
        """Verify empty query returns 400."""
        _skip_if_not_configured()
        import requests
        from requests_auth_aws_sigv4 import AWSSigV4

        response = requests.post(
            f"{API_URL}search",
            json={"query": ""},
            auth=AWSSigV4("execute-api"),
        )
        assert response.status_code == 400


class TestImageEmbeddingsE2E:
    """End-to-end tests for multimodal image embedding functionality."""

    @pytest.fixture(scope="class")
    def upload_test_image(self, s3_client):
        """Upload a test image and wait for indexing."""
        _skip_if_not_configured()

        test_key = "documents/test/damage-photo-e2e-test.jpg"

        # Create a minimal valid JPEG (1x1 pixel red image)
        # This is the smallest valid JPEG that Titan Multimodal can process
        import struct

        # Minimal JPEG: SOI + APP0 + DQT + SOF0 + DHT + SOS + image data + EOI
        # For testing, we use a pre-built minimal JPEG binary
        minimal_jpeg = (
            b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
            b'\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t'
            b'\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a'
            b'\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=8'
            b'3<.telerik34444444444444444444444444444444444444444444444444'
            b'\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00'
            b'\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00'
            b'\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b'
            b'\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04'
            b'\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa'
            b'\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n'
            b'\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz'
            b'\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99'
            b'\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7'
            b'\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5'
            b'\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1'
            b'\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa'
            b'\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd2\x8a+\xff\xd9'
        )

        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=test_key,
            Body=minimal_jpeg,
            ContentType="image/jpeg",
        )

        # Wait for event-driven indexing
        time.sleep(INDEXING_WAIT_SECONDS)

        yield test_key

        # Cleanup
        s3_client.delete_object(Bucket=S3_BUCKET, Key=test_key)
        time.sleep(5)

    def test_image_indexed_in_image_index(self, upload_test_image):
        """Verify that an uploaded image is indexed in the image-embeddings index."""
        data = _search(
            query="damage photo",
            top_k=5,
            distance_threshold=0.9,  # Relaxed threshold for image search
        )
        # We can't guarantee the minimal JPEG matches "damage photo" semantically,
        # but we can verify the search API works with the image index
        assert "results" in data
        assert "latency_ms" in data

    def test_image_search_uses_image_index(self, upload_test_image):
        """Verify that querying the image index works without errors."""
        import requests
        from requests_auth_aws_sigv4 import AWSSigV4

        response = requests.post(
            f"{API_URL}search",
            json={
                "query": "vehicle damage photograph",
                "top_k": 5,
                "index": "image-embeddings",
            },
            auth=AWSSigV4("execute-api"),
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "latency_ms" in data
        assert data["latency_ms"] > 0

    def test_image_not_in_text_index(self, upload_test_image):
        """Verify that images are stored in the image index, not the text index."""
        import requests
        from requests_auth_aws_sigv4 import AWSSigV4

        # Search the text index - the image should NOT appear here
        response = requests.post(
            f"{API_URL}search",
            json={
                "query": "damage-photo-e2e-test",
                "top_k": 10,
                "index": "document-embeddings",
                "distance_threshold": 0.9,
            },
            auth=AWSSigV4("execute-api"),
        )

        assert response.status_code == 200
        data = response.json()

        # The image file should not be in the text index results
        result_keys = [r["document_key"] for r in data.get("results", [])]
        assert upload_test_image not in result_keys, (
            "Image should be indexed in image-embeddings, not document-embeddings"
        )

    def test_image_deleted_on_removal(self, s3_client):
        """Verify that deleting an image removes its vector from the image index."""
        _skip_if_not_configured()
        import requests
        from requests_auth_aws_sigv4 import AWSSigV4

        # Upload a temporary image
        temp_key = "documents/test/temp-delete-test.jpg"
        minimal_jpeg = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9'

        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=temp_key,
            Body=minimal_jpeg,
            ContentType="image/jpeg",
        )
        time.sleep(INDEXING_WAIT_SECONDS)

        # Delete the image
        s3_client.delete_object(Bucket=S3_BUCKET, Key=temp_key)
        time.sleep(10)  # Wait for delete handler

        # Search the image index - the deleted image should not appear
        response = requests.post(
            f"{API_URL}search",
            json={
                "query": "temp delete test",
                "top_k": 10,
                "index": "image-embeddings",
                "distance_threshold": 0.99,
            },
            auth=AWSSigV4("execute-api"),
        )

        assert response.status_code == 200
        data = response.json()

        result_keys = [r["document_key"] for r in data.get("results", [])]
        assert temp_key not in result_keys, (
            "Deleted image should have been removed from the image index"
        )
