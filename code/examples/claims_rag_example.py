# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Insurance Claims Processing with RAG
Demonstrates how a motor insurance SaaS provider can use S3 vector
search to power a RAG-based claims processing assistant.

Includes guardrails to prevent hallucinations and ensure safe,
accurate claims assessments.
"""

import boto3
import json
import os

s3_client = boto3.client("s3")
bedrock_client = boto3.client("bedrock-runtime")
lambda_client = boto3.client("lambda")

# Global cross-Region inference profile for Claude Sonnet 4.5. Override with
# RAG_MODEL_ID to use a different model or a Region-scoped profile (us., eu.).
RAG_MODEL_ID = os.environ.get(
    "RAG_MODEL_ID", "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
)

# Replace with your actual bucket name after deployment
BUCKET_NAME = "vector-search-docs-ACCOUNT_ID"

# Confidence threshold: only include search results with distance below this value.
# With cosine distance, lower = more similar. 0.5 is a conservative threshold
# that prevents irrelevant context from reaching the LLM.
DISTANCE_THRESHOLD = 0.5


def process_claim_with_rag(
    claim_description: str, policy_number: str = None
):
    """
    Process an insurance claim using RAG with S3 vector search.
    1. Search for similar past claims and relevant policy docs
    2. Apply confidence threshold to filter low-relevance results
    3. Retrieve full text of top matches
    4. Generate assessment using Bedrock with retrieved context and guardrails
    """
    similar_claims = _semantic_search(
        query=claim_description,
        filters={"file_type": "txt"},
        top_k=5,
    )
    policy_docs = _semantic_search(
        query=claim_description,
        filters={"file_type": "pdf"},
        top_k=3,
    )

    # Apply confidence threshold - only use high-relevance results
    all_results = similar_claims + policy_docs
    high_confidence_results = [
        r for r in all_results
        if r.get("distance", 1.0) <= DISTANCE_THRESHOLD
    ]

    if not high_confidence_results:
        return {
            "assessment": "Insufficient similar claims found for automated assessment.",
            "confidence": "low",
            "recommendation": "Route to manual review by a senior claims adjuster.",
            "similar_claims_found": 0,
            "threshold_used": DISTANCE_THRESHOLD,
        }

    # Retrieve full documents for high-confidence matches
    context_documents = []
    for result in high_confidence_results[:6]:
        doc_text = _get_document_text(result["document_key"])
        context_documents.append({
            "source": result["document_key"],
            "distance": result.get("distance", 0),
            "text": doc_text[:3000],
        })

    return _generate_assessment(
        claim_description, context_documents, policy_number
    )


def _semantic_search(query: str, filters: dict = None, top_k: int = 5) -> list:
    """Invoke the search Lambda with confidence threshold."""
    search_body = {
        "query": query,
        "top_k": top_k,
        "distance_threshold": DISTANCE_THRESHOLD,
    }
    if filters:
        search_body["filters"] = filters

    response = lambda_client.invoke(
        FunctionName="vector-search-handler",
        InvocationType="RequestResponse",
        Payload=json.dumps({
            "body": json.dumps(search_body)
        }),
    )
    result = json.loads(response["Payload"].read())
    body = json.loads(result.get("body", "{}"))
    return body.get("results", [])


def _get_document_text(key: str) -> str:
    """Retrieve full text of a document from S3."""
    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        return response["Body"].read().decode("utf-8", errors="replace")
    except Exception as e:
        return f"[Document unavailable: {e}]"


def _generate_assessment(
    claim_description: str,
    context_documents: list,
    policy_number: str = None,
) -> dict:
    """Generate a claims assessment using Bedrock with RAG context and guardrails.

    Guardrails:
    - Only reference information explicitly present in provided documents
    - Never fabricate policy numbers, coverage amounts, or liability determinations
    - Flag potential fraud indicators for human review
    - Return structured confidence levels
    """
    context_text = "\n\n---\n\n".join([
        f"Source: {doc['source']} (similarity distance: {doc['distance']:.4f})\n"
        f"{doc['text']}"
        for doc in context_documents
    ])

    prompt = f"""You are an insurance claims processing assistant. Based on
the following claim description and relevant reference documents, provide
a structured assessment.

IMPORTANT GUARDRAILS - YOU MUST FOLLOW THESE:
1. Only reference information explicitly present in the provided documents.
2. If the documents do not contain sufficient information to make a determination, 
   state that clearly and recommend manual review.
3. Do NOT fabricate or invent policy numbers, coverage amounts, deductible values,
   or liability determinations that are not in the source documents.
4. Flag any potential fraud indicators for human review.
5. Always include a confidence level based on how closely the reference documents
   match the current claim.
6. If the similarity distances are high (>0.3), lower your confidence accordingly.

CLAIM DESCRIPTION:
{claim_description}

{f'POLICY NUMBER: {policy_number}' if policy_number else ''}

RELEVANT REFERENCE DOCUMENTS (sorted by relevance):
{context_text}

Provide your assessment as JSON with this exact structure:
{{
    "claim_category": "category of the claim (e.g., auto collision, property damage)",
    "severity_estimate": "low/medium/high based on the claim description",
    "similar_precedents": ["list of similar past claims found in the documents"],
    "applicable_policy_sections": ["relevant policy sections IF found in documents"],
    "recommended_action": "recommended next steps for the claims adjuster",
    "estimated_processing_time": "estimated time to resolve based on similar cases",
    "confidence_score": 0.0,
    "confidence_rationale": "explain why this confidence level was assigned",
    "requires_human_review": true,
    "human_review_reason": "reason for flagging (always explain)",
    "fraud_indicators": ["any fraud indicators noticed, or empty list if none"]
}}"""

    response = bedrock_client.invoke_model(
        modelId=RAG_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "temperature": 0.1,  # Low temperature for more deterministic output
            "messages": [{"role": "user", "content": prompt}],
        }),
    )

    result = json.loads(response["body"].read())
    text = result["content"][0]["text"]

    try:
        assessment = json.loads(text)
        # Add metadata about the RAG process
        assessment["_metadata"] = {
            "documents_used": len(context_documents),
            "distance_threshold": DISTANCE_THRESHOLD,
            "avg_distance": sum(d["distance"] for d in context_documents) / len(context_documents),
            "model": RAG_MODEL_ID,
        }
        return assessment
    except json.JSONDecodeError:
        return {
            "raw_assessment": text,
            "confidence_score": 0.3,
            "requires_human_review": True,
            "human_review_reason": "LLM output could not be parsed as structured JSON",
        }


if __name__ == "__main__":
    claim = """
    Vehicle collision on Highway 101 near exit 42. Policyholder's
    2022 Toyota Camry rear-ended by uninsured driver. Significant
    rear bumper and trunk damage. Policyholder reports neck pain
    and visited ER. Police report filed (#2024-PR-78432).
    Estimated vehicle repair cost: $8,500. Requesting coverage
    for vehicle repair and medical expenses.
    """

    result = process_claim_with_rag(
        claim_description=claim,
        policy_number="POL-2024-789456",
    )
    print(json.dumps(result, indent=2))
