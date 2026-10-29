# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Legal Document Discovery Example
Demonstrates semantic search across contracts, case files, and
regulatory filings using S3 Vectors.

Searches are scoped to the documents/legal/ prefix with a metadata
pre-filter on document_key ($startsWith). On an ENHANCED index, S3 Vectors
evaluates the filter before the similarity search, so top_k results come
from the legal documents only.

To filter on attributes such as jurisdiction or year, add them as
filterable metadata at ingest time (in the embedding generator's
put_vectors call), then pass filters such as:
    {"jurisdiction": "US"}
    {"year": {"$gte": 2020}}   # store year as a number
"""

import boto3
import json

LEGAL_PREFIX = "documents/legal/"


def legal_document_search(query: str, top_k: int = 20) -> dict:
    """Search legal documents using semantic similarity."""
    lambda_client = boto3.client("lambda")

    search_payload = {
        "query": query,
        "top_k": top_k,
        "filters": {"document_key": {"$startsWith": LEGAL_PREFIX}},
    }

    response = lambda_client.invoke(
        FunctionName="vector-search-handler",
        InvocationType="RequestResponse",
        Payload=json.dumps({"body": json.dumps(search_payload)}),
    )

    result = json.loads(response["Payload"].read())
    return json.loads(result.get("body", "{}"))


if __name__ == "__main__":
    searches = [
        {
            "query": "IP assignment clauses in M&A transactions",
            "note": "Finds 'technology rights', 'patent assignment', "
                    "'proprietary asset conveyance' across jurisdictions",
        },
        {
            "query": "non-compete provisions for senior executives "
                     "after acquisition",
            "note": "Matches 'restraint of trade' (UK) and "
                    "'covenant not to compete' (US)",
        },
        {
            "query": "data privacy obligations in cross-border "
                     "transactions",
            "note": "Finds GDPR, CCPA, and data protection clauses "
                    "regardless of regulation names",
        },
    ]

    for s in searches:
        print(f"\nQuery: {s['query']}")
        print(f"  Why it works: {s['note']}")
        results = legal_document_search(s["query"])
        print(f"  Found {results.get('result_count', 0)} documents")
        for r in results.get("results", [])[:3]:
            # Cosine distance: lower is more similar (0 = identical)
            print(f"    - {r['document_key']} "
                  f"(distance: {r['distance']:.4f})")
