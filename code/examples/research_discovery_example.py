# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Research Paper Discovery Example
Cross-disciplinary academic paper discovery using semantic search.

Each discipline is scoped with a metadata pre-filter on document_key
($startsWith), so on an ENHANCED index the top_k results for a discipline
come only from that discipline's folder.
"""

import boto3
import json

lambda_client = boto3.client("lambda")

RESEARCH_PREFIX = "documents/research/"


def _search(query: str, prefix: str, top_k: int) -> list:
    """Invoke the search Lambda scoped to a document_key prefix."""
    payload = {
        "query": query,
        "top_k": top_k,
        "filters": {"document_key": {"$startsWith": prefix}},
    }
    response = lambda_client.invoke(
        FunctionName="vector-search-handler",
        InvocationType="RequestResponse",
        Payload=json.dumps({"body": json.dumps(payload)}),
    )
    result = json.loads(response["Payload"].read())
    body = json.loads(result.get("body", "{}"))
    return body.get("results", [])


def discover_related_research(
    research_question: str,
    disciplines: list = None,
    top_k: int = 15,
) -> list:
    """
    Find related research papers across academic disciplines.
    Demonstrates cross-disciplinary discovery where the same concept
    may use completely different terminology.
    """
    if not disciplines:
        return _search(research_question, RESEARCH_PREFIX, top_k)

    all_results = []
    for discipline in disciplines:
        all_results.extend(
            _search(research_question, f"{RESEARCH_PREFIX}{discipline}/", top_k)
        )

    # Cosine distance: lower is more similar, so sort ascending
    all_results.sort(key=lambda x: x["distance"])
    return all_results[:top_k]


if __name__ == "__main__":
    results = discover_related_research(
        research_question=(
            "machine learning approaches to predict "
            "protein folding stability"
        ),
        disciplines=["biology", "computer-science", "chemistry"],
        top_k=10,
    )

    print(f"Found {len(results)} related papers across disciplines")
    for r in results:
        print(f"  [distance {r['distance']:.4f}] {r['document_key']}")
