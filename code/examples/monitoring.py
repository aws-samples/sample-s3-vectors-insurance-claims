# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
CloudWatch Monitoring
Custom metrics and alarms for the S3 vector search pipeline.
"""

import boto3

cloudwatch = boto3.client("cloudwatch")


def publish_search_metrics(
    query: str, result_count: int, latency_ms: float
):
    """Publish custom metrics for search operations."""
    cloudwatch.put_metric_data(
        Namespace="S3VectorSearch",
        MetricData=[
            {
                "MetricName": "SearchLatency",
                "Value": latency_ms,
                "Unit": "Milliseconds",
                "Dimensions": [
                    {"Name": "Operation", "Value": "SemanticSearch"},
                ],
            },
            {
                "MetricName": "ResultCount",
                "Value": result_count,
                "Unit": "Count",
                "Dimensions": [
                    {"Name": "Operation", "Value": "SemanticSearch"},
                ],
            },
            {
                "MetricName": "SearchRequests",
                "Value": 1,
                "Unit": "Count",
                "Dimensions": [
                    {"Name": "Operation", "Value": "SemanticSearch"},
                ],
            },
        ],
    )


def create_monitoring_alarms(sns_topic_arn: str):
    """Create CloudWatch alarms for pipeline health."""

    # High search latency
    cloudwatch.put_metric_alarm(
        AlarmName="S3VectorSearch-HighLatency",
        MetricName="SearchLatency",
        Namespace="S3VectorSearch",
        Statistic="Average",
        Period=300,
        EvaluationPeriods=3,
        Threshold=2000,
        ComparisonOperator="GreaterThanThreshold",
        AlarmActions=[sns_topic_arn],
        AlarmDescription=(
            "Search latency exceeds 2 seconds for 15 minutes"
        ),
    )

    # Embedding generation failures
    cloudwatch.put_metric_alarm(
        AlarmName="S3VectorSearch-EmbeddingFailures",
        MetricName="Errors",
        Namespace="AWS/Lambda",
        Statistic="Sum",
        Period=300,
        EvaluationPeriods=2,
        Threshold=10,
        ComparisonOperator="GreaterThanThreshold",
        Dimensions=[
            {
                "Name": "FunctionName",
                "Value": "vector-search-embedding-generator",
            },
        ],
        AlarmActions=[sns_topic_arn],
        AlarmDescription=(
            "More than 10 embedding failures in 10 minutes"
        ),
    )

    # DLQ messages (failed processing)
    cloudwatch.put_metric_alarm(
        AlarmName="S3VectorSearch-DLQMessages",
        MetricName="ApproximateNumberOfMessagesVisible",
        Namespace="AWS/SQS",
        Statistic="Maximum",
        Period=300,
        EvaluationPeriods=1,
        Threshold=0,
        ComparisonOperator="GreaterThanThreshold",
        Dimensions=[
            {
                "Name": "QueueName",
                "Value": "vector-search-processing-dlq",
            },
        ],
        AlarmActions=[sns_topic_arn],
        AlarmDescription="Messages detected in dead letter queue",
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python monitoring.py <sns-topic-arn>")
        sys.exit(1)
    create_monitoring_alarms(sys.argv[1])
    print("Alarms created successfully")
