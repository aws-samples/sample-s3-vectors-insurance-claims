# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Start batch processing of existing documents via Step Functions.

Usage: python start_batch_processing.py <state-machine-arn> [prefix]
"""

import boto3
import json
import sys


def start_batch(state_machine_arn: str, prefix: str = "documents/"):
    """Start the batch processing Step Functions execution."""
    sfn_client = boto3.client("stepfunctions")

    response = sfn_client.start_execution(
        stateMachineArn=state_machine_arn,
        input=json.dumps({
            "prefix": prefix,
            "continuation_token": None,
        }),
    )

    execution_arn = response["executionArn"]
    print(f"Batch processing started: {execution_arn}")
    print(f"Monitor at: https://console.aws.amazon.com/states/home"
          f"#/executions/details/{execution_arn}")
    return execution_arn


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python start_batch_processing.py "
              "<state-machine-arn> [prefix]")
        sys.exit(1)

    arn = sys.argv[1]
    pfx = sys.argv[2] if len(sys.argv) > 2 else "documents/"
    start_batch(arn, pfx)
