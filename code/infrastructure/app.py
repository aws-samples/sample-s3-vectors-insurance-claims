# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#!/usr/bin/env python3
import aws_cdk as cdk
from stacks.vector_search_stack import VectorSearchStack

app = cdk.App()
VectorSearchStack(app, "VectorSearchStack",
    env=cdk.Environment(
        account=cdk.Aws.ACCOUNT_ID,
        region=cdk.Aws.REGION,
    ),
)
app.synth()
