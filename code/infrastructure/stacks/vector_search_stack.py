# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import aws_cdk as cdk
from aws_cdk import (
    Stack,
    aws_s3 as s3,
    aws_s3vectors as s3vectors,
    aws_lambda as _lambda,
    aws_sqs as sqs,
    aws_iam as iam,
    aws_s3_notifications as s3n,
    aws_lambda_event_sources as lambda_events,
    aws_apigateway as apigw,
    aws_logs as logs,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
    aws_cloudwatch as cloudwatch,
    Duration,
    RemovalPolicy,
)
from constructs import Construct


class VectorSearchStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Apply resource tags to all resources in this stack
        cdk.Tags.of(self).add("Project", "S3VectorsSemanticSearch")
        cdk.Tags.of(self).add("ManagedBy", "CDK")

        # --- S3 Vectors: Vector Bucket and Indexes ---
        vector_bucket_name = f"vector-search-{cdk.Aws.ACCOUNT_ID}"

        self.vector_bucket = s3vectors.CfnVectorBucket(
            self, "VectorBucket",
            vector_bucket_name=vector_bucket_name,
        )

        # Primary index for text document embeddings
        self.vector_index = s3vectors.CfnIndex(
            self, "VectorIndex",
            vector_bucket_name=vector_bucket_name,
            index_name="document-embeddings",
            data_type="float32",
            dimension=1024,
            distance_metric="cosine",
            metadata_configuration=s3vectors.CfnIndex.MetadataConfigurationProperty(
                non_filterable_metadata_keys=["source_text"],
            ),
        )
        self.vector_index.add_dependency(self.vector_bucket)

        # Secondary index for multimodal image embeddings
        self.image_vector_index = s3vectors.CfnIndex(
            self, "ImageVectorIndex",
            vector_bucket_name=vector_bucket_name,
            index_name="image-embeddings",
            data_type="float32",
            dimension=1024,
            distance_metric="cosine",
            metadata_configuration=s3vectors.CfnIndex.MetadataConfigurationProperty(
                non_filterable_metadata_keys=["description"],
            ),
        )
        self.image_vector_index.add_dependency(self.vector_bucket)

        # S3 bucket for source documents
        self.document_bucket = s3.Bucket(
            self, "DocumentBucket",
            bucket_name=f"vector-search-docs-{cdk.Aws.ACCOUNT_ID}",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,
        )

        # SQS queue to buffer S3 event notifications
        self.dlq = sqs.Queue(
            self, "ProcessingDLQ",
            queue_name="vector-search-processing-dlq",
            retention_period=Duration.days(14),
            encryption=sqs.QueueEncryption.SQS_MANAGED,
        )

        self.processing_queue = sqs.Queue(
            self, "DocumentProcessingQueue",
            queue_name="vector-search-processing",
            visibility_timeout=Duration.minutes(15),
            retention_period=Duration.days(7),
            encryption=sqs.QueueEncryption.SQS_MANAGED,
            dead_letter_queue=sqs.DeadLetterQueue(
                max_receive_count=3,
                queue=self.dlq,
            ),
        )

        # S3 event notifications -> SQS for supported file types
        supported_suffixes = [".pdf", ".txt", ".docx", ".md", ".json",
                             ".jpg", ".jpeg", ".png", ".gif", ".webp"]
        for suffix in supported_suffixes:
            self.document_bucket.add_event_notification(
                s3.EventType.OBJECT_CREATED,
                s3n.SqsDestination(self.processing_queue),
                s3.NotificationKeyFilter(prefix="documents/", suffix=suffix),
            )

        # SQS queue for delete events -> vector cleanup
        self.delete_queue = sqs.Queue(
            self, "DeleteProcessingQueue",
            queue_name="vector-search-delete-processing",
            visibility_timeout=Duration.minutes(5),
            retention_period=Duration.days(7),
            encryption=sqs.QueueEncryption.SQS_MANAGED,
        )

        for suffix in supported_suffixes:
            self.document_bucket.add_event_notification(
                s3.EventType.OBJECT_REMOVED,
                s3n.SqsDestination(self.delete_queue),
                s3.NotificationKeyFilter(prefix="documents/", suffix=suffix),
            )

        # Bedrock model ARNs
        bedrock_text_model_arn = (
            f"arn:aws:bedrock:{cdk.Aws.REGION}::foundation-model/"
            "amazon.titan-embed-text-v2:0"
        )
        bedrock_image_model_arn = (
            f"arn:aws:bedrock:{cdk.Aws.REGION}::foundation-model/"
            "amazon.titan-embed-image-v1"
        )

        # S3 Vectors IAM policies - scoped per function (least privilege)
        s3vectors_base_resources = [
            f"arn:aws:s3vectors:{cdk.Aws.REGION}:{cdk.Aws.ACCOUNT_ID}:bucket/{vector_bucket_name}",
            f"arn:aws:s3vectors:{cdk.Aws.REGION}:{cdk.Aws.ACCOUNT_ID}:bucket/{vector_bucket_name}/*",
        ]

        s3vectors_write_policy = iam.PolicyStatement(
            actions=[
                "s3vectors:PutVectors",
                "s3vectors:GetVectors",
                "s3vectors:DeleteVectors",
                "s3vectors:ListVectors",
            ],
            resources=s3vectors_base_resources,
        )

        s3vectors_read_policy = iam.PolicyStatement(
            actions=[
                "s3vectors:QueryVectors",
                "s3vectors:GetVectors",
                "s3vectors:GetIndex",
            ],
            resources=s3vectors_base_resources,
        )

        s3vectors_delete_policy = iam.PolicyStatement(
            actions=[
                "s3vectors:GetVectors",
                "s3vectors:DeleteVectors",
                "s3vectors:ListVectors",
            ],
            resources=s3vectors_base_resources,
        )

        # CloudWatch metrics policy
        cloudwatch_policy = iam.PolicyStatement(
            actions=["cloudwatch:PutMetricData"],
            resources=["*"],
            conditions={
                "StringEquals": {
                    "cloudwatch:namespace": "S3VectorsSearch",
                },
            },
        )

        # --- Lambda: Embedding Generator ---
        self.embedding_lambda = _lambda.Function(
            self, "EmbeddingGenerator",
            function_name="vector-search-embedding-generator",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="index.handler",
            code=_lambda.Code.from_asset("../lambda/embedding_generator"),
            timeout=Duration.minutes(10),
            memory_size=1024,
            environment={
                "DOCUMENT_BUCKET_NAME": self.document_bucket.bucket_name,
                "VECTOR_BUCKET_NAME": vector_bucket_name,
                "VECTOR_INDEX_NAME": "document-embeddings",
                "IMAGE_INDEX_NAME": "image-embeddings",
                "EMBEDDING_MODEL_ID": "amazon.titan-embed-text-v2:0",
                "MULTIMODAL_MODEL_ID": "amazon.titan-embed-image-v1",
                "EMBEDDING_DIMENSIONS": "1024",
                "CHUNK_SIZE": "1000",
                "CHUNK_OVERLAP": "200",
                "LOG_LEVEL": "INFO",
            },
            log_retention=logs.RetentionDays.TWO_WEEKS,
        )
        self.document_bucket.grant_read(self.embedding_lambda)
        self.embedding_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["bedrock:InvokeModel"],
            resources=[bedrock_text_model_arn, bedrock_image_model_arn],
        ))
        self.embedding_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=[
                "textract:DetectDocumentText",
                "textract:StartDocumentTextDetection",
                "textract:GetDocumentTextDetection",
            ],
            resources=["*"],  # Textract does not support resource-level permissions
        ))
        self.embedding_lambda.add_to_role_policy(s3vectors_write_policy)
        self.embedding_lambda.add_to_role_policy(cloudwatch_policy)
        self.embedding_lambda.add_event_source(
            lambda_events.SqsEventSource(
                self.processing_queue,
                batch_size=5,
                max_batching_window=Duration.seconds(30),
            )
        )

        # --- Lambda: Search Handler ---
        self.search_lambda = _lambda.Function(
            self, "SearchHandler",
            function_name="vector-search-handler",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="index.handler",
            code=_lambda.Code.from_asset("../lambda/search_handler"),
            timeout=Duration.seconds(30),
            memory_size=512,
            environment={
                "DOCUMENT_BUCKET_NAME": self.document_bucket.bucket_name,
                "VECTOR_BUCKET_NAME": vector_bucket_name,
                "VECTOR_INDEX_NAME": "document-embeddings",
                "IMAGE_INDEX_NAME": "image-embeddings",
                "EMBEDDING_MODEL_ID": "amazon.titan-embed-text-v2:0",
                "MULTIMODAL_MODEL_ID": "amazon.titan-embed-image-v1",
                "EMBEDDING_DIMENSIONS": "1024",
                "DEFAULT_TOP_K": "10",
                "DISTANCE_THRESHOLD": "0.75",
                "LOG_LEVEL": "INFO",
            },
            log_retention=logs.RetentionDays.TWO_WEEKS,
        )
        self.document_bucket.grant_read(self.search_lambda)
        self.search_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["bedrock:InvokeModel"],
            resources=[bedrock_text_model_arn, bedrock_image_model_arn],
        ))
        self.search_lambda.add_to_role_policy(s3vectors_read_policy)
        self.search_lambda.add_to_role_policy(cloudwatch_policy)

        # --- Lambda: Batch Processor ---
        self.batch_lambda = _lambda.Function(
            self, "BatchProcessor",
            function_name="vector-search-batch-processor",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="index.handler",
            code=_lambda.Code.from_asset("../lambda/batch_processor"),
            timeout=Duration.minutes(15),
            memory_size=2048,
            environment={
                "DOCUMENT_BUCKET_NAME": self.document_bucket.bucket_name,
                "VECTOR_BUCKET_NAME": vector_bucket_name,
                "VECTOR_INDEX_NAME": "document-embeddings",
                "IMAGE_INDEX_NAME": "image-embeddings",
                "EMBEDDING_MODEL_ID": "amazon.titan-embed-text-v2:0",
                "EMBEDDING_DIMENSIONS": "1024",
                "CHUNK_SIZE": "1000",
                "CHUNK_OVERLAP": "200",
                "BATCH_SIZE": "100",
                "MAX_WORKERS": "10",
                "LOG_LEVEL": "INFO",
            },
            log_retention=logs.RetentionDays.TWO_WEEKS,
        )
        self.document_bucket.grant_read(self.batch_lambda)
        self.batch_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=["bedrock:InvokeModel"],
            resources=[bedrock_text_model_arn, bedrock_image_model_arn],
        ))
        self.batch_lambda.add_to_role_policy(iam.PolicyStatement(
            actions=[
                "textract:DetectDocumentText",
                "textract:StartDocumentTextDetection",
                "textract:GetDocumentTextDetection",
            ],
            resources=["*"],  # Textract does not support resource-level permissions
        ))
        self.batch_lambda.add_to_role_policy(s3vectors_write_policy)
        self.batch_lambda.add_to_role_policy(cloudwatch_policy)

        # --- Lambda: Delete Handler (vector cleanup on S3 object deletion) ---
        self.delete_lambda = _lambda.Function(
            self, "DeleteHandler",
            function_name="vector-search-delete-handler",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="index.handler",
            code=_lambda.Code.from_asset("../lambda/delete_handler"),
            timeout=Duration.minutes(2),
            memory_size=256,
            environment={
                "VECTOR_BUCKET_NAME": vector_bucket_name,
                "VECTOR_INDEX_NAME": "document-embeddings",
                "IMAGE_INDEX_NAME": "image-embeddings",
                # Restored objects (delete marker removed) are re-queued here
                "PROCESSING_QUEUE_URL": self.processing_queue.queue_url,
                "LOG_LEVEL": "INFO",
            },
            log_retention=logs.RetentionDays.TWO_WEEKS,
        )
        self.delete_lambda.add_to_role_policy(s3vectors_delete_policy)
        # HeadObject to tell a real delete from a version/delete-marker removal.
        # grant_read includes s3:ListBucket, so a missing key returns 404, not 403.
        self.document_bucket.grant_read(self.delete_lambda)
        self.processing_queue.grant_send_messages(self.delete_lambda)
        self.delete_lambda.add_event_source(
            lambda_events.SqsEventSource(
                self.delete_queue,
                batch_size=10,
                max_batching_window=Duration.seconds(10),
            )
        )

        # --- Step Functions: Batch Processing Workflow ---
        process_batch_task = tasks.LambdaInvoke(
            self, "ProcessBatchTask",
            lambda_function=self.batch_lambda,
            payload=sfn.TaskInput.from_object({
                "prefix": sfn.JsonPath.string_at("$.prefix"),
                "continuation_token": sfn.JsonPath.string_at("$.continuation_token"),
                "max_items": 100,
            }),
            result_path="$.batch_result",
            result_selector={
                "has_more.$": "$.Payload.has_more",
                "continuation_token.$": "$.Payload.continuation_token",
                "processed.$": "$.Payload.processed",
                "failed.$": "$.Payload.failed",
            },
        )

        wait_step = sfn.Wait(
            self, "WaitBeforeNextBatch",
            time=sfn.WaitTime.duration(Duration.seconds(5)),
        )

        prepare_next = sfn.Pass(
            self, "PrepareNextBatch",
            parameters={
                "prefix.$": "$.prefix",
                "continuation_token.$": "$.batch_result.continuation_token",
            },
        )

        batch_complete = sfn.Succeed(self, "BatchComplete")

        check_more = sfn.Choice(self, "CheckMoreItems")
        check_more.when(
            sfn.Condition.boolean_equals("$.batch_result.has_more", True),
            wait_step.next(prepare_next).next(process_batch_task),
        )
        check_more.otherwise(batch_complete)

        process_batch_task.next(check_more)

        self.batch_state_machine = sfn.StateMachine(
            self, "BatchProcessingStateMachine",
            state_machine_name="vector-search-batch-processor",
            definition_body=sfn.DefinitionBody.from_chainable(process_batch_task),
            timeout=Duration.hours(24),
        )

        # --- API Gateway: Search Endpoint ---
        api = apigw.RestApi(
            self, "VectorSearchAPI",
            rest_api_name="S3 Vector Search API",
            description="Semantic search across S3 documents",
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=apigw.Cors.ALL_ORIGINS,  # Restrict in production
                allow_methods=["POST", "OPTIONS"],
                allow_headers=["Content-Type", "Authorization"],
            ),
            deploy_options=apigw.StageOptions(
                throttling_rate_limit=100,
                throttling_burst_limit=50,
            ),
        )

        search_resource = api.root.add_resource("search")
        search_resource.add_method(
            "POST",
            apigw.LambdaIntegration(self.search_lambda),
            authorization_type=apigw.AuthorizationType.IAM,
        )

        # --- CloudWatch Dashboard ---
        dashboard = cloudwatch.Dashboard(
            self, "SearchPipelineDashboard",
            dashboard_name="S3VectorsSearchPipeline",
        )

        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Embedding Generator Latency (P50/P95/P99)",
                left=[
                    self.embedding_lambda.metric_duration(statistic="p50"),
                    self.embedding_lambda.metric_duration(statistic="p95"),
                    self.embedding_lambda.metric_duration(statistic="p99"),
                ],
                width=12,
            ),
            cloudwatch.GraphWidget(
                title="Search Handler Latency (P50/P95)",
                left=[
                    self.search_lambda.metric_duration(statistic="p50"),
                    self.search_lambda.metric_duration(statistic="p95"),
                ],
                width=12,
            ),
        )

        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Lambda Errors",
                left=[
                    self.embedding_lambda.metric_errors(),
                    self.search_lambda.metric_errors(),
                    self.delete_lambda.metric_errors(),
                    self.batch_lambda.metric_errors(),
                ],
                width=12,
            ),
            cloudwatch.GraphWidget(
                title="DLQ Depth",
                left=[
                    self.dlq.metric_approximate_number_of_messages_visible(),
                ],
                width=12,
            ),
        )

        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Custom: Indexing Latency by File Type",
                left=[
                    cloudwatch.Metric(
                        namespace="S3VectorsSearch",
                        metric_name="IndexingLatencyMs",
                        dimensions_map={"FileType": "pdf"},
                        statistic="p95",
                        label="PDF P95",
                    ),
                    cloudwatch.Metric(
                        namespace="S3VectorsSearch",
                        metric_name="IndexingLatencyMs",
                        dimensions_map={"FileType": "txt"},
                        statistic="p95",
                        label="TXT P95",
                    ),
                    cloudwatch.Metric(
                        namespace="S3VectorsSearch",
                        metric_name="IndexingLatencyMs",
                        dimensions_map={"FileType": "image"},
                        statistic="p95",
                        label="Image P95",
                    ),
                ],
                width=12,
            ),
            cloudwatch.GraphWidget(
                title="Custom: Search Latency",
                left=[
                    cloudwatch.Metric(
                        namespace="S3VectorsSearch",
                        metric_name="SearchLatencyMs",
                        statistic="p50",
                        label="P50",
                    ),
                    cloudwatch.Metric(
                        namespace="S3VectorsSearch",
                        metric_name="SearchLatencyMs",
                        statistic="p95",
                        label="P95",
                    ),
                ],
                width=12,
            ),
        )

        # --- DLQ Alarm ---
        cloudwatch.Alarm(
            self, "DLQAlarm",
            metric=self.dlq.metric_approximate_number_of_messages_visible(),
            threshold=1,
            evaluation_periods=1,
            alarm_description="Messages in the dead-letter queue indicate processing failures",
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
        )

        # --- Outputs ---
        cdk.CfnOutput(self, "SearchAPIEndpoint", value=api.url)
        cdk.CfnOutput(self, "DocumentBucketName",
                       value=self.document_bucket.bucket_name)
        cdk.CfnOutput(self, "VectorBucketName", value=vector_bucket_name)
        cdk.CfnOutput(self, "VectorIndexName", value="document-embeddings")
        cdk.CfnOutput(self, "ImageIndexName", value="image-embeddings")
        cdk.CfnOutput(
            self, "BatchStateMachineArn",
            value=self.batch_state_machine.state_machine_arn,
        )
        cdk.CfnOutput(self, "DLQUrl", value=self.dlq.queue_url)
        cdk.CfnOutput(self, "DashboardName", value="S3VectorsSearchPipeline")
