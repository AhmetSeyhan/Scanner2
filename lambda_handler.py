"""
Scanner Prime - AWS Lambda Handler
Serverless deepfake analysis endpoint for auto-scaling.

Deployment:
    1. Package with dependencies: pip install -t ./package -r requirements.txt
    2. Create Lambda function with 10GB memory, 5min timeout
    3. Attach S3 trigger for automated analysis

Environment Variables:
    SCANNER_S3_BUCKET: S3 bucket for uploads
    SCANNER_WEIGHTS_PATH: S3 path to model weights

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

import base64
import json
import os

import boto3

s3_client = boto3.client("s3")


def handler(event, context):
    """
    Lambda entry point.

    Accepts:
    - API Gateway POST /analyze (base64 video in body)
    - S3 trigger (automatic analysis on upload)
    """
    # Determine event source
    if "Records" in event:
        return _handle_s3_trigger(event, context)
    elif "body" in event:
        return _handle_api_gateway(event, context)
    else:
        return {"statusCode": 400, "body": json.dumps({"error": "Unknown event source"})}


def _handle_s3_trigger(event, context):
    """Process video uploaded to S3."""
    record = event["Records"][0]["s3"]
    bucket = record["bucket"]["name"]
    key = record["object"]["key"]

    # Download to /tmp
    local_path = f"/tmp/{os.path.basename(key)}"
    s3_client.download_file(bucket, key, local_path)

    result = _analyze_video(local_path, os.path.basename(key))

    # Write result back to S3
    result_key = key.rsplit(".", 1)[0] + "_result.json"
    s3_client.put_object(
        Bucket=bucket, Key=result_key,
        Body=json.dumps(result, default=str),
        ContentType="application/json",
    )

    return {"statusCode": 200, "body": json.dumps(result, default=str)}


def _handle_api_gateway(event, context):
    """Process API Gateway request."""
    body = event.get("body", "")
    if event.get("isBase64Encoded"):
        video_bytes = base64.b64decode(body)
    else:
        video_bytes = body.encode()

    filename = "upload.mp4"
    local_path = f"/tmp/{filename}"
    with open(local_path, "wb") as f:
        f.write(video_bytes)

    result = _analyze_video(local_path, filename)

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(result, default=str),
    }


def _analyze_video(video_path: str, filename: str) -> dict:
    """Run PRIME HYBRID analysis."""
    from services.analysis_service import AnalysisService

    service = AnalysisService()
    return service.analyze_video_v2(
        video_path=video_path,
        filename=filename,
        user="lambda",
    )
