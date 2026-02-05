"""
Scanner Prime - Storage Manager
S3/MinIO compatible cloud storage abstraction.

Provides unified interface for:
- AWS S3
- MinIO
- LocalStack
- Any S3-compatible storage

Copyright (c) 2026 Scanner Prime Team. All rights reserved.
"""

import os
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class StorageManager:
    """
    S3/MinIO compatible storage manager.

    Supports:
    - AWS S3
    - MinIO
    - LocalStack
    - Any S3-compatible storage
    """

    def __init__(
        self,
        bucket_name: str,
        endpoint_url: Optional[str] = None,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        region: str = "us-east-1"
    ):
        """
        Initialize storage manager.

        Args:
            bucket_name: S3 bucket name
            endpoint_url: Custom endpoint (for MinIO)
            aws_access_key: AWS access key (or from env)
            aws_secret_key: AWS secret key (or from env)
            region: AWS region
        """
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for S3 storage. "
                "Install with: pip install boto3"
            )

        self.bucket_name = bucket_name

        # Get credentials from env if not provided
        access_key = aws_access_key or os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = aws_secret_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        endpoint = endpoint_url or os.getenv("S3_ENDPOINT_URL")

        self.s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )

        self.endpoint_url = endpoint
        self._bucket_exists = False

    def ensure_bucket(self) -> bool:
        """Create bucket if it doesn't exist. Returns True if successful."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            self._bucket_exists = True
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code in ['404', 'NoSuchBucket']:
                try:
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                    self._bucket_exists = True
                    return True
                except ClientError as create_error:
                    if create_error.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
                        self._bucket_exists = True
                        return True
                    raise
            raise

    def upload_file(
        self,
        file_path: str,
        object_key: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Upload file to S3.

        Args:
            file_path: Local file path
            object_key: S3 object key (default: filename with timestamp)
            metadata: Optional metadata dict

        Returns:
            S3 object key
        """
        if not self._bucket_exists:
            self.ensure_bucket()

        if not object_key:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = Path(file_path).name
            object_key = f"uploads/{timestamp}_{filename}"

        extra_args = {}
        if metadata:
            extra_args["Metadata"] = metadata

        self.s3_client.upload_file(
            file_path,
            self.bucket_name,
            object_key,
            ExtraArgs=extra_args if extra_args else None
        )

        return object_key

    def upload_bytes(
        self,
        data: bytes,
        object_key: str,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Upload bytes directly to S3.

        Args:
            data: Bytes to upload
            object_key: S3 object key
            content_type: MIME type
            metadata: Optional metadata

        Returns:
            S3 object key
        """
        if not self._bucket_exists:
            self.ensure_bucket()

        extra_args = {"ContentType": content_type}
        if metadata:
            extra_args["Metadata"] = metadata

        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=object_key,
            Body=data,
            **extra_args
        )

        return object_key

    def upload_json(
        self,
        data: Dict[str, Any],
        object_key: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Upload JSON data to S3.

        Args:
            data: Dictionary to upload as JSON
            object_key: S3 object key
            metadata: Optional metadata

        Returns:
            S3 object key
        """
        json_bytes = json.dumps(data, indent=2, default=str).encode('utf-8')
        return self.upload_bytes(
            json_bytes,
            object_key,
            content_type="application/json",
            metadata=metadata
        )

    def download_file(self, object_key: str, local_path: str) -> str:
        """
        Download file from S3.

        Args:
            object_key: S3 object key
            local_path: Local destination path

        Returns:
            Local file path
        """
        self.s3_client.download_file(
            self.bucket_name,
            object_key,
            local_path
        )
        return local_path

    def download_bytes(self, object_key: str) -> bytes:
        """
        Download object as bytes.

        Args:
            object_key: S3 object key

        Returns:
            Object content as bytes
        """
        response = self.s3_client.get_object(
            Bucket=self.bucket_name,
            Key=object_key
        )
        return response['Body'].read()

    def get_presigned_url(
        self,
        object_key: str,
        expiration: int = 3600,
        method: str = "get_object"
    ) -> str:
        """
        Generate presigned URL for object.

        Args:
            object_key: S3 object key
            expiration: URL expiration in seconds
            method: S3 method (get_object or put_object)

        Returns:
            Presigned URL
        """
        return self.s3_client.generate_presigned_url(
            method,
            Params={"Bucket": self.bucket_name, "Key": object_key},
            ExpiresIn=expiration
        )

    def delete_file(self, object_key: str) -> bool:
        """Delete file from S3."""
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            return True
        except ClientError:
            return False

    def list_files(self, prefix: str = "", max_keys: int = 1000) -> List[Dict[str, Any]]:
        """
        List files with optional prefix.

        Args:
            prefix: Key prefix filter
            max_keys: Maximum files to return

        Returns:
            List of file info dictionaries
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            return [
                {
                    "key": obj["Key"],
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"].isoformat()
                }
                for obj in response.get("Contents", [])
            ]
        except ClientError:
            return []

    def get_storage_usage(self) -> Dict[str, Any]:
        """Get storage usage statistics."""
        try:
            files = self.list_files()
            total_size = sum(f["size"] for f in files)
            return {
                "total_files": len(files),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2)
            }
        except Exception:
            return {
                "total_files": 0,
                "total_size_bytes": 0,
                "total_size_mb": 0
            }

    def file_exists(self, object_key: str) -> bool:
        """Check if file exists in S3."""
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=object_key)
            return True
        except ClientError:
            return False


class LocalStorageManager:
    """
    Local filesystem storage manager (fallback when S3 not configured).
    """

    def __init__(self, base_path: str = "storage"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def upload_file(
        self,
        file_path: str,
        object_key: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """Copy file to local storage."""
        import shutil

        if not object_key:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = Path(file_path).name
            object_key = f"uploads/{timestamp}_{filename}"

        dest_path = self.base_path / object_key
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, dest_path)

        return object_key

    def upload_bytes(
        self,
        data: bytes,
        object_key: str,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """Write bytes to local storage."""
        dest_path = self.base_path / object_key
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(data)
        return object_key

    def download_file(self, object_key: str, local_path: str) -> str:
        """Copy file from local storage."""
        import shutil
        src_path = self.base_path / object_key
        shutil.copy2(src_path, local_path)
        return local_path

    def delete_file(self, object_key: str) -> bool:
        """Delete file from local storage."""
        try:
            (self.base_path / object_key).unlink()
            return True
        except Exception:
            return False

    def list_files(self, prefix: str = "") -> List[Dict[str, Any]]:
        """List files in local storage."""
        files = []
        search_path = self.base_path / prefix if prefix else self.base_path
        if search_path.exists():
            for f in search_path.rglob("*"):
                if f.is_file():
                    files.append({
                        "key": str(f.relative_to(self.base_path)),
                        "size": f.stat().st_size,
                        "last_modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                    })
        return files
