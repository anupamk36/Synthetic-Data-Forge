"""
Data Sinks — Zero-Copy Cloud Push.

Extensible sink interface for pushing generated DataFrames
to various storage backends without intermediate disk writes.
"""

import logging
import polars as pl
import os
import io
import math
from abc import ABC, abstractmethod

from core.exceptions import SinkError
from core.validation import sanitize_partition_value
from core.config import validate_output_path

logger = logging.getLogger(__name__)


class DataSink(ABC):
    """Abstract base class for data sinks."""

    @abstractmethod
    def push(self, df: pl.DataFrame, destination: str, file_format: str = "parquet",
             records_per_file: int = 250, partitions: list = None) -> list:
        """
        Push a DataFrame to the sink.

        Returns list of paths/URIs written.
        """
        pass


class LocalSink(DataSink):
    """Write data to the local filesystem."""

    def push(self, df: pl.DataFrame, destination: str, file_format: str = "parquet",
             records_per_file: int = 250, partitions: list = None) -> list:
        """Write DataFrame to local disk with optional partitioning."""
        destination = validate_output_path(destination)
        written_paths = []

        if partitions:
            groups = df.group_by(partitions)
            for group_vals, group_df in groups:
                if isinstance(group_vals, tuple):
                    path_parts = [
                        f"{sanitize_partition_value(col)}={sanitize_partition_value(val)}"
                        for col, val in zip(partitions, group_vals)
                    ]
                else:
                    path_parts = [
                        f"{sanitize_partition_value(partitions[0])}={sanitize_partition_value(group_vals)}"
                    ]
                nested_dir = os.path.join(destination, *path_parts)
                paths = self._write_batches(group_df, nested_dir, file_format, records_per_file)
                written_paths.extend(paths)
        else:
            written_paths = self._write_batches(df, destination, file_format, records_per_file)

        logger.info("LocalSink wrote %d files to %s", len(written_paths), destination)
        return written_paths

    def _write_batches(self, df: pl.DataFrame, out_dir: str,
                       file_format: str, records_per_file: int) -> list:
        """Split and write a DataFrame in batches."""
        os.makedirs(out_dir, exist_ok=True)
        paths = []
        num_files = max(1, math.ceil(len(df) / records_per_file))

        for i in range(num_files):
            batch = df.slice(i * records_per_file, records_per_file)
            if len(batch) == 0:
                continue

            ext = {"parquet": "parquet", "csv": "csv", "json": "json"}.get(file_format, "parquet")
            filepath = os.path.join(out_dir, f"part_{i}.{ext}")

            if file_format == "csv":
                batch.write_csv(filepath)
            elif file_format == "json":
                batch.write_json(filepath)
            else:
                batch.write_parquet(filepath)

            paths.append(filepath)

        return paths


class S3Sink(DataSink):
    """Push data directly to Amazon S3 without touching local disk."""

    def __init__(self, bucket: str, prefix: str = "", region: str = "us-east-1",
                 aws_access_key_id: str = "", aws_secret_access_key: str = "",
                 aws_session_token: str = ""):
        if not bucket:
            raise SinkError("S3 bucket name is required.")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.region = region
        self.aws_access_key_id = aws_access_key_id or None
        self.aws_secret_access_key = aws_secret_access_key or None
        self.aws_session_token = aws_session_token or None

    def push(self, df: pl.DataFrame, destination: str = "", file_format: str = "parquet",
             records_per_file: int = 250, partitions: list = None) -> list:
        """Stream DataFrame directly to S3."""
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for S3 sink. Install it with: pip install boto3"
            )

        client_kwargs = {"region_name": self.region}
        if self.aws_access_key_id and self.aws_secret_access_key:
            client_kwargs["aws_access_key_id"] = self.aws_access_key_id
            client_kwargs["aws_secret_access_key"] = self.aws_secret_access_key
            if self.aws_session_token:
                client_kwargs["aws_session_token"] = self.aws_session_token
        s3 = boto3.client("s3", **client_kwargs)
        written_keys = []

        base_prefix = f"{self.prefix}/{destination}".strip("/") if destination else self.prefix

        if partitions:
            groups = df.group_by(partitions)
            for group_vals, group_df in groups:
                if isinstance(group_vals, tuple):
                    path_parts = "/".join(
                        f"{sanitize_partition_value(col)}={sanitize_partition_value(val)}"
                        for col, val in zip(partitions, group_vals)
                    )
                else:
                    path_parts = (
                        f"{sanitize_partition_value(partitions[0])}="
                        f"{sanitize_partition_value(group_vals)}"
                    )
                nested_prefix = f"{base_prefix}/{path_parts}"
                keys = self._upload_batches(s3, group_df, nested_prefix, file_format, records_per_file)
                written_keys.extend(keys)
        else:
            written_keys = self._upload_batches(s3, df, base_prefix, file_format, records_per_file)

        logger.info("S3Sink wrote %d objects to s3://%s/%s", len(written_keys), self.bucket, base_prefix)
        return written_keys

    def _upload_batches(self, s3, df: pl.DataFrame, prefix: str,
                        file_format: str, records_per_file: int) -> list:
        """Upload batches directly to S3 from memory."""
        keys = []
        num_files = max(1, math.ceil(len(df) / records_per_file))

        for i in range(num_files):
            batch = df.slice(i * records_per_file, records_per_file)
            if len(batch) == 0:
                continue

            ext = {"parquet": "parquet", "csv": "csv", "json": "json"}.get(file_format, "parquet")
            key = f"{prefix}/part_{i}.{ext}"

            buf = io.BytesIO()
            if file_format == "csv":
                csv_data = batch.write_csv()
                if isinstance(csv_data, str):
                    csv_data = csv_data.encode("utf-8")
                buf.write(csv_data)
            elif file_format == "json":
                json_data = batch.write_json()
                if isinstance(json_data, str):
                    json_data = json_data.encode("utf-8")
                buf.write(json_data)
            else:
                batch.write_parquet(buf)

            buf.seek(0)
            s3.upload_fileobj(buf, self.bucket, key)
            keys.append(f"s3://{self.bucket}/{key}")

        return keys


def get_sink(sink_type: str, **kwargs) -> DataSink:
    """Factory function to create a sink by type."""
    if sink_type == "local":
        return LocalSink()
    elif sink_type == "s3":
        return S3Sink(
            bucket=kwargs.get("bucket", ""),
            prefix=kwargs.get("prefix", ""),
            region=kwargs.get("region", "us-east-1"),
            aws_access_key_id=kwargs.get("aws_access_key_id", ""),
            aws_secret_access_key=kwargs.get("aws_secret_access_key", ""),
            aws_session_token=kwargs.get("aws_session_token", ""),
        )
    else:
        raise SinkError(f"Unknown sink type: {sink_type}")
