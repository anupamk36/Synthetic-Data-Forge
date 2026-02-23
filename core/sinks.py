"""
Data Sinks — Zero-Copy Cloud Push.

Extensible sink interface for pushing generated DataFrames
to various storage backends without intermediate disk writes.
"""

import polars as pl
import os
import io
import math
from abc import ABC, abstractmethod


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
        destination = os.path.abspath(os.path.expanduser(destination))
        written_paths = []

        if partitions:
            # Group by all partition columns for full Hive-style nesting
            groups = df.group_by(partitions)
            for group_vals, group_df in groups:
                # Build nested Hive path
                if isinstance(group_vals, tuple):
                    path_parts = [f"{col}={val}" for col, val in zip(partitions, group_vals)]
                else:
                    path_parts = [f"{partitions[0]}={group_vals}"]
                nested_dir = os.path.join(destination, *path_parts)
                paths = self._write_batches(group_df, nested_dir, file_format, records_per_file)
                written_paths.extend(paths)
        else:
            written_paths = self._write_batches(df, destination, file_format, records_per_file)

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

    def __init__(self, bucket: str, prefix: str = "", region: str = "us-east-1"):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.region = region

    def push(self, df: pl.DataFrame, destination: str = "", file_format: str = "parquet",
             records_per_file: int = 250, partitions: list = None) -> list:
        """Stream DataFrame directly to S3."""
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for S3 sink. Install it with: pip install boto3"
            )

        s3 = boto3.client("s3", region_name=self.region)
        written_keys = []

        base_prefix = f"{self.prefix}/{destination}".strip("/") if destination else self.prefix

        if partitions:
            groups = df.group_by(partitions)
            for group_vals, group_df in groups:
                if isinstance(group_vals, tuple):
                    path_parts = "/".join(f"{col}={val}" for col, val in zip(partitions, group_vals))
                else:
                    path_parts = f"{partitions[0]}={group_vals}"
                nested_prefix = f"{base_prefix}/{path_parts}"
                keys = self._upload_batches(s3, group_df, nested_prefix, file_format, records_per_file)
                written_keys.extend(keys)
        else:
            written_keys = self._upload_batches(s3, df, base_prefix, file_format, records_per_file)

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
                buf.write(batch.write_csv().encode("utf-8") if isinstance(batch.write_csv(), str) else batch.write_csv())
            elif file_format == "json":
                buf.write(batch.write_json().encode("utf-8") if isinstance(batch.write_json(), str) else batch.write_json())
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
        )
    else:
        raise ValueError(f"Unknown sink type: {sink_type}")
