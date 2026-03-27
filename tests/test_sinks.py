"""Tests for core.sinks — LocalSink and S3Sink."""

import os
import polars as pl
import pytest

from core.sinks import LocalSink, S3Sink, get_sink
from core.exceptions import SinkError, ConfigError


class TestLocalSink:

    def test_write_parquet(self, sample_df, tmp_output):
        sink = LocalSink()
        paths = sink.push(sample_df, str(tmp_output / "out"), "parquet", 250)
        assert len(paths) >= 1
        assert all(p.endswith(".parquet") for p in paths)
        assert os.path.exists(paths[0])

    def test_write_csv(self, sample_df, tmp_output):
        sink = LocalSink()
        paths = sink.push(sample_df, str(tmp_output / "csv_out"), "csv", 250)
        assert all(p.endswith(".csv") for p in paths)

    def test_write_json(self, sample_df, tmp_output):
        sink = LocalSink()
        paths = sink.push(sample_df, str(tmp_output / "json_out"), "json", 250)
        assert all(p.endswith(".json") for p in paths)

    def test_batching(self, tmp_output):
        df = pl.DataFrame({"x": list(range(100))})
        sink = LocalSink()
        paths = sink.push(df, str(tmp_output / "batch"), "parquet", 30)
        assert len(paths) == 4  # ceil(100/30) = 4

    def test_partitioning(self, tmp_output):
        df = pl.DataFrame({"region": ["US", "EU", "US", "EU"], "val": [1, 2, 3, 4]})
        sink = LocalSink()
        paths = sink.push(df, str(tmp_output / "part"), "parquet", 250, partitions=["region"])
        assert len(paths) >= 2

    def test_path_traversal_blocked(self, tmp_output):
        sink = LocalSink()
        df = pl.DataFrame({"x": [1]})
        with pytest.raises(ConfigError):
            sink.push(df, "/etc/passwd", "csv", 250)

    def test_roundtrip_csv(self, sample_df, tmp_output):
        sink = LocalSink()
        paths = sink.push(sample_df, str(tmp_output / "rt"), "csv", 250)
        read_back = pl.read_csv(paths[0])
        assert set(read_back.columns) == set(sample_df.columns)
        assert len(read_back) == len(sample_df)


class TestS3Sink:

    def test_missing_bucket_raises(self):
        with pytest.raises(SinkError):
            S3Sink(bucket="")


class TestGetSink:

    def test_local(self):
        sink = get_sink("local")
        assert isinstance(sink, LocalSink)

    def test_unknown_raises(self):
        with pytest.raises(SinkError):
            get_sink("gcs")
