"use client";

import { useState, useCallback } from "react";
import { Upload, HardDrive, Cloud, Loader2, FolderOutput } from "lucide-react";
import { toast } from "sonner";
import { exportData } from "@/lib/api";
import { cn } from "@/lib/utils";

interface ExportPanelProps {
  data: Record<string, unknown>[];
  columns: string[];
}

export function ExportPanel({ data, columns }: ExportPanelProps) {
  const [sinkType, setSinkType] = useState<"local" | "s3">("local");
  const [outputPath, setOutputPath] = useState("./output_data");
  const [outputFormat, setOutputFormat] = useState("parquet");
  const [recordsPerFile, setRecordsPerFile] = useState(250);
  const [partitionCols, setPartitionCols] = useState<string[]>([]);
  const [s3Bucket, setS3Bucket] = useState("");
  const [s3Prefix, setS3Prefix] = useState("synthetic-data");
  const [s3Region, setS3Region] = useState("us-east-1");
  const [s3AccessKey, setS3AccessKey] = useState("");
  const [s3SecretKey, setS3SecretKey] = useState("");
  const [s3SessionToken, setS3SessionToken] = useState("");
  const [showCredentials, setShowCredentials] = useState(false);
  const [exporting, setExporting] = useState(false);

  const selectClass =
    "w-full h-8 rounded-md border border-input bg-background px-2 text-sm text-foreground outline-none focus:border-ring cursor-pointer appearance-none";

  const handleExport = useCallback(async () => {
    setExporting(true);
    try {
      const result = await exportData({
        data,
        sink_type: sinkType,
        output_path: sinkType === "local" ? outputPath : undefined,
        output_format: outputFormat,
        records_per_file: recordsPerFile,
        partition_on: partitionCols.length > 0 ? partitionCols : undefined,
        s3_bucket: sinkType === "s3" ? s3Bucket : undefined,
        s3_prefix: sinkType === "s3" ? s3Prefix : undefined,
        s3_region: sinkType === "s3" ? s3Region : undefined,
        s3_access_key: sinkType === "s3" && s3AccessKey ? s3AccessKey : undefined,
        s3_secret_key: sinkType === "s3" && s3SecretKey ? s3SecretKey : undefined,
        s3_session_token: sinkType === "s3" && s3SessionToken ? s3SessionToken : undefined,
      });
      toast.success(`Exported ${result.files_written.length} file(s)`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Export failed");
    } finally {
      setExporting(false);
    }
  }, [data, sinkType, outputPath, outputFormat, recordsPerFile, partitionCols, s3Bucket, s3Prefix, s3Region, s3AccessKey, s3SecretKey, s3SessionToken]);

  const togglePartition = (col: string) => {
    setPartitionCols((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );
  };

  return (
    <div className="rounded-xl border bg-card p-5 space-y-4">
      <div className="flex items-center gap-2 text-sm font-semibold">
        <FolderOutput className="size-4 text-emerald-400" />
        Save to Destination
      </div>

      {/* Sink type toggle */}
      <div className="flex gap-2">
        <button
          onClick={() => setSinkType("local")}
          className={cn(
            "flex-1 flex items-center justify-center gap-2 rounded-lg border py-2.5 text-sm font-medium transition-all",
            sinkType === "local"
              ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-400"
              : "border-border text-muted-foreground hover:text-foreground hover:border-border/80"
          )}
        >
          <HardDrive className="size-4" />
          Local Filesystem
        </button>
        <button
          onClick={() => setSinkType("s3")}
          className={cn(
            "flex-1 flex items-center justify-center gap-2 rounded-lg border py-2.5 text-sm font-medium transition-all",
            sinkType === "s3"
              ? "border-cyan-500/40 bg-cyan-500/10 text-cyan-400"
              : "border-border text-muted-foreground hover:text-foreground hover:border-border/80"
          )}
        >
          <Cloud className="size-4" />
          Amazon S3
        </button>
      </div>

      {/* Sink-specific settings */}
      {sinkType === "local" ? (
        <div>
          <label className="text-xs text-muted-foreground mb-1 block">Output Directory</label>
          <input
            type="text"
            value={outputPath}
            onChange={(e) => setOutputPath(e.target.value)}
            className={selectClass}
            placeholder="./output_data"
          />
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div>
            <label className="text-xs text-muted-foreground mb-1 block">S3 Bucket</label>
            <input
              type="text"
              value={s3Bucket}
              onChange={(e) => setS3Bucket(e.target.value)}
              className={selectClass}
              placeholder="my-bucket"
            />
          </div>
          <div>
            <label className="text-xs text-muted-foreground mb-1 block">S3 Prefix</label>
            <input
              type="text"
              value={s3Prefix}
              onChange={(e) => setS3Prefix(e.target.value)}
              className={selectClass}
              placeholder="synthetic-data"
            />
          </div>
          <div>
            <label className="text-xs text-muted-foreground mb-1 block">Region</label>
            <input
              type="text"
              value={s3Region}
              onChange={(e) => setS3Region(e.target.value)}
              className={selectClass}
              placeholder="us-east-1"
            />
          </div>

          {/* Credentials toggle */}
          <div className="col-span-1 md:col-span-3">
            <button
              type="button"
              onClick={() => setShowCredentials(!showCredentials)}
              className="text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              {showCredentials ? "▾ Hide" : "▸ Show"} AWS Credentials
              <span className="ml-1.5 text-[10px] text-muted-foreground/60">
                (leave blank to use env vars or IAM role)
              </span>
            </button>
          </div>

          {showCredentials && (
            <>
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Access Key ID</label>
                <input
                  type="password"
                  value={s3AccessKey}
                  onChange={(e) => setS3AccessKey(e.target.value)}
                  className={selectClass}
                  placeholder="AKIA..."
                />
              </div>
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Secret Access Key</label>
                <input
                  type="password"
                  value={s3SecretKey}
                  onChange={(e) => setS3SecretKey(e.target.value)}
                  className={selectClass}
                  placeholder="••••••••"
                />
              </div>
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Session Token (optional)</label>
                <input
                  type="password"
                  value={s3SessionToken}
                  onChange={(e) => setS3SessionToken(e.target.value)}
                  className={selectClass}
                  placeholder="Optional"
                />
              </div>
            </>
          )}
        </div>
      )}

      {/* Format + records per file */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="text-xs text-muted-foreground mb-1 block">Format</label>
          <select
            value={outputFormat}
            onChange={(e) => setOutputFormat(e.target.value)}
            className={selectClass}
          >
            <option value="parquet">Parquet</option>
            <option value="csv">CSV</option>
            <option value="json">JSON</option>
          </select>
        </div>
        <div>
          <label className="text-xs text-muted-foreground mb-1 block">Records per File</label>
          <input
            type="number"
            min={1}
            value={recordsPerFile}
            onChange={(e) => setRecordsPerFile(Math.max(1, parseInt(e.target.value) || 1))}
            className={cn(selectClass, "font-mono")}
          />
        </div>
      </div>

      {/* Partition columns */}
      {columns.length > 0 && (
        <div>
          <label className="text-xs text-muted-foreground mb-2 block">
            Partition Columns (Hive-style)
          </label>
          <div className="flex flex-wrap gap-1.5">
            {columns.map((col) => (
              <button
                key={col}
                onClick={() => togglePartition(col)}
                className={cn(
                  "text-xs px-2.5 py-1 rounded-md border transition-all",
                  partitionCols.includes(col)
                    ? "border-emerald-500/40 bg-emerald-500/15 text-emerald-400"
                    : "border-border text-muted-foreground hover:text-foreground"
                )}
              >
                {col}
              </button>
            ))}
          </div>
          {partitionCols.length > 0 && (
            <p className="text-[10px] text-muted-foreground mt-1.5">
              Output: {partitionCols.map((c) => `${c}=*/`).join("")}part_0.{outputFormat}
            </p>
          )}
        </div>
      )}

      {/* Export button */}
      <button
        onClick={handleExport}
        disabled={exporting || (sinkType === "s3" && !s3Bucket)}
        className={cn(
          "w-full flex items-center justify-center gap-2 py-3 rounded-lg font-semibold text-sm transition-all",
          sinkType === "s3"
            ? "bg-cyan-600 hover:bg-cyan-500 text-white"
            : "glow-button gradient-primary text-primary-foreground"
        )}
      >
        {exporting ? (
          <Loader2 className="size-4 animate-spin" />
        ) : (
          <Upload className="size-4" />
        )}
        {exporting
          ? "Exporting..."
          : sinkType === "s3"
            ? `Export to s3://${s3Bucket || "..."}/${s3Prefix}`
            : `Export to ${outputPath}`}
      </button>
    </div>
  );
}
