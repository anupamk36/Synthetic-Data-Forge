"use client";

import { useState } from "react";
import {
  Download,
  Cloud,
  Database,
  Loader2,
  CheckCircle2,
  HardDrive,
} from "lucide-react";
import { toast } from "sonner";

import { GlassCard } from "@/components/shared/glass-card";
import { exportData } from "@/lib/api";

interface ExportPanelProps {
  data: Record<string, unknown>[] | unknown;
  schema?: Record<string, string>;
  filename?: string;
}

const FORMATS = [
  { value: "csv", label: "CSV" },
  { value: "json", label: "JSON" },
  { value: "parquet", label: "Parquet" },
  { value: "ndjson", label: "NDJSON" },
] as const;

const DESTINATIONS = [
  { value: "download", label: "Download", icon: Download },
  { value: "s3", label: "Amazon S3", icon: Cloud },
  { value: "redshift", label: "Redshift", icon: Database },
] as const;

function flattenData(data: unknown): Record<string, unknown>[] {
  if (Array.isArray(data)) return data;
  if (typeof data === "object" && data !== null) {
    const obj = data as Record<string, unknown>;
    const allRows: Record<string, unknown>[] = [];
    for (const val of Object.values(obj)) {
      if (Array.isArray(val)) {
        allRows.push(...val.filter((r): r is Record<string, unknown> => typeof r === "object" && r !== null));
      }
    }
    if (allRows.length > 0) return allRows;
    return [obj];
  }
  return [];
}

export function ExportPanel({ data, schema, filename = "export" }: ExportPanelProps) {
  const [format, setFormat] = useState("csv");
  const [destination, setDestination] = useState("download");
  const [exporting, setExporting] = useState(false);
  const [exported, setExported] = useState(false);

  const [s3Bucket, setS3Bucket] = useState("");
  const [s3Prefix, setS3Prefix] = useState("");
  const [s3Region, setS3Region] = useState("us-east-1");
  const [s3AccessKey, setS3AccessKey] = useState("");
  const [s3SecretKey, setS3SecretKey] = useState("");

  const [partitionCols, setPartitionCols] = useState<string[]>([]);
  const [recordsPerFile, setRecordsPerFile] = useState(1000);

  const columns = schema ? Object.keys(schema) : [];
  const rowCount = flattenData(data).length;

  const togglePartition = (col: string) => {
    setPartitionCols((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );
  };

  const handleExport = async () => {
    setExporting(true);
    setExported(false);
    const rows = flattenData(data);
    try {
      if (destination === "download") {
        let content: string;
        let mimeType: string;
        let ext: string;

        if (format === "json") {
          content = JSON.stringify(rows, null, 2);
          mimeType = "application/json";
          ext = "json";
        } else if (format === "csv") {
          if (rows.length === 0) throw new Error("No data to export");
          const cols = Object.keys(rows[0]).filter((k) => !k.startsWith("_"));
          const header = cols.join(",");
          const lines = rows.map((row) =>
            cols.map((c) => {
              const val = row[c];
              if (val === null || val === undefined) return "";
              const str = String(val);
              return str.includes(",") || str.includes('"') || str.includes("\n")
                ? `"${str.replace(/"/g, '""')}"`
                : str;
            }).join(",")
          );
          content = [header, ...lines].join("\n");
          mimeType = "text/csv";
          ext = "csv";
        } else if (format === "ndjson") {
          content = rows.map((r) => JSON.stringify(r)).join("\n");
          mimeType = "application/x-ndjson";
          ext = "ndjson";
        } else {
          content = JSON.stringify(rows, null, 2);
          mimeType = "application/json";
          ext = "json";
          toast.info("Parquet requires server-side export — downloading as JSON");
        }

        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${filename}.${ext}`;
        a.click();
        URL.revokeObjectURL(url);
        toast.success(`Downloaded ${rows.length} rows as ${ext.toUpperCase()}`);
      } else if (destination === "s3") {
        if (!s3Bucket) {
          toast.error("S3 bucket name is required");
          return;
        }
        await exportData({
          data: rows,
          sink_type: "s3",
          output_format: format === "ndjson" ? "json" : format,
          output_path: s3Prefix || filename,
          records_per_file: recordsPerFile,
          partition_on: partitionCols.length > 0 ? partitionCols : undefined,
          s3_bucket: s3Bucket,
          s3_prefix: s3Prefix,
          s3_region: s3Region,
          s3_access_key: s3AccessKey,
          s3_secret_key: s3SecretKey,
        });
        toast.success(`Exported to s3://${s3Bucket}/${s3Prefix || filename}`);
      }
      setExported(true);
      setTimeout(() => setExported(false), 3000);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Export failed");
    } finally {
      setExporting(false);
    }
  };

  return (
    <GlassCard>
      <div className="p-5 space-y-5">
        <div className="flex items-center gap-2">
          <Download className="size-5 text-[#007AFF]" />
          <h3 className="text-[15px] font-semibold text-[#1D1D1F]">Export Data</h3>
          <span className="text-[11px] text-[#86868B] ml-1">
            {rowCount} rows
          </span>
        </div>

        {/* Format picker */}
        <div className="space-y-1.5">
          <label className="text-[11px] font-semibold text-[#86868B] uppercase tracking-wide">Format</label>
          <div className="flex gap-1.5">
            {FORMATS.map((f) => (
              <button
                key={f.value}
                onClick={() => setFormat(f.value)}
                className={`flex-1 py-2 text-[12px] font-medium rounded-lg border transition-all ${
                  format === f.value
                    ? "border-[#007AFF] bg-[#007AFF]/8 text-[#007AFF]"
                    : "border-black/[0.08] text-[#3A3A3C] hover:bg-black/[0.03]"
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>
        </div>

        {/* Destination picker */}
        <div className="space-y-1.5">
          <label className="text-[11px] font-semibold text-[#86868B] uppercase tracking-wide">Destination</label>
          <div className="flex gap-1.5">
            {DESTINATIONS.map((d) => {
              const Icon = d.icon;
              const isRedshift = d.value === "redshift";
              return (
                <button
                  key={d.value}
                  onClick={() => !isRedshift && setDestination(d.value)}
                  className={`flex-1 flex items-center justify-center gap-1.5 py-2 text-[12px] font-medium rounded-lg border transition-all ${
                    destination === d.value
                      ? "border-[#007AFF] bg-[#007AFF]/8 text-[#007AFF]"
                      : isRedshift
                      ? "border-black/[0.06] text-[#C7C7CC] cursor-not-allowed"
                      : "border-black/[0.08] text-[#3A3A3C] hover:bg-black/[0.03]"
                  }`}
                >
                  <Icon className="size-3.5" />
                  {d.label}
                  {isRedshift && (
                    <span className="text-[9px] px-1 py-0.5 rounded bg-[#FF9F0A]/15 text-[#FF9F0A] font-semibold">SOON</span>
                  )}
                </button>
              );
            })}
          </div>
        </div>

        {/* S3 config */}
        {destination === "s3" && (
          <div className="space-y-3 p-4 rounded-xl border border-black/[0.06] bg-white/40">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-[11px] font-medium text-[#86868B] mb-1">Bucket *</label>
                <input
                  type="text"
                  value={s3Bucket}
                  onChange={(e) => setS3Bucket(e.target.value)}
                  placeholder="my-data-bucket"
                  className="w-full px-3 py-2 text-[12px] rounded-lg border border-black/[0.08] bg-white/70 focus:outline-none focus:ring-2 focus:ring-[#007AFF]/30"
                />
              </div>
              <div>
                <label className="block text-[11px] font-medium text-[#86868B] mb-1">Prefix</label>
                <input
                  type="text"
                  value={s3Prefix}
                  onChange={(e) => setS3Prefix(e.target.value)}
                  placeholder="exports/test-data/"
                  className="w-full px-3 py-2 text-[12px] rounded-lg border border-black/[0.08] bg-white/70 focus:outline-none focus:ring-2 focus:ring-[#007AFF]/30"
                />
              </div>
              <div>
                <label className="block text-[11px] font-medium text-[#86868B] mb-1">Region</label>
                <input
                  type="text"
                  value={s3Region}
                  onChange={(e) => setS3Region(e.target.value)}
                  className="w-full px-3 py-2 text-[12px] rounded-lg border border-black/[0.08] bg-white/70 focus:outline-none focus:ring-2 focus:ring-[#007AFF]/30"
                />
              </div>
              <div>
                <label className="block text-[11px] font-medium text-[#86868B] mb-1">Access Key</label>
                <input
                  type="password"
                  value={s3AccessKey}
                  onChange={(e) => setS3AccessKey(e.target.value)}
                  placeholder="AKIA..."
                  className="w-full px-3 py-2 text-[12px] rounded-lg border border-black/[0.08] bg-white/70 focus:outline-none focus:ring-2 focus:ring-[#007AFF]/30"
                />
              </div>
            </div>
            <div>
              <label className="block text-[11px] font-medium text-[#86868B] mb-1">Secret Key</label>
              <input
                type="password"
                value={s3SecretKey}
                onChange={(e) => setS3SecretKey(e.target.value)}
                className="w-full px-3 py-2 text-[12px] rounded-lg border border-black/[0.08] bg-white/70 focus:outline-none focus:ring-2 focus:ring-[#007AFF]/30"
              />
            </div>
          </div>
        )}

        {/* Partition options */}
        {columns.length > 0 && destination !== "download" && (
          <div className="space-y-1.5">
            <label className="text-[11px] font-semibold text-[#86868B] uppercase tracking-wide">
              Partition by (optional)
            </label>
            <div className="flex flex-wrap gap-1.5">
              {columns.map((col) => (
                <button
                  key={col}
                  onClick={() => togglePartition(col)}
                  className={`px-2.5 py-1 text-[11px] font-medium rounded-md border transition-all ${
                    partitionCols.includes(col)
                      ? "border-[#007AFF] bg-[#007AFF]/10 text-[#007AFF]"
                      : "border-black/[0.08] text-[#86868B] hover:text-[#3A3A3C]"
                  }`}
                >
                  {col}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Records per file (S3 only) */}
        {destination === "s3" && (
          <div className="w-40">
            <label className="block text-[11px] font-medium text-[#86868B] mb-1">Records per file</label>
            <input
              type="number"
              min={1}
              max={100000}
              value={recordsPerFile}
              onChange={(e) => setRecordsPerFile(Number(e.target.value))}
              className="w-full px-3 py-2 text-[12px] rounded-lg border border-black/[0.08] bg-white/70 focus:outline-none focus:ring-2 focus:ring-[#007AFF]/30"
            />
          </div>
        )}

        {/* Export button */}
        <button
          onClick={handleExport}
          disabled={exporting || rowCount === 0 || destination === "redshift"}
          className={`w-full flex items-center justify-center gap-2 py-2.5 text-[13px] font-semibold rounded-lg transition-all disabled:opacity-40 disabled:cursor-not-allowed ${
            exported
              ? "bg-[#34C759] text-white"
              : "bg-[#007AFF] text-white hover:bg-[#0066DD]"
          }`}
        >
          {exporting ? (
            <Loader2 className="size-4 animate-spin" />
          ) : exported ? (
            <CheckCircle2 className="size-4" />
          ) : destination === "download" ? (
            <Download className="size-4" />
          ) : destination === "s3" ? (
            <Cloud className="size-4" />
          ) : (
            <HardDrive className="size-4" />
          )}
          {exporting
            ? "Exporting..."
            : exported
            ? "Exported!"
            : destination === "download"
            ? `Download ${format.toUpperCase()}`
            : destination === "s3"
            ? `Export to S3`
            : "Coming Soon"}
        </button>
      </div>
    </GlassCard>
  );
}
