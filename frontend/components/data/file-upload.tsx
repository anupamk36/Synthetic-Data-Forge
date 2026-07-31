"use client";

import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { FileText, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface FileUploadProps {
  readonly onFileAccepted?: (file: File) => void;
  readonly onFilesAccepted?: (files: File[]) => void;
  readonly multiple?: boolean;
  readonly accept?: string[];
  readonly loading?: boolean;
}

const DEFAULT_ACCEPT: Record<string, string[]> = {
  "text/csv": [".csv"],
  "application/json": [".json", ".jsonl"],
  "application/x-parquet": [".parquet"],
  "application/octet-stream": [".parquet"],
};

export function FileUpload({
  onFileAccepted,
  onFilesAccepted,
  multiple = false,
  accept,
  loading = false,
}: FileUploadProps) {
  const acceptMap = accept
    ? accept.reduce<Record<string, string[]>>((acc, ext) => {
        acc["application/octet-stream"] = [
          ...(acc["application/octet-stream"] || []),
          ext.startsWith(".") ? ext : `.${ext}`,
        ];
        return acc;
      }, {})
    : DEFAULT_ACCEPT;

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length === 0) return;
      if (multiple && onFilesAccepted) {
        onFilesAccepted(acceptedFiles);
      } else if (onFileAccepted) {
        onFileAccepted(acceptedFiles[0]);
      }
    },
    [onFileAccepted, onFilesAccepted, multiple]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: acceptMap,
    multiple,
    disabled: loading,
  });

  return (
    <div
      {...getRootProps()}
      className={cn(
        "relative flex flex-col items-center justify-center gap-3 rounded-[14px] border-2 border-dashed px-6 py-12 transition-all cursor-pointer overflow-hidden",
        isDragActive
          ? "border-[#007AFF]/40 bg-[#007AFF]/[0.03] scale-[1.005]"
          : "border-[#007AFF]/20 bg-[#007AFF]/[0.015] hover:border-[#007AFF]/40 hover:bg-[#007AFF]/[0.03] hover:scale-[1.005]",
        loading && "pointer-events-none opacity-60"
      )}
    >
      <input {...getInputProps()} />

      {/* Scanning light sweep */}
      {!loading && (
        <div
          className="absolute inset-[-2px] rounded-[14px] animate-scan pointer-events-none"
          style={{
            background: "linear-gradient(90deg, transparent, rgba(0,122,255,0.08), transparent)",
          }}
        />
      )}

      {loading ? (
        <Loader2 className="size-9 animate-spin text-[#007AFF]" />
      ) : (
        <FileText className="size-9 text-[#007AFF]/60 animate-bob relative z-10" />
      )}

      <div className="text-center relative z-10">
        <p className="text-[14px] font-medium text-[#1D1D1F]">
          {loading && "Processing..."}
          {!loading && isDragActive && `Drop your file${multiple ? "s" : ""} here`}
          {!loading && !isDragActive && (
            <>Drop file here or <span className="text-[#007AFF] font-semibold">browse</span></>
          )}
        </p>
        {!loading && !isDragActive && (
          <p className="mt-1 text-[12px] text-[#86868B]">
            CSV, Parquet, JSON, JSONL · Up to 200MB
          </p>
        )}
      </div>
    </div>
  );
}
