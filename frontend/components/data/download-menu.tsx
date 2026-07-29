"use client";

import { useCallback } from "react";
import { Download, FileJson, FileSpreadsheet } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

interface DownloadMenuProps {
  data: Record<string, unknown>[];
  filename?: string;
}

function downloadBlob(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function toCsv(data: Record<string, unknown>[]): string {
  if (data.length === 0) return "";
  const headers = Object.keys(data[0]);
  const headerLine = headers
    .map((h) => `"${h.replace(/"/g, '""')}"`)
    .join(",");
  const rows = data.map((row) =>
    headers
      .map((h) => {
        const val = row[h];
        if (val === null || val === undefined) return "";
        const str = String(val);
        return `"${str.replace(/"/g, '""')}"`;
      })
      .join(",")
  );
  return [headerLine, ...rows].join("\n");
}

export function DownloadMenu({
  data,
  filename = "synthetic_data",
}: DownloadMenuProps) {
  const handleCsv = useCallback(() => {
    downloadBlob(toCsv(data), `${filename}.csv`, "text/csv;charset=utf-8;");
  }, [data, filename]);

  const handleJson = useCallback(() => {
    downloadBlob(
      JSON.stringify(data, null, 2),
      `${filename}.json`,
      "application/json"
    );
  }, [data, filename]);

  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        className="inline-flex shrink-0 items-center justify-center gap-1.5 rounded-lg border border-input bg-transparent px-2.5 py-1.5 text-sm font-medium whitespace-nowrap transition-colors outline-none hover:bg-muted hover:text-foreground focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 dark:border-input dark:bg-input/30 dark:hover:bg-input/50"
        disabled={data.length === 0}
      >
        <Download className="size-4" />
        Download
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem onClick={handleCsv}>
          <FileSpreadsheet className="size-4 text-emerald-400" />
          Download CSV
        </DropdownMenuItem>
        <DropdownMenuItem onClick={handleJson}>
          <FileJson className="size-4 text-cyan-400" />
          Download JSON
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
