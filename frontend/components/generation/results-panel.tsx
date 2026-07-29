"use client";

import { CheckCircle2 } from "lucide-react";
import { DataTable } from "@/components/data/data-table";
import { DownloadMenu } from "@/components/data/download-menu";
import { StatCard } from "@/components/shared/stat-card";
import type { Schema } from "@/lib/types";

interface ResultsPanelProps {
  readonly data: Record<string, unknown>[];
  readonly schema: Schema;
  readonly elapsed: number;
  readonly status: string;
}

export function ResultsPanel({
  data,
  schema,
  elapsed,
  status,
}: ResultsPanelProps) {
  const columns = Object.keys(schema).length;
  const records = data.length;
  const recPerSec = elapsed > 0 ? (records / elapsed).toFixed(0) : "0";
  const isSuccess = status === "complete";

  return (
    <div className="space-y-5">
      {isSuccess && (
        <div className="flex items-center gap-3 rounded-xl border border-[#34C759]/20 bg-[#34C759]/[0.04] px-5 py-3">
          <CheckCircle2 className="w-5 h-5 text-[#34C759] shrink-0" />
          <p className="text-[13px] text-[#1D1D1F]">
            Generation complete — <span className="font-semibold">{records.toLocaleString()}</span> records in {elapsed.toFixed(1)}s
          </p>
        </div>
      )}

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatCard label="Records" value={records.toLocaleString()} color="blue" />
        <StatCard label="Columns" value={columns} color="default" delay={80} />
        <StatCard label="Time" value={`${elapsed.toFixed(1)}s`} color="green" delay={160} />
        <StatCard label="Rec/sec" value={recPerSec} color="amber" delay={240} />
      </div>

      {data.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <p className="text-[11px] font-semibold uppercase tracking-[0.5px] text-[#86868B]">
              Data Preview
            </p>
            <DownloadMenu data={data} />
          </div>
          <DataTable data={data} maxRows={20} />
        </div>
      )}
    </div>
  );
}
