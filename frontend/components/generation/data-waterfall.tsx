"use client";

import { useRef, useEffect } from "react";
import { cn } from "@/lib/utils";

interface DataWaterfallProps {
  readonly data: Record<string, unknown>[];
  readonly schema: Record<string, string>;
  readonly maxRows?: number;
}

const TYPE_COLORS: Record<string, string> = {
  Int64: "text-[#007AFF]",
  Float64: "text-[#3A9CFF]",
  String: "text-[#AF82FF]",
  Date: "text-[#FF9F0A]",
  Boolean: "text-[#34C759]",
};

export function DataWaterfall({ data, schema, maxRows = 50 }: DataWaterfallProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const columns = Object.keys(schema);
  const visibleRows = data.slice(-maxRows);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [data.length]);

  return (
    <div className="rounded-xl border border-black/[0.06] overflow-hidden bg-white/50">
      {/* Header */}
      <div className="grid gap-0 border-b border-black/[0.06] bg-black/[0.02] px-4 py-2"
        style={{ gridTemplateColumns: `repeat(${Math.min(columns.length, 6)}, 1fr)` }}
      >
        {columns.slice(0, 6).map((col) => (
          <div key={col} className="text-[10px] font-semibold uppercase tracking-[0.3px] text-[#86868B] truncate px-1">
            {col}
          </div>
        ))}
      </div>

      {/* Rows */}
      <div ref={containerRef} className="max-h-[320px] overflow-y-auto">
        {visibleRows.map((row, i) => (
          <div
            key={i}
            className={cn(
              "grid gap-0 px-4 py-[6px] border-b border-black/[0.03] animate-slide-up",
              i % 2 === 0 ? "bg-transparent" : "bg-black/[0.01]"
            )}
            style={{
              gridTemplateColumns: `repeat(${Math.min(columns.length, 6)}, 1fr)`,
              animationDelay: `${Math.max(0, (i - visibleRows.length + 5)) * 50}ms`,
            }}
          >
            {columns.slice(0, 6).map((col) => {
              const dtype = schema[col] ?? "String";
              const colorClass = TYPE_COLORS[dtype] ?? "text-[#1D1D1F]";
              const val = row[col];

              return (
                <div
                  key={col}
                  className={cn("text-[12px] truncate px-1 tabular-nums", colorClass)}
                >
                  {val == null ? <span className="text-[#D1D1D6]">null</span> : String(val)}
                </div>
              );
            })}
          </div>
        ))}

        {visibleRows.length === 0 && (
          <div className="flex items-center justify-center py-12 text-[13px] text-[#86868B]">
            Waiting for data...
          </div>
        )}
      </div>

      {/* Footer */}
      {data.length > maxRows && (
        <div className="px-4 py-2 border-t border-black/[0.06] bg-black/[0.02] text-[10px] text-[#86868B]">
          Showing latest {maxRows} of {data.length.toLocaleString()} rows
        </div>
      )}
    </div>
  );
}
