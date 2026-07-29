"use client";

import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { Table2 } from "lucide-react";
import { cn } from "@/lib/utils";

export interface TableNodeData {
  label: string;
  columnCount: number;
  columns?: string[];
  expanded?: boolean;
  generating?: boolean;
  progress?: number;
  complete?: boolean;
}

function TableNodeComponent({ data, selected }: NodeProps) {
  const d = data as unknown as TableNodeData;

  return (
    <div
      className={cn(
        "relative rounded-2xl bg-white/80 backdrop-blur-sm border px-5 py-4 min-w-[160px] transition-all duration-300",
        selected
          ? "border-[#007AFF] shadow-[0_2px_8px_rgba(0,122,255,0.3)]"
          : "border-black/[0.08] shadow-[0_1px_3px_rgba(0,0,0,0.04)]",
        d.generating && "animate-pulse",
        d.complete && "border-[#34C759]/50"
      )}
    >
      <Handle type="target" position={Position.Top} className="!bg-[#007AFF] !w-2 !h-2 !border-white !border-2" />
      <Handle type="source" position={Position.Bottom} className="!bg-[#007AFF] !w-2 !h-2 !border-white !border-2" />

      {d.generating && d.progress != null && (
        <div className="absolute top-0 left-0 right-0 h-[3px] rounded-t-2xl overflow-hidden bg-[#007AFF]/10">
          <div
            className="h-full bg-[#007AFF] transition-all duration-500 ease-out"
            style={{ width: `${d.progress}%` }}
          />
        </div>
      )}

      <div className="flex items-center gap-2.5">
        <div className={cn(
          "flex items-center justify-center w-8 h-8 rounded-lg",
          d.complete ? "bg-[#34C759]/10" : "bg-[#007AFF]/10"
        )}>
          <Table2 className={cn(
            "w-4 h-4",
            d.complete ? "text-[#34C759]" : "text-[#007AFF]"
          )} />
        </div>
        <div>
          <p className="text-[13px] font-semibold text-[#1D1D1F] leading-tight">{d.label}</p>
          <p className="text-[10px] text-[#86868B]">{d.columnCount} columns</p>
        </div>
        {d.complete && (
          <span className="ml-auto text-[#34C759] text-sm">&#10003;</span>
        )}
      </div>

      {d.expanded && d.columns && d.columns.length > 0 && (
        <div className="mt-3 pt-3 border-t border-black/[0.06] space-y-1">
          {d.columns.map((col) => (
            <p key={col} className="text-[11px] text-[#3A3A3C] font-mono truncate">{col}</p>
          ))}
        </div>
      )}
    </div>
  );
}

export const TableNode = memo(TableNodeComponent);
