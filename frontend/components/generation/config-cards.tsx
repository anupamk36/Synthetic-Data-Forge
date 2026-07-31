"use client";

import { cn } from "@/lib/utils";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Hash, FileOutput, ShieldCheck, DollarSign } from "lucide-react";

interface ConfigCardsProps {
  records: number;
  setRecords: (n: number) => void;
  format: string;
  setFormat: (f: string) => void;
  validationEnabled: boolean;
  setValidationEnabled: (v: boolean) => void;
  validationSample: number;
  setValidationSample: (n: number) => void;
  estimatedCost: number;
}

function ConfigCard({
  icon: Icon,
  label,
  children,
  className,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-col gap-3 rounded-xl border border-border/40 bg-card/60 p-4",
        className
      )}
    >
      <div className="flex items-center gap-2">
        <Icon className="size-4 text-muted-foreground" />
        <span className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
          {label}
        </span>
      </div>
      {children}
    </div>
  );
}

export function ConfigCards({
  records,
  setRecords,
  format,
  setFormat,
  validationEnabled,
  setValidationEnabled,
  validationSample,
  setValidationSample,
  estimatedCost,
}: ConfigCardsProps) {
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
      {/* Records */}
      <ConfigCard icon={Hash} label="Records">
        <Input
          type="number"
          min={1}
          max={1_000_000}
          value={records}
          onChange={(e) => setRecords(Number(e.target.value) || 1)}
          className="font-mono tabular-nums"
        />
      </ConfigCard>

      {/* Format */}
      <ConfigCard icon={FileOutput} label="Format">
        <Select value={format} onValueChange={(v) => v && setFormat(v)}>
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="parquet">Parquet</SelectItem>
            <SelectItem value="csv">CSV</SelectItem>
            <SelectItem value="json">JSON</SelectItem>
          </SelectContent>
        </Select>
      </ConfigCard>

      {/* LLM Validation */}
      <ConfigCard icon={ShieldCheck} label="Post-Gen Validation">
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">
            {validationEnabled ? "On" : "Off"}
          </span>
          <button
            type="button"
            role="switch"
            aria-checked={validationEnabled}
            onClick={() => setValidationEnabled(!validationEnabled)}
            className={cn(
              "relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors",
              validationEnabled ? "bg-emerald-500" : "bg-muted"
            )}
          >
            <span
              className={cn(
                "inline-block h-3.5 w-3.5 rounded-full bg-white transition-transform",
                validationEnabled ? "translate-x-[18px]" : "translate-x-[3px]"
              )}
            />
          </button>
        </div>
        <p className="text-[10px] text-muted-foreground/70 leading-tight">
          Sends rows to the LLM to fix cross-column inconsistencies. Turn off for faster, LLM-free generation.
        </p>
        {validationEnabled && (
          <div className="space-y-1.5 pt-1">
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">Sample %</span>
              <span className="text-xs font-mono text-foreground">
                {validationSample}%
              </span>
            </div>
            <input
              type="range"
              min={10}
              max={100}
              step={10}
              value={validationSample}
              onChange={(e) => setValidationSample(parseInt(e.target.value))}
              className="w-full accent-emerald-500 h-1.5"
            />
          </div>
        )}
      </ConfigCard>

      {/* Estimated Cost */}
      <ConfigCard icon={DollarSign} label="Est. Cost">
        <p className="text-2xl font-semibold font-mono tabular-nums text-emerald-400">
          ${estimatedCost.toFixed(4)}
        </p>
      </ConfigCard>
    </div>
  );
}
