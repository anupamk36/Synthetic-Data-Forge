"use client";

import { cn } from "@/lib/utils";

interface ProgressBarProps {
  progress: number;
  recordsDone: number;
  totalRecords: number;
  elapsed: number;
  status: string;
}

function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  const s = (seconds % 60).toFixed(0);
  return `${m}m ${s}s`;
}

export function ProgressBar({
  progress,
  recordsDone,
  totalRecords,
  elapsed,
  status,
}: ProgressBarProps) {
  const pct = Math.min(Math.max(progress * 100, 0), 100);
  const recPerSec = elapsed > 0 ? Math.round(recordsDone / elapsed) : 0;
  const isRunning = status === "running";

  return (
    <div className="space-y-2">
      {/* Stats line */}
      <div className="flex items-center justify-between text-xs text-muted-foreground font-mono">
        <span className={cn(isRunning && "text-emerald-400")}>
          {isRunning ? "Generating..." : status === "complete" ? "Complete" : status}
        </span>
        <div className="flex items-center gap-3">
          <span>
            {recordsDone.toLocaleString()} / {totalRecords.toLocaleString()}
          </span>
          <span>{formatElapsed(elapsed)}</span>
          {isRunning && <span>{recPerSec.toLocaleString()} rec/s</span>}
        </div>
      </div>

      {/* Progress track */}
      <div className="relative h-2.5 w-full overflow-hidden rounded-full bg-muted">
        <div
          className={cn(
            "h-full rounded-full transition-all duration-300 ease-out gradient-primary",
            isRunning && "animate-pulse"
          )}
          style={{ width: `${pct}%` }}
        />
      </div>

      {/* Percentage */}
      <div className="flex justify-end">
        <span className="text-xs font-mono tabular-nums text-muted-foreground">
          {pct.toFixed(1)}%
        </span>
      </div>
    </div>
  );
}
