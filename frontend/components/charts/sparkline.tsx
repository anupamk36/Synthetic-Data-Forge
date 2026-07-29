import { cn } from "@/lib/utils";

interface SparklineProps {
  values: number[];
  color?: string;
}

export function Sparkline({ values, color = "bg-emerald-500" }: SparklineProps) {
  const max = Math.max(...values, 0.01);

  return (
    <div className="inline-flex items-end gap-px h-5">
      {values.map((v, i) => (
        <div
          key={i}
          className={cn("w-1.5 rounded-sm transition-all", color)}
          style={{ height: `${Math.max((v / max) * 100, 4)}%` }}
        />
      ))}
    </div>
  );
}
