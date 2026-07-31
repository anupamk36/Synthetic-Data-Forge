"use client";

import { cn } from "@/lib/utils";

interface StatCardProps {
  readonly value: string | number;
  readonly label: string;
  readonly color?: "default" | "blue" | "green" | "amber" | "red";
  readonly className?: string;
  readonly children?: React.ReactNode;
  readonly delay?: number;
}

const VALUE_COLORS = {
  default: "text-[#1D1D1F]",
  blue: "text-[#007AFF]",
  green: "text-[#34C759]",
  amber: "text-[#FF9F0A]",
  red: "text-[#FF3B30]",
};

export function StatCard({ value, label, color = "default", className, children, delay = 0 }: StatCardProps) {
  return (
    <div
      className={cn(
        "flex-1 p-[14px_16px] glass-stat rounded-xl hover-lift cursor-default animate-count-up",
        className
      )}
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className={cn("text-[22px] font-bold tabular-nums", VALUE_COLORS[color])}>
        {value}
      </div>
      <div className="text-[11px] text-[#86868B] font-medium mt-[2px]">
        {label}
      </div>
      {children}
    </div>
  );
}
