"use client";

import { cn } from "@/lib/utils";

type StatusLevel = "low" | "medium" | "high";
type QualityGrade = "A" | "B" | "C" | "D" | "F";

interface StatusPillProps {
  readonly level?: StatusLevel;
  readonly grade?: QualityGrade;
  readonly label?: string;
  readonly size?: "sm" | "lg";
}

function getStyleFromLevel(level: StatusLevel) {
  switch (level) {
    case "low": return "status-low";
    case "medium": return "status-medium";
    case "high": return "status-high";
  }
}

function getStyleFromGrade(grade: QualityGrade) {
  if (grade === "A" || grade === "B") return "status-low";
  if (grade === "C") return "status-medium";
  return "status-high";
}

function getLabelFromLevel(level: StatusLevel) {
  switch (level) {
    case "low": return "Low Risk";
    case "medium": return "Medium Risk";
    case "high": return "High Risk";
  }
}

export function StatusPill({ level, grade, label, size = "sm" }: StatusPillProps) {
  const style = level ? getStyleFromLevel(level) : grade ? getStyleFromGrade(grade) : "status-low";
  const displayLabel = label ?? (level ? getLabelFromLevel(level) : grade ? `Grade ${grade}` : "");

  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full font-semibold animate-pop",
        style,
        size === "sm" ? "px-3 py-1 text-[12px]" : "px-5 py-2 text-[16px]"
      )}
    >
      {displayLabel}
    </span>
  );
}
