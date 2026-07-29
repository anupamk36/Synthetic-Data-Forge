"use client";

interface ProgressRingProps {
  readonly progress: number;
  readonly size?: number;
  readonly strokeWidth?: number;
  readonly label?: string;
  readonly hideValue?: boolean;
}

export function ProgressRing({ progress, size = 80, strokeWidth = 6, label, hideValue }: ProgressRingProps) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (progress / 100) * circumference;

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg width={size} height={size} className="-rotate-90">
        {/* Background track */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="rgba(0,0,0,0.06)"
          strokeWidth={strokeWidth}
        />
        {/* Progress arc */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#007AFF"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 0.5s ease" }}
        />
      </svg>
      {!hideValue && (
        <div className="absolute flex flex-col items-center">
          <span className="text-[16px] font-bold text-[#1D1D1F] tabular-nums">
            {Math.round(progress)}%
          </span>
          {label && (
            <span className="text-[9px] text-[#86868B] font-medium">{label}</span>
          )}
        </div>
      )}
    </div>
  );
}
