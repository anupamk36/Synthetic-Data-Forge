"use client";

import { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

interface DcrHistogramProps {
  readonly values: number[];
  readonly threshold?: number;
}

function binValues(values: number[], binCount = 30) {
  if (values.length === 0) return [];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const binWidth = range / binCount;

  return Array.from({ length: binCount }, (_, i) => {
    const lo = min + binWidth * i;
    const hi = lo + binWidth;
    const count = values.filter((v) => v >= lo && (i === binCount - 1 ? v <= hi : v < hi)).length;
    return {
      bin: +(lo + binWidth / 2).toPrecision(4),
      label: `${lo.toFixed(2)}-${hi.toFixed(2)}`,
      count,
    };
  });
}

export function DcrHistogram({ values, threshold }: DcrHistogramProps) {
  const chartData = useMemo(() => binValues(values), [values]);

  if (values.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-[13px] text-[#86868B]">
        No DCR values to display
      </div>
    );
  }

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={240}>
        <BarChart
          data={chartData}
          margin={{ top: 8, right: 8, left: -10, bottom: 0 }}
        >
          <defs>
            <linearGradient id="dcrGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#007AFF" stopOpacity={0.8} />
              <stop offset="100%" stopColor="#007AFF" stopOpacity={0.3} />
            </linearGradient>
          </defs>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="rgba(0,0,0,0.04)"
            vertical={false}
          />
          <XAxis
            dataKey="bin"
            tick={{ fill: "#86868B", fontSize: 10 }}
            axisLine={{ stroke: "rgba(0,0,0,0.06)" }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: "#86868B", fontSize: 10 }}
            axisLine={{ stroke: "rgba(0,0,0,0.06)" }}
            tickLine={false}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "rgba(255,255,255,0.95)",
              backdropFilter: "blur(12px)",
              border: "1px solid rgba(0,0,0,0.06)",
              borderRadius: "10px",
              fontSize: "12px",
              color: "#1D1D1F",
              boxShadow: "0 4px 16px rgba(0,0,0,0.08)",
            }}
            labelFormatter={(label) => `DCR: ${label}`}
          />
          <Bar
            dataKey="count"
            name="Frequency"
            fill="url(#dcrGradient)"
            radius={[3, 3, 0, 0]}
          />
          {threshold !== undefined && (
            <ReferenceLine
              x={threshold}
              stroke="#FF9F0A"
              strokeWidth={2}
              strokeDasharray="6 3"
              label={{
                value: `Threshold: ${threshold}`,
                position: "top",
                fill: "#FF9F0A",
                fontSize: 11,
              }}
            />
          )}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
