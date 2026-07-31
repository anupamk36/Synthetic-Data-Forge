"use client";

import { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface DistributionChartProps {
  readonly realData?: number[];
  readonly syntheticData: number[];
  readonly columnName: string;
}

function binData(values: number[], binCount = 30) {
  if (values.length === 0) return [];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const binWidth = range / binCount;

  const bins = Array.from({ length: binCount }, (_, i) => ({
    bin: +(min + binWidth * (i + 0.5)).toPrecision(4),
    count: 0,
  }));

  for (const v of values) {
    const idx = Math.min(Math.floor((v - min) / binWidth), binCount - 1);
    bins[idx].count++;
  }

  return bins;
}

export function DistributionChart({
  realData,
  syntheticData,
  columnName,
}: DistributionChartProps) {
  const chartData = useMemo(() => {
    const synBins = binData(syntheticData);
    if (!realData || realData.length === 0) {
      return synBins.map((b) => ({
        bin: b.bin,
        synthetic: b.count,
      }));
    }

    const realBins = binData(realData);
    return synBins.map((b, i) => ({
      bin: b.bin,
      synthetic: b.count,
      real: realBins[i]?.count ?? 0,
    }));
  }, [realData, syntheticData]);

  return (
    <div className="w-full">
      <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.5px] text-[#86868B]">
        {columnName} Distribution
      </p>
      <ResponsiveContainer width="100%" height={240}>
        <BarChart
          data={chartData}
          margin={{ top: 8, right: 8, left: -10, bottom: 0 }}
        >
          <defs>
            <linearGradient id="distRealGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#AF82FF" stopOpacity={0.6} />
              <stop offset="100%" stopColor="#AF82FF" stopOpacity={0.2} />
            </linearGradient>
            <linearGradient id="distSyntheticGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#007AFF" stopOpacity={0.7} />
              <stop offset="100%" stopColor="#007AFF" stopOpacity={0.25} />
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
          />
          <Legend
            wrapperStyle={{ fontSize: "11px", color: "#86868B" }}
          />
          {realData && realData.length > 0 && (
            <Bar
              dataKey="real"
              name="Real"
              fill="url(#distRealGradient)"
              radius={[2, 2, 0, 0]}
            />
          )}
          <Bar
            dataKey="synthetic"
            name="Synthetic"
            fill="url(#distSyntheticGradient)"
            radius={[2, 2, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
