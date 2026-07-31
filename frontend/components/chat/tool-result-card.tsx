"use client";

import type { ToolResult } from "@/lib/chat-types";
import { Database, Shield, BarChart3, Lightbulb, Activity, Download } from "lucide-react";

interface ToolResultCardProps {
  toolResult: ToolResult;
}

export function ToolResultCard({ toolResult }: ToolResultCardProps) {
  const { tool, result } = toolResult;

  switch (tool) {
    case "generate_schema":
      return <SchemaCard result={result} />;
    case "generate_data":
      return <GenerationCard result={result} />;
    case "run_privacy_audit":
      return <PrivacyCard result={result} />;
    case "run_quality_check":
      return <QualityCard result={result} />;
    case "profile_data":
      return <ProfileCard result={result} />;
    case "suggest_improvements":
      return <SuggestionsCard result={result} />;
    default:
      return <GenericCard tool={tool} result={result} />;
  }
}

function SchemaCard({ result }: { result: Record<string, unknown> }) {
  const schema = (result.schema || {}) as Record<string, string>;
  const descriptions = (result.field_descriptions || {}) as Record<string, string>;
  const columns = Object.entries(schema);

  if (result.error) return <ErrorCard message={result.error as string} />;

  return (
    <div className="my-2 rounded-xl bg-white/70 border border-black/[0.06] shadow-sm overflow-hidden">
      <div className="flex items-center gap-2 px-3 py-2 border-b border-black/[0.06] bg-[#007AFF]/[0.04]">
        <Database className="size-3.5 text-[#007AFF]" />
        <span className="text-[11px] font-semibold text-[#007AFF]">
          Schema ({columns.length} columns)
        </span>
      </div>
      <div className="divide-y divide-black/[0.04]">
        {columns.map(([name, type]) => (
          <div key={name} className="flex items-center justify-between px-3 py-1.5">
            <div className="flex flex-col">
              <span className="text-[12px] font-medium text-[#1D1D1F]">{name}</span>
              {descriptions[name] && (
                <span className="text-[10px] text-[#86868B]">{descriptions[name]}</span>
              )}
            </div>
            <TypeBadge type={type} />
          </div>
        ))}
      </div>
    </div>
  );
}

function GenerationCard({ result }: { result: Record<string, unknown> }) {
  if (result.error) return <ErrorCard message={result.error as string} />;

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8100";
  const sessionId = result.session_id as string | undefined;
  const dataKey = (result.data_key as string) || "generated";
  const formats = (result.download_formats as string[]) || ["csv", "json"];
  const preview = result.preview as Record<string, unknown>[] | undefined;
  const columns = preview && preview.length > 0 ? Object.keys(preview[0]) : [];
  const recordCount = Number(result.record_count || 0);
  const outputFormat = String(result.format || "csv");
  const qualityGrade = result.quality_grade ? String(result.quality_grade) : null;
  const qualityScore = result.quality_score !== undefined ? Number(result.quality_score) : null;

  return (
    <div className="my-2 rounded-xl bg-white/70 border border-black/[0.06] shadow-sm overflow-hidden">
      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <Activity className="size-3.5 text-[#34C759]" />
          <span className="text-[11px] font-semibold text-[#34C759]">Generation Complete</span>
          <span className="ml-auto text-[10px] text-[#86868B]">
            {recordCount} records
          </span>
        </div>
        <div className="grid grid-cols-2 gap-2 text-[11px]">
          <MetricRow label="Format" value={outputFormat.toUpperCase()} />
          {qualityGrade ? (
            <MetricRow label="Quality" value={`${qualityGrade} (${qualityScore}/100)`} />
          ) : null}
        </div>
      </div>

      {preview && preview.length > 0 && (
        <div className="border-t border-black/[0.06] overflow-x-auto">
          <table className="w-full text-[10px]">
            <thead>
              <tr className="bg-black/[0.02]">
                {columns.map((col) => (
                  <th
                    key={col}
                    className="px-2 py-1.5 text-left font-semibold text-[#86868B] whitespace-nowrap"
                  >
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {preview.map((row, i) => (
                <tr
                  key={`row-${i}`}
                  className={i % 2 === 0 ? "" : "bg-black/[0.015]"}
                >
                  {columns.map((col) => (
                    <td
                      key={`${i}-${col}`}
                      className="px-2 py-1 text-[#3A3A3C] whitespace-nowrap max-w-[120px] truncate"
                      title={String(row[col] as string ?? "")}
                    >
                      {String(row[col] as string ?? "")}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <div className="px-2 py-1 text-[9px] text-[#86868B] bg-black/[0.02] border-t border-black/[0.04]">
            Showing {preview.length} of {String(result.record_count || 0)} rows
          </div>
        </div>
      )}

      {sessionId && (
        <div className="flex items-center gap-1.5 px-3 py-2.5 border-t border-black/[0.06]">
          <Download className="size-3 text-[#86868B]" />
          <span className="text-[10px] text-[#86868B] mr-1">Download:</span>
          {formats.map((fmt) => (
            <a
              key={fmt}
              href={`${API_URL}/api/v1/chat/download/${sessionId}/${dataKey}?format=${fmt}`}
              download
              className="text-[10px] font-medium text-[#007AFF] hover:text-[#0056CC] px-1.5 py-0.5 rounded-md bg-[#007AFF]/[0.06] hover:bg-[#007AFF]/[0.12] transition-colors"
            >
              {fmt.toUpperCase()}
            </a>
          ))}
        </div>
      )}
    </div>
  );
}

function PrivacyCard({ result }: { result: Record<string, unknown> }) {
  if (result.error) return <ErrorCard message={result.error as string} />;

  const riskLevel = String(result.risk_level || "Unknown");
  const riskColor =
    riskLevel === "Low" ? "#34C759" : riskLevel === "Medium" ? "#FF9F0A" : "#FF3B30";

  return (
    <div className="my-2 rounded-xl bg-white/70 border border-black/[0.06] shadow-sm p-3">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Shield className="size-3.5" style={{ color: riskColor }} />
          <span className="text-[11px] font-semibold" style={{ color: riskColor }}>
            Privacy Audit
          </span>
        </div>
        <span
          className="text-[10px] font-semibold px-2 py-0.5 rounded-full"
          style={{
            backgroundColor: `${riskColor}15`,
            color: riskColor,
          }}
        >
          {riskLevel} Risk
        </span>
      </div>
      <div className="grid grid-cols-2 gap-2 text-[11px]">
        <MetricRow label="Min DCR" value={String(result.min_dcr || 0)} />
        <MetricRow label="Mean DCR" value={String(result.mean_dcr || 0)} />
        <MetricRow label="Median DCR" value={String(result.median_dcr || 0)} />
        <MetricRow label="Exact Matches" value={`${result.pct_exact_matches || 0}%`} />
      </div>
    </div>
  );
}

function QualityCard({ result }: { result: Record<string, unknown> }) {
  if (result.error) return <ErrorCard message={result.error as string} />;

  const score = Number(result.overall_score || 0);
  const grade = String(result.grade || "N/A");
  const gradeColor =
    grade === "A" ? "#34C759" : grade === "B" ? "#007AFF" : grade === "C" ? "#FF9F0A" : "#FF3B30";

  return (
    <div className="my-2 rounded-xl bg-white/70 border border-black/[0.06] shadow-sm p-3">
      <div className="flex items-center gap-2 mb-2">
        <BarChart3 className="size-3.5 text-[#007AFF]" />
        <span className="text-[11px] font-semibold text-[#007AFF]">Quality Assessment</span>
        <span
          className="ml-auto text-[14px] font-bold"
          style={{ color: gradeColor }}
        >
          {grade}
        </span>
        <span className="text-[10px] text-[#86868B]">{score}/100</span>
      </div>
      <div className="grid grid-cols-2 gap-2 text-[11px]">
        <MetricRow label="Completeness" value={`${Math.round(Number(result.completeness || 0) * 100)}%`} />
        <MetricRow label="Uniqueness" value={`${Math.round(Number(result.uniqueness || 0) * 100)}%`} />
        <MetricRow label="Distribution" value={`${Math.round(Number(result.distribution_score || 0) * 100)}%`} />
        <MetricRow label="Correlation" value={`${Math.round(Number(result.correlation_preservation || 0) * 100)}%`} />
      </div>
    </div>
  );
}

function ProfileCard({ result }: { result: Record<string, unknown> }) {
  if (result.error) return <ErrorCard message={result.error as string} />;

  return (
    <div className="my-2 rounded-xl bg-white/70 border border-black/[0.06] shadow-sm p-3">
      <div className="flex items-center gap-2 mb-2">
        <BarChart3 className="size-3.5 text-[#AF82FF]" />
        <span className="text-[11px] font-semibold text-[#AF82FF]">Data Profile</span>
      </div>
      <div className="text-[11px] text-[#3A3A3C]">
        <MetricRow label="Rows" value={String(result.row_count || 0)} />
        {Array.isArray(result.key_correlations) ? (
          <div className="mt-1.5">
            <span className="text-[10px] font-medium text-[#86868B]">Key Correlations:</span>
            {(result.key_correlations as string[]).slice(0, 3).map((c, i) => (
              <div key={`corr-${i}`} className="text-[10px] text-[#3A3A3C] ml-2">• {c}</div>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function SuggestionsCard({ result }: { result: Record<string, unknown> }) {
  if (result.error) return <ErrorCard message={result.error as string} />;

  const suggestions = (result.suggestions || []) as string[];

  return (
    <div className="my-2 rounded-xl bg-white/70 border border-black/[0.06] shadow-sm p-3">
      <div className="flex items-center gap-2 mb-2">
        <Lightbulb className="size-3.5 text-[#FF9F0A]" />
        <span className="text-[11px] font-semibold text-[#FF9F0A]">Suggestions</span>
      </div>
      <div className="space-y-1">
        {suggestions.map((s, i) => (
          <div key={i} className="text-[11px] text-[#3A3A3C] flex gap-1.5">
            <span className="text-[#86868B] shrink-0">{i + 1}.</span>
            <span>{s}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function GenericCard({ tool, result }: { tool: string; result: Record<string, unknown> }) {
  return (
    <div className="my-2 rounded-xl bg-white/70 border border-black/[0.06] shadow-sm p-3">
      <span className="text-[10px] font-medium text-[#86868B]">{tool}</span>
      <pre className="text-[10px] mt-1 overflow-auto max-h-24 text-[#3A3A3C]">
        {JSON.stringify(result, null, 2)}
      </pre>
    </div>
  );
}

function ErrorCard({ message }: { message: string }) {
  return (
    <div className="my-2 rounded-xl bg-[#FF3B30]/[0.06] border border-[#FF3B30]/20 p-3">
      <span className="text-[11px] font-medium text-[#FF3B30]">{message}</span>
    </div>
  );
}

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-[#86868B]">{label}</span>
      <span className="font-medium text-[#1D1D1F]">{value}</span>
    </div>
  );
}

function TypeBadge({ type }: { type: string }) {
  const colors: Record<string, string> = {
    Int64: "#007AFF",
    Float64: "#5AC8FA",
    String: "#AF82FF",
    Date: "#FF9F0A",
    Boolean: "#34C759",
  };
  const color = colors[type] || "#86868B";

  return (
    <span
      className="text-[10px] font-medium px-1.5 py-0.5 rounded"
      style={{ backgroundColor: `${color}15`, color }}
    >
      {type}
    </span>
  );
}
