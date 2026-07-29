"use client";

import { useState, useCallback, useMemo } from "react";
import {
  Loader2,
  ChevronDown,
  ChevronRight,
  Clock,
  Database,
  Cpu,
  AlertCircle,
  CheckCircle2,
  CircleDot,
  CirclePause,
  XCircle,
} from "lucide-react";

import { TopBar } from "@/components/layout/top-bar";
import { StatCard } from "@/components/shared/stat-card";
import { useHistory } from "@/hooks/use-history";

function statusConfig(status: string) {
  switch (status) {
    case "complete":
      return { bg: "bg-[#34C759]/10", text: "text-[#34C759]", icon: CheckCircle2, label: "Complete" };
    case "running":
      return { bg: "bg-[#007AFF]/10", text: "text-[#007AFF]", icon: CircleDot, label: "Running" };
    case "stopped":
      return { bg: "bg-[#FF9F0A]/10", text: "text-[#FF9F0A]", icon: CirclePause, label: "Stopped" };
    case "error":
      return { bg: "bg-[#FF3B30]/10", text: "text-[#FF3B30]", icon: XCircle, label: "Error" };
    default:
      return { bg: "bg-black/[0.04]", text: "text-[#86868B]", icon: CircleDot, label: status };
  }
}

function featureLabel(feature: string) {
  switch (feature) {
    case "single": return "Single Table";
    case "relational": return "Relational";
    default: return feature;
  }
}

export default function HistoryPage() {
  const [feature, setFeature] = useState<string>("all");
  const [limit, setLimit] = useState(50);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const { data: runs, isLoading } = useHistory(limit, feature === "all" ? undefined : feature);

  const summary = useMemo(() => {
    if (!runs) return { total: 0, records: 0, completed: 0, errors: 0 };
    return {
      total: runs.length,
      records: runs.reduce((acc, r) => acc + (r.record_count || 0), 0),
      completed: runs.filter((r) => r.status === "complete").length,
      errors: runs.filter((r) => r.status === "error").length,
    };
  }, [runs]);

  const toggleExpand = useCallback((id: string) => {
    setExpandedId((prev) => (prev === id ? null : id));
  }, []);

  return (
    <div className="flex flex-col h-full">
      <TopBar title="Generation History" />

      <div className="flex-1 overflow-y-auto relative z-10">
        <div className="px-6 py-6 max-w-5xl mx-auto space-y-6">
          {/* Filters */}
          <div className="flex flex-col sm:flex-row gap-3">
            <div className="flex-1">
              <span className="text-[11px] text-[#86868B] mb-1 block">Feature Type</span>
              <div className="flex rounded-lg border border-black/[0.08] overflow-hidden">
                {[
                  { value: "all", label: "All" },
                  { value: "single", label: "Single" },
                  { value: "relational", label: "Relational" },
                ].map((f) => (
                  <button
                    key={f.value}
                    onClick={() => setFeature(f.value)}
                    className={`flex-1 py-2 text-[12px] font-medium transition-all ${
                      feature === f.value
                        ? "bg-[#007AFF] text-white"
                        : "bg-white/80 text-[#3A3A3C] hover:bg-[#007AFF]/[0.05]"
                    }`}
                  >
                    {f.label}
                  </button>
                ))}
              </div>
            </div>
            <div className="w-28">
              <span className="text-[11px] text-[#86868B] mb-1 block">Limit</span>
              <input
                type="number"
                min={1}
                max={500}
                value={limit}
                onChange={(e) => setLimit(Math.max(1, Number.parseInt(e.target.value) || 50))}
                className="w-full rounded-lg border border-black/[0.08] bg-white/80 px-3 py-2 text-[13px] font-mono focus:border-[#007AFF] focus:ring-1 focus:ring-[#007AFF]/20 outline-none transition-all"
              />
            </div>
          </div>

          {/* Summary */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <StatCard label="Total Runs" value={summary.total} color="blue" />
            <StatCard label="Total Records" value={summary.records.toLocaleString()} color="default" delay={80} />
            <StatCard label="Completed" value={summary.completed} color="green" delay={160} />
            <StatCard label="Errors" value={summary.errors} color="red" delay={240} />
          </div>

          {/* Run list */}
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.5px] text-[#86868B] mb-3">
              Run History
            </p>

            {isLoading && (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="h-6 w-6 animate-spin text-[#007AFF]" />
              </div>
            )}
            {!isLoading && runs && runs.length > 0 && (
              <div className="space-y-2">
                {runs.map((run, i) => {
                  const badge = statusConfig(run.status);
                  const Icon = badge.icon;
                  const isExpanded = expandedId === run.id;

                  return (
                    <div
                      key={run.id}
                      className="rounded-xl border border-black/[0.06] bg-white/60 backdrop-blur-sm overflow-hidden animate-slide-up"
                      style={{ animationDelay: `${Math.min(i, 10) * 40}ms` }}
                    >
                      <button
                        onClick={() => toggleExpand(run.id)}
                        className="w-full px-4 py-3 flex items-center gap-3 text-left hover:bg-black/[0.01] transition-colors"
                      >
                        {isExpanded ? (
                          <ChevronDown className="h-3.5 w-3.5 text-[#86868B] flex-shrink-0" />
                        ) : (
                          <ChevronRight className="h-3.5 w-3.5 text-[#86868B] flex-shrink-0" />
                        )}

                        <span className={`inline-flex items-center gap-1 text-[10px] font-medium px-2 py-0.5 rounded-full ${badge.bg} ${badge.text}`}>
                          <Icon className="h-3 w-3" />
                          {badge.label}
                        </span>

                        <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-black/[0.03] text-[#3A3A3C]">
                          {featureLabel(run.feature)}
                        </span>

                        <span className="text-[11px] text-[#86868B] flex items-center gap-1">
                          <Database className="h-3 w-3" />
                          {run.record_count.toLocaleString()}
                        </span>

                        <span className="text-[11px] text-[#86868B] flex items-center gap-1">
                          <Cpu className="h-3 w-3" />
                          {run.engine}
                          {run.model_name && ` / ${run.model_name}`}
                        </span>

                        <span className="text-[11px] text-[#86868B] flex items-center gap-1 ml-auto">
                          <Clock className="h-3 w-3" />
                          {run.elapsed_sec.toFixed(1)}s
                        </span>

                        <span className="text-[10px] text-[#86868B] flex-shrink-0">
                          {new Date(run.created_at).toLocaleString()}
                        </span>
                      </button>

                      {isExpanded && (
                        <div className="px-4 pb-4 pt-1 border-t border-black/[0.04] space-y-3">
                          {run.schema && (
                            <div>
                              <p className="text-[10px] font-semibold uppercase tracking-[0.3px] text-[#86868B] mb-1">Schema</p>
                              <pre className="text-[11px] font-mono bg-black/[0.02] rounded-lg p-3 overflow-x-auto max-h-48 overflow-y-auto text-[#3A3A3C]">
                                {JSON.stringify(run.schema, null, 2)}
                              </pre>
                            </div>
                          )}

                          {run.settings && Object.keys(run.settings).length > 0 && (
                            <div>
                              <p className="text-[10px] font-semibold uppercase tracking-[0.3px] text-[#86868B] mb-1">Settings</p>
                              <pre className="text-[11px] font-mono bg-black/[0.02] rounded-lg p-3 overflow-x-auto max-h-48 overflow-y-auto text-[#3A3A3C]">
                                {JSON.stringify(run.settings, null, 2)}
                              </pre>
                            </div>
                          )}

                          {run.error_msg && (
                            <div className="rounded-lg border border-[#FF3B30]/20 bg-[#FF3B30]/[0.04] p-3">
                              <div className="flex items-center gap-1.5 mb-1">
                                <AlertCircle className="h-3.5 w-3.5 text-[#FF3B30]" />
                                <span className="text-[11px] font-semibold text-[#FF3B30]">Error</span>
                              </div>
                              <p className="text-[11px] font-mono text-[#3A3A3C]">{run.error_msg}</p>
                            </div>
                          )}

                          {run.output_path && (
                            <div>
                              <p className="text-[10px] font-semibold uppercase tracking-[0.3px] text-[#86868B] mb-1">Output</p>
                              <p className="text-[11px] font-mono text-[#86868B]">{run.output_path}</p>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
            {!isLoading && (!runs || runs.length === 0) && (
              <div className="text-center py-12 text-[13px] text-[#86868B]">
                No generation history found
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
