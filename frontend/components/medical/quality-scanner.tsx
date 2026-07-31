"use client";

import { useState, useCallback } from "react";
import {
  ShieldCheck,
  ShieldAlert,
  AlertTriangle,
  CheckCircle2,
  Loader2,
  Brain,
} from "lucide-react";
import { toast } from "sonner";

import { GlassCard } from "@/components/shared/glass-card";
import { StatCard } from "@/components/shared/stat-card";
import { scanMedicalData } from "@/lib/api";
import { useForgeStore } from "@/lib/store";

interface ScanResult {
  data_type: string;
  total_resources: number;
  resource_types: Record<string, number>;
  issues: { severity: string; category: string; resource_type: string; description: string; fix: string }[];
  issue_count: number;
  score: number;
  summary: { high: number; medium: number; low: number };
}

interface QualityScannerProps {
  data: unknown;
  dataType: "fhir" | "sdtm" | "dicom";
}

function scoreColor(score: number) {
  if (score >= 80) return "#34C759";
  if (score >= 60) return "#FF9F0A";
  return "#FF3B30";
}

function severityBadge(severity: string) {
  switch (severity) {
    case "high":
      return "bg-[#FF3B30]/10 text-[#FF3B30]";
    case "medium":
      return "bg-[#FF9F0A]/10 text-[#FF9F0A]";
    default:
      return "bg-[#34C759]/10 text-[#34C759]";
  }
}

export function QualityScanner({ data, dataType }: QualityScannerProps) {
  const { provider, apiKey, model } = useForgeStore();
  const [scanning, setScanning] = useState(false);
  const [result, setResult] = useState<ScanResult | null>(null);

  const handleScan = useCallback(async () => {
    setScanning(true);
    try {
      const report = await scanMedicalData(data, dataType, provider, apiKey, model ?? undefined);
      setResult(report);
      if (report.issue_count === 0) {
        toast.success("No quality issues found!");
      } else {
        toast.info(`Found ${report.issue_count} quality issue${report.issue_count > 1 ? "s" : ""}`);
      }
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Scan failed");
    } finally {
      setScanning(false);
    }
  }, [data, dataType, provider, apiKey, model]);

  return (
    <GlassCard>
      <div className="p-5 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Brain className="size-5 text-[#007AFF]" />
            <h3 className="text-[15px] font-semibold text-[#1D1D1F]">
              AI Quality Scanner
            </h3>
          </div>
          <button
            onClick={handleScan}
            disabled={scanning}
            className="flex items-center gap-2 px-4 py-2 bg-[#007AFF] text-white text-[13px] font-medium rounded-lg hover:bg-[#0066DD] disabled:opacity-50 transition-colors"
          >
            {scanning ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <ShieldCheck className="size-4" />
            )}
            {scanning ? "Scanning..." : "Scan for Quality Issues"}
          </button>
        </div>

        {result && (
          <div className="space-y-4 animate-in fade-in slide-in-from-bottom-2 duration-300">
            {/* Score + Summary */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              <div className="flex flex-col items-center justify-center p-4 rounded-xl border border-black/[0.06] bg-white/50">
                <div
                  className="text-[32px] font-bold"
                  style={{ color: scoreColor(result.score) }}
                >
                  {result.score}
                </div>
                <div className="text-[11px] text-[#86868B]">Quality Score</div>
              </div>
              <StatCard label="Resources" value={result.total_resources} />
              <StatCard label="Issues" value={result.issue_count} />
              <StatCard label="High" value={result.summary.high} />
              <StatCard label="Medium" value={result.summary.medium} />
            </div>

            {/* Issues list */}
            {result.issues.length > 0 ? (
              <div className="space-y-2">
                <h4 className="text-[13px] font-semibold text-[#1D1D1F]">
                  Issues Found ({result.issue_count})
                </h4>
                <div className="max-h-[300px] overflow-y-auto space-y-1.5">
                  {result.issues.map((issue, i) => (
                    <div
                      key={i}
                      className="flex items-start gap-3 p-3 rounded-lg border border-black/[0.05] bg-white/40"
                    >
                      {issue.severity === "high" ? (
                        <ShieldAlert className="size-4 shrink-0 text-[#FF3B30] mt-0.5" />
                      ) : issue.severity === "medium" ? (
                        <AlertTriangle className="size-4 shrink-0 text-[#FF9F0A] mt-0.5" />
                      ) : (
                        <CheckCircle2 className="size-4 shrink-0 text-[#34C759] mt-0.5" />
                      )}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-0.5">
                          <span
                            className={`text-[10px] font-semibold px-1.5 py-0.5 rounded-full ${severityBadge(issue.severity)}`}
                          >
                            {issue.severity.toUpperCase()}
                          </span>
                          <span className="text-[10px] text-[#86868B]">
                            {issue.resource_type} / {issue.category}
                          </span>
                        </div>
                        <p className="text-[12px] text-[#1D1D1F]">{issue.description}</p>
                        <p className="text-[11px] text-[#007AFF] mt-0.5">Fix: {issue.fix}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="flex items-center gap-2 p-4 rounded-xl bg-[#34C759]/10 border border-[#34C759]/20">
                <CheckCircle2 className="size-5 text-[#34C759]" />
                <span className="text-[13px] font-medium text-[#34C759]">
                  All quality checks passed — no issues found
                </span>
              </div>
            )}
          </div>
        )}
      </div>
    </GlassCard>
  );
}
