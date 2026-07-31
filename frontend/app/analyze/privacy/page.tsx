"use client";

import { useState, useCallback, useMemo } from "react";
import { Loader2, ShieldCheck, ShieldAlert, ShieldX, Upload, Download, CheckCircle2, XCircle } from "lucide-react";
import { toast } from "sonner";

import { TopBar } from "@/components/layout/top-bar";
import { FileUpload } from "@/components/data/file-upload";
import { DataTable } from "@/components/data/data-table";
import { DcrHistogram } from "@/components/charts/dcr-histogram";
import { GlassCard } from "@/components/shared/glass-card";
import { StatCard } from "@/components/shared/stat-card";
import { StatusPill } from "@/components/shared/status-pill";

import { uploadFile, auditPrivacy, auditPrivacyFull } from "@/lib/api";
import { useForgeStore } from "@/lib/store";
import type { PrivacyResult, PrivacyReport } from "@/lib/types";

/* ── Color helpers for metric thresholds ── */
function kAnonColor(minK: number) {
  if (minK >= 5) return "green";
  if (minK >= 2) return "amber";
  return "red";
}

function lDiversityColor(minL: number) {
  if (minL >= 3) return "green";
  if (minL >= 2) return "amber";
  return "red";
}

function epsilonColor(eps: number) {
  if (eps <= 1) return "green";
  if (eps <= 5) return "amber";
  return "red";
}

export default function PrivacyPage() {
  /* ── Real data state ── */
  const [realData, setRealData] = useState<Record<string, unknown>[] | null>(null);
  const [realName, setRealName] = useState("");
  const [uploadingReal, setUploadingReal] = useState(false);

  /* ── Synthetic data state ── */
  const [syntheticData, setSyntheticData] = useState<Record<string, unknown>[] | null>(null);
  const [syntheticName, setSyntheticName] = useState("");
  const [uploadingSynthetic, setUploadingSynthetic] = useState(false);

  /* ── Audit state ── */
  const [auditing, setAuditing] = useState(false);
  const [result, setResult] = useState<PrivacyResult | null>(null);
  const [fullReport, setFullReport] = useState<PrivacyReport | null>(null);

  /* ── Compliance config state ── */
  const [quasiIdentifiers, setQuasiIdentifiers] = useState<Set<string>>(new Set());
  const [sensitiveColumn, setSensitiveColumn] = useState<string | null>(null);

  const store = useForgeStore();

  /* ── Derive columns from uploaded data ── */
  const columns = useMemo(() => {
    if (realData && realData.length > 0) {
      return Object.keys(realData[0]);
    }
    if (syntheticData && syntheticData.length > 0) {
      return Object.keys(syntheticData[0]);
    }
    return [];
  }, [realData, syntheticData]);

  /* ── Auto-suggest high-cardinality columns as quasi-identifiers ── */
  const highCardinalityColumns = useMemo(() => {
    if (!realData || realData.length === 0) return new Set<string>();
    const suggested = new Set<string>();
    for (const col of columns) {
      const uniqueValues = new Set(realData.map((row) => {
        const val = row[col];
        if (val === null || val === undefined) return "";
        if (typeof val === "object") return JSON.stringify(val);
        return String(val);
      }));
      if (uniqueValues.size > 10) {
        suggested.add(col);
      }
    }
    return suggested;
  }, [realData, columns]);

  /* ── Upload handlers ── */
  const handleUploadReal = useCallback(async (file: File) => {
    setUploadingReal(true);
    try {
      const res = await uploadFile(file);
      setRealData(res.sample_rows);
      setRealName(file.name);
      toast.success(`Real data loaded: ${file.name}`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploadingReal(false);
    }
  }, []);

  const handleUploadSynthetic = useCallback(async (file: File) => {
    setUploadingSynthetic(true);
    try {
      const res = await uploadFile(file);
      setSyntheticData(res.sample_rows);
      setSyntheticName(file.name);
      toast.success(`Synthetic data loaded: ${file.name}`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploadingSynthetic(false);
    }
  }, []);

  const handleUseLastGenerated = useCallback(() => {
    if (store.lastGeneratedData) {
      setSyntheticData(store.lastGeneratedData);
      setSyntheticName("Last Generated Data");
      toast.success("Loaded last generated data");
    }
  }, [store.lastGeneratedData]);

  /* ── Run basic DCR audit ── */
  const handleAudit = useCallback(async () => {
    if (!realData || !syntheticData) {
      toast.error("Upload both datasets first");
      return;
    }
    setAuditing(true);
    try {
      const res = await auditPrivacy(realData, syntheticData);
      setResult(res);
      if (res.error) {
        toast.error(res.error);
      } else {
        toast.success("Privacy audit complete");
      }
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Audit failed");
    } finally {
      setAuditing(false);
    }
  }, [realData, syntheticData]);

  /* ── Run full compliance audit ── */
  const handleFullAudit = useCallback(async () => {
    if (!realData || !syntheticData) {
      toast.error("Upload both datasets first");
      return;
    }
    setAuditing(true);
    try {
      const qiArray = quasiIdentifiers.size > 0 ? Array.from(quasiIdentifiers) : null;
      const res = await auditPrivacyFull(realData, syntheticData, qiArray, sensitiveColumn);
      setFullReport(res);
      setResult(res.dcr);
      toast.success("Full compliance audit complete");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Audit failed");
    } finally {
      setAuditing(false);
    }
  }, [realData, syntheticData, quasiIdentifiers, sensitiveColumn]);

  /* ── Toggle quasi-identifier ── */
  const toggleQuasiIdentifier = useCallback((col: string) => {
    setQuasiIdentifiers((prev) => {
      const next = new Set(prev);
      if (next.has(col)) {
        next.delete(col);
      } else {
        next.add(col);
      }
      return next;
    });
  }, []);

  /* ── Download compliance report as JSON ── */
  const handleDownloadReport = useCallback(() => {
    if (!fullReport) return;
    const blob = new Blob([JSON.stringify(fullReport, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "privacy_compliance_report.json";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }, [fullReport]);

  /* ── Risk config ── */
  const riskConfig = {
    Low: { icon: ShieldCheck, color: "text-[#34C759]", bg: "bg-[#34C759]/[0.06]", border: "border-[#34C759]/20" },
    Medium: { icon: ShieldAlert, color: "text-[#FF9F0A]", bg: "bg-[#FF9F0A]/[0.06]", border: "border-[#FF9F0A]/20" },
    High: { icon: ShieldX, color: "text-[#FF3B30]", bg: "bg-[#FF3B30]/[0.06]", border: "border-[#FF3B30]/20" },
  };

  let topBarStatus: "running" | "complete" | undefined;
  if (auditing) {
    topBarStatus = "running";
  } else if (result) {
    topBarStatus = "complete";
  }

  return (
    <div className="flex flex-col h-full">
      <TopBar title="Privacy Audit" status={topBarStatus} />

      <div className="flex-1 overflow-y-auto relative z-10">
        <div className="px-6 py-6 max-w-5xl mx-auto space-y-6">
          {/* Upload Section */}
          <GlassCard animatedBorder>
            <div className="p-7 space-y-5">
              <div>
                <h2 className="text-[16px] font-semibold text-[#1D1D1F]">Upload Datasets</h2>
                <p className="text-[13px] text-[#86868B] mt-1">Compare real data against synthetic to measure privacy risk (DCR analysis).</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Real data */}
                <div className="rounded-xl border border-black/[0.06] bg-white/60 p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <div className="h-2.5 w-2.5 rounded-full bg-[#AF82FF]" />
                    <span className="text-[13px] font-medium text-[#1D1D1F]">Real Data</span>
                    {realName && (
                      <span className="text-[11px] text-[#86868B] ml-auto truncate max-w-[120px]">{realName}</span>
                    )}
                  </div>
                  <FileUpload onFileAccepted={handleUploadReal} loading={uploadingReal} />
                  {realData && realData.length > 0 && (
                    <div className="mt-3">
                      <DataTable data={realData} maxRows={4} />
                    </div>
                  )}
                </div>

                {/* Synthetic data */}
                <div className="rounded-xl border border-black/[0.06] bg-white/60 p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <div className="h-2.5 w-2.5 rounded-full bg-[#007AFF]" />
                    <span className="text-[13px] font-medium text-[#1D1D1F]">Synthetic Data</span>
                    {syntheticName && (
                      <span className="text-[11px] text-[#86868B] ml-auto truncate max-w-[120px]">{syntheticName}</span>
                    )}
                  </div>
                  <FileUpload onFileAccepted={handleUploadSynthetic} loading={uploadingSynthetic} />
                  {store.lastGeneratedData && !syntheticData && (
                    <button
                      onClick={handleUseLastGenerated}
                      className="mt-2 w-full text-[11px] font-medium text-[#007AFF] hover:text-[#005EC4] transition-colors flex items-center justify-center gap-1"
                    >
                      <Upload className="h-3 w-3" />
                      Use last generated data
                    </button>
                  )}
                  {syntheticData && syntheticData.length > 0 && (
                    <div className="mt-3">
                      <DataTable data={syntheticData} maxRows={4} />
                    </div>
                  )}
                </div>
              </div>

              {/* Run button */}
              {realData && syntheticData && !result && (
                <button
                  onClick={handleAudit}
                  disabled={auditing}
                  className="w-full glow-button bg-[#007AFF] text-white font-semibold text-[13px] py-3 rounded-lg flex items-center justify-center gap-2 btn-shimmer disabled:opacity-40"
                >
                  {auditing ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <ShieldCheck className="h-4 w-4" />
                  )}
                  Run Privacy Audit
                </button>
              )}
            </div>
          </GlassCard>

          {/* Compliance Configuration Section */}
          {realData && syntheticData && !result && columns.length > 0 && (
            <GlassCard>
              <div className="p-7 space-y-5">
                <div>
                  <h2 className="text-[16px] font-semibold text-[#1D1D1F]">Full Compliance Audit</h2>
                  <p className="text-[13px] text-[#86868B] mt-1">Configure quasi-identifiers and sensitive columns for k-anonymity, l-diversity, and epsilon analysis.</p>
                </div>

                {/* Quasi-Identifier Selection */}
                <div className="space-y-3">
                  <div>
                    <h3 className="text-[13px] font-semibold text-[#1D1D1F]">Quasi-Identifiers</h3>
                    <p className="text-[11px] text-[#86868B] mt-0.5">
                      Select columns that could be combined to re-identify individuals. High-cardinality columns are auto-suggested.
                    </p>
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                    {columns.map((col) => {
                      const isHighCardinality = highCardinalityColumns.has(col);
                      const isSelected = quasiIdentifiers.has(col);
                      return (
                        <label
                          key={col}
                          className={`flex items-center gap-2 px-3 py-2 rounded-lg border cursor-pointer transition-all text-[12px] ${
                            isSelected
                              ? "border-[#007AFF]/40 bg-[#007AFF]/[0.06] text-[#007AFF]"
                              : "border-black/[0.06] bg-white/60 text-[#1D1D1F] hover:border-black/[0.12]"
                          }`}
                        >
                          <input
                            type="checkbox"
                            checked={isSelected}
                            onChange={() => toggleQuasiIdentifier(col)}
                            className="sr-only"
                          />
                          <div
                            className={`h-3.5 w-3.5 rounded border flex items-center justify-center flex-shrink-0 ${
                              isSelected
                                ? "bg-[#007AFF] border-[#007AFF]"
                                : "border-[#86868B]/40 bg-white"
                            }`}
                          >
                            {isSelected && (
                              <svg className="h-2.5 w-2.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                              </svg>
                            )}
                          </div>
                          <span className="truncate">{col}</span>
                          {isHighCardinality && (
                            <span className="ml-auto text-[9px] font-medium px-1.5 py-0.5 rounded-full bg-[#FF9F0A]/10 text-[#FF9F0A] flex-shrink-0">
                              suggested
                            </span>
                          )}
                        </label>
                      );
                    })}
                  </div>
                </div>

                {/* Sensitive Column Selection */}
                <div className="space-y-3">
                  <div>
                    <h3 className="text-[13px] font-semibold text-[#1D1D1F]">Sensitive Column</h3>
                    <p className="text-[11px] text-[#86868B] mt-0.5">
                      Select the column containing sensitive information for l-diversity analysis.
                    </p>
                  </div>
                  <select
                    value={sensitiveColumn ?? ""}
                    onChange={(e) => setSensitiveColumn(e.target.value || null)}
                    className="w-full md:w-1/2 px-3 py-2 rounded-lg border border-black/[0.06] bg-white/60 text-[12px] text-[#1D1D1F] focus:outline-none focus:ring-2 focus:ring-[#007AFF]/30 focus:border-[#007AFF]/40 transition-all"
                  >
                    <option value="">None selected</option>
                    {columns.map((col) => (
                      <option key={col} value={col}>
                        {col}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Run Full Compliance Audit button */}
                <button
                  onClick={handleFullAudit}
                  disabled={auditing}
                  className="w-full glow-button bg-[#34C759] text-white font-semibold text-[13px] py-3 rounded-lg flex items-center justify-center gap-2 btn-shimmer disabled:opacity-40"
                >
                  {auditing ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <ShieldCheck className="h-4 w-4" />
                  )}
                  Run Full Compliance Audit
                </button>
              </div>
            </GlassCard>
          )}

          {/* Loading */}
          {auditing && (
            <div className="flex flex-col items-center justify-center py-12">
              <Loader2 className="h-10 w-10 animate-spin text-[#007AFF] mb-4" />
              <p className="text-[14px] font-medium text-[#1D1D1F]">Analyzing privacy risks...</p>
              <p className="text-[12px] text-[#86868B] mt-1">Computing distance to closest record</p>
            </div>
          )}

          {/* Results */}
          {result && (
            <GlassCard>
              <div className="p-7 space-y-5">
                {/* Risk badge */}
                {(() => {
                  const cfg = riskConfig[result.risk_level];
                  const Icon = cfg.icon;
                  return (
                    <div className={`rounded-xl border ${cfg.border} ${cfg.bg} p-8 flex flex-col items-center justify-center`}>
                      <Icon className={`h-12 w-12 ${cfg.color} mb-3`} />
                      <StatusPill level={result.risk_level.toLowerCase() as "low" | "medium" | "high"} size="lg" />
                      <span className="text-[12px] text-[#86868B] mt-2">Privacy Risk Assessment</span>
                    </div>
                  );
                })()}

                {/* Compliance Badge */}
                {fullReport && (
                  <div className={`rounded-xl border p-4 flex items-center justify-center gap-3 ${
                    fullReport.compliant
                      ? "border-[#34C759]/20 bg-[#34C759]/[0.06]"
                      : "border-[#FF3B30]/20 bg-[#FF3B30]/[0.06]"
                  }`}>
                    {fullReport.compliant ? (
                      <CheckCircle2 className="h-6 w-6 text-[#34C759]" />
                    ) : (
                      <XCircle className="h-6 w-6 text-[#FF3B30]" />
                    )}
                    <span className={`text-[14px] font-semibold ${
                      fullReport.compliant ? "text-[#34C759]" : "text-[#FF3B30]"
                    }`}>
                      {fullReport.compliant ? "Compliant" : "Non-Compliant"}
                    </span>
                  </div>
                )}

                {/* Stat cards */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <StatCard label="Min DCR" value={result.min_dcr.toFixed(4)} color="blue" />
                  <StatCard label="Mean DCR" value={result.mean_dcr.toFixed(4)} color="default" delay={80} />
                  <StatCard label="Median DCR" value={result.median_dcr.toFixed(4)} color="default" delay={160} />
                  <StatCard label="Exact Matches" value={`${result.pct_exact_matches.toFixed(2)}%`} color="red" delay={240} />
                </div>

                {/* Full report additional metrics */}
                {fullReport && (
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                    {fullReport.k_anonymity && (
                      <StatCard
                        label="k-Anonymity (min k)"
                        value={String(fullReport.k_anonymity.min_k)}
                        color={kAnonColor(fullReport.k_anonymity.min_k)}
                        delay={320}
                      />
                    )}
                    {fullReport.l_diversity && (
                      <StatCard
                        label="l-Diversity (min l)"
                        value={String(fullReport.l_diversity.min_l)}
                        color={lDiversityColor(fullReport.l_diversity.min_l)}
                        delay={400}
                      />
                    )}
                    <StatCard
                      label="Epsilon"
                      value={fullReport.epsilon.estimated_epsilon.toFixed(2)}
                      color={epsilonColor(fullReport.epsilon.estimated_epsilon)}
                      delay={480}
                    />
                  </div>
                )}

                {/* Epsilon interpretation */}
                {fullReport && (
                  <div className="rounded-xl border border-black/[0.06] bg-white/50 p-4">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.5px] text-[#86868B] mb-1">
                      Epsilon Interpretation
                    </p>
                    <p className="text-[13px] text-[#1D1D1F]">{fullReport.epsilon.interpretation}</p>
                  </div>
                )}

                {/* DCR Histogram */}
                <div className="rounded-xl border border-black/[0.06] bg-white/50 p-5">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.5px] text-[#86868B] mb-3">
                    DCR Distribution
                  </p>
                  <DcrHistogram values={[result.min_dcr, result.mean_dcr, result.median_dcr, result.std_dcr]} />
                </div>

                {/* Recommendations */}
                {fullReport && fullReport.recommendations.length > 0 && (
                  <div className="rounded-xl border border-black/[0.06] bg-white/50 p-5 space-y-3">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.5px] text-[#86868B]">
                      Recommendations
                    </p>
                    <ul className="space-y-2">
                      {fullReport.recommendations.map((rec, idx) => (
                        <li key={idx} className="flex items-start gap-2 text-[13px] text-[#1D1D1F]">
                          <span className="text-[#007AFF] mt-0.5 flex-shrink-0">&#8226;</span>
                          <span>{rec}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Download Compliance Report */}
                {fullReport && (
                  <button
                    onClick={handleDownloadReport}
                    className="w-full border border-[#007AFF]/20 bg-[#007AFF]/[0.04] text-[#007AFF] font-semibold text-[13px] py-3 rounded-lg flex items-center justify-center gap-2 hover:bg-[#007AFF]/[0.08] transition-colors"
                  >
                    <Download className="h-4 w-4" />
                    Download Compliance Report
                  </button>
                )}

                {/* Reset */}
                <div className="flex justify-center pt-2">
                  <button
                    onClick={() => {
                      setResult(null);
                      setFullReport(null);
                      setRealData(null);
                      setSyntheticData(null);
                      setRealName("");
                      setSyntheticName("");
                      setQuasiIdentifiers(new Set());
                      setSensitiveColumn(null);
                    }}
                    className="text-[13px] font-medium text-[#86868B] hover:text-[#1D1D1F] transition-colors"
                  >
                    ← Run Another Audit
                  </button>
                </div>
              </div>
            </GlassCard>
          )}
        </div>
      </div>
    </div>
  );
}
