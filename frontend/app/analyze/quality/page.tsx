"use client";

import { useState, useCallback, useMemo } from "react";
import {
  Loader2,
  BarChart3,
  Upload,
  AlertTriangle,
  CheckCircle2,
  XCircle,
} from "lucide-react";
import { toast } from "sonner";

import { TopBar } from "@/components/layout/top-bar";
import { FileUpload } from "@/components/data/file-upload";
import { DataTable } from "@/components/data/data-table";
import { DistributionChart } from "@/components/charts/distribution-chart";
import { GlassCard } from "@/components/shared/glass-card";
import { ProgressRing } from "@/components/shared/progress-ring";

import { uploadFile } from "@/lib/api";
import { useQualityAssessment } from "@/hooks/use-quality";
import { useForgeStore } from "@/lib/store";
import type { QualityReport } from "@/lib/types";

function gradeColor(grade: string) {
  switch (grade.toUpperCase()) {
    case "A":
      return "#34C759";
    case "B":
      return "#007AFF";
    case "C":
      return "#FF9F0A";
    case "D":
      return "#FF9F0A";
    default:
      return "#FF3B30";
  }
}

export default function QualityPage() {
  /* ── Original data state ── */
  const [originalData, setOriginalData] = useState<Record<string, unknown>[] | null>(null);
  const [originalName, setOriginalName] = useState("");
  const [uploadingOriginal, setUploadingOriginal] = useState(false);

  /* ── Generated data state ── */
  const [generatedData, setGeneratedData] = useState<Record<string, unknown>[] | null>(null);
  const [generatedName, setGeneratedName] = useState("");
  const [uploadingGenerated, setUploadingGenerated] = useState(false);

  /* ── Results ── */
  const [report, setReport] = useState<QualityReport | null>(null);
  const [selectedColumn, setSelectedColumn] = useState<string | null>(null);

  const assessment = useQualityAssessment();
  const store = useForgeStore();

  /* ── Upload handlers ── */
  const handleUploadOriginal = useCallback(async (file: File) => {
    setUploadingOriginal(true);
    try {
      const res = await uploadFile(file);
      setOriginalData(res.sample_rows);
      setOriginalName(file.name);
      toast.success(`Original data loaded: ${file.name}`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploadingOriginal(false);
    }
  }, []);

  const handleUploadGenerated = useCallback(async (file: File) => {
    setUploadingGenerated(true);
    try {
      const res = await uploadFile(file);
      setGeneratedData(res.sample_rows);
      setGeneratedName(file.name);
      toast.success(`Generated data loaded: ${file.name}`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploadingGenerated(false);
    }
  }, []);

  const handleUseLastGenerated = useCallback(() => {
    if (store.lastGeneratedData) {
      setGeneratedData(store.lastGeneratedData);
      setGeneratedName("Last Generated Data");
      toast.success("Loaded last generated data");
    }
  }, [store.lastGeneratedData]);

  /* ── Run assessment ── */
  const handleAssess = useCallback(async () => {
    if (!generatedData) {
      toast.error("Upload generated data first");
      return;
    }
    try {
      const res = await assessment.mutateAsync({
        generatedData,
        originalData,
        expectedSchema: store.lastSchema,
      });
      setReport(res);
      toast.success("Quality assessment complete");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Assessment failed");
    }
  }, [generatedData, originalData, store.lastSchema, assessment]);

  /* ── Numeric columns for distribution chart ── */
  const numericColumns = useMemo(() => {
    if (!originalData || originalData.length === 0) return [];
    return Object.keys(originalData[0]).filter((key) => {
      const val = originalData[0][key];
      return typeof val === "number";
    });
  }, [originalData]);

  const getNumericValues = useCallback(
    (data: Record<string, unknown>[], col: string): number[] => {
      return data
        .map((row) => row[col])
        .filter((v): v is number => typeof v === "number");
    },
    []
  );

  let topBarStatus: "running" | "complete" | undefined;
  if (assessment.isPending) {
    topBarStatus = "running";
  } else if (report) {
    topBarStatus = "complete";
  }

  return (
    <div className="flex flex-col h-full">
      <TopBar title="Data Quality" status={topBarStatus} />

      <div className="flex-1 overflow-y-auto relative z-10">
        <div className="px-6 py-6 max-w-5xl mx-auto space-y-6">
          {/* Upload Section */}
          <GlassCard animatedBorder>
            <div className="p-7 space-y-5">
              <div>
                <h2 className="text-[16px] font-semibold text-[#1D1D1F]">Upload Datasets</h2>
                <p className="text-[13px] text-[#86868B] mt-1">Compare original and generated data to assess synthetic data quality.</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Original */}
                <div className="rounded-xl border border-black/[0.06] bg-white/60 p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <div className="h-2.5 w-2.5 rounded-full bg-[#AF82FF]" />
                    <span className="text-[13px] font-medium text-[#1D1D1F]">Original Data</span>
                    {originalName && (
                      <span className="text-[11px] text-[#86868B] ml-auto truncate max-w-[120px]">{originalName}</span>
                    )}
                  </div>
                  <FileUpload onFileAccepted={handleUploadOriginal} loading={uploadingOriginal} />
                  {originalData && originalData.length > 0 && (
                    <div className="mt-3">
                      <DataTable data={originalData} maxRows={4} />
                    </div>
                  )}
                </div>

                {/* Generated */}
                <div className="rounded-xl border border-black/[0.06] bg-white/60 p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <div className="h-2.5 w-2.5 rounded-full bg-[#007AFF]" />
                    <span className="text-[13px] font-medium text-[#1D1D1F]">Generated Data</span>
                    {generatedName && (
                      <span className="text-[11px] text-[#86868B] ml-auto truncate max-w-[120px]">{generatedName}</span>
                    )}
                  </div>
                  <FileUpload onFileAccepted={handleUploadGenerated} loading={uploadingGenerated} />
                  {store.lastGeneratedData && !generatedData && (
                    <button
                      onClick={handleUseLastGenerated}
                      className="mt-2 w-full text-[11px] font-medium text-[#007AFF] hover:text-[#005EC4] transition-colors flex items-center justify-center gap-1"
                    >
                      <Upload className="h-3 w-3" />
                      Use last generated data
                    </button>
                  )}
                  {generatedData && generatedData.length > 0 && (
                    <div className="mt-3">
                      <DataTable data={generatedData} maxRows={4} />
                    </div>
                  )}
                </div>
              </div>

              {/* Run button */}
              {generatedData && !report && (
                <button
                  onClick={handleAssess}
                  disabled={assessment.isPending}
                  className="w-full glow-button bg-[#007AFF] text-white font-semibold text-[13px] py-3 rounded-lg flex items-center justify-center gap-2 btn-shimmer disabled:opacity-40"
                >
                  {assessment.isPending ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <BarChart3 className="h-4 w-4" />
                  )}
                  Run Quality Assessment
                </button>
              )}
            </div>
          </GlassCard>

          {/* Loading */}
          {assessment.isPending && (
            <div className="flex flex-col items-center justify-center py-12">
              <Loader2 className="h-10 w-10 animate-spin text-[#007AFF] mb-4" />
              <p className="text-[14px] font-medium text-[#1D1D1F]">Analyzing data quality...</p>
              <p className="text-[12px] text-[#86868B] mt-1">Running statistical tests and distribution analysis</p>
            </div>
          )}

          {/* Results */}
          {report && (
            <GlassCard>
              <div className="p-7 space-y-5">
                {/* Grade + Ring */}
                <div className="flex items-center justify-center py-6">
                  <div className="relative">
                    <ProgressRing
                      progress={report.overall_score}
                      size={140}
                      strokeWidth={10}
                      hideValue
                    />
                    <div className="absolute inset-0 flex flex-col items-center justify-center">
                      <span
                        className="text-3xl font-bold"
                        style={{ color: gradeColor(report.realism_grade) }}
                      >
                        {report.overall_score.toFixed(0)}%
                      </span>
                    </div>
                  </div>
                </div>

                {/* Metric cards */}
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  {[
                    { label: "Distribution", value: report.distribution_score, color: "blue" as const },
                    { label: "Correlation", value: report.correlation_preservation, color: "blue" as const },
                    { label: "Completeness", value: report.completeness, color: "green" as const },
                    { label: "Schema Match", value: report.schema_match, color: "green" as const },
                    { label: "Uniqueness", value: report.uniqueness, color: "default" as const },
                    { label: "Dependency", value: report.dependency_score, color: "default" as const },
                  ].map(({ label, value }, i) => (
                    <div key={label} className="rounded-xl border border-black/[0.06] bg-white/60 p-4 animate-slide-up" style={{ animationDelay: `${i * 60}ms` }}>
                      <p className="text-[11px] text-[#86868B] mb-1">{label}</p>
                      <p className="text-[20px] font-bold font-mono text-[#1D1D1F]">{value.toFixed(1)}%</p>
                      <div className="mt-2 h-[3px] bg-black/[0.04] rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full bg-[#007AFF] transition-all duration-700 ease-out"
                          style={{ width: `${Math.min(value, 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                {/* Statistical Tests */}
                {report.statistical_tests.length > 0 && (
                  <div className="rounded-xl border border-black/[0.06] bg-white/50 overflow-hidden">
                    <div className="px-5 py-3 border-b border-black/[0.06] bg-black/[0.02]">
                      <p className="text-[11px] font-semibold uppercase tracking-[0.5px] text-[#86868B]">Statistical Tests</p>
                    </div>
                    <div className="overflow-x-auto">
                      <table className="w-full text-[12px]">
                        <thead>
                          <tr className="border-b border-black/[0.06]">
                            <th className="px-4 py-2.5 text-left text-[10px] font-semibold uppercase tracking-[0.3px] text-[#86868B]">Column</th>
                            <th className="px-4 py-2.5 text-left text-[10px] font-semibold uppercase tracking-[0.3px] text-[#86868B]">Test</th>
                            <th className="px-4 py-2.5 text-left text-[10px] font-semibold uppercase tracking-[0.3px] text-[#86868B]">Statistic</th>
                            <th className="px-4 py-2.5 text-left text-[10px] font-semibold uppercase tracking-[0.3px] text-[#86868B]">p-value</th>
                            <th className="px-4 py-2.5 text-left text-[10px] font-semibold uppercase tracking-[0.3px] text-[#86868B]">Result</th>
                          </tr>
                        </thead>
                        <tbody>
                          {report.statistical_tests.map((test) => (
                            <tr key={`${test.column}-${test.test}`} className="border-b border-black/[0.03] last:border-0">
                              <td className="px-4 py-2 font-mono text-[#1D1D1F]">{test.column}</td>
                              <td className="px-4 py-2 text-[#3A3A3C]">{test.test}</td>
                              <td className="px-4 py-2 font-mono text-[#3A3A3C]">{test.statistic.toFixed(4)}</td>
                              <td className="px-4 py-2 font-mono text-[#3A3A3C]">{test.p_value.toFixed(4)}</td>
                              <td className="px-4 py-2">
                                {test.pass ? (
                                  <span className="inline-flex items-center gap-1 text-[#34C759]">
                                    <CheckCircle2 className="h-3.5 w-3.5" />
                                    Pass
                                  </span>
                                ) : (
                                  <span className="inline-flex items-center gap-1 text-[#FF3B30]">
                                    <XCircle className="h-3.5 w-3.5" />
                                    Fail
                                  </span>
                                )}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Warnings */}
                {report.warnings.length > 0 && (
                  <div className="rounded-xl border border-[#FF9F0A]/20 bg-[#FF9F0A]/[0.04] p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <AlertTriangle className="h-4 w-4 text-[#FF9F0A]" />
                      <span className="text-[12px] font-semibold text-[#FF9F0A]">Warnings</span>
                    </div>
                    <ul className="space-y-1">
                      {report.warnings.map((w) => (
                        <li key={w} className="text-[12px] text-[#3A3A3C]">{w}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Distribution Comparison */}
                {originalData && generatedData && numericColumns.length > 0 && (
                  <div className="rounded-xl border border-black/[0.06] bg-white/50 p-5">
                    <div className="flex items-center justify-between mb-4">
                      <p className="text-[11px] font-semibold uppercase tracking-[0.5px] text-[#86868B]">Distribution Comparison</p>
                      <select
                        value={selectedColumn || ""}
                        onChange={(e) => setSelectedColumn(e.target.value || null)}
                        className="rounded-lg border border-black/[0.08] bg-white/80 px-3 py-1.5 text-[12px] focus:border-[#007AFF] focus:ring-1 focus:ring-[#007AFF]/20 outline-none"
                      >
                        <option value="">Select column...</option>
                        {numericColumns.map((col) => (
                          <option key={col} value={col}>{col}</option>
                        ))}
                      </select>
                    </div>
                    {selectedColumn && (
                      <DistributionChart
                        syntheticData={getNumericValues(generatedData, selectedColumn)}
                        realData={getNumericValues(originalData, selectedColumn)}
                        columnName={selectedColumn}
                      />
                    )}
                  </div>
                )}

                {/* Reset */}
                <div className="flex justify-center pt-2">
                  <button
                    onClick={() => {
                      setReport(null);
                      setOriginalData(null);
                      setGeneratedData(null);
                      setOriginalName("");
                      setGeneratedName("");
                      setSelectedColumn(null);
                    }}
                    className="text-[13px] font-medium text-[#86868B] hover:text-[#1D1D1F] transition-colors"
                  >
                    ← Run Another Assessment
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
