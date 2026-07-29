"use client";

import { useState, useCallback } from "react";
import {
  Brain, CheckCircle2, AlertTriangle,
  Shield, Loader2, Zap, Database,
} from "lucide-react";
import { toast } from "sonner";

import { TopBar } from "@/components/layout/top-bar";
import { GlassCard } from "@/components/shared/glass-card";
import { StatCard } from "@/components/shared/stat-card";
import { ExportPanel } from "@/components/shared/export-panel";
import { FileUpload } from "@/components/data/file-upload";

import { uploadFile, generateTestSuite, fixTestGaps, scoreTestData } from "@/lib/api";
import { useForgeStore } from "@/lib/store";
import type { TestAnalysis, TestCoverageResult } from "@/lib/types";

const CATEGORIES = [
  "original",
  "happy_path",
  "boundary",
  "invalid",
  "security",
  "unicode",
  "nulls",
] as const;

const CATEGORY_LABELS: Record<string, string> = {
  original: "Original Data",
  happy_path: "Happy Path",
  boundary: "Boundary",
  invalid: "Invalid",
  security: "Security",
  unicode: "Unicode",
  nulls: "Nulls",
};

const CATEGORY_ICONS: Record<string, React.ReactNode> = {
  original: <Database className="w-3.5 h-3.5" />,
  happy_path: <CheckCircle2 className="w-3.5 h-3.5" />,
  boundary: <AlertTriangle className="w-3.5 h-3.5" />,
  invalid: <Shield className="w-3.5 h-3.5" />,
  security: <Shield className="w-3.5 h-3.5" />,
  unicode: <Zap className="w-3.5 h-3.5" />,
  nulls: <Brain className="w-3.5 h-3.5" />,
};

function scoreColor(score: number) {
  if (score >= 80) return "#34C759";
  if (score >= 60) return "#FF9F0A";
  return "#FF3B30";
}

function severityColor(severity: string) {
  if (severity === "high") return "bg-[#FF3B30]/10 text-[#FF3B30]";
  if (severity === "medium") return "bg-[#FF9F0A]/10 text-[#FF9F0A]";
  return "bg-[#34C759]/10 text-[#34C759]";
}

export default function TestIntelligencePage() {
  const { provider, apiKey, model } = useForgeStore();

  /* ── Upload state ── */
  const [schema, setSchema] = useState<Record<string, string> | null>(null);
  const [sampleRows, setSampleRows] = useState<Record<string, unknown>[]>([]);
  const [uploadScore, setUploadScore] = useState<{ score: number; issues: string[] } | null>(null);

  /* ── Generation state ── */
  const [generating, setGenerating] = useState(false);
  const [analysis, setAnalysis] = useState<TestAnalysis | null>(null);
  const [testData, setTestData] = useState<Record<string, Record<string, unknown>[]> | null>(null);
  const [coverage, setCoverage] = useState<TestCoverageResult | null>(null);
  const [totalRows, setTotalRows] = useState(0);
  const [activeCategory, setActiveCategory] = useState<string>("original");

  /* ── Fix gaps state ── */
  const [fixing, setFixing] = useState(false);
  const [fixResult, setFixResult] = useState<{
    previousScore: number;
    newScore: number;
    addedSummary: Record<string, number>;
    totalAdded: number;
    gapsFixed: number;
    addedRows: Record<string, Record<string, unknown>[]>;
  } | null>(null);

  /* ── Upload handler ── */
  const [scoring, setScoring] = useState(false);

  const handleUpload = useCallback(async (file: File) => {
    try {
      const result = await uploadFile(file);
      setSchema(result.schema);
      setSampleRows(result.sample_rows);
      toast.success(`Schema inferred — ${Object.keys(result.schema).length} columns, ${result.row_count} rows`);

      // Call AI to score the uploaded data (limit payload size)
      setScoring(true);
      const dataForScoring = result.sample_rows.slice(0, 200).map((row) => {
        const trimmed: Record<string, unknown> = {};
        for (const [k, v] of Object.entries(row)) {
          trimmed[k] = typeof v === "string" && v.length > 100 ? v.slice(0, 100) + "..." : v;
        }
        return trimmed;
      });
      try {
        const scoreResult = await scoreTestData({
          schema: result.schema,
          data: dataForScoring,
          provider,
          api_key: apiKey,
          model: model ?? undefined,
        });
        setUploadScore({
          score: scoreResult.score,
          issues: scoreResult.gaps.map((g) => `[${g.severity.toUpperCase()}] ${g.description}`),
        });
      } catch {
        setUploadScore({ score: 30, issues: ["AI scoring unavailable — connect an LLM provider for accurate scoring"] });
      } finally {
        setScoring(false);
      }
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Upload failed");
    }
  }, [provider, apiKey, model]);

  /* ── Generate handler ── */
  const handleGenerate = useCallback(async () => {
    if (!schema) return;
    setGenerating(true);
    try {
      const result = await generateTestSuite({
        schema,
        sample_data: sampleRows,
        provider,
        api_key: apiKey,
        model: model ?? undefined,
      });
      setAnalysis(result.analysis);
      setTestData(result.test_data);
      setCoverage(result.coverage);
      setTotalRows(result.total_rows);
      setActiveCategory("original");
      toast.success(`Test suite generated — ${result.total_rows} scenarios across ${CATEGORIES.length} categories`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Generation failed");
    } finally {
      setGenerating(false);
    }
  }, [schema, sampleRows, provider, apiKey, model]);

  /* ── Fix gaps handler ── */
  const handleFixGaps = useCallback(async () => {
    if (!schema || !analysis || !coverage) return;
    const previousScore = coverage.score;
    setFixing(true);
    setFixResult(null);
    try {
      const result = await fixTestGaps({
        schema,
        analysis,
        gaps: coverage.gaps,
        existing_test_data: testData ?? undefined,
        provider,
        api_key: apiKey,
        model: model ?? undefined,
      });
      const merged = { ...testData };
      for (const [cat, rows] of Object.entries(result.additional_data)) {
        merged[cat] = [...(merged[cat] || []), ...rows];
      }
      const totalAdded = result.total_added ?? Object.values(result.additional_data).reduce((s, r) => s + r.length, 0);
      setTestData(merged);
      setCoverage(result.new_coverage);
      setTotalRows((prev) => prev + totalAdded);
      setFixResult({
        previousScore,
        newScore: result.new_coverage.score,
        addedSummary: result.added_summary ?? {},
        totalAdded,
        gapsFixed: result.gaps_fixed ?? coverage.gaps.length,
        addedRows: result.additional_data,
      });
      toast.success(`Fixed ${coverage.gaps.length} gaps — added ${totalAdded} new test cases!`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Fix gaps failed");
    } finally {
      setFixing(false);
    }
  }, [schema, analysis, coverage, testData, provider, apiKey, model]);


  /* ── Helpers ── */
  const schemaEntries = schema ? Object.entries(schema) : [];
  const activeCategoryRows = testData?.[activeCategory] ?? [];
  const dataColumns = activeCategoryRows.length > 0
    ? Object.keys(activeCategoryRows[0]).filter((k) => k !== "_scenario").slice(0, 4)
    : [];

  return (
    <div className="flex flex-col h-full">
      <TopBar title="AI Test Intelligence" />

      <div className="flex-1 overflow-y-auto relative z-10">
        <div className="px-6 pb-8 pt-6 max-w-5xl mx-auto space-y-6">

          {/* ═══════════ Input Section ═══════════ */}
          <GlassCard animatedBorder>
            <div className="p-7 space-y-5">
              <div className="flex items-center gap-3">
                <div className="flex items-center justify-center w-10 h-10 rounded-full bg-[#007AFF]/10">
                  <Brain className="w-5 h-5 text-[#007AFF]" />
                </div>
                <div>
                  <h2 className="text-[16px] font-semibold text-[#1D1D1F]">Upload Source Data</h2>
                  <p className="text-[13px] text-[#86868B] mt-0.5">
                    Upload a CSV or JSON file to infer schema and generate intelligent test cases.
                  </p>
                </div>
              </div>

              <FileUpload onFileAccepted={handleUpload} />

              {/* Inferred schema table */}
              {schema && (
                <div className="rounded-xl border border-black/[0.06] bg-white/50 overflow-hidden animate-slide-up">
                  <div className="px-4 py-3 border-b border-black/[0.06] bg-black/[0.02]">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.5px] text-[#86868B]">
                      Inferred Schema — {schemaEntries.length} columns
                    </p>
                  </div>
                  <div className="max-h-[200px] overflow-y-auto">
                    <table className="w-full text-[12px]">
                      <thead>
                        <tr className="border-b border-black/[0.04]">
                          <th className="text-left px-4 py-2 text-[#86868B] font-medium">Column</th>
                          <th className="text-left px-4 py-2 text-[#86868B] font-medium">Type</th>
                        </tr>
                      </thead>
                      <tbody>
                        {schemaEntries.map(([col, type]) => (
                          <tr key={col} className="border-b border-black/[0.03] last:border-0">
                            <td className="px-4 py-2 text-[#1D1D1F] font-medium">{col}</td>
                            <td className="px-4 py-2">
                              <span className="inline-block px-2 py-0.5 rounded-md bg-[#007AFF]/8 text-[#007AFF] text-[11px] font-medium">
                                {type}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* AI Scoring loading */}
              {scoring && !uploadScore && (
                <div className="flex items-center gap-3 p-4 rounded-xl border border-[#007AFF]/20 bg-[#007AFF]/5">
                  <Loader2 className="w-5 h-5 animate-spin text-[#007AFF]" />
                  <span className="text-[13px] text-[#007AFF] font-medium">AI is analyzing your data for test coverage...</span>
                </div>
              )}

              {/* Upload score */}
              {uploadScore && !testData && (
                <div className="rounded-xl border border-black/[0.06] bg-white/50 p-5 space-y-4">
                  <div className="flex items-start gap-5">
                    <div className="flex flex-col items-center">
                      <div
                        className="w-[72px] h-[72px] rounded-full flex items-center justify-center border-[3px]"
                        style={{
                          borderColor: scoreColor(uploadScore.score),
                          background: `${scoreColor(uploadScore.score)}08`,
                        }}
                      >
                        <span className="text-[24px] font-bold" style={{ color: scoreColor(uploadScore.score) }}>
                          {uploadScore.score}
                        </span>
                      </div>
                      <p className="text-[10px] text-[#86868B] font-medium mt-1.5">Test Readiness</p>
                    </div>
                    <div className="flex-1">
                      <h4 className="text-[14px] font-semibold text-[#1D1D1F] mb-1">Your data needs edge case coverage</h4>
                      <p className="text-[12px] text-[#86868B] mb-3">
                        Your uploaded data provides a baseline, but it&apos;s missing critical test scenarios. Generate a test suite to fill the gaps.
                      </p>
                      <div className="space-y-1">
                        {uploadScore.issues.map((issue, i) => (
                          <div key={i} className="flex items-center gap-2 text-[12px] text-[#FF9F0A]">
                            <AlertTriangle className="w-3 h-3 shrink-0" />
                            <span>{issue}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Generate button */}
              <div className="flex justify-end">
                <button
                  onClick={handleGenerate}
                  disabled={!schema || generating}
                  className="glow-button bg-[#007AFF] text-white font-semibold text-[13px] px-6 py-2.5 rounded-lg disabled:opacity-40 disabled:cursor-not-allowed btn-shimmer flex items-center gap-2"
                >
                  {generating ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Brain className="w-4 h-4" />
                  )}
                  Generate Test Suite
                </button>
              </div>

              {/* Loading indicator */}
              {generating && (
                <div className="flex flex-col items-center justify-center py-6 animate-slide-up">
                  <Loader2 className="h-10 w-10 animate-spin text-[#007AFF] mb-3" />
                  <p className="text-[14px] font-medium text-[#1D1D1F]">Analyzing schema and generating edge cases...</p>
                  <p className="text-[12px] text-[#86868B] mt-1">This may take a few seconds depending on schema complexity</p>
                </div>
              )}
            </div>
          </GlassCard>

          {/* ═══════════ Results Section ═══════════ */}
          {testData && coverage && (
            <>
              {/* Coverage score + category stats */}
              <GlassCard animatedBorder>
                <div className="p-7 space-y-5">
                  <div className="flex items-start gap-6">
                    {/* Large coverage score */}
                    <div className="flex flex-col items-center">
                      <div
                        className="w-[100px] h-[100px] rounded-full flex items-center justify-center border-4"
                        style={{
                          borderColor: scoreColor(coverage.score),
                          background: `${scoreColor(coverage.score)}08`,
                        }}
                      >
                        <span
                          className="text-[32px] font-bold tabular-nums"
                          style={{ color: scoreColor(coverage.score) }}
                        >
                          {coverage.score}
                        </span>
                      </div>
                      <p className="text-[11px] text-[#86868B] font-medium mt-2">Coverage Score</p>
                    </div>

                    {/* Summary */}
                    <div className="flex-1 pt-2">
                      <div className="flex items-center gap-2 mb-1">
                        <CheckCircle2 className="w-4 h-4 text-[#34C759]" />
                        <h2 className="text-[16px] font-semibold text-[#1D1D1F]">Test Suite Generated</h2>
                      </div>
                      <p className="text-[13px] text-[#86868B]">
                        {totalRows} test scenarios across {CATEGORIES.length} categories covering {analysis?.domain || "general"} domain data.
                      </p>
                    </div>
                  </div>

                  {/* Category stat cards */}
                  <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
                    {CATEGORIES.map((cat, i) => (
                      <StatCard
                        key={cat}
                        label={CATEGORY_LABELS[cat]}
                        value={testData[cat]?.length ?? 0}
                        color={cat === "happy_path" ? "green" : cat === "security" ? "red" : "default"}
                        delay={i * 60}
                      />
                    ))}
                  </div>
                </div>
              </GlassCard>

              {/* Category tabs + data */}
              <GlassCard>
                <div className="overflow-hidden rounded-xl">
                  {/* Tab bar */}
                  <div className="flex gap-0 border-b border-black/[0.06] bg-black/[0.02] overflow-x-auto">
                    {CATEGORIES.map((cat) => (
                      <button
                        key={cat}
                        onClick={() => setActiveCategory(cat)}
                        className={`px-4 py-2.5 text-[12px] font-medium transition-all border-b-2 whitespace-nowrap flex items-center gap-1.5 ${
                          activeCategory === cat
                            ? "border-[#007AFF] text-[#007AFF] bg-[#007AFF]/[0.03]"
                            : "border-transparent text-[#86868B] hover:text-[#1D1D1F]"
                        }`}
                      >
                        {CATEGORY_ICONS[cat]}
                        {CATEGORY_LABELS[cat]}
                        <span className="ml-1 text-[10px] px-1.5 py-0.5 rounded-full bg-black/[0.05] font-semibold">
                          {testData[cat]?.length ?? 0}
                        </span>
                      </button>
                    ))}
                  </div>

                  {/* Data table */}
                  <div className="p-4">
                    {activeCategoryRows.length === 0 ? (
                      <p className="text-[13px] text-[#86868B] text-center py-8">
                        No test cases in this category yet.
                      </p>
                    ) : (
                      <div className="overflow-x-auto">
                        <table className="w-full text-[12px]">
                          <thead>
                            <tr className="border-b border-black/[0.06]">
                              <th className="text-left px-3 py-2.5 text-[#86868B] font-medium">Scenario</th>
                              {dataColumns.map((col) => (
                                <th key={col} className="text-left px-3 py-2.5 text-[#86868B] font-medium">{col}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {activeCategoryRows.map((row, i) => (
                              <tr
                                key={i}
                                className="border-b border-black/[0.03] last:border-0 hover:bg-[#007AFF]/[0.02] transition-colors"
                              >
                                <td className="px-3 py-2.5 text-[#1D1D1F] font-medium max-w-[240px]">
                                  <span className="inline-block px-2 py-0.5 rounded-md bg-[#007AFF]/8 text-[#007AFF] text-[11px]">
                                    {String(row._scenario ?? "")}
                                  </span>
                                </td>
                                {dataColumns.map((col) => (
                                  <td key={col} className="px-3 py-2.5 text-[#1D1D1F] font-mono text-[11px] max-w-[180px] truncate">
                                    {row[col] === null ? (
                                      <span className="text-[#86868B] italic">null</span>
                                    ) : (
                                      String(row[col] ?? "")
                                    )}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                </div>
              </GlassCard>

              {/* ═══════════ Fix Result Banner ═══════════ */}
              {fixResult && (
                <GlassCard>
                  <div className="p-7 space-y-5">
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="w-5 h-5 text-[#34C759]" />
                      <h3 className="text-[15px] font-semibold text-[#1D1D1F]">Gaps Fixed Successfully</h3>
                    </div>

                    {/* Score before → after */}
                    <div className="flex items-center justify-center gap-6 py-4">
                      <div className="text-center">
                        <div className="text-[28px] font-bold text-[#86868B] line-through decoration-2">{fixResult.previousScore}</div>
                        <div className="text-[10px] text-[#86868B] mt-1">Before</div>
                      </div>
                      <div className="text-[24px] text-[#86868B]">&rarr;</div>
                      <div className="text-center">
                        <div className="text-[36px] font-bold" style={{ color: scoreColor(fixResult.newScore) }}>{fixResult.newScore}</div>
                        <div className="text-[10px] font-medium" style={{ color: scoreColor(fixResult.newScore) }}>After</div>
                      </div>
                      <div className="ml-4 px-3 py-1.5 rounded-full bg-[#34C759]/10 text-[#34C759] text-[13px] font-semibold">
                        +{fixResult.newScore - fixResult.previousScore} pts
                      </div>
                    </div>

                    {/* What was added */}
                    <div className="space-y-2">
                      <h4 className="text-[12px] font-semibold text-[#86868B] uppercase tracking-wide">What was added</h4>
                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                        {Object.entries(fixResult.addedSummary).map(([cat, count]) => (
                          <div key={cat} className="flex items-center gap-2 px-3 py-2.5 rounded-lg border border-[#34C759]/20 bg-[#34C759]/5">
                            <CheckCircle2 className="w-3.5 h-3.5 text-[#34C759] shrink-0" />
                            <span className="text-[12px] text-[#1D1D1F] font-medium">{CATEGORY_LABELS[cat] ?? cat}</span>
                            <span className="ml-auto text-[12px] font-bold text-[#34C759]">+{count}</span>
                          </div>
                        ))}
                      </div>
                      <p className="text-[12px] text-[#86868B] pt-1">
                        {fixResult.gapsFixed} gap{fixResult.gapsFixed !== 1 ? "s" : ""} addressed, {fixResult.totalAdded} new test rows added.
                      </p>
                    </div>

                    {/* Git-diff style preview of added records */}
                    <div className="space-y-2">
                      <h4 className="text-[12px] font-semibold text-[#86868B] uppercase tracking-wide">Added Records Preview</h4>
                      <div className="max-h-[300px] overflow-y-auto rounded-xl border border-[#34C759]/20 bg-[#f0fdf4]">
                        {Object.entries(fixResult.addedRows).map(([cat, rows]) =>
                          rows.length > 0 && (
                            <div key={cat}>
                              <div className="sticky top-0 px-3 py-1.5 bg-[#34C759]/10 border-b border-[#34C759]/15 text-[11px] font-semibold text-[#166534]">
                                + {CATEGORY_LABELS[cat] ?? cat} ({rows.length} rows)
                              </div>
                              {rows.slice(0, 5).map((row, i) => {
                                const scenario = String(row._scenario ?? "");
                                const dataCols = Object.keys(row).filter((k) => !k.startsWith("_")).slice(0, 3);
                                return (
                                  <div key={`${cat}-${i}`} className="flex items-center gap-3 px-3 py-2 border-b border-[#34C759]/10 font-mono text-[11px]">
                                    <span className="text-[#34C759] font-bold shrink-0">+</span>
                                    <span className="text-[#166534] shrink-0 max-w-[200px] truncate">{scenario}</span>
                                    <span className="text-[#3A3A3C] truncate">
                                      {dataCols.map((c) => `${c}=${row[c] === null ? "null" : String(row[c] ?? "").slice(0, 20)}`).join(" | ")}
                                    </span>
                                  </div>
                                );
                              })}
                              {rows.length > 5 && (
                                <div className="px-3 py-1.5 text-[10px] text-[#86868B] italic">
                                  ... and {rows.length - 5} more rows
                                </div>
                              )}
                            </div>
                          )
                        )}
                      </div>
                    </div>
                  </div>
                </GlassCard>
              )}

              {/* ═══════════ Gaps Section ═══════════ */}
              {coverage.gaps.length > 0 && (
                <GlassCard>
                  <div className="p-7 space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="text-[15px] font-semibold text-[#1D1D1F]">Coverage Gaps</h3>
                        <p className="text-[12px] text-[#86868B] mt-0.5">
                          {coverage.gaps.length} gap{coverage.gaps.length !== 1 ? "s" : ""} identified — fix them to improve coverage.
                        </p>
                      </div>
                      <button
                        onClick={handleFixGaps}
                        disabled={fixing}
                        className="flex items-center gap-2 bg-[#34C759] text-white font-semibold text-[13px] px-5 py-2.5 rounded-lg hover:bg-[#2DB84E] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {fixing ? (
                          <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                          <Zap className="w-4 h-4" />
                        )}
                        Fix Gaps
                      </button>
                    </div>

                    <div className="space-y-2">
                      {coverage.gaps.map((gap, i) => (
                        <div
                          key={i}
                          className="flex items-start gap-3 rounded-xl border border-black/[0.06] bg-white/60 p-4"
                        >
                          <span className={`inline-flex items-center px-2 py-0.5 rounded-md text-[10px] font-semibold uppercase tracking-wide ${severityColor(gap.severity)}`}>
                            {gap.severity}
                          </span>
                          <div className="flex-1">
                            <p className="text-[13px] text-[#1D1D1F] font-medium">{gap.description}</p>
                            <p className="text-[11px] text-[#86868B] mt-0.5">Category: {CATEGORY_LABELS[gap.category] ?? gap.category}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </GlassCard>
              )}

              {/* All gaps fixed */}
              {coverage.gaps.length === 0 && fixResult && (
                <div className="flex items-center gap-2 p-4 rounded-xl bg-[#34C759]/10 border border-[#34C759]/20">
                  <CheckCircle2 className="size-5 text-[#34C759]" />
                  <span className="text-[13px] font-medium text-[#34C759]">
                    All coverage gaps resolved — test suite is comprehensive
                  </span>
                </div>
              )}

              {/* ═══════════ Export Section ═══════════ */}
              <ExportPanel data={testData} schema={schema ?? undefined} filename="test-suite" />
            </>
          )}
        </div>
      </div>
    </div>
  );
}
