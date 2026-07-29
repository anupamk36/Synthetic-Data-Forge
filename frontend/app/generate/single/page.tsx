"use client";

import { useState, useCallback } from "react";
import { Loader2, Sparkles, BrainCircuit, Dices, Zap, Download, CheckCircle2 } from "lucide-react";
import { toast } from "sonner";

import { TopBar } from "@/components/layout/top-bar";
import { GlassCard } from "@/components/shared/glass-card";
import { WizardStepper } from "@/components/shared/wizard-stepper";
import { StatCard } from "@/components/shared/stat-card";
import { ProgressRing } from "@/components/shared/progress-ring";
import { FileUpload } from "@/components/data/file-upload";
import { DataTable } from "@/components/data/data-table";
import { SchemaEditor } from "@/components/data/schema-editor";
import { FieldHints } from "@/components/generation/field-hints";
import { DataWaterfall } from "@/components/generation/data-waterfall";
import { ExportPanel } from "@/components/shared/export-panel";
import { DownloadMenu } from "@/components/data/download-menu";

import { uploadFile, profileData, estimateCost } from "@/lib/api";
import { useGenerate } from "@/hooks/use-generate";
import { useForgeStore } from "@/lib/store";
import type { Schema, DataProfile, CostEstimate, GenerateRequest } from "@/lib/types";

const WIZARD_STEPS = [
  { label: "Upload" },
  { label: "Schema" },
  { label: "Configure" },
  { label: "Generate" },
];

export default function SingleTablePage() {
  const [step, setStep] = useState(0);

  /* ── Upload state ── */
  const [uploading, setUploading] = useState(false);
  const [schema, setSchema] = useState<Schema | null>(null);
  const [sampleRows, setSampleRows] = useState<Record<string, unknown>[]>([]);
  const [profile, setProfile] = useState<DataProfile | null>(null);

  /* ── Config state ── */
  const [records, setRecords] = useState(1000);
  const [format, setFormat] = useState("json");
  const [useLlm, setUseLlm] = useState(false);
  const [validationEnabled, setValidationEnabled] = useState(true);
  const [validationSample, setValidationSample] = useState(100);
  const [hints, setHints] = useState<Record<string, string>>({});
  const [seed, setSeed] = useState<number | null>(null);

  /* ── Cost state ── */
  const [cost, setCost] = useState<CostEstimate | null>(null);
  const [costLoading, setCostLoading] = useState(false);

  /* ── Generation hook ── */
  const genState = useGenerate();

  /* ── Store ── */
  const store = useForgeStore();

  /* ── Upload handler ── */
  const handleUpload = useCallback(async (file: File) => {
    setUploading(true);
    try {
      const result = await uploadFile(file);
      setSchema(result.schema);
      setSampleRows(result.sample_rows);
      toast.success(`Uploaded ${file.name} — ${result.row_count} rows detected`);
      setStep(1);

      try {
        const prof = await profileData(result.sample_rows);
        setProfile(prof);
      } catch {
        toast.error("Profiling failed — schema loaded without stats");
      }
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }, []);

  /* ── Cost estimation ── */
  const handleEstimateCost = useCallback(async () => {
    if (!schema) return;
    setCostLoading(true);
    try {
      const est = await estimateCost(schema, records, store.provider, store.model);
      setCost(est);
    } catch {
      toast.error("Could not estimate cost");
    } finally {
      setCostLoading(false);
    }
  }, [schema, records, store.provider, store.model]);

  /* ── Generate handler ── */
  const handleGenerate = useCallback(async () => {
    if (!schema) return;

    const req: GenerateRequest = {
      schema,
      count: records,
      output_format: format,
      use_llm: useLlm,
      field_descriptions: useLlm ? hints : undefined,
      seed,
      provider: store.provider,
      model: store.model,
      api_key: store.apiKey || undefined,
      llm_validation: validationEnabled,
      validation_sample_rate: validationSample / 100,
    };

    toast.success("Generation started");
    await genState.start(req);
  }, [schema, records, format, useLlm, hints, seed, store, genState, validationEnabled, validationSample]);

  /* ── Persist results ── */
  if (genState.status === "complete" && genState.data && schema) {
    if (store.lastGeneratedData !== genState.data) {
      store.setLastGenerated(genState.data, schema);
    }
  }

  const elapsed = genState.elapsed;
  const recPerSec = elapsed > 0 ? Math.round(genState.recordsDone / elapsed) : 0;

  return (
    <div className="flex flex-col h-full">
      <TopBar title="Single Table Generator" />

      <div className="flex-1 overflow-y-auto p-7 relative z-10">
        {/* Wizard Stepper */}
        <WizardStepper steps={WIZARD_STEPS} current={step} />

        {/* ═══════════ STEP 0: Upload ═══════════ */}
        {step === 0 && (
          <GlassCard>
            <h2 className="text-[19px] font-semibold text-[#1D1D1F] mb-1">Upload Sample Data</h2>
            <p className="text-[13px] text-[#86868B] mb-5 leading-relaxed">
              Drop a CSV, Parquet, or JSON file to get started. We&apos;ll infer the schema automatically.
            </p>
            <FileUpload onFileAccepted={handleUpload} loading={uploading} />
            {sampleRows.length > 0 && (
              <div className="mt-5">
                <p className="text-[12px] text-[#86868B] mb-2">Preview (first {Math.min(sampleRows.length, 5)} rows)</p>
                <DataTable data={sampleRows} maxRows={5} />
              </div>
            )}
          </GlassCard>
        )}

        {/* ═══════════ STEP 1: Schema ═══════════ */}
        {step === 1 && schema && (
          <GlassCard>
            <h2 className="text-[19px] font-semibold text-[#1D1D1F] mb-1">Define Your Schema</h2>
            <p className="text-[13px] text-[#86868B] mb-5 leading-relaxed">
              We detected {Object.keys(schema).length} columns. Review the types and add descriptions to help the LLM generate realistic data.
            </p>
            <SchemaEditor
              schema={schema}
              onChange={setSchema}
              profile={profile ?? undefined}
            />
            <div className="flex justify-between mt-6">
              <button
                onClick={() => setStep(0)}
                className="px-5 py-[9px] bg-black/[0.04] border border-black/[0.08] rounded-[9px] text-[13px] font-medium text-[#3A3A3C] hover:bg-black/[0.06] transition-colors"
              >
                ← Back
              </button>
              <button
                onClick={() => setStep(2)}
                className="px-5 py-[9px] bg-[#007AFF] text-white rounded-[9px] text-[13px] font-semibold glow-button btn-shimmer"
              >
                Continue →
              </button>
            </div>
          </GlassCard>
        )}

        {/* ═══════════ STEP 2: Configure ═══════════ */}
        {step === 2 && schema && (
          <GlassCard>
            <h2 className="text-[19px] font-semibold text-[#1D1D1F] mb-1">Configure Generation</h2>
            <p className="text-[13px] text-[#86868B] mb-5 leading-relaxed">
              Set how many records to generate, output format, and optional LLM enhancement.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Record count */}
              <div className="p-4 rounded-xl border border-black/[0.06] bg-white/50">
                <label htmlFor="record-count" className="text-[11px] font-semibold text-[#86868B] uppercase tracking-[0.5px] block mb-2">
                  Record Count
                </label>
                <input
                  id="record-count"
                  type="number"
                  value={records}
                  onChange={(e) => setRecords(Math.max(1, Number.parseInt(e.target.value) || 1))}
                  className="w-full rounded-lg border border-black/[0.06] bg-white/60 px-3 py-2 text-[14px] font-medium tabular-nums focus:border-[#007AFF] focus:ring-1 focus:ring-[#007AFF]/30 outline-none"
                />
              </div>

              {/* Output format */}
              <div className="p-4 rounded-xl border border-black/[0.06] bg-white/50">
                <span className="text-[11px] font-semibold text-[#86868B] uppercase tracking-[0.5px] block mb-2">
                  Output Format
                </span>
                <div className="inline-flex gap-1 p-1 bg-black/[0.04] rounded-lg">
                  {["parquet", "csv", "json"].map((f) => (
                    <button
                      key={f}
                      onClick={() => setFormat(f)}
                      className={`px-4 py-1.5 rounded-md text-[12px] font-medium transition-all ${
                        format === f
                          ? "bg-white text-[#007AFF] shadow-sm font-semibold"
                          : "text-[#86868B] hover:text-[#3A3A3C]"
                      }`}
                    >
                      {f.toUpperCase()}
                    </button>
                  ))}
                </div>
              </div>

              {/* LLM toggle */}
              <div className="p-4 rounded-xl border border-black/[0.06] bg-white/50 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <BrainCircuit className="h-4 w-4 text-[#86868B]" />
                  <span className="text-[13px] text-[#1D1D1F] font-medium">LLM-Enhanced</span>
                </div>
                <button
                  onClick={() => setUseLlm((v) => !v)}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${useLlm ? "bg-[#007AFF]" : "bg-black/[0.12]"}`}
                >
                  <span
                    className={`inline-block h-4.5 w-4.5 rounded-full bg-white shadow-sm transition-transform ${useLlm ? "translate-x-[22px]" : "translate-x-[3px]"}`}
                  />
                </button>
              </div>

              {/* Seed */}
              <div className="p-4 rounded-xl border border-black/[0.06] bg-white/50">
                <label htmlFor="seed-input" className="text-[11px] font-semibold text-[#86868B] uppercase tracking-[0.5px] mb-2 flex items-center gap-1.5">
                  <Dices className="h-3 w-3" />
                  Seed (optional)
                </label>
                <input
                  id="seed-input"
                  type="number"
                  value={seed ?? ""}
                  onChange={(e) => setSeed(e.target.value ? Number.parseInt(e.target.value) : null)}
                  placeholder="Random"
                  className="w-full rounded-lg border border-black/[0.06] bg-white/60 px-3 py-2 text-[13px] font-mono focus:border-[#007AFF] focus:ring-1 focus:ring-[#007AFF]/30 outline-none"
                />
              </div>
            </div>

            {/* LLM options */}
            {useLlm && (
              <div className="mt-4 space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-4 rounded-xl border border-black/[0.06] bg-white/50 flex items-center justify-between">
                    <span className="text-[13px] text-[#1D1D1F]">Semantic Validation</span>
                    <button
                      onClick={() => setValidationEnabled((v) => !v)}
                      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${validationEnabled ? "bg-[#007AFF]" : "bg-black/[0.12]"}`}
                    >
                      <span className={`inline-block h-4.5 w-4.5 rounded-full bg-white shadow-sm transition-transform ${validationEnabled ? "translate-x-[22px]" : "translate-x-[3px]"}`} />
                    </button>
                  </div>
                  <div className="p-4 rounded-xl border border-black/[0.06] bg-white/50">
                    <label className="text-[11px] font-semibold text-[#86868B] uppercase tracking-[0.5px] block mb-2">
                      Validation Sample %
                    </label>
                    <input
                      type="range"
                      min={10}
                      max={100}
                      step={10}
                      value={validationSample}
                      onChange={(e) => setValidationSample(Number.parseInt(e.target.value))}
                      className="w-full accent-[#007AFF]"
                    />
                    <span className="text-[11px] text-[#86868B]">{validationSample}%</span>
                  </div>
                </div>

                <FieldHints schema={schema} hints={hints} onChange={setHints} />

                {/* Cost estimate */}
                <div className="flex items-center gap-4">
                  <button
                    onClick={handleEstimateCost}
                    disabled={costLoading}
                    className="text-[13px] text-[#007AFF] hover:text-[#0056CC] transition-colors flex items-center gap-1.5 font-medium"
                  >
                    {costLoading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Zap className="h-3.5 w-3.5" />}
                    Estimate Cost
                  </button>
                  {cost && (
                    <span className="text-[13px] font-mono text-[#34C759] font-medium">
                      ~${cost.estimated_cost_usd.toFixed(4)} USD
                    </span>
                  )}
                </div>
              </div>
            )}

            <div className="flex justify-between mt-6">
              <button
                onClick={() => setStep(1)}
                className="px-5 py-[9px] bg-black/[0.04] border border-black/[0.08] rounded-[9px] text-[13px] font-medium text-[#3A3A3C] hover:bg-black/[0.06] transition-colors"
              >
                ← Back
              </button>
              <button
                onClick={() => { setStep(3); handleGenerate(); }}
                className="px-6 py-[9px] bg-[#007AFF] text-white rounded-[9px] text-[13px] font-semibold glow-button btn-shimmer flex items-center gap-2"
              >
                <Sparkles className="h-4 w-4" />
                Generate {records.toLocaleString()} Records
              </button>
            </div>
          </GlassCard>
        )}

        {/* ═══════════ STEP 3: Generate / Results ═══════════ */}
        {step === 3 && (
          <GlassCard animatedBorder={genState.status === "running"}>
            {/* Running state */}
            {genState.status === "running" && (
              <>
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h2 className="text-[19px] font-semibold text-[#1D1D1F]">Generating...</h2>
                    <p className="text-[13px] text-[#86868B]">
                      {genState.recordsDone.toLocaleString()} of {genState.totalRecords.toLocaleString()} records
                    </p>
                  </div>
                  <ProgressRing progress={genState.progress * 100} />
                </div>

                {/* Live stats */}
                <div className="flex gap-3 mb-5">
                  <StatCard value={genState.recordsDone.toLocaleString()} label="Records" color="blue" />
                  <StatCard value={`${elapsed.toFixed(1)}s`} label="Elapsed" />
                  <StatCard value={`${recPerSec}/s`} label="Throughput" color="green" />
                </div>

                {/* Data waterfall */}
                {genState.data && schema && (
                  <DataWaterfall data={genState.data} schema={schema} />
                )}

                <button
                  onClick={genState.stop}
                  className="mt-5 w-full border border-[#FF3B30]/30 text-[#FF3B30] font-medium py-2.5 rounded-[9px] hover:bg-[#FF3B30]/[0.05] transition-colors text-[13px]"
                >
                  Stop Generation
                </button>
              </>
            )}

            {/* Error state */}
            {genState.status === "error" && (
              <div className="text-center py-8">
                <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-[#FF3B30]/10 mb-4">
                  <span className="text-[#FF3B30] text-xl">✕</span>
                </div>
                <h2 className="text-[19px] font-semibold text-[#1D1D1F] mb-2">Generation Failed</h2>
                <p className="text-[13px] text-[#86868B] mb-5">{genState.error || "An unexpected error occurred"}</p>
                <button
                  onClick={() => { genState.reset(); setStep(2); }}
                  className="px-5 py-[9px] bg-black/[0.04] border border-black/[0.08] rounded-[9px] text-[13px] font-medium text-[#3A3A3C] hover:bg-black/[0.06] transition-colors"
                >
                  Back to Configure
                </button>
              </div>
            )}

            {/* Complete state */}
            {genState.status === "complete" && genState.data && schema && (
              <>
                {/* Success header */}
                <div className="flex items-center gap-3 mb-6">
                  <div className="flex items-center justify-center w-10 h-10 rounded-full bg-[#34C759]/10">
                    <CheckCircle2 className="w-5 h-5 text-[#34C759]" />
                  </div>
                  <div>
                    <h2 className="text-[19px] font-semibold text-[#1D1D1F]">Generation Complete</h2>
                    <p className="text-[13px] text-[#86868B]">
                      {genState.data.length.toLocaleString()} records generated successfully
                    </p>
                  </div>
                </div>

                {/* Stats */}
                <div className="flex gap-3 mb-5">
                  <StatCard value={genState.data.length.toLocaleString()} label="Records" color="blue" delay={0} />
                  <StatCard value={Object.keys(schema).length} label="Columns" delay={100} />
                  <StatCard value={`${elapsed.toFixed(1)}s`} label="Elapsed" delay={200} />
                  <StatCard value={`${recPerSec}/s`} label="Rec/sec" color="green" delay={300} />
                </div>

                {/* Data preview */}
                <div className="mb-5">
                  <p className="text-[12px] font-medium text-[#86868B] mb-2">Data Preview</p>
                  <DataTable data={genState.data} maxRows={20} />
                </div>

                {/* Download */}
                <div className="flex items-center gap-3 mb-5">
                  <Download className="w-4 h-4 text-[#86868B]" />
                  <span className="text-[13px] font-medium text-[#1D1D1F]">Download</span>
                  <DownloadMenu data={genState.data} filename="synthetic_data" />
                </div>

                {/* Export */}
                <ExportPanel data={genState.data} schema={schema} filename="synthetic-data" />

                {/* New generation button */}
                <div className="flex justify-between mt-6 border-t border-black/[0.06] pt-5">
                  <button
                    onClick={() => { genState.reset(); setStep(2); }}
                    className="px-5 py-[9px] bg-black/[0.04] border border-black/[0.08] rounded-[9px] text-[13px] font-medium text-[#3A3A3C] hover:bg-black/[0.06] transition-colors"
                  >
                    ← Configure Again
                  </button>
                  <button
                    onClick={() => { genState.reset(); setStep(0); setSchema(null); setSampleRows([]); setProfile(null); }}
                    className="px-5 py-[9px] bg-[#007AFF] text-white rounded-[9px] text-[13px] font-semibold glow-button btn-shimmer"
                  >
                    New Generation
                  </button>
                </div>
              </>
            )}

            {/* Idle state (shouldn't normally show) */}
            {genState.status === "idle" && (
              <div className="text-center py-8">
                <p className="text-[13px] text-[#86868B]">Ready to generate. Go back to configure settings.</p>
                <button
                  onClick={() => setStep(2)}
                  className="mt-4 px-5 py-[9px] bg-black/[0.04] border border-black/[0.08] rounded-[9px] text-[13px] font-medium text-[#3A3A3C] hover:bg-black/[0.06] transition-colors"
                >
                  ← Back to Configure
                </button>
              </div>
            )}
          </GlassCard>
        )}
      </div>
    </div>
  );
}
