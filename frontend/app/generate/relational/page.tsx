"use client";

import { useState, useCallback } from "react";
import {
  Loader2, Sparkles, Plus, Trash2, Table2, Link2, CheckCircle2,
} from "lucide-react";
import { toast } from "sonner";

import { TopBar } from "@/components/layout/top-bar";
import { FileUpload } from "@/components/data/file-upload";
import { SchemaEditor } from "@/components/data/schema-editor";
import { DataTable } from "@/components/data/data-table";
import { DownloadMenu } from "@/components/data/download-menu";
import { GlassCard } from "@/components/shared/glass-card";
import { WizardStepper } from "@/components/shared/wizard-stepper";
import { StatCard } from "@/components/shared/stat-card";
import { ExportPanel } from "@/components/shared/export-panel";
import { GraphCanvas } from "@/components/relational/graph-canvas";

import { uploadFile, generateRelational } from "@/lib/api";
import type { Schema } from "@/lib/types";

interface TableEntry {
  name: string;
  schema: Schema;
  sampleRows: Record<string, unknown>[];
  rowCount: number;
}

interface Relationship {
  parent_table: string;
  parent_col: string;
  child_table: string;
  child_col: string;
}

const STEPS = ["Upload", "Relationships", "Configure", "Generate"];

export default function RelationalPage() {
  const [step, setStep] = useState(0);

  /* ── Table state ── */
  const [tables, setTables] = useState<TableEntry[]>([]);
  const [uploading, setUploading] = useState(false);
  const [activeTab, setActiveTab] = useState(0);

  /* ── Relationship state ── */
  const [relationships, setRelationships] = useState<Relationship[]>([]);
  const [newRel, setNewRel] = useState<Relationship>({
    parent_table: "",
    parent_col: "",
    child_table: "",
    child_col: "",
  });

  /* ── Configure state ── */
  const [counts, setCounts] = useState<Record<string, number>>({});

  /* ── Generation state ── */
  const [generating, setGenerating] = useState(false);
  const [results, setResults] = useState<Record<string, Record<string, unknown>[]> | null>(null);
  const [resultTab, setResultTab] = useState(0);
  const [elapsed, setElapsed] = useState(0);

  /* ── Upload handler ── */
  const handleUploadMultiple = useCallback(async (files: File[]) => {
    setUploading(true);
    try {
      for (const file of files) {
        const result = await uploadFile(file);
        const name = file.name.replace(/\.(csv|json|parquet|xlsx?)$/i, "");
        setTables((prev) => [
          ...prev,
          { name, schema: result.schema, sampleRows: result.sample_rows, rowCount: result.row_count },
        ]);
        setCounts((prev) => ({ ...prev, [name]: 1000 }));
        toast.success(`"${name}" uploaded — ${result.row_count} rows`);
      }
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }, []);

  /* ── Relationship actions ── */
  const addRelationship = useCallback(() => {
    if (!newRel.parent_table || !newRel.parent_col || !newRel.child_table || !newRel.child_col) {
      toast.error("Fill in all relationship fields");
      return;
    }
    setRelationships((prev) => [...prev, { ...newRel }]);
    setNewRel({ parent_table: "", parent_col: "", child_table: "", child_col: "" });
    toast.success("Relationship added");
  }, [newRel]);

  const removeRelationship = useCallback((index: number) => {
    setRelationships((prev) => prev.filter((_, i) => i !== index));
  }, []);

  /* ── Generate handler ── */
  const handleGenerate = useCallback(async () => {
    if (tables.length < 2) {
      toast.error("Upload at least 2 tables");
      return;
    }
    if (relationships.length === 0) {
      toast.error("Define at least one relationship");
      return;
    }

    setGenerating(true);
    setStep(3);
    const startTime = Date.now();

    try {
      const tablesMap: Record<string, Schema> = {};
      const sourceData: Record<string, Record<string, unknown>[]> = {};
      tables.forEach((t) => {
        tablesMap[t.name] = t.schema;
        sourceData[t.name] = t.sampleRows;
      });

      const data = await generateRelational({
        tables: tablesMap,
        relationships,
        counts,
        source_data: sourceData,
      });

      setResults(data);
      setElapsed((Date.now() - startTime) / 1000);
      setResultTab(0);
      toast.success("Relational data generated!");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Generation failed");
    } finally {
      setGenerating(false);
    }
  }, [tables, relationships, counts]);

  const tableNames = tables.map((t) => t.name);
  const resultTableNames = results ? Object.keys(results) : [];
  const graphTables = tables.map((t) => ({
    name: t.name,
    columnCount: Object.keys(t.schema).length,
    columns: Object.keys(t.schema),
  }));

  let topBarStatus: "running" | "complete" | undefined;
  if (generating) {
    topBarStatus = "running";
  } else if (results) {
    topBarStatus = "complete";
  }

  return (
    <div className="flex flex-col h-full">
      <TopBar title="Multi-Table Generator" status={topBarStatus} />

      <div className="flex-1 overflow-y-auto relative z-10">
        {/* Wizard Stepper */}
        <div className="pt-6 pb-4 flex justify-center">
          <WizardStepper steps={STEPS} current={step} />
        </div>

        <div className="px-6 pb-8 max-w-5xl mx-auto">
          {/* ═══════════ Step 0: Upload ═══════════ */}
          {step === 0 && (
            <GlassCard animatedBorder>
              <div className="p-7 space-y-5">
                <div>
                  <h2 className="text-[16px] font-semibold text-[#1D1D1F]">Upload Tables</h2>
                  <p className="text-[13px] text-[#86868B] mt-1">Upload 2 or more related tables to define their schema and relationships.</p>
                </div>

                <FileUpload multiple onFilesAccepted={handleUploadMultiple} loading={uploading} />

                {tables.length > 0 && (
                  <div className="space-y-4">
                    {/* Uploaded table cards */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                      {tables.map((t, i) => (
                        <button
                          key={t.name}
                          onClick={() => setActiveTab(i)}
                          className={`rounded-xl border p-4 text-left transition-all hover-lift ${
                            activeTab === i
                              ? "border-[#007AFF]/30 bg-[#007AFF]/[0.03]"
                              : "border-black/[0.06] bg-white/50"
                          }`}
                        >
                          <div className="flex items-center gap-2.5">
                            <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-[#007AFF]/10">
                              <Table2 className="w-4 h-4 text-[#007AFF]" />
                            </div>
                            <div>
                              <p className="text-[13px] font-semibold text-[#1D1D1F]">{t.name}</p>
                              <p className="text-[10px] text-[#86868B]">{Object.keys(t.schema).length} cols · {t.rowCount} rows</p>
                            </div>
                          </div>
                        </button>
                      ))}
                    </div>

                    {/* Schema editor for selected table */}
                    <div className="rounded-xl border border-black/[0.06] bg-white/50 p-4">
                      <p className="text-[11px] font-semibold uppercase tracking-[0.5px] text-[#86868B] mb-3">
                        Schema — {tables[activeTab]?.name}
                      </p>
                      {tables[activeTab] && (
                        <SchemaEditor
                          schema={tables[activeTab].schema}
                          onChange={(s) => {
                            setTables((prev) =>
                              prev.map((t, i) => (i === activeTab ? { ...t, schema: s } : t))
                            );
                          }}
                          profile={undefined}
                        />
                      )}
                    </div>

                    {/* Data preview */}
                    {tables[activeTab]?.sampleRows.length > 0 && (
                      <div className="rounded-xl border border-black/[0.06] bg-white/50 p-4">
                        <p className="text-[11px] font-semibold uppercase tracking-[0.5px] text-[#86868B] mb-3">
                          Preview — {tables[activeTab].name}
                        </p>
                        <DataTable data={tables[activeTab].sampleRows} maxRows={8} />
                      </div>
                    )}
                  </div>
                )}

                {/* Navigation */}
                <div className="flex justify-end pt-2">
                  <button
                    onClick={() => setStep(1)}
                    disabled={tables.length < 2}
                    className="glow-button bg-[#007AFF] text-white font-semibold text-[13px] px-6 py-2.5 rounded-lg disabled:opacity-40 disabled:cursor-not-allowed btn-shimmer"
                  >
                    Continue to Relationships
                  </button>
                </div>
              </div>
            </GlassCard>
          )}

          {/* ═══════════ Step 1: Relationships ═══════════ */}
          {step === 1 && (
            <GlassCard animatedBorder>
              <div className="p-7 space-y-5">
                <div>
                  <h2 className="text-[16px] font-semibold text-[#1D1D1F]">Define Relationships</h2>
                  <p className="text-[13px] text-[#86868B] mt-1">Connect tables with foreign key relationships. Drag nodes to rearrange.</p>
                </div>

                {/* Graph Canvas */}
                <GraphCanvas
                  tables={graphTables}
                  relationships={relationships}
                />

                {/* Existing relationships list */}
                {relationships.length > 0 && (
                  <div className="space-y-2">
                    {relationships.map((rel, i) => (
                      <div
                        key={`${rel.parent_table}.${rel.parent_col}-${rel.child_table}.${rel.child_col}`}
                        className="rounded-xl border border-black/[0.06] bg-white/60 p-3 flex items-center justify-between animate-slide-up"
                        style={{ animationDelay: `${i * 50}ms` }}
                      >
                        <div className="flex items-center gap-2 text-[12px]">
                          <span className="font-semibold text-[#007AFF]">{rel.parent_table}</span>
                          <span className="text-[#86868B]">.{rel.parent_col}</span>
                          <Link2 className="h-3.5 w-3.5 text-[#AF82FF]" />
                          <span className="font-semibold text-[#AF82FF]">{rel.child_table}</span>
                          <span className="text-[#86868B]">.{rel.child_col}</span>
                        </div>
                        <button
                          onClick={() => removeRelationship(i)}
                          className="text-[#86868B] hover:text-[#FF3B30] transition-colors"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    ))}
                  </div>
                )}

                {/* Add relationship form */}
                <div className="rounded-xl border border-black/[0.06] bg-white/50 p-5">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.5px] text-[#86868B] mb-3">
                    Add Relationship
                  </p>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    <div>
                      <span className="text-[11px] text-[#86868B] mb-1 block">Parent Table</span>
                      <select
                        value={newRel.parent_table}
                        onChange={(e) => setNewRel((r) => ({ ...r, parent_table: e.target.value, parent_col: "" }))}
                        className="w-full rounded-lg border border-black/[0.08] bg-white/80 px-3 py-2 text-[12px] focus:border-[#007AFF] focus:ring-1 focus:ring-[#007AFF]/20 outline-none transition-all"
                      >
                        <option value="">Select...</option>
                        {tableNames.map((n) => (
                          <option key={n} value={n}>{n}</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <span className="text-[11px] text-[#86868B] mb-1 block">Parent Column</span>
                      <select
                        value={newRel.parent_col}
                        onChange={(e) => setNewRel((r) => ({ ...r, parent_col: e.target.value }))}
                        className="w-full rounded-lg border border-black/[0.08] bg-white/80 px-3 py-2 text-[12px] focus:border-[#007AFF] focus:ring-1 focus:ring-[#007AFF]/20 outline-none transition-all"
                      >
                        <option value="">Select...</option>
                        {newRel.parent_table &&
                          Object.keys(
                            tables.find((t) => t.name === newRel.parent_table)?.schema ?? {}
                          ).map((col) => (
                            <option key={col} value={col}>{col}</option>
                          ))}
                      </select>
                    </div>
                    <div>
                      <span className="text-[11px] text-[#86868B] mb-1 block">Child Table</span>
                      <select
                        value={newRel.child_table}
                        onChange={(e) => setNewRel((r) => ({ ...r, child_table: e.target.value, child_col: "" }))}
                        className="w-full rounded-lg border border-black/[0.08] bg-white/80 px-3 py-2 text-[12px] focus:border-[#007AFF] focus:ring-1 focus:ring-[#007AFF]/20 outline-none transition-all"
                      >
                        <option value="">Select...</option>
                        {tableNames.map((n) => (
                          <option key={n} value={n}>{n}</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <span className="text-[11px] text-[#86868B] mb-1 block">Child Column</span>
                      <select
                        value={newRel.child_col}
                        onChange={(e) => setNewRel((r) => ({ ...r, child_col: e.target.value }))}
                        className="w-full rounded-lg border border-black/[0.08] bg-white/80 px-3 py-2 text-[12px] focus:border-[#007AFF] focus:ring-1 focus:ring-[#007AFF]/20 outline-none transition-all"
                      >
                        <option value="">Select...</option>
                        {newRel.child_table &&
                          Object.keys(
                            tables.find((t) => t.name === newRel.child_table)?.schema ?? {}
                          ).map((col) => (
                            <option key={col} value={col}>{col}</option>
                          ))}
                      </select>
                    </div>
                  </div>
                  <button
                    onClick={addRelationship}
                    className="mt-3 flex items-center gap-1.5 text-[12px] font-semibold text-[#007AFF] hover:text-[#005EC4] transition-colors"
                  >
                    <Plus className="h-4 w-4" />
                    Add Relationship
                  </button>
                </div>

                {/* Navigation */}
                <div className="flex justify-between pt-2">
                  <button
                    onClick={() => setStep(0)}
                    className="text-[13px] font-medium text-[#86868B] hover:text-[#1D1D1F] transition-colors px-4 py-2.5"
                  >
                    ← Back
                  </button>
                  <button
                    onClick={() => setStep(2)}
                    disabled={relationships.length === 0}
                    className="glow-button bg-[#007AFF] text-white font-semibold text-[13px] px-6 py-2.5 rounded-lg disabled:opacity-40 disabled:cursor-not-allowed btn-shimmer"
                  >
                    Continue to Configure
                  </button>
                </div>
              </div>
            </GlassCard>
          )}

          {/* ═══════════ Step 2: Configure ═══════════ */}
          {step === 2 && (
            <GlassCard animatedBorder>
              <div className="p-7 space-y-5">
                <div>
                  <h2 className="text-[16px] font-semibold text-[#1D1D1F]">Configure Generation</h2>
                  <p className="text-[13px] text-[#86868B] mt-1">Set the number of records to generate for each table.</p>
                </div>

                {/* Per-table count cards */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                  {tables.map((t, i) => (
                    <div
                      key={t.name}
                      className="rounded-xl border border-black/[0.06] bg-white/60 p-4 animate-slide-up"
                      style={{ animationDelay: `${i * 80}ms` }}
                    >
                      <div className="flex items-center gap-2 mb-3">
                        <div className="flex items-center justify-center w-7 h-7 rounded-lg bg-[#007AFF]/10">
                          <Table2 className="w-3.5 h-3.5 text-[#007AFF]" />
                        </div>
                        <span className="text-[13px] font-semibold text-[#1D1D1F]">{t.name}</span>
                      </div>
                      <input
                        type="number"
                        min={1}
                        value={counts[t.name] ?? 1000}
                        onChange={(e) =>
                          setCounts((prev) => ({
                            ...prev,
                            [t.name]: Math.max(1, Number.parseInt(e.target.value) || 1),
                          }))
                        }
                        className="w-full rounded-lg border border-black/[0.08] bg-white/80 px-3 py-2 text-[13px] font-mono focus:border-[#007AFF] focus:ring-1 focus:ring-[#007AFF]/20 outline-none transition-all"
                      />
                      <p className="text-[10px] text-[#86868B] mt-1.5">{Object.keys(t.schema).length} columns</p>
                    </div>
                  ))}
                </div>

                {/* Summary stats */}
                <div className="grid grid-cols-3 gap-3">
                  <StatCard label="Tables" value={tables.length} color="blue" />
                  <StatCard
                    label="Total Records"
                    value={Object.values(counts).reduce((a, b) => a + b, 0).toLocaleString()}
                    color="green"
                    delay={100}
                  />
                  <StatCard label="Relationships" value={relationships.length} delay={200} />
                </div>

                {/* Navigation */}
                <div className="flex justify-between pt-2">
                  <button
                    onClick={() => setStep(1)}
                    className="text-[13px] font-medium text-[#86868B] hover:text-[#1D1D1F] transition-colors px-4 py-2.5"
                  >
                    ← Back
                  </button>
                  <button
                    onClick={handleGenerate}
                    disabled={generating}
                    className="glow-button bg-[#007AFF] text-white font-semibold text-[13px] px-6 py-2.5 rounded-lg disabled:opacity-40 disabled:cursor-not-allowed btn-shimmer flex items-center gap-2"
                  >
                    <Sparkles className="w-4 h-4" />
                    Generate All Tables
                  </button>
                </div>
              </div>
            </GlassCard>
          )}

          {/* ═══════════ Step 3: Generate ═══════════ */}
          {step === 3 && (
            <GlassCard animatedBorder>
              <div className="p-7 space-y-5">
                {/* Generating state */}
                {generating && !results && (
                  <div className="space-y-5">
                    <div className="flex flex-col items-center justify-center py-8">
                      <Loader2 className="h-10 w-10 animate-spin text-[#007AFF] mb-4" />
                      <p className="text-[14px] font-medium text-[#1D1D1F]">Generating relational data...</p>
                      <p className="text-[12px] text-[#86868B] mt-1">Maintaining FK integrity across tables</p>
                    </div>

                    {/* Graph with generating state */}
                    <GraphCanvas
                      tables={graphTables}
                      relationships={relationships}
                      generating={true}
                    />
                  </div>
                )}

                {/* Results state */}
                {results && (
                  <div className="space-y-5">
                    <div className="flex items-center gap-3">
                      <div className="flex items-center justify-center w-10 h-10 rounded-full bg-[#34C759]/10">
                        <CheckCircle2 className="w-5 h-5 text-[#34C759]" />
                      </div>
                      <div>
                        <h2 className="text-[16px] font-semibold text-[#1D1D1F]">Generation Complete</h2>
                        <p className="text-[12px] text-[#86868B]">{elapsed.toFixed(1)}s · All FK constraints satisfied</p>
                      </div>
                    </div>

                    {/* Stats */}
                    <div className="grid grid-cols-3 gap-3">
                      <StatCard label="Tables" value={resultTableNames.length} color="blue" />
                      <StatCard
                        label="Total Rows"
                        value={resultTableNames.reduce((sum, name) => sum + results[name].length, 0).toLocaleString()}
                        color="green"
                        delay={100}
                      />
                      <StatCard label="Time" value={`${elapsed.toFixed(1)}s`} delay={200} />
                    </div>

                    {/* Graph in complete state */}
                    <GraphCanvas
                      tables={graphTables}
                      relationships={relationships}
                      completedTables={resultTableNames}
                    />

                    {/* Result table tabs */}
                    <div className="rounded-xl border border-black/[0.06] bg-white/50 overflow-hidden">
                      <div className="flex gap-0 border-b border-black/[0.06] bg-black/[0.02]">
                        {resultTableNames.map((name, i) => (
                          <button
                            key={name}
                            onClick={() => setResultTab(i)}
                            className={`px-5 py-2.5 text-[12px] font-medium transition-all border-b-2 ${
                              resultTab === i
                                ? "border-[#007AFF] text-[#007AFF] bg-[#007AFF]/[0.03]"
                                : "border-transparent text-[#86868B] hover:text-[#1D1D1F]"
                            }`}
                          >
                            {name}
                            <span className="ml-1.5 text-[10px] opacity-60">
                              ({results[name].length})
                            </span>
                          </button>
                        ))}
                      </div>

                      <div className="p-4">
                        {resultTableNames[resultTab] && (
                          <DataTable data={results[resultTableNames[resultTab]]} maxRows={20} />
                        )}
                      </div>
                    </div>

                    {/* Download */}
                    {resultTableNames[resultTab] && (
                      <DownloadMenu
                        data={results[resultTableNames[resultTab]]}
                        filename={resultTableNames[resultTab]}
                      />
                    )}

                    <ExportPanel data={results} filename="relational-data" />

                    {/* Start over */}
                    <div className="flex justify-center pt-2">
                      <button
                        onClick={() => {
                          setResults(null);
                          setStep(0);
                        }}
                        className="text-[13px] font-medium text-[#86868B] hover:text-[#1D1D1F] transition-colors"
                      >
                        ← Start New Generation
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </GlassCard>
          )}
        </div>
      </div>
    </div>
  );
}
