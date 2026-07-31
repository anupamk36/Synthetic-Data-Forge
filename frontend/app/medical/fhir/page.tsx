"use client";

import { useState, useCallback, useEffect } from "react";
import {
  HeartPulse,
  Download,
  CheckCircle2,
  Loader2,
  FileJson,
  Activity,
  Pill,
  Stethoscope,
  Building2,
  User,
  CalendarDays,
  ScanLine,
  ClipboardList,
  ShieldAlert,
} from "lucide-react";
import { toast } from "sonner";

import { TopBar } from "@/components/layout/top-bar";
import { GlassCard } from "@/components/shared/glass-card";
import { WizardStepper } from "@/components/shared/wizard-stepper";
import { StatCard } from "@/components/shared/stat-card";
import { ExportPanel } from "@/components/shared/export-panel";
import { QualityScanner } from "@/components/medical/quality-scanner";

import {
  generateFHIR,
  generateFHIRAsync,
  getFHIRJob,
  type FHIRGenerateRequest,
  type FHIRGenerateResponse,
} from "@/lib/api";

const WIZARD_STEPS = [
  { label: "Resources" },
  { label: "Configure" },
  { label: "Generate" },
  { label: "Results" },
];

const RESOURCE_TYPES = [
  { type: "Organization", icon: Building2, description: "Hospitals, clinics, labs" },
  { type: "Practitioner", icon: User, description: "Doctors, nurses, specialists" },
  { type: "Patient", icon: HeartPulse, description: "Demographics, identifiers" },
  { type: "Encounter", icon: CalendarDays, description: "Visits, admissions" },
  { type: "Condition", icon: ClipboardList, description: "Diagnoses (ICD-10)" },
  { type: "Observation", icon: Activity, description: "Lab results, vitals (LOINC)" },
  { type: "MedicationRequest", icon: Pill, description: "Prescriptions (RxNorm)" },
  { type: "Procedure", icon: Stethoscope, description: "Surgeries, procedures (SNOMED)" },
  { type: "DiagnosticReport", icon: FileJson, description: "Lab report summaries" },
  { type: "AllergyIntolerance", icon: ShieldAlert, description: "Allergies, reactions" },
  { type: "ImagingStudy", icon: ScanLine, description: "Imaging metadata (DICOM)" },
];

const PRESETS = {
  full: RESOURCE_TYPES.map((r) => r.type),
  labs: ["Organization", "Practitioner", "Patient", "Encounter", "Observation", "DiagnosticReport"],
  oncology: ["Organization", "Practitioner", "Patient", "Encounter", "Condition", "Observation", "MedicationRequest", "Procedure", "ImagingStudy"],
  minimal: ["Patient", "Encounter", "Condition", "Observation"],
};

const DENSITY_OPTIONS = ["low", "moderate", "high"] as const;
const FORMAT_OPTIONS = ["bundle", "ndjson", "individual", "tabular"] as const;
const FOCUS_OPTIONS = [null, "oncology", "cardiovascular", "respiratory", "endocrine", "neurological", "musculoskeletal"] as const;

export default function FHIRGeneratorPage() {
  const [step, setStep] = useState(0);

  // Step 1: Resource selection
  const [selectedTypes, setSelectedTypes] = useState<string[]>(PRESETS.full);

  // Step 2: Configuration
  const [patientCount, setPatientCount] = useState(100);
  const [encMin, setEncMin] = useState(1);
  const [encMax, setEncMax] = useState(5);
  const [density, setDensity] = useState<string>("moderate");
  const [format, setFormat] = useState<string>("bundle");
  const [bundleType] = useState<string>("collection");
  const [focus, setFocus] = useState<string | null>(null);
  const [seed, setSeed] = useState<number | null>(null);
  const [includeHl7v2, setIncludeHl7v2] = useState(false);

  // Step 3: Generation
  const [generating, setGenerating] = useState(false);
  const [progress, setProgress] = useState<Record<string, number>>({});
  const [jobId, setJobId] = useState<string | null>(null);

  // Step 4: Results
  const [result, setResult] = useState<FHIRGenerateResponse | null>(null);

  const toggleResource = (type: string) => {
    setSelectedTypes((prev) =>
      prev.includes(type) ? prev.filter((t) => t !== type) : [...prev, type]
    );
  };

  const applyPreset = (preset: keyof typeof PRESETS) => {
    setSelectedTypes(PRESETS[preset]);
  };

  const handleGenerate = useCallback(async () => {
    setGenerating(true);
    setProgress({});
    setStep(2);

    const req: FHIRGenerateRequest = {
      resource_types: selectedTypes,
      patient_count: patientCount,
      encounters_per_patient: { min: encMin, max: encMax },
      clinical_density: density,
      output_format: format,
      bundle_type: bundleType,
      include_narrative: false,
      terminology_focus: focus,
      seed: seed,
      include_hl7v2: includeHl7v2,
    };

    try {
      if (patientCount > 500) {
        const asyncRes = await generateFHIRAsync(req);
        setJobId(asyncRes.job_id);
      } else {
        const res = await generateFHIR(req);
        setResult(res);
        setGenerating(false);
        setStep(3);
        toast.success(`Generated ${res.stats.total} FHIR resources`);
      }
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Generation failed");
      setGenerating(false);
    }
  }, [selectedTypes, patientCount, encMin, encMax, density, format, bundleType, focus, seed, includeHl7v2]);

  // Poll async job
  useEffect(() => {
    if (!jobId) return;
    const interval = setInterval(async () => {
      try {
        const status = await getFHIRJob(jobId);
        setProgress(status.progress);
        if (status.status === "completed" && status.result) {
          setResult(status.result);
          setGenerating(false);
          setStep(3);
          setJobId(null);
          toast.success(`Generated ${status.result.stats.total} FHIR resources`);
        } else if (status.status === "failed") {
          toast.error(status.error || "Generation failed");
          setGenerating(false);
          setJobId(null);
        }
      } catch {
        // ignore transient poll errors
      }
    }, 1500);
    return () => clearInterval(interval);
  }, [jobId]);


  const handleDownloadHl7v2 = useCallback(() => {
    if (!result?.hl7v2_messages) return;
    const content = result.hl7v2_messages.join("\n\n");
    const blob = new Blob([content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "hl7v2_messages.hl7";
    a.click();
    URL.revokeObjectURL(url);
  }, [result]);

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      <TopBar title="FHIR Generator" />

      <div className="flex-1 px-6 pb-8 space-y-6">
        <WizardStepper steps={WIZARD_STEPS} current={step} />

        {/* Step 0: Resource Selection */}
        {step === 0 && (
          <div className="space-y-4">
            <GlassCard>
              <div className="p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-[15px] font-semibold text-[#1D1D1F]">Select Resource Types</h3>
                  <div className="flex gap-2">
                    {(Object.keys(PRESETS) as Array<keyof typeof PRESETS>).map((preset) => (
                      <button
                        key={preset}
                        onClick={() => applyPreset(preset)}
                        className="px-3 py-1 text-[11px] font-medium rounded-full border border-black/10 hover:bg-black/5 transition-colors capitalize"
                      >
                        {preset}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                  {RESOURCE_TYPES.map(({ type, icon: Icon, description }) => {
                    const selected = selectedTypes.includes(type);
                    return (
                      <button
                        key={type}
                        onClick={() => toggleResource(type)}
                        className={`flex items-center gap-3 p-3 rounded-xl border transition-all text-left ${
                          selected
                            ? "border-[#007AFF] bg-[rgba(0,122,255,0.05)]"
                            : "border-black/10 hover:border-black/20 hover:bg-black/[0.02]"
                        }`}
                      >
                        <Icon className={`size-5 shrink-0 ${selected ? "text-[#007AFF]" : "text-[#86868B]"}`} />
                        <div>
                          <div className={`text-[13px] font-medium ${selected ? "text-[#007AFF]" : "text-[#1D1D1F]"}`}>{type}</div>
                          <div className="text-[11px] text-[#86868B]">{description}</div>
                        </div>
                      </button>
                    );
                  })}
                </div>
                <div className="flex justify-between items-center pt-2">
                  <span className="text-[12px] text-[#86868B]">{selectedTypes.length} resource types selected</span>
                  <button
                    onClick={() => setStep(1)}
                    disabled={selectedTypes.length === 0}
                    className="px-5 py-2 bg-[#007AFF] text-white text-[13px] font-medium rounded-lg hover:bg-[#0066DD] disabled:opacity-40 transition-colors"
                  >
                    Continue
                  </button>
                </div>
              </div>
            </GlassCard>
          </div>
        )}

        {/* Step 1: Configuration */}
        {step === 1 && (
          <div className="space-y-4">
            <GlassCard>
              <div className="p-5 space-y-5">
                <h3 className="text-[15px] font-semibold text-[#1D1D1F]">Generation Configuration</h3>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                  {/* Patient count */}
                  <div>
                    <label className="block text-[12px] font-medium text-[#3A3A3C] mb-1">Patient Count</label>
                    <input
                      type="number"
                      min={1}
                      max={50000}
                      value={patientCount}
                      onChange={(e) => setPatientCount(Number(e.target.value))}
                      className="w-full px-3 py-2 text-[13px] rounded-lg border border-black/10 bg-white/70 focus:outline-none focus:ring-2 focus:ring-[#007AFF]/30"
                    />
                  </div>

                  {/* Encounters per patient */}
                  <div>
                    <label className="block text-[12px] font-medium text-[#3A3A3C] mb-1">Encounters per Patient</label>
                    <div className="flex gap-2 items-center">
                      <input
                        type="number"
                        min={1}
                        max={20}
                        value={encMin}
                        onChange={(e) => setEncMin(Number(e.target.value))}
                        className="w-20 px-3 py-2 text-[13px] rounded-lg border border-black/10 bg-white/70 focus:outline-none focus:ring-2 focus:ring-[#007AFF]/30"
                      />
                      <span className="text-[12px] text-[#86868B]">to</span>
                      <input
                        type="number"
                        min={1}
                        max={20}
                        value={encMax}
                        onChange={(e) => setEncMax(Number(e.target.value))}
                        className="w-20 px-3 py-2 text-[13px] rounded-lg border border-black/10 bg-white/70 focus:outline-none focus:ring-2 focus:ring-[#007AFF]/30"
                      />
                    </div>
                  </div>

                  {/* Clinical density */}
                  <div>
                    <label className="block text-[12px] font-medium text-[#3A3A3C] mb-1">Clinical Density</label>
                    <div className="flex gap-2">
                      {DENSITY_OPTIONS.map((d) => (
                        <button
                          key={d}
                          onClick={() => setDensity(d)}
                          className={`flex-1 px-3 py-2 text-[12px] font-medium rounded-lg border transition-colors capitalize ${
                            density === d
                              ? "border-[#007AFF] bg-[rgba(0,122,255,0.08)] text-[#007AFF]"
                              : "border-black/10 text-[#3A3A3C] hover:bg-black/[0.03]"
                          }`}
                        >
                          {d}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Terminology focus */}
                  <div>
                    <label className="block text-[12px] font-medium text-[#3A3A3C] mb-1">Terminology Focus</label>
                    <select
                      value={focus || ""}
                      onChange={(e) => setFocus(e.target.value || null)}
                      className="w-full px-3 py-2 text-[13px] rounded-lg border border-black/10 bg-white/70 focus:outline-none focus:ring-2 focus:ring-[#007AFF]/30"
                    >
                      <option value="">General (all domains)</option>
                      {FOCUS_OPTIONS.filter(Boolean).map((f) => (
                        <option key={f} value={f!}>{f!.charAt(0).toUpperCase() + f!.slice(1)}</option>
                      ))}
                    </select>
                  </div>

                  {/* Output format */}
                  <div>
                    <label className="block text-[12px] font-medium text-[#3A3A3C] mb-1">Output Format</label>
                    <select
                      value={format}
                      onChange={(e) => setFormat(e.target.value)}
                      className="w-full px-3 py-2 text-[13px] rounded-lg border border-black/10 bg-white/70 focus:outline-none focus:ring-2 focus:ring-[#007AFF]/30"
                    >
                      {FORMAT_OPTIONS.map((f) => (
                        <option key={f} value={f}>{f === "ndjson" ? "NDJSON (Bulk FHIR)" : f === "bundle" ? "FHIR Bundle" : f === "individual" ? "Individual Resources" : "Tabular (CSV/Parquet)"}</option>
                      ))}
                    </select>
                  </div>

                  {/* Seed */}
                  <div>
                    <label className="block text-[12px] font-medium text-[#3A3A3C] mb-1">Random Seed (optional)</label>
                    <input
                      type="number"
                      value={seed ?? ""}
                      onChange={(e) => setSeed(e.target.value ? Number(e.target.value) : null)}
                      placeholder="Leave empty for random"
                      className="w-full px-3 py-2 text-[13px] rounded-lg border border-black/10 bg-white/70 focus:outline-none focus:ring-2 focus:ring-[#007AFF]/30"
                    />
                  </div>
                </div>

                {/* HL7v2 toggle */}
                <div className="flex items-center gap-3 pt-2">
                  <input
                    type="checkbox"
                    id="hl7v2"
                    checked={includeHl7v2}
                    onChange={(e) => setIncludeHl7v2(e.target.checked)}
                    className="rounded border-black/20"
                  />
                  <label htmlFor="hl7v2" className="text-[13px] text-[#3A3A3C]">
                    Also generate HL7v2 messages (ADT, ORU)
                  </label>
                </div>

                <div className="flex justify-between pt-2">
                  <button
                    onClick={() => setStep(0)}
                    className="px-5 py-2 text-[13px] font-medium text-[#3A3A3C] rounded-lg border border-black/10 hover:bg-black/[0.03] transition-colors"
                  >
                    Back
                  </button>
                  <button
                    onClick={handleGenerate}
                    className="px-5 py-2 bg-[#007AFF] text-white text-[13px] font-medium rounded-lg hover:bg-[#0066DD] transition-colors flex items-center gap-2"
                  >
                    <HeartPulse className="size-4" />
                    Generate FHIR Data
                  </button>
                </div>
              </div>
            </GlassCard>
          </div>
        )}

        {/* Step 2: Generation Progress */}
        {step === 2 && (
          <GlassCard>
            <div className="p-5 space-y-4">
              <div className="flex items-center gap-3">
                {generating ? (
                  <Loader2 className="size-5 text-[#007AFF] animate-spin" />
                ) : (
                  <CheckCircle2 className="size-5 text-green-500" />
                )}
                <h3 className="text-[15px] font-semibold text-[#1D1D1F]">
                  {generating ? "Generating FHIR Resources..." : "Generation Complete"}
                </h3>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                {selectedTypes.map((type) => {
                  const count = progress[type];
                  const done = count !== undefined;
                  return (
                    <div
                      key={type}
                      className={`flex items-center gap-2 px-3 py-2 rounded-lg border ${
                        done ? "border-green-200 bg-green-50/50" : "border-black/5 bg-black/[0.02]"
                      }`}
                    >
                      {done ? (
                        <CheckCircle2 className="size-4 text-green-500 shrink-0" />
                      ) : (
                        <Loader2 className="size-4 text-[#86868B] animate-spin shrink-0" />
                      )}
                      <div>
                        <div className="text-[12px] font-medium text-[#1D1D1F]">{type}</div>
                        {done && <div className="text-[11px] text-[#86868B]">{count} generated</div>}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </GlassCard>
        )}

        {/* Step 3: Results */}
        {step === 3 && result && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <StatCard label="Total Resources" value={result.stats.total} />
              <StatCard label="Resource Types" value={Object.keys(result.stats.by_type).length} />
              <StatCard label="Patients" value={result.stats.by_type.Patient || 0} />
              <StatCard label="Format" value={result.format.toUpperCase()} />
            </div>

            <GlassCard>
              <div className="p-5 space-y-4">
                <h3 className="text-[15px] font-semibold text-[#1D1D1F]">Resource Breakdown</h3>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
                  {Object.entries(result.stats.by_type).map(([type, count]) => (
                    <div key={type} className="flex justify-between items-center px-3 py-2 rounded-lg bg-black/[0.02] border border-black/5">
                      <span className="text-[12px] text-[#3A3A3C]">{type}</span>
                      <span className="text-[13px] font-semibold text-[#1D1D1F]">{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </GlassCard>

            {result.hl7v2_messages && result.hl7v2_messages.length > 0 && (
              <div className="flex">
                <button
                  onClick={handleDownloadHl7v2}
                  className="flex items-center gap-2 px-4 py-2 bg-emerald-600 text-white text-[13px] font-medium rounded-lg hover:bg-emerald-700 transition-colors"
                >
                  <Download className="size-4" />
                  HL7v2 Messages ({result.hl7v2_count})
                </button>
              </div>
            )}

            <ExportPanel data={result.data} filename="fhir-bundle" />

            {/* AI Quality Scanner */}
            <QualityScanner data={result.data} dataType="fhir" />

            {/* JSON Preview */}
            <GlassCard>
              <div className="p-5 space-y-3">
                <h3 className="text-[15px] font-semibold text-[#1D1D1F]">Preview</h3>
                <pre className="max-h-[400px] overflow-auto p-4 bg-[#1D1D1F] text-green-300 text-[11px] rounded-xl font-mono leading-relaxed">
                  {JSON.stringify(
                    format === "bundle"
                      ? { ...(result.data as Record<string, unknown>), entry: ((result.data as Record<string, unknown>)?.entry as unknown[] || []).slice(0, 3) }
                      : result.data,
                    null,
                    2
                  ).slice(0, 5000)}
                  {JSON.stringify(result.data, null, 2).length > 5000 ? "\n... (truncated)" : ""}
                </pre>
              </div>
            </GlassCard>
          </div>
        )}
      </div>
    </div>
  );
}
