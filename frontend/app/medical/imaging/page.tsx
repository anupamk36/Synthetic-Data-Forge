"use client";

import { useState, useCallback, useEffect } from "react";
import { ScanLine, Download, Loader2 } from "lucide-react";
import { toast } from "sonner";

import { TopBar } from "@/components/layout/top-bar";
import { GlassCard } from "@/components/shared/glass-card";
import { WizardStepper } from "@/components/shared/wizard-stepper";
import { StatCard } from "@/components/shared/stat-card";
import { ExportPanel } from "@/components/shared/export-panel";
import { QualityScanner } from "@/components/medical/quality-scanner";

import {
  getImagingModalities,
  generateImaging,
  type ImagingGenerateRequest,
  type ImagingGenerateResponse,
  type ImagingModality,
  type ImagingBodyPart,
} from "@/lib/api";

const WIZARD_STEPS = [
  { label: "Modality" },
  { label: "Configure" },
  { label: "Results" },
];

const FORMAT_OPTIONS = [
  { value: "dicom_json", label: "DICOM JSON" },
  { value: "fhir", label: "FHIR ImagingStudy" },
  { value: "csv", label: "Flat CSV" },
];

export default function ImagingDataPage() {
  const [step, setStep] = useState(0);
  const [modalities, setModalities] = useState<ImagingModality[]>([]);
  const [bodyParts, setBodyParts] = useState<ImagingBodyPart[]>([]);

  const [selectedModalities, setSelectedModalities] = useState<string[]>(["CT"]);
  const [selectedBodyParts, setSelectedBodyParts] = useState<string[]>([]);
  const [numStudies, setNumStudies] = useState(50);
  const [includeInstances, setIncludeInstances] = useState(true);
  const [outputFormat, setOutputFormat] = useState("dicom_json");
  const [seed, setSeed] = useState<number | null>(null);

  const [generating, setGenerating] = useState(false);
  const [result, setResult] = useState<ImagingGenerateResponse | null>(null);

  useEffect(() => {
    getImagingModalities().then((data) => {
      setModalities(data.modalities);
      setBodyParts(data.body_parts);
    }).catch(() => {});
  }, []);

  const compatibleBodyParts = bodyParts.filter((bp) =>
    selectedModalities.some((m) => bp.modalities.includes(m))
  );

  const toggleModality = (code: string) => {
    setSelectedModalities((prev) =>
      prev.includes(code) ? prev.filter((m) => m !== code) : [...prev, code]
    );
  };

  const handleGenerate = useCallback(async () => {
    setGenerating(true);
    setStep(2);

    const req: ImagingGenerateRequest = {
      modalities: selectedModalities,
      body_parts: selectedBodyParts.length > 0 ? selectedBodyParts : null,
      num_studies: numStudies,
      include_instance_metadata: includeInstances,
      output_format: outputFormat,
      seed,
    };

    try {
      const res = await generateImaging(req);
      setResult(res);
      toast.success(`Generated ${res.stats.num_studies} imaging studies`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Generation failed");
    } finally {
      setGenerating(false);
    }
  }, [selectedModalities, selectedBodyParts, numStudies, includeInstances, outputFormat, seed]);

  const handleDownload = useCallback(() => {
    if (!result?.data) return;
    const json = JSON.stringify(result.data, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `imaging_${outputFormat}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [result, outputFormat]);

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      <TopBar title="Imaging Data" />

      <div className="flex-1 px-6 pb-8 space-y-6">
        <WizardStepper steps={WIZARD_STEPS} current={step} />

        {/* Step 0: Modality Selection */}
        {step === 0 && (
          <div className="space-y-4">
            <GlassCard>
              <div className="p-5 space-y-4">
                <h3 className="text-[15px] font-semibold text-[#1D1D1F]">Generate DICOM Imaging Metadata</h3>
                <p className="text-[13px] text-[#86868B]">
                  Generate synthetic DICOM imaging metadata including study, series, and instance-level attributes (UIDs, patient demographics, modality-specific parameters). Select one or more modalities to get started.
                </p>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                  {modalities.length === 0 && (
                    <p className="col-span-full text-[13px] text-[#86868B] italic py-4">Loading modalities... Ensure the backend API is running.</p>
                  )}
                  {modalities.map((mod) => {
                    const selected = selectedModalities.includes(mod.code);
                    return (
                      <button
                        key={mod.code}
                        onClick={() => toggleModality(mod.code)}
                        className={`p-3 rounded-xl border text-left transition-all ${
                          selected ? "border-[#007AFF] bg-[rgba(0,122,255,0.05)]" : "border-black/10 hover:border-black/20"
                        }`}
                      >
                        <div className={`text-[14px] font-semibold ${selected ? "text-[#007AFF]" : "text-[#1D1D1F]"}`}>{mod.code}</div>
                        <div className="text-[11px] text-[#86868B]">{mod.display}</div>
                      </button>
                    );
                  })}
                </div>

                {compatibleBodyParts.length > 0 && (
                  <>
                    <h4 className="text-[13px] font-medium text-[#3A3A3C] pt-2">Body Parts (optional filter)</h4>
                    <div className="flex flex-wrap gap-2">
                      {compatibleBodyParts.map((bp) => (
                        <button
                          key={bp.code}
                          onClick={() => setSelectedBodyParts((prev) => prev.includes(bp.code) ? prev.filter((b) => b !== bp.code) : [...prev, bp.code])}
                          className={`px-3 py-1.5 text-[12px] rounded-full border transition-colors ${
                            selectedBodyParts.includes(bp.code)
                              ? "border-[#007AFF] bg-[rgba(0,122,255,0.08)] text-[#007AFF]"
                              : "border-black/10 text-[#3A3A3C] hover:bg-black/[0.03]"
                          }`}
                        >
                          {bp.display}
                        </button>
                      ))}
                    </div>
                  </>
                )}

                <div className="flex justify-end pt-2">
                  <button onClick={() => setStep(1)} disabled={selectedModalities.length === 0}
                    className="px-5 py-2 bg-[#007AFF] text-white text-[13px] font-medium rounded-lg hover:bg-[#0066DD] disabled:opacity-40 transition-colors">
                    Continue
                  </button>
                </div>
              </div>
            </GlassCard>
          </div>
        )}

        {/* Step 1: Configuration */}
        {step === 1 && (
          <GlassCard>
            <div className="p-5 space-y-5">
              <h3 className="text-[15px] font-semibold text-[#1D1D1F]">Imaging Configuration</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                <div>
                  <label className="block text-[12px] font-medium text-[#3A3A3C] mb-1">Number of Studies</label>
                  <input type="number" min={1} max={10000} value={numStudies} onChange={(e) => setNumStudies(Number(e.target.value))}
                    className="w-full px-3 py-2 text-[13px] rounded-lg border border-black/10 bg-white/70 focus:outline-none focus:ring-2 focus:ring-[#007AFF]/30" />
                </div>
                <div>
                  <label className="block text-[12px] font-medium text-[#3A3A3C] mb-1">Output Format</label>
                  <select value={outputFormat} onChange={(e) => setOutputFormat(e.target.value)}
                    className="w-full px-3 py-2 text-[13px] rounded-lg border border-black/10 bg-white/70 focus:outline-none focus:ring-2 focus:ring-[#007AFF]/30">
                    {FORMAT_OPTIONS.map((f) => <option key={f.value} value={f.value}>{f.label}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-[12px] font-medium text-[#3A3A3C] mb-1">Random Seed (optional)</label>
                  <input type="number" value={seed ?? ""} onChange={(e) => setSeed(e.target.value ? Number(e.target.value) : null)} placeholder="Leave empty for random"
                    className="w-full px-3 py-2 text-[13px] rounded-lg border border-black/10 bg-white/70 focus:outline-none focus:ring-2 focus:ring-[#007AFF]/30" />
                </div>
                <div className="flex items-center gap-3">
                  <input type="checkbox" id="instances" checked={includeInstances} onChange={(e) => setIncludeInstances(e.target.checked)} className="rounded border-black/20" />
                  <label htmlFor="instances" className="text-[13px] text-[#3A3A3C]">Include instance-level metadata</label>
                </div>
              </div>
              <div className="flex justify-between pt-2">
                <button onClick={() => setStep(0)} className="px-5 py-2 text-[13px] font-medium text-[#3A3A3C] rounded-lg border border-black/10 hover:bg-black/[0.03] transition-colors">Back</button>
                <button onClick={handleGenerate} className="px-5 py-2 bg-[#007AFF] text-white text-[13px] font-medium rounded-lg hover:bg-[#0066DD] transition-colors flex items-center gap-2">
                  <ScanLine className="size-4" /> Generate Imaging Data
                </button>
              </div>
            </div>
          </GlassCard>
        )}

        {/* Step 2: Results */}
        {step === 2 && (
          <div className="space-y-4">
            {generating ? (
              <GlassCard>
                <div className="p-5 flex items-center gap-3">
                  <Loader2 className="size-5 text-[#007AFF] animate-spin" />
                  <span className="text-[14px] text-[#1D1D1F]">Generating imaging metadata...</span>
                </div>
              </GlassCard>
            ) : result && (
              <>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <StatCard label="Studies" value={result.stats.num_studies} />
                  <StatCard label="Series" value={result.stats.total_series} />
                  <StatCard label="Instances" value={result.stats.total_instances} />
                  <StatCard label="Time" value={`${result.stats.elapsed_seconds}s`} />
                </div>

                <GlassCard>
                  <div className="p-5 space-y-4">
                    <div className="flex items-center justify-between">
                      <h3 className="text-[15px] font-semibold text-[#1D1D1F]">Generated Data</h3>
                      <button onClick={handleDownload} className="flex items-center gap-2 px-4 py-2 bg-[#007AFF] text-white text-[13px] font-medium rounded-lg hover:bg-[#0066DD] transition-colors">
                        <Download className="size-4" /> Download {outputFormat === "dicom_json" ? "DICOM JSON" : outputFormat === "fhir" ? "FHIR Bundle" : "CSV"}
                      </button>
                    </div>
                    <div className="flex gap-2 flex-wrap">
                      {result.stats.modalities.map((m) => (
                        <span key={m} className="px-3 py-1 text-[12px] font-medium rounded-full bg-black/5 text-[#3A3A3C]">{m}</span>
                      ))}
                    </div>
                    <pre className="max-h-[400px] overflow-auto p-4 bg-[#1D1D1F] text-green-300 text-[11px] rounded-xl font-mono leading-relaxed">
                      {JSON.stringify(
                        Array.isArray(result.data) ? result.data.slice(0, 2) : result.data,
                        null, 2
                      ).slice(0, 5000)}
                      {JSON.stringify(result.data).length > 5000 ? "\n... (truncated)" : ""}
                    </pre>
                  </div>
                </GlassCard>

                <ExportPanel data={result.data} filename="imaging-data" />

                {/* AI Quality Scanner */}
                <QualityScanner data={result.data} dataType="dicom" />
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
