"use client";

import { useState, useCallback } from "react";
import {
  FileText,
  Loader2,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Copy,
  Download,
} from "lucide-react";
import { toast } from "sonner";

import { TopBar } from "@/components/layout/top-bar";
import { GlassCard } from "@/components/shared/glass-card";
import { WizardStepper } from "@/components/shared/wizard-stepper";
import { StatCard } from "@/components/shared/stat-card";

import {
  generateFHIR,
  type NarrativeDocument,
} from "@/lib/api";
import { useForgeStore } from "@/lib/store";

const WIZARD_STEPS = [
  { label: "Doc Types" },
  { label: "Configure" },
  { label: "Generating" },
  { label: "Results" },
];

const DOC_TYPE_OPTIONS = [
  { value: "discharge_summary", label: "Discharge Summary", description: "End-of-stay clinical summary for inpatient encounters" },
  { value: "radiology_report", label: "Radiology Report", description: "Imaging findings and impressions (CT, MRI, X-Ray)" },
  { value: "pathology_report", label: "Pathology Report", description: "Tissue and specimen analysis results" },
  { value: "clinical_note", label: "Clinical Note", description: "Provider progress notes and assessments" },
  { value: "operative_note", label: "Operative Note", description: "Surgical procedure documentation" },
];

const DENSITY_OPTIONS = ["low", "moderate", "high"] as const;

export default function ClinicalNotesPage() {
  const { provider, model, apiKey } = useForgeStore();
  const [step, setStep] = useState(0);

  // Step 0: doc type selection
  const [selectedDocTypes, setSelectedDocTypes] = useState<string[]>(
    DOC_TYPE_OPTIONS.map((d) => d.value)
  );

  // Step 1: configuration
  const [patientCount, setPatientCount] = useState(5);
  const [density, setDensity] = useState<string>("moderate");

  // Step 2: generating
  const [generating, setGenerating] = useState(false);

  // Step 3: results
  const [documents, setDocuments] = useState<NarrativeDocument[]>([]);
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());

  const toggleDocType = (value: string) => {
    setSelectedDocTypes((prev) =>
      prev.includes(value) ? prev.filter((t) => t !== value) : [...prev, value]
    );
  };

  const toggleExpanded = (id: string) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const handleCopy = useCallback((text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      toast.success("Copied to clipboard");
    });
  }, []);

  const handleDownloadAll = useCallback(() => {
    if (documents.length === 0) return;
    const content = documents
      .map((doc) => `=== ${doc.type} (${doc.id}) ===\n\n${doc.text}`)
      .join("\n\n" + "─".repeat(60) + "\n\n");
    const blob = new Blob([content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "clinical-notes.txt";
    a.click();
    URL.revokeObjectURL(url);
    toast.success(`Downloaded ${documents.length} clinical documents`);
  }, [documents]);

  const handleGenerate = useCallback(async () => {
    setGenerating(true);
    setDocuments([]);
    setExpandedIds(new Set());
    setStep(2);

    try {
      const res = await generateFHIR({
        resource_types: ["Organization", "Practitioner", "Patient", "Encounter",
          "Condition", "Observation", "MedicationRequest", "Procedure",
          "AllergyIntolerance", "ImagingStudy", "DocumentReference"],
        patient_count: patientCount,
        encounters_per_patient: { min: 1, max: 3 },
        clinical_density: density,
        output_format: "bundle",
        bundle_type: "collection",
        include_narrative: true,
        terminology_focus: null,
        seed: null,
        include_hl7v2: false,
        narrative_doc_types: selectedDocTypes.length > 0 ? selectedDocTypes : undefined,
        narrative_provider: provider,
        narrative_api_key: apiKey || undefined,
        narrative_model: model || undefined,
      });

      // Parse DocumentReference resources from the bundle
      const bundle = res.data as { entry?: { resource?: Record<string, unknown> }[] };
      const entries = bundle?.entry ?? [];
      const docs: NarrativeDocument[] = [];

      for (const entry of entries) {
        const resource = entry.resource;
        if (!resource || resource.resourceType !== "DocumentReference") continue;

        const docId = resource.id as string;
        const typeCoding = (resource.type as { coding?: { display?: string }[] })?.coding ?? [];
        const docType = typeCoding[0]?.display ?? "Clinical Document";

        const contentArr = resource.content as { attachment?: { data?: string } }[] | undefined;
        let text = "";
        if (contentArr && contentArr.length > 0) {
          const b64 = contentArr[0]?.attachment?.data ?? "";
          if (b64) {
            try {
              text = atob(b64);
            } catch {
              text = b64;
            }
          }
        }

        docs.push({ id: docId, type: docType, text, document_reference: resource });
      }

      setDocuments(docs);
      setGenerating(false);
      setStep(3);

      if (docs.length > 0) {
        toast.success(`Generated ${docs.length} clinical document${docs.length !== 1 ? "s" : ""}`);
      } else {
        toast.warning("No clinical documents were generated. Check that the LLM provider is reachable.");
      }
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Generation failed");
      setGenerating(false);
      setStep(1);
    }
  }, [selectedDocTypes, patientCount, density, provider, model, apiKey]);

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      <TopBar title="Clinical Notes" />

      <div className="flex-1 px-6 pb-8 space-y-6">
        <WizardStepper steps={WIZARD_STEPS} current={step} />

        {/* Step 0: Document Type Selection */}
        {step === 0 && (
          <GlassCard>
            <div className="p-5 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-[15px] font-semibold text-[#1D1D1F]">Select Document Types</h3>
                <div className="flex gap-2">
                  <button
                    onClick={() => setSelectedDocTypes(DOC_TYPE_OPTIONS.map((d) => d.value))}
                    className="px-3 py-1 text-[11px] font-medium rounded-full border border-black/10 hover:bg-black/5 transition-colors"
                  >
                    All
                  </button>
                  <button
                    onClick={() => setSelectedDocTypes([])}
                    className="px-3 py-1 text-[11px] font-medium rounded-full border border-black/10 hover:bg-black/5 transition-colors"
                  >
                    None
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {DOC_TYPE_OPTIONS.map(({ value, label, description }) => {
                  const selected = selectedDocTypes.includes(value);
                  return (
                    <button
                      key={value}
                      onClick={() => toggleDocType(value)}
                      className={`flex items-start gap-3 p-4 rounded-xl border text-left transition-all ${
                        selected
                          ? "border-[#007AFF] bg-[rgba(0,122,255,0.05)]"
                          : "border-black/10 hover:border-black/20 hover:bg-black/[0.02]"
                      }`}
                    >
                      <FileText
                        className={`size-5 shrink-0 mt-[1px] ${selected ? "text-[#007AFF]" : "text-[#86868B]"}`}
                      />
                      <div>
                        <div className={`text-[13px] font-medium ${selected ? "text-[#007AFF]" : "text-[#1D1D1F]"}`}>
                          {label}
                        </div>
                        <div className="text-[11px] text-[#86868B] mt-[2px]">{description}</div>
                      </div>
                    </button>
                  );
                })}
              </div>

              <div className="flex justify-between items-center pt-2">
                <span className="text-[12px] text-[#86868B]">
                  {selectedDocTypes.length} type{selectedDocTypes.length !== 1 ? "s" : ""} selected
                </span>
                <button
                  onClick={() => setStep(1)}
                  disabled={selectedDocTypes.length === 0}
                  className="px-5 py-2 bg-[#007AFF] text-white text-[13px] font-medium rounded-lg hover:bg-[#0066DD] disabled:opacity-40 transition-colors"
                >
                  Continue
                </button>
              </div>
            </div>
          </GlassCard>
        )}

        {/* Step 1: Configuration */}
        {step === 1 && (
          <GlassCard>
            <div className="p-5 space-y-5">
              <h3 className="text-[15px] font-semibold text-[#1D1D1F]">Generation Configuration</h3>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                {/* Patient count */}
                <div>
                  <label className="block text-[12px] font-medium text-[#3A3A3C] mb-1">
                    Patient Count <span className="text-[#86868B] font-normal">(1–50)</span>
                  </label>
                  <input
                    type="number"
                    min={1}
                    max={50}
                    value={patientCount}
                    onChange={(e) => setPatientCount(Math.min(50, Math.max(1, Number(e.target.value))))}
                    className="w-full px-3 py-2 text-[13px] rounded-lg border border-black/10 bg-white/70 focus:outline-none focus:ring-2 focus:ring-[#007AFF]/30"
                  />
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

                {/* LLM provider info */}
                <div className="md:col-span-2">
                  <div className="flex items-center gap-2 px-3 py-2 text-[12px] text-[#86868B] bg-black/[0.02] rounded-lg border border-black/5">
                    <span>Using LLM provider <span className="font-medium text-[#1D1D1F]">{provider}</span>{model ? <> / <span className="font-medium text-[#1D1D1F]">{model}</span></> : null} — change in sidebar</span>
                  </div>
                </div>
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
                  <FileText className="size-4" />
                  Generate Clinical Notes
                </button>
              </div>
            </div>
          </GlassCard>
        )}

        {/* Step 2: Generating spinner */}
        {step === 2 && (
          <GlassCard>
            <div className="p-10 flex flex-col items-center gap-5 text-center">
              {generating ? (
                <Loader2 className="size-10 text-[#007AFF] animate-spin" />
              ) : (
                <CheckCircle2 className="size-10 text-green-500" />
              )}
              <div>
                <h3 className="text-[16px] font-semibold text-[#1D1D1F] mb-1">
                  {generating ? "Generating Clinical Notes..." : "Generation Complete"}
                </h3>
                <p className="text-[13px] text-[#86868B]">
                  {generating
                    ? `Building FHIR resources and calling ${provider} for narrative generation`
                    : "All documents have been generated successfully"}
                </p>
              </div>
            </div>
          </GlassCard>
        )}

        {/* Step 3: Results */}
        {step === 3 && (
          <div className="space-y-4">
            {/* Stats row */}
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              <StatCard label="Documents Generated" value={documents.length} />
              <StatCard label="Document Types" value={new Set(documents.map((d) => d.type)).size} />
              <StatCard label="Provider" value={provider} />
            </div>

            {/* Download all button */}
            {documents.length > 0 && (
              <div className="flex gap-3">
                <button
                  onClick={handleDownloadAll}
                  className="flex items-center gap-2 px-4 py-2 bg-[#007AFF] text-white text-[13px] font-medium rounded-lg hover:bg-[#0066DD] transition-colors"
                >
                  <Download className="size-4" />
                  Download All ({documents.length})
                </button>
                <button
                  onClick={() => { setStep(0); }}
                  className="flex items-center gap-2 px-4 py-2 text-[13px] font-medium rounded-lg border border-black/10 hover:bg-black/[0.03] transition-colors"
                >
                  Generate Again
                </button>
              </div>
            )}

            {/* Document list */}
            {documents.length === 0 ? (
              <GlassCard>
                <div className="p-8 text-center text-[13px] text-[#86868B]">
                  No clinical documents were generated. Ensure the LLM provider is reachable and try again.
                </div>
              </GlassCard>
            ) : (
              <div className="space-y-3">
                {documents.map((doc) => {
                  const expanded = expandedIds.has(doc.id);
                  return (
                    <GlassCard key={doc.id}>
                      <div className="p-4">
                        {/* Header row */}
                        <div className="flex items-center justify-between gap-3">
                          <div className="flex items-center gap-3 min-w-0">
                            <FileText className="size-4 shrink-0 text-[#007AFF]" />
                            <div className="min-w-0">
                              <div className="text-[13px] font-semibold text-[#1D1D1F] truncate">{doc.type}</div>
                              <div className="text-[11px] text-[#86868B] font-mono truncate">{doc.id}</div>
                            </div>
                          </div>
                          <div className="flex items-center gap-2 shrink-0">
                            <button
                              onClick={() => handleCopy(doc.text)}
                              className="flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-medium rounded-lg border border-black/10 hover:bg-black/[0.03] transition-colors text-[#3A3A3C]"
                              title="Copy text"
                            >
                              <Copy className="size-3" />
                              Copy
                            </button>
                            <button
                              onClick={() => toggleExpanded(doc.id)}
                              className="flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-medium rounded-lg border border-black/10 hover:bg-black/[0.03] transition-colors text-[#3A3A3C]"
                            >
                              {expanded ? (
                                <>
                                  <ChevronUp className="size-3" />
                                  Collapse
                                </>
                              ) : (
                                <>
                                  <ChevronDown className="size-3" />
                                  Expand
                                </>
                              )}
                            </button>
                          </div>
                        </div>

                        {/* Expandable text */}
                        {expanded && (
                          <div className="mt-4">
                            {doc.text ? (
                              <pre className="whitespace-pre-wrap text-[12px] text-[#1D1D1F] leading-relaxed font-sans p-4 bg-black/[0.02] rounded-xl border border-black/[0.04] max-h-[500px] overflow-y-auto">
                                {doc.text}
                              </pre>
                            ) : (
                              <p className="text-[12px] text-[#86868B] italic px-4">
                                No narrative text available for this document.
                              </p>
                            )}
                          </div>
                        )}
                      </div>
                    </GlassCard>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
