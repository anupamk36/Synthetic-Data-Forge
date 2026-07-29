"use client";

import { useState, useCallback, useEffect } from "react";
import {
  FlaskConical,
  Download,
  CheckCircle2,
  Loader2,
} from "lucide-react";
import { toast } from "sonner";

import { TopBar } from "@/components/layout/top-bar";
import { GlassCard } from "@/components/shared/glass-card";
import { WizardStepper } from "@/components/shared/wizard-stepper";
import { StatCard } from "@/components/shared/stat-card";
import { ExportPanel } from "@/components/shared/export-panel";
import { QualityScanner } from "@/components/medical/quality-scanner";

import {
  getTrialProfiles,
  generateTrial,
  generateTrialAsync,
  getTrialJob,
  type TrialGenerateRequest,
  type TrialGenerateResponse,
  type TrialProfile,
} from "@/lib/api";

const WIZARD_STEPS = [
  { label: "Profile" },
  { label: "Design" },
  { label: "Generate" },
  { label: "Results" },
];

export default function ClinicalTrialsPage() {
  const [step, setStep] = useState(0);
  const [profiles, setProfiles] = useState<TrialProfile[]>([]);

  // Step 1: Profile selection
  const [selectedProfile, setSelectedProfile] = useState<string>("oncology_phase2");

  // Step 2: Study design
  const [numSites, setNumSites] = useState(5);
  const [subjectsPerArm, setSubjectsPerArm] = useState(50);
  const [dropoutRate, setDropoutRate] = useState(0.15);
  const [effectSize, setEffectSize] = useState(0.3);
  const [seed, setSeed] = useState<number | null>(null);
  const [outputFormats, setOutputFormats] = useState<string[]>(["sdtm", "fhir"]);

  // Step 3: Generation
  const [generating, setGenerating] = useState(false);
  const [progress, setProgress] = useState<Record<string, number>>({});
  const [jobId, setJobId] = useState<string | null>(null);

  // Step 4: Results
  const [result, setResult] = useState<TrialGenerateResponse | null>(null);

  useEffect(() => {
    getTrialProfiles().then(setProfiles).catch(() => {});
  }, []);

  const handleGenerate = useCallback(async () => {
    setGenerating(true);
    setProgress({});
    setStep(2);

    const req: TrialGenerateRequest = {
      profile: selectedProfile,
      num_sites: numSites,
      subjects_per_arm: subjectsPerArm,
      dropout_rate: dropoutRate,
      effect_size: effectSize,
      seed,
      output_formats: outputFormats,
    };

    try {
      if (subjectsPerArm > 200) {
        const asyncRes = await generateTrialAsync(req);
        setJobId(asyncRes.job_id);
      } else {
        const res = await generateTrial(req);
        setResult(res);
        setGenerating(false);
        setStep(3);
        toast.success(`Trial generated: ${res.stats.total} resources`);
      }
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Generation failed");
      setGenerating(false);
    }
  }, [selectedProfile, numSites, subjectsPerArm, dropoutRate, effectSize, seed, outputFormats]);

  useEffect(() => {
    if (!jobId) return;
    const interval = setInterval(async () => {
      try {
        const status = await getTrialJob(jobId);
        setProgress(status.progress);
        if (status.status === "completed" && status.result) {
          setResult(status.result);
          setGenerating(false);
          setStep(3);
          setJobId(null);
          toast.success(`Trial generated: ${status.result.stats.total} resources`);
        } else if (status.status === "failed") {
          toast.error(status.error || "Generation failed");
          setGenerating(false);
          setJobId(null);
        }
      } catch { /* ignore */ }
    }, 1500);
    return () => clearInterval(interval);
  }, [jobId]);

  const handleDownloadSDTM = useCallback(() => {
    if (!result?.sdtm) return;
    const json = JSON.stringify(result.sdtm, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `trial_sdtm_${selectedProfile}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [result, selectedProfile]);

  const handleDownloadFHIR = useCallback(() => {
    if (!result?.fhir) return;
    const json = JSON.stringify(result.fhir.data, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `trial_fhir_${selectedProfile}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [result, selectedProfile]);

  const currentProfile = profiles.find((p) => p.id === selectedProfile);

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      <TopBar title="Clinical Trials" />

      <div className="flex-1 px-6 pb-8 space-y-6">
        <WizardStepper steps={WIZARD_STEPS} current={step} />

        {/* Step 0: Profile Selection */}
        {step === 0 && (
          <GlassCard>
            <div className="p-5 space-y-4">
              <h3 className="text-[15px] font-semibold text-[#1D1D1F]">Select Trial Profile</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {profiles.map((profile) => (
                  <button
                    key={profile.id}
                    onClick={() => setSelectedProfile(profile.id)}
                    className={`p-4 rounded-xl border text-left transition-all ${
                      selectedProfile === profile.id
                        ? "border-[#007AFF] bg-[rgba(0,122,255,0.05)]"
                        : "border-black/10 hover:border-black/20"
                    }`}
                  >
                    <div className={`text-[14px] font-semibold ${selectedProfile === profile.id ? "text-[#007AFF]" : "text-[#1D1D1F]"}`}>
                      {profile.display_name}
                    </div>
                    <div className="text-[12px] text-[#86868B] mt-1">{profile.description}</div>
                    <div className="flex gap-2 mt-2">
                      <span className="px-2 py-0.5 text-[10px] font-medium rounded-full bg-black/5 text-[#3A3A3C]">{profile.phase}</span>
                      <span className="px-2 py-0.5 text-[10px] font-medium rounded-full bg-black/5 text-[#3A3A3C]">{profile.therapeutic_area}</span>
                    </div>
                  </button>
                ))}
              </div>
              <div className="flex justify-end pt-2">
                <button
                  onClick={() => setStep(1)}
                  disabled={!selectedProfile}
                  className="px-5 py-2 bg-[#007AFF] text-white text-[13px] font-medium rounded-lg hover:bg-[#0066DD] disabled:opacity-40 transition-colors"
                >
                  Continue
                </button>
              </div>
            </div>
          </GlassCard>
        )}

        {/* Step 1: Study Design */}
        {step === 1 && (
          <GlassCard>
            <div className="p-5 space-y-5">
              <h3 className="text-[15px] font-semibold text-[#1D1D1F]">Study Design — {currentProfile?.display_name}</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                <div>
                  <label className="block text-[12px] font-medium text-[#3A3A3C] mb-1">Number of Sites</label>
                  <input type="number" min={1} max={50} value={numSites} onChange={(e) => setNumSites(Number(e.target.value))}
                    className="w-full px-3 py-2 text-[13px] rounded-lg border border-black/10 bg-white/70 focus:outline-none focus:ring-2 focus:ring-[#007AFF]/30" />
                </div>
                <div>
                  <label className="block text-[12px] font-medium text-[#3A3A3C] mb-1">Subjects per Arm</label>
                  <input type="number" min={5} max={5000} value={subjectsPerArm} onChange={(e) => setSubjectsPerArm(Number(e.target.value))}
                    className="w-full px-3 py-2 text-[13px] rounded-lg border border-black/10 bg-white/70 focus:outline-none focus:ring-2 focus:ring-[#007AFF]/30" />
                </div>
                <div>
                  <label className="block text-[12px] font-medium text-[#3A3A3C] mb-1">Dropout Rate ({Math.round(dropoutRate * 100)}%)</label>
                  <input type="range" min={0} max={0.8} step={0.05} value={dropoutRate} onChange={(e) => setDropoutRate(Number(e.target.value))}
                    className="w-full" />
                </div>
                <div>
                  <label className="block text-[12px] font-medium text-[#3A3A3C] mb-1">Effect Size ({Math.round(effectSize * 100)}%)</label>
                  <input type="range" min={0} max={1} step={0.05} value={effectSize} onChange={(e) => setEffectSize(Number(e.target.value))}
                    className="w-full" />
                </div>
                <div>
                  <label className="block text-[12px] font-medium text-[#3A3A3C] mb-1">Random Seed (optional)</label>
                  <input type="number" value={seed ?? ""} onChange={(e) => setSeed(e.target.value ? Number(e.target.value) : null)} placeholder="Leave empty for random"
                    className="w-full px-3 py-2 text-[13px] rounded-lg border border-black/10 bg-white/70 focus:outline-none focus:ring-2 focus:ring-[#007AFF]/30" />
                </div>
                <div>
                  <label className="block text-[12px] font-medium text-[#3A3A3C] mb-1">Output Formats</label>
                  <div className="flex gap-3">
                    {["sdtm", "fhir"].map((fmt) => (
                      <label key={fmt} className="flex items-center gap-2 text-[13px] text-[#3A3A3C]">
                        <input type="checkbox" checked={outputFormats.includes(fmt)} onChange={(e) => {
                          setOutputFormats(e.target.checked ? [...outputFormats, fmt] : outputFormats.filter((f) => f !== fmt));
                        }} className="rounded border-black/20" />
                        {fmt === "sdtm" ? "CDISC SDTM" : "FHIR Bundle"}
                      </label>
                    ))}
                  </div>
                </div>
              </div>
              <div className="flex justify-between pt-2">
                <button onClick={() => setStep(0)} className="px-5 py-2 text-[13px] font-medium text-[#3A3A3C] rounded-lg border border-black/10 hover:bg-black/[0.03] transition-colors">Back</button>
                <button onClick={handleGenerate} className="px-5 py-2 bg-[#007AFF] text-white text-[13px] font-medium rounded-lg hover:bg-[#0066DD] transition-colors flex items-center gap-2">
                  <FlaskConical className="size-4" /> Generate Trial
                </button>
              </div>
            </div>
          </GlassCard>
        )}

        {/* Step 2: Generation Progress */}
        {step === 2 && (
          <GlassCard>
            <div className="p-5 space-y-4">
              <div className="flex items-center gap-3">
                {generating ? <Loader2 className="size-5 text-[#007AFF] animate-spin" /> : <CheckCircle2 className="size-5 text-green-500" />}
                <h3 className="text-[15px] font-semibold text-[#1D1D1F]">{generating ? "Generating Trial Data..." : "Generation Complete"}</h3>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                {Object.entries(progress).map(([step_name, count]) => (
                  <div key={step_name} className="flex items-center gap-2 px-3 py-2 rounded-lg border border-green-200 bg-green-50/50">
                    <CheckCircle2 className="size-4 text-green-500 shrink-0" />
                    <div>
                      <div className="text-[12px] font-medium text-[#1D1D1F]">{step_name}</div>
                      <div className="text-[11px] text-[#86868B]">{count} generated</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </GlassCard>
        )}

        {/* Step 3: Results */}
        {step === 3 && result && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <StatCard label="Total Resources" value={result.stats.total} />
              <StatCard label="SDTM Domains" value={result.sdtm ? Object.keys(result.sdtm).length : 0} />
              <StatCard label="Profile" value={currentProfile?.display_name || selectedProfile} />
              <StatCard label="Subjects" value={result.stats.by_type?.ResearchSubject || 0} />
            </div>

            {result.sdtm && (
              <GlassCard>
                <div className="p-5 space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-[15px] font-semibold text-[#1D1D1F]">CDISC SDTM Domains</h3>
                    <button onClick={handleDownloadSDTM} className="flex items-center gap-2 px-4 py-2 bg-[#007AFF] text-white text-[13px] font-medium rounded-lg hover:bg-[#0066DD] transition-colors">
                      <Download className="size-4" /> Download SDTM
                    </button>
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-2">
                    {Object.entries(result.sdtm).map(([domain, data]) => (
                      <div key={domain} className="flex justify-between items-center px-3 py-2 rounded-lg bg-black/[0.02] border border-black/5">
                        <span className="text-[13px] font-mono font-semibold text-[#1D1D1F]">{domain}</span>
                        <span className="text-[12px] text-[#86868B]">{data.rows} rows</span>
                      </div>
                    ))}
                  </div>
                </div>
              </GlassCard>
            )}

            {result.fhir && (
              <GlassCard>
                <div className="p-5 space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-[15px] font-semibold text-[#1D1D1F]">FHIR Bundle</h3>
                    <button onClick={handleDownloadFHIR} className="flex items-center gap-2 px-4 py-2 bg-emerald-600 text-white text-[13px] font-medium rounded-lg hover:bg-emerald-700 transition-colors">
                      <Download className="size-4" /> Download FHIR
                    </button>
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                    {Object.entries(result.stats.by_type || {}).map(([type, count]) => (
                      <div key={type} className="flex justify-between items-center px-3 py-2 rounded-lg bg-black/[0.02] border border-black/5">
                        <span className="text-[12px] text-[#3A3A3C]">{type}</span>
                        <span className="text-[13px] font-semibold text-[#1D1D1F]">{count}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </GlassCard>
            )}

            <ExportPanel data={result.sdtm || result.fhir?.data || {}} filename="trial-data" />

            {/* AI Quality Scanner */}
            <QualityScanner
              data={result.sdtm || result.fhir?.data || {}}
              dataType={result.sdtm ? "sdtm" : "fhir"}
            />
          </div>
        )}
      </div>
    </div>
  );
}
