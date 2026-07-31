"use client";

import Link from "next/link";
import {
  HeartPulse,
  FlaskConical,
  ScanLine,
  Table2,
  Link2,
  Shield,
  BarChart3,
  ArrowRight,
} from "lucide-react";
import { GlassCard } from "@/components/shared/glass-card";

const CLINICAL_FEATURES = [
  {
    title: "FHIR R4 Generator",
    description:
      "Complete FHIR bundles with ICD-10, LOINC, SNOMED, RxNorm terminologies and cross-resource referential integrity.",
    icon: HeartPulse,
    href: "/medical/fhir",
    color: "#FF3B30",
  },
  {
    title: "Clinical Trials",
    description:
      "Phase I-III trial simulation with CDISC SDTM export, randomization, dropout modeling, and adverse events.",
    icon: FlaskConical,
    href: "/medical/trials",
    color: "#AF82FF",
  },
  {
    title: "Medical Imaging",
    description:
      "DICOM metadata for CT, MR, US, DX, MG, PT with modality-specific templates and compliant UID generation.",
    icon: ScanLine,
    href: "/medical/imaging",
    color: "#007AFF",
  },
];

const GENERIC_FEATURES = [
  {
    title: "Single Table",
    description: "Schema inference, Faker generation with Gaussian copula correlation preservation, LLM validation.",
    icon: Table2,
    href: "/generate/single",
  },
  {
    title: "Multi-Table Relational",
    description: "DAG-based generation with topological sorting and guaranteed foreign key integrity.",
    icon: Link2,
    href: "/generate/relational",
  },
];

const ANALYZE_FEATURES = [
  {
    title: "Privacy Compliance",
    description: "DCR, k-anonymity, l-diversity, and epsilon estimation with downloadable compliance reports.",
    icon: Shield,
    href: "/analyze/privacy",
  },
  {
    title: "Data Quality",
    description: "KS tests, chi-squared tests, correlation preservation scoring, and realism grading (A-F).",
    icon: BarChart3,
    href: "/analyze/quality",
  },
];

export default function Home() {
  return (
    <div className="flex flex-col h-full overflow-y-auto">
      <div className="px-8 py-12 max-w-5xl mx-auto space-y-12">
        {/* Hero */}
        <div className="space-y-4">
          <h1 className="text-[36px] font-bold tracking-tight text-[#1D1D1F]">
            Generate Compliant Clinical{" "}
            <span
              className="bg-clip-text text-transparent"
              style={{
                backgroundImage: "linear-gradient(135deg, #007AFF, #34C759)",
              }}
            >
              & Life Sciences
            </span>{" "}
            Data
          </h1>
          <p className="text-[16px] text-[#86868B] max-w-2xl leading-relaxed">
            HIPAA-safe, FHIR R4 compliant synthetic data for development, testing, and research.
            Six terminology systems, three clinical standards, and built-in privacy compliance.
          </p>
        </div>

        {/* Clinical Data — Primary */}
        <section className="space-y-4">
          <h2 className="text-[13px] font-semibold uppercase tracking-[0.8px] text-[#86868B]">
            Clinical Data Generation
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {CLINICAL_FEATURES.map((feature) => {
              const Icon = feature.icon;
              return (
                <Link key={feature.href} href={feature.href}>
                  <GlassCard className="h-full transition-transform hover:scale-[1.02] cursor-pointer">
                    <div className="space-y-3">
                      <div
                        className="flex size-10 items-center justify-center rounded-xl"
                        style={{ backgroundColor: `${feature.color}15` }}
                      >
                        <Icon className="size-5" style={{ color: feature.color }} />
                      </div>
                      <h3 className="text-[15px] font-semibold text-[#1D1D1F]">
                        {feature.title}
                      </h3>
                      <p className="text-[12px] text-[#86868B] leading-relaxed">
                        {feature.description}
                      </p>
                      <div className="flex items-center gap-1 text-[12px] font-medium text-[#007AFF]">
                        <span>Open</span>
                        <ArrowRight className="size-3" />
                      </div>
                    </div>
                  </GlassCard>
                </Link>
              );
            })}
          </div>
        </section>

        {/* Generic Data — Secondary */}
        <section className="space-y-4">
          <h2 className="text-[13px] font-semibold uppercase tracking-[0.8px] text-[#86868B]">
            Also Generates Any Tabular Data
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {GENERIC_FEATURES.map((feature) => {
              const Icon = feature.icon;
              return (
                <Link key={feature.href} href={feature.href}>
                  <GlassCard className="h-full transition-transform hover:scale-[1.02] cursor-pointer">
                    <div className="flex items-start gap-4">
                      <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-black/[0.04]">
                        <Icon className="size-4 text-[#3A3A3C]" />
                      </div>
                      <div className="space-y-1">
                        <h3 className="text-[14px] font-semibold text-[#1D1D1F]">
                          {feature.title}
                        </h3>
                        <p className="text-[12px] text-[#86868B] leading-relaxed">
                          {feature.description}
                        </p>
                      </div>
                    </div>
                  </GlassCard>
                </Link>
              );
            })}
          </div>
        </section>

        {/* Analyze */}
        <section className="space-y-4">
          <h2 className="text-[13px] font-semibold uppercase tracking-[0.8px] text-[#86868B]">
            Privacy & Quality Assurance
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {ANALYZE_FEATURES.map((feature) => {
              const Icon = feature.icon;
              return (
                <Link key={feature.href} href={feature.href}>
                  <GlassCard className="h-full transition-transform hover:scale-[1.02] cursor-pointer">
                    <div className="flex items-start gap-4">
                      <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-black/[0.04]">
                        <Icon className="size-4 text-[#3A3A3C]" />
                      </div>
                      <div className="space-y-1">
                        <h3 className="text-[14px] font-semibold text-[#1D1D1F]">
                          {feature.title}
                        </h3>
                        <p className="text-[12px] text-[#86868B] leading-relaxed">
                          {feature.description}
                        </p>
                      </div>
                    </div>
                  </GlassCard>
                </Link>
              );
            })}
          </div>
        </section>
      </div>
    </div>
  );
}
