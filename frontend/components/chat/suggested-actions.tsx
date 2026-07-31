"use client";

import { Sparkles } from "lucide-react";

const SUGGESTIONS = [
  "Generate employee data with names, departments, and salaries",
  "Create a schema for patient records",
  "Run a privacy audit on my data",
  "Explain my quality score",
];

interface SuggestedActionsProps {
  onSelect: (message: string) => void;
}

export function SuggestedActions({ onSelect }: SuggestedActionsProps) {
  return (
    <div className="flex flex-col items-center py-8 px-4">
      <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-[#007AFF]/20 to-[#AF82FF]/20 flex items-center justify-center mb-4">
        <Sparkles className="size-5 text-[#007AFF]" />
      </div>
      <h3 className="text-[14px] font-semibold text-[#1D1D1F] mb-1">
        Forge AI Assistant
      </h3>
      <p className="text-[12px] text-[#86868B] text-center mb-6">
        Describe what data you need, and I&apos;ll help you generate it.
      </p>
      <div className="w-full space-y-2">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            onClick={() => onSelect(s)}
            className="w-full text-left text-[12px] text-[#3A3A3C] px-3 py-2.5 rounded-xl border border-black/[0.06] bg-white/40 hover:bg-white/70 hover:border-[#007AFF]/20 transition-all"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}
