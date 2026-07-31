"use client";

import { cn } from "@/lib/utils";

export type WizardStep = string | { readonly label: string };

interface WizardStepperProps {
  readonly steps: WizardStep[];
  readonly current: number;
}

function getLabel(step: WizardStep): string {
  return typeof step === "string" ? step : step.label;
}

export function WizardStepper({ steps, current }: WizardStepperProps) {
  return (
    <div className="inline-flex gap-[3px] p-[5px] bg-white/70 border border-black/[0.06] rounded-[14px] shadow-[0_2px_8px_rgba(0,0,0,0.04),0_0_0_1px_rgba(255,255,255,0.8)_inset]">
      {steps.map((step, i) => {
        const label = getLabel(step);
        const isActive = i === current;
        const isCompleted = i < current;

        return (
          <div
            key={label}
            className={cn(
              "px-[18px] py-[7px] rounded-[10px] text-[12px] font-medium transition-all duration-300 cursor-default select-none",
              isActive && "bg-[#007AFF] text-white font-semibold shadow-[0_2px_8px_rgba(0,122,255,0.35)]",
              isCompleted && "text-[#007AFF]",
              !isActive && !isCompleted && "text-[#86868B]"
            )}
          >
            {`${i + 1}. ${label}`}
            {isCompleted && <span className="ml-1 text-[10px]">✓</span>}
          </div>
        );
      })}
    </div>
  );
}
