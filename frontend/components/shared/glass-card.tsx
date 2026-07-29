"use client";

import { cn } from "@/lib/utils";

interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
  animatedBorder?: boolean;
}

export function GlassCard({ children, className, animatedBorder = true }: GlassCardProps) {
  return (
    <div className={cn("relative rounded-2xl overflow-hidden", className)}>
      {animatedBorder && (
        <div
          className="absolute inset-0 rounded-2xl animate-border-flow"
          style={{
            padding: 1,
            background: "linear-gradient(135deg, rgba(0,122,255,0.2), rgba(52,199,89,0.1), rgba(175,130,255,0.15), rgba(0,122,255,0.1))",
            backgroundSize: "300% 300%",
            WebkitMask: "linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)",
            WebkitMaskComposite: "xor",
            maskComposite: "exclude",
          }}
        />
      )}
      <div className="glass-card rounded-2xl p-7">
        {children}
      </div>
    </div>
  );
}
