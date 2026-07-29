"use client";

import { Input } from "@/components/ui/input";

interface FieldHintsProps {
  schema: Record<string, string>;
  hints: Record<string, string>;
  onChange: (hints: Record<string, string>) => void;
}

export function FieldHints({ schema, hints, onChange }: FieldHintsProps) {
  const columns = Object.keys(schema);

  return (
    <div>
      <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">
        LLM Semantic Hints
      </h3>
      <p className="text-xs text-muted-foreground mb-4">
        Describe each field to help the LLM generate semantically coherent data.
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {columns.map((col) => (
          <div key={col}>
            <label className="text-xs text-muted-foreground mb-1 block">{col}</label>
            <Input
              placeholder={`Describe ${col}...`}
              value={hints[col] || ""}
              onChange={(e) =>
                onChange({ ...hints, [col]: e.target.value })
              }
              className="bg-card border-border text-sm"
            />
          </div>
        ))}
      </div>
    </div>
  );
}
