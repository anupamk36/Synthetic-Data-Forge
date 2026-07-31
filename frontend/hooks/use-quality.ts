"use client";

import { useMutation } from "@tanstack/react-query";
import { assessQuality } from "@/lib/api";

export function useQualityAssessment() {
  return useMutation({
    mutationFn: ({
      generatedData,
      originalData,
      expectedSchema,
    }: {
      generatedData: Record<string, unknown>[];
      originalData?: Record<string, unknown>[] | null;
      expectedSchema?: Record<string, string> | null;
    }) => assessQuality(generatedData, originalData, expectedSchema),
  });
}
