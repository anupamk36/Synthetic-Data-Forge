"use client";

import { useMutation } from "@tanstack/react-query";
import { profileData } from "@/lib/api";

export function useProfile() {
  return useMutation({
    mutationFn: (data: Record<string, unknown>[]) => profileData(data),
  });
}
