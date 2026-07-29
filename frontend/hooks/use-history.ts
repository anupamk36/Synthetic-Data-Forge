"use client";

import { useQuery } from "@tanstack/react-query";
import { listHistory } from "@/lib/api";

export function useHistory(limit = 50, feature?: string) {
  return useQuery({
    queryKey: ["history", limit, feature],
    queryFn: () => listHistory(limit, feature),
    staleTime: 10_000,
  });
}
