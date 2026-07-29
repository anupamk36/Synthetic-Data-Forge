"use client";

import { useQuery } from "@tanstack/react-query";
import { getProviders } from "@/lib/api";

export function useProviders() {
  return useQuery({
    queryKey: ["providers"],
    queryFn: getProviders,
    staleTime: 60_000,
    retry: 1,
  });
}
