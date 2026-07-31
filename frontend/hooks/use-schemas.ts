"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { listSchemas, createSchema, deleteSchema } from "@/lib/api";

export function useSchemas(search = "") {
  return useQuery({
    queryKey: ["schemas", search],
    queryFn: () => listSchemas(search),
    staleTime: 10_000,
  });
}

export function useSaveSchema() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: createSchema,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["schemas"] }),
  });
}

export function useDeleteSchema() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: deleteSchema,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["schemas"] }),
  });
}
