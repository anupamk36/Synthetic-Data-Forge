"use client";

import { useState, useCallback, useRef } from "react";
import { generateAsync, getJobStatus, getJobData, stopJob } from "@/lib/api";
import type { GenerateRequest, JobStatus } from "@/lib/types";

interface GenerationState {
  status: "idle" | "running" | "complete" | "error" | "stopped";
  jobId: string | null;
  progress: number;
  recordsDone: number;
  totalRecords: number;
  elapsed: number;
  data: Record<string, unknown>[] | null;
  error: string | null;
}

export function useGenerate() {
  const [state, setState] = useState<GenerationState>({
    status: "idle",
    jobId: null,
    progress: 0,
    recordsDone: 0,
    totalRecords: 0,
    elapsed: 0,
    data: null,
    error: null,
  });

  const startTimeRef = useRef<number>(0);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const clearPoll = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const start = useCallback(async (req: GenerateRequest) => {
    clearPoll();
    startTimeRef.current = Date.now();

    setState({
      status: "running",
      jobId: null,
      progress: 0,
      recordsDone: 0,
      totalRecords: req.count,
      elapsed: 0,
      data: null,
      error: null,
    });

    try {
      const { job_id } = await generateAsync(req);
      setState((s) => ({ ...s, jobId: job_id }));

      pollRef.current = setInterval(async () => {
        try {
          const job: JobStatus = await getJobStatus(job_id);
          const elapsed = (Date.now() - startTimeRef.current) / 1000;

          if (job.status === "complete") {
            clearPoll();
            // Small delay to ensure backend has finished writing data
            await new Promise((r) => setTimeout(r, 300));
            const result = await getJobData(job_id);
            const data = Array.isArray(result.data) ? result.data : [];
            setState((s) => ({
              ...s,
              status: "complete",
              progress: 1,
              recordsDone: data.length,
              elapsed,
              data,
            }));
          } else if (job.status === "error") {
            clearPoll();
            setState((s) => ({
              ...s,
              status: "error",
              elapsed,
              error: job.error || "Generation failed",
            }));
          } else if (job.status === "stopped") {
            clearPoll();
            setState((s) => ({
              ...s,
              status: "stopped",
              elapsed,
            }));
          } else {
            setState((s) => ({
              ...s,
              progress: job.progress,
              recordsDone: job.records_done,
              elapsed,
              data: job.partial_data && job.partial_data.length > 0 ? job.partial_data : s.data,
            }));
          }
        } catch {
          clearPoll();
          setState((s) => ({ ...s, status: "error", error: "Lost connection to server" }));
        }
      }, 500);
    } catch (e) {
      setState((s) => ({
        ...s,
        status: "error",
        error: e instanceof Error ? e.message : "Failed to start generation",
      }));
    }
  }, [clearPoll]);

  const stop = useCallback(async () => {
    if (state.jobId) {
      await stopJob(state.jobId).catch(() => {});
    }
    clearPoll();
    setState((s) => ({ ...s, status: "stopped" }));
  }, [state.jobId, clearPoll]);

  const reset = useCallback(() => {
    clearPoll();
    setState({
      status: "idle",
      jobId: null,
      progress: 0,
      recordsDone: 0,
      totalRecords: 0,
      elapsed: 0,
      data: null,
      error: null,
    });
  }, [clearPoll]);

  return { ...state, start, stop, reset };
}
