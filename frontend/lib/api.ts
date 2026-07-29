import type {
  AsyncJobResponse,
  CostEstimate,
  DataProfile,
  GenerateRequest,
  GenerateResponse,
  HistoryRun,
  JobStatus,
  PrivacyReport,
  PrivacyResult,
  ProviderInfo,
  QualityReport,
  RelationalRequest,
  SavedSchema,
  TestAnalysis,
  TestCoverageGap,
  TestCoverageResult,
  TestIntelligenceResult,
  UploadResponse,
} from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`API ${res.status}: ${detail}`);
  }
  return res.json();
}

function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, { method: "POST", body: JSON.stringify(body) });
}

// Health
export const getHealth = () => request<{ status: string; ollama_available: boolean }>("/health");

// Providers
export const getProviders = () => request<ProviderInfo[]>("/api/v1/providers");

// Upload
export async function uploadFile(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_URL}/api/v1/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`);
  return res.json();
}

// Profile
export const profileData = (data: Record<string, unknown>[]) =>
  post<DataProfile>("/api/v1/profile", { data });

// Cost estimate
export const estimateCost = (schema: Record<string, string>, count: number, provider: string, model?: string | null) =>
  post<CostEstimate>("/api/v1/estimate", { schema, count, provider, model });

// Generate (sync)
export const generateSync = (req: GenerateRequest) =>
  post<GenerateResponse>("/api/v1/generate", req);

// Generate (async)
export const generateAsync = (req: GenerateRequest) =>
  post<AsyncJobResponse>("/api/v1/generate/async", req);

// Jobs
export const getJobStatus = (jobId: string) =>
  request<JobStatus>(`/api/v1/jobs/${jobId}`);

export const getJobData = (jobId: string, format = "json") =>
  request<{ format: string; data: Record<string, unknown>[] | string }>(`/api/v1/jobs/${jobId}/data?format=${format}`);

export const stopJob = (jobId: string) =>
  post<{ job_id: string; stopped: boolean }>(`/api/v1/jobs/${jobId}/stop`, {});

// Relational
export const generateRelational = (req: RelationalRequest) =>
  post<Record<string, Record<string, unknown>[]>>("/api/v1/generate/relational", req);

// Quality
export const assessQuality = (
  generatedData: Record<string, unknown>[],
  originalData?: Record<string, unknown>[] | null,
  expectedSchema?: Record<string, string> | null,
) => post<QualityReport>("/api/v1/quality/assess", {
  generated_data: generatedData,
  original_data: originalData ?? null,
  expected_schema: expectedSchema ?? null,
});

// Privacy
export const auditPrivacy = (realData: Record<string, unknown>[], syntheticData: Record<string, unknown>[]) =>
  post<PrivacyResult>("/api/v1/privacy/audit", { real_data: realData, synthetic_data: syntheticData });

export const auditPrivacyFull = (
  realData: Record<string, unknown>[],
  syntheticData: Record<string, unknown>[],
  quasiIdentifiers?: string[] | null,
  sensitiveColumn?: string | null,
) =>
  post<PrivacyReport>("/api/v1/privacy/report", {
    real_data: realData,
    synthetic_data: syntheticData,
    quasi_identifiers: quasiIdentifiers ?? null,
    sensitive_column: sensitiveColumn ?? null,
  });

// Test Intelligence
export const generateTestSuite = (req: {
  schema: Record<string, string>;
  sample_data?: Record<string, unknown>[];
  provider?: string;
  api_key?: string;
  model?: string;
}) => post<TestIntelligenceResult>("/api/v1/test-intelligence/generate", req);

export const scoreTestData = (req: {
  schema: Record<string, string>;
  data: Record<string, unknown>[];
  provider?: string;
  api_key?: string;
  model?: string;
}) => post<{
  score: number;
  total_rows: number;
  gaps: { category: string; description: string; severity: string }[];
  suggestions: string[];
  analysis: TestAnalysis;
}>("/api/v1/test-intelligence/score", req);

export const fixTestGaps = (req: {
  schema: Record<string, string>;
  analysis: TestAnalysis;
  gaps: TestCoverageGap[];
  existing_test_data?: Record<string, Record<string, unknown>[]>;
  provider?: string;
  api_key?: string;
  model?: string;
}) => post<{
  additional_data: Record<string, Record<string, unknown>[]>;
  new_coverage: TestCoverageResult;
  added_summary: Record<string, number>;
  total_added: number;
  gaps_fixed: number;
}>("/api/v1/test-intelligence/fix-gaps", req);

// Medical Data Scanning
export const scanMedicalData = (data: unknown, dataType: string, provider?: string, apiKey?: string, model?: string) =>
  post<{
    data_type: string;
    total_resources: number;
    resource_types: Record<string, number>;
    issues: { severity: string; category: string; resource_type: string; description: string; fix: string }[];
    issue_count: number;
    score: number;
    summary: { high: number; medium: number; low: number };
  }>("/api/v1/test-intelligence/scan-medical", { data, data_type: dataType, provider: provider ?? "ollama", model, api_key: apiKey });

// Schemas
export const listSchemas = (search = "") =>
  request<SavedSchema[]>(`/api/v1/schemas?search=${encodeURIComponent(search)}`);

export const getSchema = (id: string) =>
  request<SavedSchema>(`/api/v1/schemas/${id}`);

export const createSchema = (schema: { name: string; schema: Record<string, string>; description?: string; tags?: string }) =>
  post<{ id: string; name: string }>("/api/v1/schemas", schema);

export const deleteSchema = (id: string) =>
  request<{ id: string; deleted: boolean }>(`/api/v1/schemas/${id}`, { method: "DELETE" });

// History
export const listHistory = (limit = 50, feature?: string) => {
  const params = new URLSearchParams({ limit: String(limit) });
  if (feature && feature !== "all") params.set("feature", feature);
  return request<HistoryRun[]>(`/api/v1/history?${params}`);
};

export const getRunDetail = (runId: string) =>
  request<HistoryRun>(`/api/v1/history/${runId}`);

// Export
export const exportData = (req: {
  data: Record<string, unknown>[];
  sink_type?: string;
  output_path?: string;
  output_format?: string;
  records_per_file?: number;
  partition_on?: string[];
  s3_bucket?: string;
  s3_prefix?: string;
  s3_region?: string;
  s3_access_key?: string;
  s3_secret_key?: string;
  s3_session_token?: string;
}) => post<{ files_written: string[] }>("/api/v1/export", req);

// Chat
export async function chatUpload(sessionId: string, file: File) {
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("file", file);
  const res = await fetch(`${API_URL}/api/v1/chat/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`);
  return res.json() as Promise<{
    data_key: string;
    rows: number;
    columns: string[];
  }>;
}

export const getChatModels = () =>
  request<{ models: string[]; default: string; provider: string }>(
    "/api/v1/chat/models"
  );

export const clearChatSession = (sessionId: string) =>
  post<{ cleared: boolean }>("/api/v1/chat/clear", {
    session_id: sessionId,
  });

// Medical / FHIR
export interface FHIRGenerateRequest {
  resource_types: string[];
  patient_count: number;
  encounters_per_patient: { min: number; max: number };
  clinical_density: string;
  output_format: string;
  bundle_type: string;
  include_narrative: boolean;
  terminology_focus: string | null;
  seed: number | null;
  include_hl7v2: boolean;
  narrative_doc_types?: string[];
  narrative_provider?: string;
  narrative_api_key?: string;
  narrative_model?: string;
}

export interface FHIRGenerateResponse {
  status: string;
  stats: { total: number; by_type: Record<string, number> };
  format: string;
  data: unknown;
  hl7v2_messages?: string[];
  hl7v2_count?: number;
}

export interface FHIRResourceType {
  type: string;
  dependencies: string[];
}

export const getFHIRResourceTypes = () =>
  request<FHIRResourceType[]>("/api/v1/medical/fhir/resource-types");

export const generateFHIR = (req: FHIRGenerateRequest) =>
  post<FHIRGenerateResponse>("/api/v1/medical/fhir/generate", req);

export const generateFHIRAsync = (req: FHIRGenerateRequest) =>
  post<{ job_id: string; status: string }>("/api/v1/medical/fhir/generate/async", req);

export const getFHIRJob = (jobId: string) =>
  request<{ job_id: string; status: string; progress: Record<string, number>; result?: FHIRGenerateResponse; error?: string }>(`/api/v1/medical/fhir/jobs/${jobId}`);

export const searchTerminology = (system: string, query: string) =>
  request<{ system: string; query: string; results: unknown[]; count: number }>(`/api/v1/medical/terminologies/search?system=${encodeURIComponent(system)}&query=${encodeURIComponent(query)}`);

// Clinical Trials
export interface TrialGenerateRequest {
  profile: string;
  num_sites: number;
  subjects_per_arm: number;
  dropout_rate: number;
  effect_size: number;
  seed: number | null;
  output_formats: string[];
}

export interface TrialProfile {
  id: string;
  display_name: string;
  description: string;
  therapeutic_area: string;
  phase: string;
  target_enrollment: number;
}

export interface TrialGenerateResponse {
  status: string;
  stats: { total: number; by_type: Record<string, number> };
  sdtm?: Record<string, { rows: number; data: Record<string, unknown>[] }>;
  fhir?: { format: string; data: unknown; stats: unknown };
}

export const getTrialProfiles = () =>
  request<TrialProfile[]>("/api/v1/medical/trials/profiles");

export const generateTrial = (req: TrialGenerateRequest) =>
  post<TrialGenerateResponse>("/api/v1/medical/trials/generate", req);

export const generateTrialAsync = (req: TrialGenerateRequest) =>
  post<{ job_id: string; status: string }>("/api/v1/medical/trials/generate/async", req);

export const getTrialJob = (jobId: string) =>
  request<{ job_id: string; status: string; progress: Record<string, number>; result?: TrialGenerateResponse; error?: string }>(`/api/v1/medical/trials/jobs/${jobId}`);

// Imaging / DICOM
export interface ImagingGenerateRequest {
  modalities: string[];
  body_parts: string[] | null;
  num_studies: number;
  include_instance_metadata: boolean;
  output_format: string;
  seed: number | null;
}

export interface ImagingModality {
  code: string;
  display: string;
  weight: number;
}

export interface ImagingBodyPart {
  code: string;
  display: string;
  modalities: string[];
  weight: number;
}

export interface ImagingGenerateResponse {
  status: string;
  format: string;
  data: unknown;
  stats: { num_studies: number; total_series: number; total_instances: number; modalities: string[]; elapsed_seconds: number };
}

export const getImagingModalities = () =>
  request<{ modalities: ImagingModality[]; body_parts: ImagingBodyPart[] }>("/api/v1/medical/imaging/modalities");

export const generateImaging = (req: ImagingGenerateRequest) =>
  post<ImagingGenerateResponse>("/api/v1/medical/imaging/generate", req);

export const generateImagingAsync = (req: ImagingGenerateRequest) =>
  post<{ job_id: string; status: string }>("/api/v1/medical/imaging/generate/async", req);

export const getImagingJob = (jobId: string) =>
  request<{ job_id: string; status: string; progress: Record<string, number>; result?: ImagingGenerateResponse; error?: string }>(`/api/v1/medical/imaging/jobs/${jobId}`);

// Clinical Narratives
export interface NarrativeGenerateRequest {
  bundle: unknown;
  doc_types?: string[];
  provider: string;
  api_key?: string;
  model?: string;
  encounter_ids?: string[];
}

export interface NarrativeDocument {
  id: string;
  type: string;
  text: string;
  document_reference: unknown;
}

export interface NarrativeGenerateResponse {
  status: string;
  documents: NarrativeDocument[];
  count: number;
}

export const generateNarratives = (req: NarrativeGenerateRequest) =>
  post<NarrativeGenerateResponse>("/api/v1/medical/narratives/generate", req);
