export interface Schema {
  [column: string]: string;
}

export interface GenerateRequest {
  schema: Schema;
  count: number;
  use_llm?: boolean;
  field_descriptions?: Record<string, string>;
  seed?: number | null;
  output_format?: string;
  provider?: string;
  model?: string | null;
  api_key?: string | null;
  llm_validation?: boolean;
  validation_sample_rate?: number;
  token_budget_usd?: number;
}

export interface GenerateResponse {
  run_id: string;
  format: string;
  data: Record<string, unknown>[] | string;
}

export interface AsyncJobResponse {
  job_id: string;
  status: string;
}

export interface JobStatus {
  status: "running" | "complete" | "error" | "stopped";
  progress: number;
  records_done: number;
  total: number;
  run_id?: string;
  record_count?: number;
  error?: string;
  partial_data?: Record<string, unknown>[];
}

export interface UploadResponse {
  schema: Schema;
  sample_rows: Record<string, unknown>[];
  row_count: number;
}

export interface ColumnStats {
  name: string;
  dtype: string;
  null_rate: number;
  unique_rate: number;
  unique_count: number;
  is_numeric: boolean;
  is_categorical: boolean;
  is_date: boolean;
  min_val: number | string | null;
  max_val: number | string | null;
  mean: number | null;
  std: number | null;
  percentiles: Record<string, number> | null;
  distribution_type: string;
  top_values: { value: string; count: number; pct: number }[] | null;
  cardinality: number;
  entropy: number;
}

export interface CorrelationEntry {
  col_a: string;
  col_b: string;
  method: string;
  value: number;
  significant: boolean;
}

export interface Constraint {
  constraint_type: string;
  columns: string[];
  details: string;
}

export interface DataProfile {
  row_count: number;
  col_count: number;
  column_stats: ColumnStats[];
  correlations: CorrelationEntry[];
  conditional_distributions: unknown[];
  constraints: Constraint[];
}

export interface QualityReport {
  overall_score: number;
  realism_grade: string;
  completeness: number;
  uniqueness: number;
  schema_match: number;
  distribution_score: number;
  correlation_preservation: number;
  dependency_score: number;
  column_details: Record<string, unknown>[];
  statistical_tests: {
    column: string;
    test: string;
    statistic: number;
    p_value: number;
    pass: boolean;
  }[];
  warnings: string[];
}

export interface PrivacyResult {
  min_dcr: number;
  mean_dcr: number;
  median_dcr: number;
  std_dcr: number;
  pct_exact_matches: number;
  risk_level: "Low" | "Medium" | "High";
  error: string | null;
}

export interface PrivacyReport {
  dcr: PrivacyResult;
  k_anonymity: {
    min_k: number;
    mean_group_size: number;
    vulnerable_groups: number;
    total_groups: number;
  } | null;
  l_diversity: {
    min_l: number;
    mean_l: number;
    vulnerable_groups: number;
  } | null;
  epsilon: {
    estimated_epsilon: number;
    interpretation: string;
    per_column: Record<string, number>;
  };
  overall_risk: "Low" | "Medium" | "High";
  recommendations: string[];
  compliant: boolean;
}

export interface CostEstimate {
  provider: string;
  model: string | null;
  count: number;
  estimated_cost_usd: number;
}

export interface ProviderInfo {
  name: string;
  models: string[];
  available: boolean | null;
}

export interface SavedSchema {
  id: string;
  name: string;
  schema: Schema;
  description: string;
  field_descriptions: Record<string, string>;
  tags: string;
  created_at: string;
  updated_at: string;
}

export interface HistoryRun {
  id: string;
  created_at: string;
  feature: string;
  status: string;
  record_count: number;
  columns: number;
  elapsed_sec: number;
  engine: string;
  model_name: string | null;
  output_path: string;
  error_msg: string;
  schema?: Schema;
  settings?: Record<string, unknown>;
}

export interface TestAnalysis {
  columns: Record<string, { type: string; semantic: string; categories: string[] }>;
  domain: string;
}

export interface TestCoverageGap {
  category: string;
  description: string;
  severity: "high" | "medium" | "low";
}

export interface TestCoverageResult {
  score: number;
  total_rows: number;
  gaps: TestCoverageGap[];
  suggestions: string[];
}

export interface TestIntelligenceResult {
  analysis: TestAnalysis;
  test_data: Record<string, Record<string, unknown>[]>;
  coverage: TestCoverageResult;
  total_rows: number;
}

export interface RelationalRequest {
  tables: Record<string, Schema>;
  relationships: {
    parent_table: string;
    parent_col: string;
    child_table: string;
    child_col: string;
  }[];
  counts: Record<string, number>;
  source_data?: Record<string, Record<string, unknown>[]>;
}

