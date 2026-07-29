export const SAMPLE_SCHEMA = {
  name: "String",
  age: "Int64",
  city: "String",
};

export const SAMPLE_ROWS = [
  { name: "Alice Johnson", age: 34, city: "Boston" },
  { name: "Bob Smith", age: 45, city: "Chicago" },
  { name: "Carol White", age: 28, city: "Denver" },
  { name: "David Brown", age: 52, city: "Seattle" },
  { name: "Eva Martinez", age: 39, city: "Austin" },
];

export const SAMPLE_FHIR_TYPES = [
  { type: "Patient", dependencies: [] },
  { type: "Encounter", dependencies: ["Patient"] },
  { type: "Observation", dependencies: ["Patient", "Encounter"] },
  { type: "Condition", dependencies: ["Patient"] },
];

export const SAMPLE_FHIR_RESPONSE = {
  status: "complete",
  stats: {
    total: 100,
    by_type: {
      Patient: 50,
      Encounter: 50,
    },
  },
  format: "bundle",
  data: {},
};

export const SAMPLE_PRIVACY_REPORT = {
  risk_level: "LOW_RISK",
  compliant: true,
  metrics: {
    dcr: {
      score: 0.92,
      threshold: 0.8,
      passed: true,
      description: "Distance to Closest Record",
    },
    k_anonymity: {
      score: 5,
      threshold: 3,
      passed: true,
      description: "k-Anonymity level",
    },
    epsilon: {
      score: 1.2,
      threshold: 5.0,
      passed: true,
      description: "Differential privacy epsilon",
    },
  },
  summary: "Synthetic data meets all privacy thresholds.",
};

export const SAMPLE_UPLOAD_RESPONSE = {
  schema: SAMPLE_SCHEMA,
  sample_rows: SAMPLE_ROWS,
  row_count: 100,
};
