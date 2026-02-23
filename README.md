# 🛠️ Synthetic Data Forge

A powerful Streamlit-based platform for generating realistic, privacy-safe synthetic datasets. Upload sample files, define schemas, inject business logic via LLM, generate multi-table relational data, simulate temporal trends, and push output to local storage or S3 — all through an intuitive tabbed web UI.

---

## ✨ Features

### 📊 Single Table Generation
- **Schema Inference** — Upload CSV or Parquet files to auto-detect column types
- **Interactive Schema Editor** — Modify types (`Int64`, `Float64`, `String`, `Date`) before generation
- **Output Format Selection** — Export as **Parquet**, **CSV**, or **JSON**
- **Hive-Style Partitioning** — Nest output by multiple partition columns (e.g., `region=US/year=2024/part_0.parquet`)
- **Scalable Generation** — Produce thousands of realistic records using [Faker](https://faker.readthedocs.io/)

### 🧠 LLM-Powered Business Logic Injection
- Write natural language rules like *"discount_price must be less than original_price"*
- Rules are translated into Python filters via a local [Ollama](https://ollama.ai/) LLM (runs in Docker)
- Generated data is automatically filtered to satisfy all constraints
- Graceful degradation when Ollama is unavailable

### 🛡️ Privacy Scorecard (DCR Metric)
- Computes **Distance to Closest Record** between real and synthetic datasets
- Flags near-exact matches as potential privacy leaks
- Color-coded risk assessment: 🟢 Low / 🟡 Medium / 🔴 High
- Distribution histogram and detailed metrics dashboard

### 🔗 Multi-Table Relational Integrity (Hydra)
- Upload multiple related files and define **foreign key relationships** via UI
- DAG-based generation order (parents before children) ensures FK consistency
- Mermaid diagram visualization of the relational map
- Per-table row counts and independent schema editing

### ⏰ Time-Travel Simulation
- Generate data across configurable time periods (daily / weekly / monthly)
- **Trend slider** — simulate growth or decline (-20% to +20% per period)
- **Spike injection** — add date-specific volume multipliers (e.g., Black Friday = 3×)
- Volume preview chart before generation
- Auto-partitioned by time period

### 📤 Zero-Copy Cloud Push (Data Sinks)
- **Local Filesystem** — write to any local directory with `~/` path expansion
- **Amazon S3** — stream data directly from memory to S3 (requires AWS credentials)
- Extensible sink architecture for future targets (Snowflake, BigQuery, Kafka)

---

## 📂 Project Structure

```
Synthetic-Data-Forge/
├── app/
│   ├── main.py                # Tab-based Streamlit orchestrator
│   ├── ui_schema.py           # Reusable schema editor component
│   ├── ui_privacy.py          # Privacy scorecard dashboard
│   ├── ui_relational.py       # Multi-table relational map UI
│   └── ui_time_travel.py      # Time-travel trend config UI
├── core/
│   ├── generator.py           # ForgeEngine — synthetic record generation
│   ├── llm_logic.py           # LLM business logic injection (Ollama)
│   ├── privacy.py             # DCR metric computation
│   ├── relational.py          # DAG-based multi-table generation
│   ├── time_travel.py         # Temporal trend/spike simulation
│   └── sinks.py               # Output sinks (Local + S3)
├── .streamlit/
│   └── config.toml            # Streamlit server config
├── docker-compose.yml         # Ollama LLM server (Docker)
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **Docker** (for LLM business logic feature)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Synthetic-Data-Forge.git
   cd Synthetic-Data-Forge
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Ollama (for LLM features)**
   ```bash
   docker compose up -d
   docker exec forge-ollama ollama pull llama3
   ```

4. **Run the app**
   ```bash
   python3 -m streamlit run app/main.py
   ```

5. Open **http://localhost:8501**

---

## 📖 Usage

### Single Table Generation
1. Upload a CSV/Parquet sample file
2. Review and edit the inferred schema
3. Choose output format (Parquet/CSV/JSON), record count, and partitioning
4. Optionally add LLM business logic rules
5. Select output sink (Local or S3) and click **Generate**

### Multi-Table (Hydra)
1. Upload 2+ related files
2. Define FK relationships (parent column → child column)
3. Set per-table row counts
4. Generate — parents are created first, children get valid FK values

### Time-Travel Simulation
1. Upload a sample file for schema
2. Configure date range, frequency, and trend percentage
3. Add volume spikes on specific dates
4. Preview the volume distribution chart
5. Generate temporal data partitioned by period

### Privacy Scorecard
1. Upload the original (real) dataset
2. Upload synthetic data or use the last generated output
3. View DCR metrics, risk level, and distribution histogram

---

## ⚙️ Configuration

### Streamlit (`.streamlit/config.toml`)
```toml
[server]
enableXsrfProtection = false
enableCORS = false
maxUploadSize = 200
```

### S3 Sink
Set AWS credentials via environment variables:
```bash
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
export AWS_DEFAULT_REGION=us-east-1
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| [Streamlit](https://streamlit.io/) | Interactive web UI |
| [Polars](https://pola.rs/) | Fast DataFrame operations |
| [Faker](https://faker.readthedocs.io/) | Realistic synthetic data generation |
| [PyArrow](https://arrow.apache.org/docs/python/) | Parquet file I/O |
| [NumPy](https://numpy.org/) + [SciPy](https://scipy.org/) | DCR distance computation |
| [Requests](https://requests.readthedocs.io/) | Ollama API communication |
| [Boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/) | Amazon S3 integration |

---

## 📄 License

This project is open source. Feel free to use and modify it as needed.
