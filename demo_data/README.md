# Demo Data Files

Sample datasets for demonstrating all four features of **ForgeFlow AI / Synthetic-Data-Forge**.

---

## 1. Single Table Generator

**File:** `patients.csv` (15 rows × 10 columns)

Upload this file on the **Single Table Generator** page.

| Column | Description |
|---|---|
| patient_id | Unique patient identifier |
| first_name / last_name | Patient name |
| age | Patient age |
| sex | M or F |
| diagnosis | Common chronic disease |
| blood_type | ABO blood group |
| region | Geographic region (North America, Europe, Asia Pacific, etc.) |
| admission_date | Hospital admission date |
| billing_amount | Treatment cost in USD |

**Demo tips:**
- Generate 1,000+ records to see partitioning in action
- Try **Partition by `region`** after generation to see Hive-style folder output
- Enable **LLM Smart Mode** and add hints like:
  - `diagnosis`: "common chronic diseases such as Diabetes, Hypertension, COPD"
  - `sex`: "M or F"
  - `blood_type`: "one of A+, A-, B+, B-, AB+, AB-, O+, O-"

---

## 2. Multi-Table Generator (Hydra)

**Files (upload all 3 together):**

| File | Rows | Role | Key Columns |
|---|---|---|---|
| `clinical_sites.csv` | 8 | Parent table | `site_id` (PK) |
| `clinical_subjects.csv` | 15 | Child of sites | `subject_id` (PK), `site_id` (FK → sites) |
| `clinical_visits.csv` | 15 | Child of subjects | `visit_id` (PK), `subject_id` (FK → subjects) |

**Relationships to define:**
1. `clinical_subjects.site_id` → `clinical_sites.site_id`
2. `clinical_visits.subject_id` → `clinical_subjects.subject_id`

This models a **clinical trial** hierarchy: Sites → Subjects → Visits.

**Demo tips:**
- The Mermaid diagram will show the FK chain automatically
- Generate 50–100 records per table to keep it fast
- Verify FK consistency in the output — child tables will only reference valid parent keys

---

## 3. Time Travel Simulator

**File:** `pharma_orders.csv` (15 rows × 8 columns)

Upload this file on the **Time Travel Simulator** page.

| Column | Description |
|---|---|
| order_id | Order identifier |
| order_date | Date of order |
| product_category | Oncology, Immunology, Diagnostics, etc. |
| product_name | Drug or device name |
| quantity | Units ordered |
| unit_price | Price per unit |
| customer_region | Geographic region |
| channel | Hospital, Pharmacy, or Direct |

**Recommended settings:**
- **Date range:** Jan 1, 2024 → Dec 31, 2024
- **Frequency:** Monthly
- **Base records/period:** 100
- **Growth trend:** +5% per period (simulates business growth)
- **Add a spike:** Dec 2024, multiplier 2.5× ("year-end budget flush")

**Demo tips:**
- The volume preview chart shows how record counts change over time
- Try different trend percentages to see growth vs. decline patterns
- Add multiple spikes for seasonal patterns

---

## 4. Privacy Audit

**Files (upload separately into left and right panels):**

| File | Role | Rows |
|---|---|---|
| `privacy_real_employees.csv` | **Real (original) data** — upload on the left | 20 |
| `privacy_synthetic_employees.csv` | **Synthetic data** — upload on the right | 20 |

Both files share the same schema:

| Column | Description |
|---|---|
| employee_id | Employee identifier |
| department | Engineering, Sales, Marketing, HR, Finance |
| job_title | Role title |
| salary | Annual salary in USD |
| years_experience | Years of experience |
| performance_score | Rating 1.0–5.0 |
| city | City name |

The synthetic file has realistic variation (different cities, slightly adjusted salaries, shifted experience) to produce a **Medium / Low privacy risk** result.

**Demo tips:**
- Look at the DCR (Distance to Closest Record) histogram for distribution
- The metric cards show Min, Mean, Median DCR and exact-match percentage
- A green "Low Risk" badge means the synthetic data is well-differentiated from real data
- You can also generate synthetic data on the Single Table page first, then use "Use last generated" on the Privacy Audit page to compare against the real file

---

## Quick-Start Checklist

1. Start the app: `streamlit run app/main.py`
2. Navigate via the sidebar to each feature
3. Upload the corresponding file(s) from this `demo_data/` folder
4. Configure settings as described above
5. Click **Generate** and explore the results
