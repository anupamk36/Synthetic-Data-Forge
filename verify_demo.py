import polars as pl

files = [
    'demo_data/patients.csv',
    'demo_data/clinical_sites.csv',
    'demo_data/clinical_subjects.csv',
    'demo_data/clinical_visits.csv',
    'demo_data/pharma_orders.csv',
    'demo_data/privacy_real_employees.csv',
    'demo_data/privacy_synthetic_employees.csv',
]

for f in files:
    df = pl.read_csv(f)
    print(f"{f}: {df.shape[0]} rows x {df.shape[1]} cols  OK")

sites = pl.read_csv('demo_data/clinical_sites.csv')
subjects = pl.read_csv('demo_data/clinical_subjects.csv')
visits = pl.read_csv('demo_data/clinical_visits.csv')

site_ids = set(sites['site_id'].to_list())
subj_site_ids = set(subjects['site_id'].to_list())
assert subj_site_ids.issubset(site_ids), 'FK broken: subjects.site_id'

subj_ids = set(subjects['subject_id'].to_list())
visit_subj_ids = set(visits['subject_id'].to_list())
assert visit_subj_ids.issubset(subj_ids), 'FK broken: visits.subject_id'

print("FK integrity: ALL OK")
