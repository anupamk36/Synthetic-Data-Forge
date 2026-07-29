"""TNM staging and RECIST response helpers."""

import random as _random

from core.medical.terminologies.loader import _load_codeset

_TNM_FILE = "tnm_staging.json"
_RECIST_FILE = "recist_criteria.json"


def _load_tnm() -> dict:
    return _load_codeset(_TNM_FILE)


def _load_recist() -> dict:
    return _load_codeset(_RECIST_FILE)


def random_tnm_stage(cancer_type: str = "lung", rng=None) -> dict:
    """Generate a random TNM stage for a given cancer type."""
    rng = rng or _random.Random()
    data = _load_tnm()
    cancer = data.get("cancer_types", {}).get(cancer_type)
    if not cancer:
        cancer = list(data.get("cancer_types", {}).values())[0]

    t_stages = cancer.get("t_stages", [])
    n_stages = cancer.get("n_stages", [])
    m_stages = cancer.get("m_stages", [])

    t = rng.choices(t_stages, weights=[s["weight"] for s in t_stages], k=1)[0]
    n = rng.choices(n_stages, weights=[s["weight"] for s in n_stages], k=1)[0]
    m = rng.choices(m_stages, weights=[s["weight"] for s in m_stages], k=1)[0]

    return {
        "t": t["code"],
        "t_description": t["description"],
        "n": n["code"],
        "n_description": n["description"],
        "m": m["code"],
        "m_description": m["description"],
        "stage_string": f"{t['code']}{n['code']}{m['code']}",
    }


def recist_response_rates(treatment_type: str = "immunotherapy") -> dict:
    """Get response rate distribution for a treatment type."""
    data = _load_recist()
    rates = data.get("treatment_response_rates", {})
    return rates.get(treatment_type, rates.get("chemotherapy", {}))


def assign_recist_trajectory(arm: str, treatment_type: str = "immunotherapy", rng=None) -> str:
    """Assign a subject to a RECIST response trajectory based on arm."""
    rng = rng or _random.Random()
    if arm in ("placebo", "SOC", "control"):
        rates = recist_response_rates("placebo")
    else:
        rates = recist_response_rates(treatment_type)

    categories = list(rates.keys())
    weights = list(rates.values())
    response = rng.choices(categories, weights=weights, k=1)[0]

    trajectory_map = {"CR": "responder", "PR": "responder", "SD": "stable", "PD": "progressor"}
    return trajectory_map.get(response, "stable")


def generate_tumor_measurements(
    trajectory: str,
    num_timepoints: int,
    rng=None,
) -> list[dict]:
    """Generate tumor sum-of-diameters over time following a trajectory."""
    rng = rng or _random.Random()
    data = _load_recist()
    trajectories = data.get("response_trajectories", {})
    traj_params = trajectories.get(trajectory, trajectories.get("stable", {}))

    baseline_range = traj_params.get("baseline_sum_mm", [40, 100])
    change_range = traj_params.get("change_per_cycle_pct", [-5, 5])
    noise_pct = traj_params.get("noise_pct", 5)

    baseline = rng.uniform(baseline_range[0], baseline_range[1])
    measurements = [{"timepoint": 0, "sum_mm": round(baseline, 1), "pct_change": 0.0}]

    current = baseline
    for tp in range(1, num_timepoints):
        change_pct = rng.uniform(change_range[0], change_range[1])
        noise = rng.uniform(-noise_pct, noise_pct)
        total_change = (change_pct + noise) / 100.0
        current = max(0, current * (1 + total_change))
        pct_from_baseline = ((current - baseline) / baseline) * 100

        measurements.append({
            "timepoint": tp,
            "sum_mm": round(current, 1),
            "pct_change": round(pct_from_baseline, 1),
        })

    return measurements


def classify_recist_response(pct_change: float, nadir_pct_change: float | None = None) -> str:
    """Classify RECIST response based on percent change from baseline."""
    if pct_change <= -100:
        return "CR"
    elif pct_change <= -30:
        return "PR"
    elif pct_change >= 20:
        return "PD"
    else:
        return "SD"
