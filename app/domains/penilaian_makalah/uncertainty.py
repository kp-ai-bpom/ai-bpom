"""Uncertainty sampling helpers for penilaian_makalah."""

from __future__ import annotations

from statistics import mean, stdev

from app.domains.penilaian_makalah.constant import CRITERIA_KEYS, CRITERIA_SHORT, NSV_THRESHOLD, SCORE_RANGE


def calculate_uncertainty_metrics(scores_list: list) -> dict:
    """
    Calculate NSV, WAU, and uncertainty metrics from M samples.
    """
    score_distributions = {k: [] for k in CRITERIA_SHORT}
    
    for res in scores_list:
        if not res:
            continue
        for i, key_short in enumerate(CRITERIA_SHORT):
            key_full = CRITERIA_KEYS[i]
            if key_full in res:
                score_distributions[key_short].append(res[key_full])
    
    evaluation_results = {
        "consensus_scores": {},
        "uncertainty": {"per_criteria": {}},
        "valid_samples": len([s for s in scores_list if s])
    }
    
    nsv_dict = {}
    
    for i, (criteria_short, criteria_full) in enumerate(zip(CRITERIA_SHORT, CRITERIA_KEYS)):
        values = score_distributions[criteria_short]
        
        if not values:
            evaluation_results["consensus_scores"][criteria_full] = 0
            evaluation_results["uncertainty"]["per_criteria"][criteria_short] = {
                "mean": 0, "std": 0, "nsv": 0, "status": "❌ NO DATA", "raw_samples": []
            }
            nsv_dict[criteria_short] = 0
            continue
        
        mean_score = mean(values)
        std_dev = stdev(values) if len(values) > 1 else 0.0
        nsv = std_dev / SCORE_RANGE
        nsv_dict[criteria_short] = nsv
        
        status = "⚠️ PERLU REVIEW" if nsv > NSV_THRESHOLD else "✅ YAKIN"
        
        evaluation_results["consensus_scores"][criteria_full] = round(mean_score, 2)
        evaluation_results["uncertainty"]["per_criteria"][criteria_short] = {
            "mean": round(mean_score, 2),
            "std": round(std_dev, 2),
            "nsv": round(nsv, 3),
            "status": status,
            "raw_samples": [round(v, 1) for v in values]
        }
    
    # Calculate WAU (Weighted Aggregate Uncertainty)
    if all(k in nsv_dict for k in CRITERIA_SHORT):
        wau = (nsv_dict["n1"] + nsv_dict["n2"] + nsv_dict["n3"] + 
               (2 * nsv_dict["n4"]) + nsv_dict["n5"]) / 6
        
        evaluation_results["uncertainty"]["weighted_aggregate"] = round(wau, 3)
        evaluation_results["uncertainty"]["overall_status"] = (
            "⚠️ BUTUH REVIEW HUMAN" if wau > NSV_THRESHOLD else "✅ YAKIN (Konsisten)"
        )
        evaluation_results["uncertainty"]["most_uncertain_criteria"] = max(nsv_dict, key=nsv_dict.get)
    
    return evaluation_results