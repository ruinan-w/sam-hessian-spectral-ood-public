import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS = ["final_test_acc", "cifarc_mean_acc", "ood_gap", "top_eigenvalue", "trace_estimate", "participation_ratio_approx", "lambda_max_over_trace", "top_5_mass_ratio", "top_10_mass_ratio", "effective_rank_entropy", "spectral_entropy"]
TARGETS = ["cifarc_mean_acc", "ood_gap"]
PREDICTORS = ["top_eigenvalue", "trace_estimate", "participation_ratio_approx", "lambda_max_over_trace", "top_5_mass_ratio", "top_10_mass_ratio", "effective_rank_entropy", "spectral_entropy"]
MODELS = {
    "Flatness only": ["top_eigenvalue"],
    "Curvature magnitude": ["top_eigenvalue", "trace_estimate"],
    "Participation only": ["participation_ratio_approx"],
    "Concentration": ["lambda_max_over_trace", "top_5_mass_ratio"],
    "Spectral redistribution": ["top_eigenvalue", "participation_ratio_approx", "lambda_max_over_trace"],
    "Full geometry": ["top_eigenvalue", "trace_estimate", "participation_ratio_approx", "lambda_max_over_trace", "top_5_mass_ratio"],
}


def parse_args():
    p = argparse.ArgumentParser(description="Analyze geometry statistics and explanatory power.")
    p.add_argument("--input", default="outputs/results/final_analysis_table.csv")
    p.add_argument("--tag-filter", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--output-dir", default="outputs/results")
    return p.parse_args()


def adj_r2(r2, n, p):
    return math.nan if n <= p + 1 else 1 - (1 - r2) * (n - 1) / (n - p - 1)


def main():
    args = parse_args()
    out = PROJECT_ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROJECT_ROOT / args.input)
    if args.tag_filter:
        df = df[df["run_name"].astype(str).str.contains(args.tag_filter, na=False)]
    if args.dataset:
        df = df[df["dataset"].astype(str).str.lower() == args.dataset.lower()]
    for col in METRICS + PREDICTORS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    group = df.groupby(["dataset", "optimizer"])[[c for c in METRICS if c in df.columns]].agg(["mean", "std"])
    group.columns = ["_".join(col).strip() for col in group.columns.values]
    group.reset_index().to_csv(out / "group_statistics.csv", index=False)
    corrs = []
    for target in TARGETS:
        for pred in PREDICTORS:
            sub = df[[target, pred]].dropna() if target in df and pred in df else pd.DataFrame()
            corrs.append({"target": target, "predictor": pred, "pearson_corr": sub[target].corr(sub[pred], method="pearson") if len(sub) >= 2 else math.nan, "spearman_corr": sub[target].corr(sub[pred], method="spearman") if len(sub) >= 2 else math.nan, "n": len(sub), "note": "low_n" if len(sub) < 6 else ""})
    pd.DataFrame(corrs).to_csv(out / "geometry_correlation_summary.csv", index=False)
    regs = []
    warnings = []
    target = "cifarc_mean_acc"
    for name, preds in MODELS.items():
        cols = [target] + preds
        sub = df[cols].dropna() if all(c in df for c in cols) else pd.DataFrame()
        if len(sub) <= len(preds) + 1:
            regs.append({"model_name": name, "target": target, "predictors": "+".join(preds), "r2": math.nan, "adjusted_r2": math.nan, "n": len(sub), "note": "insufficient_samples"})
            warnings.append(f"{name}: insufficient samples n={len(sub)} for {len(preds)} predictors.")
            continue
        X = np.column_stack([np.ones(len(sub)), sub[preds].to_numpy(dtype=float)])
        y = sub[target].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = math.nan if ss_tot == 0 else 1 - ss_res / ss_tot
        regs.append({"model_name": name, "target": target, "predictors": "+".join(preds), "r2": r2, "adjusted_r2": adj_r2(r2, len(sub), len(preds)), "n": len(sub), "note": "low_n" if len(sub) < 10 else ""})
    pd.DataFrame(regs).to_csv(out / "regression_explanatory_power.csv", index=False)
    if len(df) < 10:
        warnings.append(f"Regression/correlation sample size is small: n={len(df)}.")
    with (out / "analysis_report.json").open("w", encoding="utf-8") as f:
        json.dump({"n_rows": len(df), "warnings": warnings}, f, indent=2)
    print(f"Saved: {(out / 'group_statistics.csv').resolve()}")
    print(f"Saved: {(out / 'geometry_correlation_summary.csv').resolve()}")
    print(f"Saved: {(out / 'regression_explanatory_power.csv').resolve()}")
    print(f"Saved: {(out / 'analysis_report.json').resolve()}")


if __name__ == "__main__":
    main()
