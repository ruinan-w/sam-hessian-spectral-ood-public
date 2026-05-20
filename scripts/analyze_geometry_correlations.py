import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"


def corr_or_note(frame, target, predictor, method):
    cols = frame[[target, predictor]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(cols) < 3:
        return "", f"skipped: n={len(cols)}"
    if cols[target].nunique() < 2 or cols[predictor].nunique() < 2:
        return "", "skipped: constant column"
    return float(cols[target].corr(cols[predictor], method=method)), ""


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    input_path = RESULTS_DIR / "final_analysis_table.csv"
    output_csv = RESULTS_DIR / "geometry_correlation_summary.csv"
    output_json = RESULTS_DIR / "geometry_correlation_summary.json"

    rows = []
    if not input_path.exists():
        note = f"skipped: missing {input_path}"
        rows.append(
            {
                "target": "",
                "predictor": "",
                "pearson_corr": "",
                "spearman_corr": "",
                "n": 0,
                "note": note,
            }
        )
    else:
        frame = pd.read_csv(input_path)
        targets = ["cifar10c_mean_acc", "ood_gap"]
        predictors = ["top_eigenvalue", "trace_estimate", "participation_ratio_approx"]
        for target in targets:
            for predictor in predictors:
                valid = frame[[target, predictor]].apply(pd.to_numeric, errors="coerce").dropna()
                pearson, pearson_note = corr_or_note(frame, target, predictor, "pearson")
                spearman, spearman_note = corr_or_note(frame, target, predictor, "spearman")
                rows.append(
                    {
                        "target": target,
                        "predictor": predictor,
                        "pearson_corr": pearson,
                        "spearman_corr": spearman,
                        "n": len(valid),
                        "note": pearson_note or spearman_note,
                    }
                )

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Saved: {output_csv}")
    print(f"Saved: {output_json}")


if __name__ == "__main__":
    main()
