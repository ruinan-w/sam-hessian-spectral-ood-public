import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / "matplotlib_cache"))
(PROJECT_ROOT / "outputs" / "matplotlib_cache").mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Make EI experiment figures.")
    p.add_argument("--tag-filter", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--results-dir", default="outputs/results")
    p.add_argument("--output-dir", default="outputs/figures")
    return p.parse_args()


def unique_path(path):
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    i = 2
    while True:
        candidate = path.with_name(f"{stem}_v{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def savefig(outdir, name):
    for suffix in [".png", ".pdf"]:
        plt.savefig(unique_path(outdir / f"{name}{suffix}"), bbox_inches="tight", dpi=180)
    plt.close()


def filter_df(df, tag, dataset):
    if tag and "run_name" in df:
        df = df[df["run_name"].astype(str).str.contains(tag, na=False)]
    if dataset and "dataset" in df:
        df = df[df["dataset"].astype(str).str.lower() == dataset.lower()]
    return df


def bar_metric(df, metric, title, outdir, name):
    sub = df.dropna(subset=[metric])
    if sub.empty:
        return
    grouped = sub.groupby(["dataset", "optimizer"])[metric].agg(["mean", "std"]).reset_index()
    labels = grouped["dataset"].astype(str) + "\n" + grouped["optimizer"].astype(str)
    plt.figure(figsize=(max(6, len(grouped) * 1.2), 4))
    plt.bar(labels, grouped["mean"], yerr=grouped["std"].fillna(0), capsize=4, color="#4C78A8")
    plt.ylabel(metric)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    savefig(outdir, name)


def scatter(df, x, y, outdir, name):
    sub = df.dropna(subset=[x, y])
    if sub.empty:
        return
    plt.figure(figsize=(5.5, 4))
    for opt, part in sub.groupby("optimizer"):
        plt.scatter(part[x], part[y], label=str(opt), s=50)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    savefig(outdir, name)


def main():
    args = parse_args()
    results = PROJECT_ROOT / args.results_dir
    outdir = PROJECT_ROOT / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    final = filter_df(pd.read_csv(results / "final_analysis_table.csv"), args.tag_filter, args.dataset)
    for col in final.columns:
        if col not in {"run_name", "dataset", "model", "optimizer", "checkpoint_path", "train_log_path", "cifarc_csv_path", "hessian_json_path", "topk_eigen_csv_path"}:
            final[col] = pd.to_numeric(final[col], errors="coerce")
    if not final.empty:
        clean_ood = final.melt(id_vars=["dataset", "optimizer"], value_vars=[c for c in ["final_test_acc", "cifarc_mean_acc"] if c in final], var_name="metric", value_name="accuracy").dropna()
        if not clean_ood.empty:
            grouped = clean_ood.groupby(["dataset", "optimizer", "metric"])["accuracy"].agg(["mean", "std"]).reset_index()
            labels = grouped["dataset"] + "\n" + grouped["optimizer"] + "\n" + grouped["metric"]
            plt.figure(figsize=(max(7, len(grouped) * 0.9), 4))
            plt.bar(labels, grouped["mean"], yerr=grouped["std"].fillna(0), capsize=3, color="#59A14F")
            plt.ylabel("accuracy")
            plt.xticks(rotation=35, ha="right")
            savefig(outdir, "fig_clean_vs_ood_bar")
        for metric, name in [("ood_gap", "fig_ood_gap_bar"), ("top_eigenvalue", "fig_lambda_max_bar"), ("lambda_max_over_trace", "fig_lambda_max_over_trace_bar"), ("participation_ratio_approx", "fig_participation_ratio_bar")]:
            bar_metric(final, metric, metric, outdir, name)
        scatter(final, "top_eigenvalue", "cifarc_mean_acc", outdir, "fig_cifarc_vs_lambda_max_scatter")
        scatter(final, "lambda_max_over_trace", "cifarc_mean_acc", outdir, "fig_cifarc_vs_concentration_scatter")
        scatter(final, "participation_ratio_approx", "cifarc_mean_acc", outdir, "fig_cifarc_vs_pr_scatter")
    topk_path = results / "all_topk_eigenvalues.csv"
    if topk_path.exists() and topk_path.stat().st_size > 0:
        try:
            topk = pd.read_csv(topk_path)
        except pd.errors.EmptyDataError:
            topk = pd.DataFrame()
        if not topk.empty and {"optimizer_name", "rank", "eigenvalue"}.issubset(topk.columns):
            plt.figure(figsize=(5.5, 4))
            for opt, part in topk.groupby("optimizer_name"):
                curve = part.groupby("rank")["eigenvalue"].mean()
                plt.plot(curve.index, curve.values, marker="o", label=str(opt))
            plt.xlabel("rank")
            plt.ylabel("eigenvalue")
            plt.legend()
            savefig(outdir, "fig_topk_spectrum")
    details_path = results / "all_cifarc_details.csv"
    if details_path.exists() and details_path.stat().st_size > 0:
        details = filter_df(pd.read_csv(details_path), args.tag_filter, args.dataset)
        if not details.empty and {"optimizer", "severity", "accuracy"}.issubset(details.columns):
            plt.figure(figsize=(5.5, 4))
            for opt, part in details.groupby("optimizer"):
                curve = part.groupby("severity")["accuracy"].mean()
                plt.plot(curve.index, curve.values, marker="o", label=str(opt))
            plt.xlabel("severity")
            plt.ylabel("mean accuracy")
            plt.legend()
            savefig(outdir, "fig_severity_curves")
    print(f"FIGURE_DIR={outdir.resolve()}")


if __name__ == "__main__":
    main()
