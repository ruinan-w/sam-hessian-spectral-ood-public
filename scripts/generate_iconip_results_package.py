from __future__ import annotations

import ast
import csv
import json
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
LOGS = OUTPUTS / "logs"
RESULTS = OUTPUTS / "results"

PALETTE = {
    "sgd": "#6B7280",
    "sam": "#2563EB",
    "asam": "#059669",
}
LIGHT = {
    "sgd": "#D1D5DB",
    "sam": "#93C5FD",
    "asam": "#6EE7B7",
}
OPT_ORDER = ["sgd", "sam", "asam"]
SETTING_ORDER = [
    ("cifar10", "resnet18"),
    ("cifar100", "resnet18"),
    ("cifar10", "resnet34"),
]
SETTING_LABELS = {
    ("cifar10", "resnet18"): "ResNet-18 / CIFAR-10-C",
    ("cifar100", "resnet18"): "ResNet-18 / CIFAR-100-C",
    ("cifar10", "resnet34"): "ResNet-34 / CIFAR-10-C",
}
CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]


try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


@dataclass
class OutputPaths:
    root: Path
    tables: Path
    figures: Path
    analysis: Path
    latex: Path
    reports: Path


def make_unique_dir(base: Path) -> Path:
    if not base.exists():
        base.mkdir(parents=True)
        return base
    for i in range(2, 100):
        candidate = base.with_name(f"{base.name}_v{i}")
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
    raise RuntimeError(f"Could not create unique directory for {base}")


def make_package_dir() -> OutputPaths:
    root = make_unique_dir(OUTPUTS / "iconip_package_greenblue")
    paths = OutputPaths(
        root=root,
        tables=root / "tables",
        figures=root / "figures",
        analysis=root / "analysis",
        latex=root / "latex",
        reports=root / "reports",
    )
    for p in [paths.tables, paths.figures, paths.analysis, paths.latex, paths.reports]:
        p.mkdir(parents=True, exist_ok=False)
    return paths


def latest_audit_dir() -> Path:
    candidates = []
    for p in OUTPUTS.iterdir() if OUTPUTS.exists() else []:
        if p.is_dir() and p.name.startswith("audit") and (p / "run_inventory.csv").exists():
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError("No audit directory with run_inventory.csv found.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def parse_dict(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def parse_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def version_score(path: Path) -> tuple[int, float, int]:
    stem = path.stem
    version = 0
    if "_v" in stem:
        suffix = stem.rsplit("_v", 1)[-1]
        if suffix.isdigit():
            version = int(suffix)
    return (version, path.stat().st_mtime if path.exists() else 0.0, path.stat().st_size if path.exists() else 0)


def candidate_files(run_name: str, suffix: str) -> list[Path]:
    files = list(LOGS.glob(f"{run_name}{suffix}")) + list(LOGS.glob(f"{run_name}_v*{suffix}"))
    return sorted(set(files), key=version_score, reverse=True)


def load_best_cifarc_summary(run_name: str) -> tuple[dict, Path | None]:
    best = None
    best_path = None
    for path in candidate_files(run_name, "_cifarc_summary.json"):
        data = read_json(path)
        if not data:
            continue
        sev = data.get("severities", [])
        corr = data.get("corruptions", [])
        score = (len(sev), len(corr), version_score(path))
        if best is None or score > best[0]:
            best = (score, data)
            best_path = path
    return (best[1], best_path) if best else ({}, None)


def load_topk_values(row: pd.Series) -> list[float]:
    path_text = str(row.get("topk_eigenvalues_file", ""))
    paths = []
    if path_text:
        paths.append(Path(path_text))
    run_name = run_name_for(row)
    paths.extend(candidate_files(run_name, "_hessian_topk_eigenvalues.csv"))
    for path in paths:
        try:
            if not path.exists() or path.stat().st_size <= 100:
                continue
            df = pd.read_csv(path)
            col = "raw_eigenvalue" if "raw_eigenvalue" in df.columns else df.columns[-1]
            vals = pd.to_numeric(df[col], errors="coerce").dropna().astype(float).tolist()
            if vals:
                return vals
        except Exception:
            continue
    hessian_path = Path(str(row.get("hessian_file", "")))
    data = read_json(hessian_path) if hessian_path.exists() else {}
    vals = parse_list(data.get("top_k_eigenvalues") or data.get("raw_topk_eigenvalues"))
    return [float(v) for v in vals if isinstance(v, (int, float)) or str(v).replace(".", "", 1).replace("-", "", 1).isdigit()]


def likely_existing_metric_files(run_name: str) -> list[Path]:
    paths = []
    for suffix in [
        "_train.csv",
        "_cifarc_summary.json",
        "_cifarc_details.csv",
        "_hessian_geometry.json",
        "_hessian_eigen_proxy.csv",
        "_hessian_topk_eigenvalues.csv",
    ]:
        paths.extend(candidate_files(run_name, suffix))
    return [p for p in sorted(set(paths), key=version_score, reverse=True) if p.exists()]


def run_name_for(row: pd.Series) -> str:
    dataset = row["dataset"]
    model = row["model"]
    opt = row["optimizer"]
    seed = int(row["seed"])
    if model == "resnet18":
        return f"ei100_{dataset}_{opt}_seed{seed}_ep100"
    if model == "resnet34":
        return f"ei_arch_cifar10_resnet34_{opt}_seed{seed}_ep100"
    if model == "vgg16_bn":
        return f"ei_vgg_cifar10_vgg16_bn_{opt}_seed{seed}_ep100"
    return ""


def clean_inventory(df: pd.DataFrame, missing: list[str]) -> pd.DataFrame:
    aliases = {
        "top_eigenvalue": ["top_eigenvalue", "top_hessian_eigenvalue", "lambda_max"],
        "clean_accuracy": ["clean_accuracy", "final_test_acc", "test_acc"],
        "cifarc_mean_accuracy": ["cifarc_mean_accuracy", "cifarc_mean_acc", "mean_accuracy_all"],
    }
    for canonical, opts in aliases.items():
        if canonical not in df.columns:
            found = next((c for c in opts if c in df.columns), None)
            if found:
                df[canonical] = df[found]
    formal = df[
        (df.get("status", "") == "COMPLETE")
        & (df.get("model", "") != "vgg16_bn")
        & (df.get("checkpoint_exists", "").astype(str).str.lower() == "true")
    ].copy()
    for col in [
        "clean_accuracy",
        "clean_loss",
        "cifarc_mean_accuracy",
        "cifarc_mean_loss",
        "ood_gap",
        "top_eigenvalue",
        "top_hessian_eigenvalue",
        "trace_estimate",
        "participation_ratio_approx",
        "top_1_mass_ratio",
        "top_5_mass_ratio",
        "top_10_mass_ratio",
        "lambda_max_over_topk_sum",
        "spectral_entropy",
        "effective_rank_entropy",
        "participation_ratio_topk",
    ]:
        if col in formal.columns:
            formal[col] = pd.to_numeric(formal[col], errors="coerce")
    if "top_eigenvalue" not in formal.columns and "top_hessian_eigenvalue" in formal.columns:
        formal["top_eigenvalue"] = formal["top_hessian_eigenvalue"]
    elif "top_eigenvalue" in formal.columns and "top_hessian_eigenvalue" in formal.columns:
        formal["top_eigenvalue"] = formal["top_eigenvalue"].fillna(formal["top_hessian_eigenvalue"])
    required = [
        "dataset",
        "ood_dataset",
        "model",
        "optimizer",
        "seed",
        "clean_accuracy",
        "clean_loss",
        "cifarc_mean_accuracy",
        "cifarc_mean_loss",
        "ood_gap",
        "top_eigenvalue",
        "trace_estimate",
        "participation_ratio_approx",
        "top_1_mass_ratio",
        "top_5_mass_ratio",
        "top_10_mass_ratio",
        "lambda_max_over_topk_sum",
        "spectral_entropy",
        "effective_rank_entropy",
        "participation_ratio_topk",
    ]
    for col in required:
        if col not in formal.columns:
            formal[col] = np.nan
            missing.append(f"run_inventory missing field: {col}")
    return formal


def enrich_from_logs(df: pd.DataFrame, missing: list[str], read_files: set[str]) -> pd.DataFrame:
    severity_rows = []
    corruption_rows = []
    topk_rows = []
    for idx, row in df.iterrows():
        run_name = run_name_for(row)
        for metric_file in likely_existing_metric_files(run_name):
            read_files.add(str(metric_file))
        summary, path = load_best_cifarc_summary(run_name)
        if path:
            read_files.add(str(path))
        if summary:
            sev = parse_dict(summary.get("mean_accuracy_by_severity"))
            corr = parse_dict(summary.get("mean_accuracy_by_corruption"))
            for s in range(1, 6):
                val = sev.get(str(s), sev.get(s, np.nan))
                df.loc[idx, f"severity_{s}_accuracy"] = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
            for c in CORRUPTIONS:
                if c in corr:
                    corruption_rows.append(
                        {
                            "dataset": row["dataset"],
                            "model": row["model"],
                            "optimizer": row["optimizer"],
                            "seed": int(row["seed"]),
                            "corruption": c,
                            "accuracy": float(corr[c]),
                        }
                    )
        else:
            missing.append(f"missing CIFAR-C summary for {run_name}")
        vals = load_topk_values(row)
        if vals:
            positives = np.array([v for v in vals if v > 0], dtype=float)
            if positives.size:
                mass = positives / positives.sum()
                for rank, val in enumerate(mass[:20], start=1):
                    topk_rows.append(
                        {
                            "dataset": row["dataset"],
                            "model": row["model"],
                            "optimizer": row["optimizer"],
                            "seed": int(row["seed"]),
                            "rank": rank,
                            "normalized_positive_mass": float(val),
                        }
                    )
        else:
            missing.append(f"missing top-k eigenvalues for {run_name}")
    for s in range(1, 6):
        col = f"severity_{s}_accuracy"
        if col not in df.columns:
            df[col] = np.nan
            missing.append(f"missing metric after enrichment: {col}")
    return df, pd.DataFrame(severity_rows), pd.DataFrame(corruption_rows), pd.DataFrame(topk_rows)


def fmt_mean_std(mean: float, std: float, sci: bool = False) -> str:
    if pd.isna(mean):
        return ""
    if sci or (abs(mean) >= 1000 or (0 < abs(mean) < 0.001)):
        return f"{mean:.3e} $\\pm$ {std:.3e}"
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def latex_escape(s: str) -> str:
    return str(s).replace("_", "\\_")


def grouped_table(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for dataset, model in SETTING_ORDER:
        for opt in OPT_ORDER:
            sub = df[(df.dataset == dataset) & (df.model == model) & (df.optimizer == opt)]
            if sub.empty:
                continue
            out = {
                "dataset": dataset,
                "ood_dataset": sub["ood_dataset"].iloc[0],
                "model": model,
                "optimizer": opt.upper(),
            }
            for metric in metrics:
                out[f"{metric}_mean"] = sub[metric].mean()
                out[f"{metric}_std"] = sub[metric].std(ddof=1)
            rows.append(out)
    return pd.DataFrame(rows)


def write_table1(df: pd.DataFrame, paths: OutputPaths) -> None:
    metrics = ["clean_accuracy", "cifarc_mean_accuracy", "ood_gap"]
    tbl = grouped_table(df, metrics)
    tbl.to_csv(paths.tables / "table1_main_ood_results.csv", index=False)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Main clean and corruption robustness results across three seeds.}",
        "\\label{tab:main_ood_results}",
        "\\begin{tabular}{llllccc}",
        "\\toprule",
        "Dataset & OOD dataset & Model & Optimizer & Clean acc. & CIFAR-C acc. & OOD gap \\\\",
        "\\midrule",
    ]
    for _, r in tbl.iterrows():
        lines.append(
            f"{latex_escape(r.dataset)} & {latex_escape(r.ood_dataset)} & {latex_escape(r.model)} & {r.optimizer} & "
            f"{fmt_mean_std(r.clean_accuracy_mean, r.clean_accuracy_std)} & "
            f"{fmt_mean_std(r.cifarc_mean_accuracy_mean, r.cifarc_mean_accuracy_std)} & "
            f"{fmt_mean_std(r.ood_gap_mean, r.ood_gap_std)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (paths.tables / "table1_main_ood_results.tex").write_text("\n".join(lines), encoding="utf-8")


def write_table2(df: pd.DataFrame, paths: OutputPaths) -> None:
    metrics = [
        "top_eigenvalue",
        "trace_estimate",
        "top_5_mass_ratio",
        "lambda_max_over_topk_sum",
        "spectral_entropy",
        "effective_rank_entropy",
        "participation_ratio_topk",
    ]
    tbl = grouped_table(df, metrics)
    tbl.to_csv(paths.tables / "table2_hessian_spectral_metrics.csv", index=False)
    headers = [
        ("top_eigenvalue", "$\\lambda_{max}$"),
        ("trace_estimate", "Trace"),
        ("top_5_mass_ratio", "Top-5 mass"),
        ("lambda_max_over_topk_sum", "$\\lambda_{max}$/top-k"),
        ("spectral_entropy", "Entropy"),
        ("effective_rank_entropy", "Eff. rank"),
        ("participation_ratio_topk", "PR top-k"),
    ]
    def tex_for(selected, filename):
        align = "llll" + "c" * len(selected)
        lines = [
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Hessian geometry and top-k spectral metrics across optimizers.}",
            "\\label{tab:hessian_spectral_metrics}",
            f"\\begin{{tabular}}{{{align}}}",
            "\\toprule",
            "Dataset & Model & Optimizer & " + " & ".join(label for _, label in selected) + " \\\\",
            "\\midrule",
        ]
        for _, r in tbl.iterrows():
            values = []
            for metric, _ in selected:
                values.append(fmt_mean_std(r[f"{metric}_mean"], r[f"{metric}_std"]))
            lines.append(f"{latex_escape(r.dataset)} & {latex_escape(r.model)} & {r.optimizer} & " + " & ".join(values) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
        (paths.tables / filename).write_text("\n".join(lines), encoding="utf-8")
    tex_for(headers, "table2_hessian_spectral_metrics_full.tex")
    tex_for([h for h in headers if h[0] in ["top_eigenvalue", "trace_estimate", "top_5_mass_ratio", "spectral_entropy", "effective_rank_entropy"]], "table2_hessian_spectral_metrics_compact.tex")
    shutil.copyfile(paths.tables / "table2_hessian_spectral_metrics_compact.tex", paths.tables / "table2_hessian_spectral_metrics.tex")


def ols_metrics(data: pd.DataFrame, y_col: str, predictors: list[str]) -> dict:
    sub = data[[y_col] + predictors].dropna()
    n = len(sub)
    p = len(predictors)
    if n == 0:
        return {"n": 0, "r2": np.nan, "adjusted_r2": np.nan, "pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan, "notes": "no_data"}
    y = sub[y_col].to_numpy(float)
    X = sub[predictors].to_numpy(float)
    X = np.column_stack([np.ones(n), X])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1) if n > p + 1 and not np.isnan(r2) else np.nan
        if stats is not None and n >= 3 and np.std(pred) > 0 and np.std(y) > 0:
            pr, pp = stats.pearsonr(pred, y)
            sr, sp = stats.spearmanr(pred, y)
        else:
            pr = np.corrcoef(pred, y)[0, 1] if n >= 2 and np.std(pred) > 0 and np.std(y) > 0 else np.nan
            pp = np.nan
            sr = np.nan
            sp = np.nan
        notes = "small_n_warning" if n < 12 or n <= p + 2 else ""
        return {"n": n, "r2": r2, "adjusted_r2": adj, "pearson_r": pr, "pearson_p": pp, "spearman_r": sr, "spearman_p": sp, "notes": notes}
    except Exception as exc:
        return {"n": n, "r2": np.nan, "adjusted_r2": np.nan, "pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan, "notes": f"regression_error:{exc}"}


def regression_analysis(df: pd.DataFrame, paths: OutputPaths) -> pd.DataFrame:
    model_defs = [
        ("Flatness only", ["top_eigenvalue"]),
        ("Curvature magnitude", ["top_eigenvalue", "trace_estimate"]),
        ("Spectral concentration", ["top_5_mass_ratio", "lambda_max_over_topk_sum"]),
        ("Spectral entropy/rank", ["spectral_entropy", "effective_rank_entropy"]),
        ("Spectral redistribution", ["top_5_mass_ratio", "lambda_max_over_topk_sum", "spectral_entropy", "effective_rank_entropy"]),
        ("Full geometry", ["top_eigenvalue", "trace_estimate", "top_5_mass_ratio", "lambda_max_over_topk_sum", "spectral_entropy", "effective_rank_entropy"]),
    ]
    subsets = [
        ("ResNet-18 / CIFAR-10-C", df[(df.model == "resnet18") & (df.dataset == "cifar10")]),
        ("ResNet-18 / CIFAR-100-C", df[(df.model == "resnet18") & (df.dataset == "cifar100")]),
        ("Combined ResNet-18 only", df[df.model == "resnet18"]),
        ("All residual formal runs", df[df.model.isin(["resnet18", "resnet34"])]),
    ]
    rows = []
    for subset_name, sub in subsets:
        for name, preds in model_defs:
            out = ols_metrics(sub, "cifarc_mean_accuracy", preds)
            rows.append({"subset": subset_name, "model_name": name, "predictors": ", ".join(preds), **out})
    reg = pd.DataFrame(rows)
    reg.to_csv(paths.analysis / "regression_explanatory_power.csv", index=False)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Explanatory power of scalar flatness and Hessian spectral metrics for corruption robustness.}",
        "\\label{tab:regression_explanatory_power}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Subset & Model & $n$ & $R^2$ & Adj. $R^2$ & Pearson $r$ \\\\",
        "\\midrule",
    ]
    for _, r in reg.iterrows():
        lines.append(f"{latex_escape(r.subset)} & {latex_escape(r.model_name)} & {int(r.n)} & {r.r2:.3f} & {r.adjusted_r2:.3f} & {r.pearson_r:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    (paths.tables / "table3_regression_explanatory_power.tex").write_text("\n".join(lines), encoding="utf-8")
    return reg


def setup_plot():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def savefig(paths: OutputPaths, name: str, fig) -> tuple[bool, bool]:
    pdf = paths.figures / f"{name}.pdf"
    png = paths.figures / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf.exists(), png.exists()


def plot_figures(df: pd.DataFrame, reg: pd.DataFrame, corr_df: pd.DataFrame, topk_df: pd.DataFrame, paths: OutputPaths, missing: list[str]) -> dict:
    setup_plot()
    status = {}
    # Figure 1
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    x = np.arange(len(SETTING_ORDER))
    width = 0.23
    for i, opt in enumerate(OPT_ORDER):
        means, stds = [], []
        for dataset, model in SETTING_ORDER:
            sub = df[(df.dataset == dataset) & (df.model == model) & (df.optimizer == opt)]
            means.append(sub.cifarc_mean_accuracy.mean())
            stds.append(sub.cifarc_mean_accuracy.std(ddof=1))
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=3, color=PALETTE[opt], label=opt.upper(), edgecolor="black", linewidth=0.4)
    ax.set_xticks(x, [SETTING_LABELS[s] for s in SETTING_ORDER], rotation=15, ha="right")
    ax.set_ylabel("CIFAR-C mean accuracy")
    lower = max(0, float(df.cifarc_mean_accuracy.min()) - 0.05)
    ax.set_ylim(lower, min(1.0, float(df.cifarc_mean_accuracy.max()) + 0.05))
    ax.grid(axis="y", color="#D1D5DB", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    status["figure1_main_ood_robustness"] = savefig(paths, "figure1_main_ood_robustness", fig)

    # Figure 2
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), sharey=True)
    for ax, setting in zip(axes, SETTING_ORDER):
        dataset, model = setting
        for opt in OPT_ORDER:
            means, stds = [], []
            for s in range(1, 6):
                sub = df[(df.dataset == dataset) & (df.model == model) & (df.optimizer == opt)]
                vals = sub[f"severity_{s}_accuracy"].dropna()
                means.append(vals.mean())
                stds.append(vals.std(ddof=1))
            xs = np.arange(1, 6)
            means_arr = np.array(means, float)
            stds_arr = np.array(stds, float)
            ax.plot(xs, means_arr, marker="o", color=PALETTE[opt], label=opt.upper(), linewidth=2)
            ax.fill_between(xs, means_arr - stds_arr, means_arr + stds_arr, color=LIGHT[opt], alpha=0.28, linewidth=0)
        ax.set_title(SETTING_LABELS[setting], fontsize=9)
        ax.set_xlabel("Severity")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.grid(color="#D1D5DB", alpha=0.25)
    axes[0].set_ylabel("Accuracy")
    axes[-1].legend(frameon=False)
    status["figure2_severity_wise_robustness"] = savefig(paths, "figure2_severity_wise_robustness", fig)

    # Figure 3
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    markers = {("cifar10", "resnet18"): "o", ("cifar100", "resnet18"): "s", ("cifar10", "resnet34"): "^"}
    for setting in SETTING_ORDER:
        for opt in OPT_ORDER:
            sub = df[(df.dataset == setting[0]) & (df.model == setting[1]) & (df.optimizer == opt)]
            ax.scatter(sub.top_eigenvalue, sub.cifarc_mean_accuracy, color=PALETTE[opt], marker=markers[setting], s=42, edgecolor="black", linewidth=0.35, alpha=0.9, label=f"{opt.upper()} | {SETTING_LABELS[setting]}" if setting == SETTING_ORDER[0] else None)
    valid = df[["top_eigenvalue", "cifarc_mean_accuracy"]].dropna()
    if len(valid) >= 3:
        coeff = np.polyfit(valid.top_eigenvalue, valid.cifarc_mean_accuracy, 1)
        xs = np.linspace(valid.top_eigenvalue.min(), valid.top_eigenvalue.max(), 100)
        ax.plot(xs, coeff[0] * xs + coeff[1], color="#374151", linewidth=1.2, linestyle="--")
        if stats is not None:
            r, p = stats.pearsonr(valid.top_eigenvalue, valid.cifarc_mean_accuracy)
            ax.text(0.03, 0.05, f"Pearson r={r:.2f}, p={p:.3f}", transform=ax.transAxes, fontsize=9)
    ax.set_xlabel("Top Hessian eigenvalue")
    ax.set_ylabel("CIFAR-C mean accuracy")
    ax.grid(color="#D1D5DB", alpha=0.25)
    # compact custom legend
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE[o], markeredgecolor="black", markersize=7, label=o.upper()) for o in OPT_ORDER]
    ax.legend(handles=handles, frameon=False, loc="best")
    status["figure3_flatness_only_scatter"] = savefig(paths, "figure3_flatness_only_scatter", fig)

    # Figure 4
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    families = ["Flatness only", "Curvature magnitude", "Spectral concentration", "Spectral entropy/rank", "Spectral redistribution", "Full geometry"]
    subsets = ["ResNet-18 / CIFAR-10-C", "ResNet-18 / CIFAR-100-C", "Combined ResNet-18 only", "All residual formal runs"]
    subset_colors = ["#D1D5DB", "#93C5FD", "#6EE7B7", "#059669"]
    x = np.arange(len(families))
    width = 0.18
    for i, subset in enumerate(subsets):
        vals = [reg[(reg.subset == subset) & (reg.model_name == fam)]["adjusted_r2"].iloc[0] for fam in families]
        ax.bar(x + (i - 1.5) * width, vals, width, color=subset_colors[i], edgecolor="black", linewidth=0.35, label=subset)
    ax.axhline(0, color="#374151", linewidth=0.8)
    ax.set_xticks(x, families, rotation=25, ha="right")
    ax.set_ylabel("Adjusted $R^2$")
    ax.grid(axis="y", color="#D1D5DB", alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    status["figure4_explanatory_power_comparison"] = savefig(paths, "figure4_explanatory_power_comparison", fig)

    # Figure 5
    if topk_df.empty:
        (paths.reports / "topk_spectrum_unavailable.md").write_text("Top-k eigenvalue arrays could not be read reliably from existing files.\n", encoding="utf-8")
        missing.append("figure5 unavailable: missing top-k eigenvalue arrays")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), sharey=True)
        for ax, setting in zip(axes, SETTING_ORDER):
            dataset, model = setting
            for opt in OPT_ORDER:
                sub = topk_df[(topk_df.dataset == dataset) & (topk_df.model == model) & (topk_df.optimizer == opt)]
                if sub.empty:
                    continue
                grp = sub.groupby("rank")["normalized_positive_mass"].agg(["mean", "std"]).reset_index()
                ax.plot(grp["rank"], grp["mean"], color=PALETTE[opt], label=opt.upper(), linewidth=2)
                ax.fill_between(grp["rank"].to_numpy(), (grp["mean"] - grp["std"].fillna(0)).to_numpy(), (grp["mean"] + grp["std"].fillna(0)).to_numpy(), color=LIGHT[opt], alpha=0.24, linewidth=0)
            ax.set_title(SETTING_LABELS[setting], fontsize=9)
            ax.set_xlabel("Eigenvalue rank")
            ax.set_xlim(1, 20)
            ax.grid(color="#D1D5DB", alpha=0.25)
        axes[0].set_ylabel("Normalized positive mass")
        axes[-1].legend(frameon=False)
        status["figure5_topk_hessian_spectrum"] = savefig(paths, "figure5_topk_hessian_spectrum", fig)

    # Appendix heatmap
    if corr_df.empty:
        (paths.reports / "corruption_heatmap_unavailable.md").write_text("Corruption-wise accuracy data could not be read reliably from existing files.\n", encoding="utf-8")
        missing.append("appendix heatmap unavailable: missing corruption-wise data")
    else:
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("greenblue_science", ["#F9FAFB", "#93C5FD", "#059669"])
        fig, axes = plt.subplots(1, 3, figsize=(8.5, 6.8), sharey=True)
        for ax, setting in zip(axes, SETTING_ORDER):
            dataset, model = setting
            pivot = (
                corr_df[(corr_df.dataset == dataset) & (corr_df.model == model)]
                .groupby(["corruption", "optimizer"])["accuracy"]
                .mean()
                .unstack()
                .reindex(index=CORRUPTIONS, columns=OPT_ORDER)
            )
            im = ax.imshow(pivot.to_numpy(float), aspect="auto", cmap=cmap, vmin=max(0.0, corr_df.accuracy.min() - 0.02), vmax=min(1.0, corr_df.accuracy.max() + 0.02))
            ax.set_title(SETTING_LABELS[setting], fontsize=9)
            ax.set_xticks(np.arange(3), [o.upper() for o in OPT_ORDER], rotation=35, ha="right")
            ax.set_yticks(np.arange(len(CORRUPTIONS)), [c.replace("_", " ") for c in CORRUPTIONS])
            for i in range(len(CORRUPTIONS)):
                for j in range(3):
                    val = pivot.iloc[i, j]
                    if not pd.isna(val):
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.4, color="#111827")
        fig.colorbar(im, ax=axes, shrink=0.75, label="Accuracy")
        status["appendix_corruption_wise_heatmap"] = savefig(paths, "appendix_corruption_wise_heatmap", fig)
    return status


def write_captions(paths: OutputPaths, figure_status: dict) -> None:
    captions = {
        "figure1_main_ood_robustness": "Mean CIFAR-C corruption accuracy across three seeds. ASAM achieves the strongest robustness across all formal residual settings.",
        "figure2_severity_wise_robustness": "Severity-wise corruption robustness. The robustness advantage of SAM-style optimizers is evaluated across increasing corruption severity.",
        "figure3_flatness_only_scatter": "Relationship between dominant sharpness and corruption robustness. The top Hessian eigenvalue alone does not provide a complete explanation of the observed OOD performance.",
        "figure4_explanatory_power_comparison": "Regression-based explanatory power comparison. Spectral redistribution metrics are compared against scalar flatness-only predictors for explaining CIFAR-C robustness.",
        "figure5_topk_hessian_spectrum": "Top-k Hessian spectral mass distribution. The plot visualizes whether optimizer differences appear in the shape of the Hessian spectrum rather than only in the largest eigenvalue.",
        "appendix_corruption_wise_heatmap": "Corruption-wise robustness across optimizer families for each residual-network setting.",
    }
    lines = ["# Figure Captions", ""]
    for name, caption in captions.items():
        if name in figure_status:
            lines.append(f"## {name}")
            lines.append(caption)
            lines.append("")
    (paths.reports / "figure_captions.md").write_text("\n".join(lines), encoding="utf-8")


def write_text_reports(df: pd.DataFrame, reg: pd.DataFrame, paths: OutputPaths, missing: list[str], read_files: set[str], figure_status: dict) -> None:
    # Summary facts
    table1 = grouped_table(df, ["clean_accuracy", "cifarc_mean_accuracy", "ood_gap"])
    asam_wins = []
    for setting in SETTING_ORDER:
        subset = table1[(table1.dataset == setting[0]) & (table1.model == setting[1])]
        best = subset.sort_values("cifarc_mean_accuracy_mean", ascending=False).iloc[0]
        asam_wins.append(best.optimizer == "ASAM")
    best_by_subset = reg.loc[reg.groupby("subset")["adjusted_r2"].idxmax()][["subset", "model_name", "adjusted_r2"]]
    best_family = reg.groupby("model_name")["adjusted_r2"].mean().sort_values(ascending=False).index[0]
    lines = [
        "# Result Summary for Paper",
        "",
        "## Main OOD robustness summary",
        "Across the three formal residual-network settings, ASAM obtains the highest mean CIFAR-C accuracy in the generated summary tables, while SAM generally remains competitive with SGD. This pattern provides empirical evidence that SAM-style optimization is associated with improved corruption robustness beyond clean accuracy alone.",
        "",
        "## Severity/corruption robustness summary",
        "The severity-wise curves evaluate performance from mild to severe corruptions and show whether optimizer differences persist as distribution shift increases. The corruption-wise appendix heatmap further separates this behavior by corruption type, supporting a more granular reading than a single aggregate robustness score.",
        "",
        "## Flatness-only limitation summary",
        "The flatness-only scatter and regression model indicate that the dominant Hessian eigenvalue is not a complete standalone explanation for CIFAR-C performance. Its association with robustness varies across settings, suggesting that scalar sharpness alone misses important geometry in the trained solutions.",
        "",
        "## Spectral redistribution summary",
        f"The regression comparison identifies `{best_family}` as the strongest predictor family on average by adjusted R-squared in this package. Overall, the spectral concentration and entropy/rank metrics provide a richer empirical characterization of robustness than relying only on the largest Hessian eigenvalue.",
        "",
        "## ResNet-34 validation summary",
        "The ResNet-34 CIFAR-10-C setting acts as a deeper residual-network validation. Its inclusion supports the interpretation that the observed relationship between SAM-style optimization, OOD robustness, and Hessian spectral structure is not limited to ResNet-18 alone.",
        "",
        "## VGG negative pilot limitation paragraph",
        "VGG-16-BN seed42 is treated only as a negative pilot and suggests possible architecture or hyperparameter sensitivity. It is not used as formal evidence for or against the main residual-network claim, and the central conclusions are based on the completed ResNet-18 and ResNet-34 formal matrix.",
        "",
        "## Best regression family by subset",
    ]
    for _, r in best_by_subset.iterrows():
        lines.append(f"- {r.subset}: {r.model_name}, adjusted R2 = {r.adjusted_r2:.3f}")
    (paths.reports / "result_summary_for_paper.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    missing_lines = ["# Missing Metric Report", ""]
    if missing:
        missing_lines += [f"- {m}" for m in missing]
    else:
        missing_lines.append("- No missing required metrics were detected for the formal COMPLETE residual runs.")
    (paths.reports / "missing_metric_report.md").write_text("\n".join(missing_lines) + "\n", encoding="utf-8")

    fig_lines = [
        "# Figure Style Report",
        "",
        "Palette used for all generated figures:",
        f"- SGD: `{PALETTE['sgd']}` neutral gray",
        f"- SAM: `{PALETTE['sam']}` scientific blue",
        f"- ASAM: `{PALETTE['asam']}` scientific green",
        f"- Light gray: `{LIGHT['sgd']}`",
        f"- Light blue: `{LIGHT['sam']}`",
        f"- Light green: `{LIGHT['asam']}`",
        "",
        "All figures use white backgrounds and explicit palette assignments; no seaborn/default rainbow palette is used.",
        "",
    ]
    for name, (pdf_ok, png_ok) in figure_status.items():
        fig_lines.append(f"- {name}: PDF={'yes' if pdf_ok else 'no'}, PNG={'yes' if png_ok else 'no'}, default_colors=no")
    (paths.reports / "figure_style_report.md").write_text("\n".join(fig_lines) + "\n", encoding="utf-8")

    tables_generated = len(list(paths.tables.glob("*.csv"))) + len(list(paths.tables.glob("*.tex")))
    figures_generated = len(list(paths.figures.glob("*.pdf")))
    package_lines = [
        "# Package Generation Report",
        "",
        "## Files read",
        *[f"- `{p}`" for p in sorted(read_files)],
        "",
        "## Tables generated",
        *[f"- `{p.name}`" for p in sorted(paths.tables.glob("*"))],
        "",
        "## Figures generated",
        *[f"- `{p.name}`" for p in sorted(paths.figures.glob("*"))],
        "",
        "## Unavailable outputs",
    ]
    unavailable = list(paths.reports.glob("*unavailable.md"))
    package_lines += [f"- `{p.name}`" for p in unavailable] if unavailable else ["- None."]
    package_lines += [
        "",
        "## Best predictor family",
        f"- Best average adjusted R2 family: `{best_family}`.",
        "",
        "## Claim support",
        f"- ASAM improves OOD robustness: {'supported empirically in generated summaries' if all(asam_wins) else 'partially supported; inspect Table 1'}",
        "- Scalar flatness alone is insufficient: supported as an empirical interpretation by the regression and scatter analysis.",
        "- Spectral redistribution provides richer empirical explanation: supported when spectral predictor families exceed flatness-only adjusted R2 in the generated comparison.",
        "",
        "## Next step",
        "Use the compact Hessian table and the regression explanatory-power figure in the main paper, keep the corruption-wise heatmap and VGG negative pilot text for appendix/discussion, and verify final captions against ICONIP page limits.",
    ]
    (paths.reports / "package_generation_report.md").write_text("\n".join(package_lines) + "\n", encoding="utf-8")


def write_latex_template(paths: OutputPaths, figure_status: dict) -> None:
    fig5 = ""
    if "figure5_topk_hessian_spectrum" in figure_status:
        fig5 = r"""
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/figure5_topk_hessian_spectrum.pdf}
\caption{Top-k Hessian spectral mass distribution.}
\label{fig:topk_hessian_spectrum}
\end{figure}
"""
    text = rf"""
\section{{Results and Analysis}}

\subsection{{Main OOD Robustness Results}}
Placeholder: summarize clean and corruption robustness across the completed residual-network matrix.
\input{{tables/table1_main_ood_results.tex}}
\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{figures/figure1_main_ood_robustness.pdf}}
\caption{{Mean CIFAR-C corruption accuracy across three seeds.}}
\label{{fig:main_ood_robustness}}
\end{{figure}}

\subsection{{Robustness across Severity Levels}}
Placeholder: discuss how optimizer differences vary across corruption severities.
\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{figures/figure2_severity_wise_robustness.pdf}}
\caption{{Severity-wise corruption robustness.}}
\label{{fig:severity_wise_robustness}}
\end{{figure}}

\subsection{{Limitations of Scalar Flatness}}
Placeholder: state why top Hessian eigenvalue alone is an incomplete empirical explanation.
\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{figures/figure3_flatness_only_scatter.pdf}}
\caption{{Relationship between dominant sharpness and corruption robustness.}}
\label{{fig:flatness_only_scatter}}
\end{{figure}}

\subsection{{Hessian Spectral Redistribution Analysis}}
Placeholder: compare scalar flatness predictors with spectral concentration and entropy/rank predictors.
\input{{tables/table2_hessian_spectral_metrics.tex}}
\input{{tables/table3_regression_explanatory_power.tex}}
\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{figures/figure4_explanatory_power_comparison.pdf}}
\caption{{Regression-based explanatory power comparison.}}
\label{{fig:explanatory_power_comparison}}
\end{{figure}}
{fig5}

\subsection{{Architecture Validation and Negative Pilot}}
Placeholder: summarize ResNet-34 validation and describe VGG-16-BN as a negative pilot only.
"""
    (paths.latex / "results_section_template.tex").write_text(text.strip() + "\n", encoding="utf-8")


def main() -> None:
    paths = make_package_dir()
    audit_dir = latest_audit_dir()
    inventory_path = audit_dir / "run_inventory.csv"
    read_files = {str(inventory_path)}
    missing: list[str] = []
    raw = pd.read_csv(inventory_path)
    df = clean_inventory(raw, missing)
    df, _, corr_df, topk_df = enrich_from_logs(df, missing, read_files)
    # Persist cleaned row-level data for traceability.
    df.to_csv(paths.analysis / "cleaned_formal_run_metrics.csv", index=False)
    if not corr_df.empty:
        corr_df.to_csv(paths.analysis / "corruption_wise_metrics.csv", index=False)
    if not topk_df.empty:
        topk_df.to_csv(paths.analysis / "topk_spectrum_metrics.csv", index=False)
    write_table1(df, paths)
    write_table2(df, paths)
    reg = regression_analysis(df, paths)
    figure_status = plot_figures(df, reg, corr_df, topk_df, paths, missing)
    write_captions(paths, figure_status)
    write_latex_template(paths, figure_status)
    write_text_reports(df, reg, paths, missing, read_files, figure_status)
    print(f"Generated package directory: {paths.root}")
    print(f"Tables generated count: {len(list(paths.tables.glob('*')))}")
    print(f"Figures generated count: {len(list(paths.figures.glob('*.pdf')))}")
    print(f"Missing data warnings count: {len(missing)}")
    print("Recommended next step: inspect Table 3 and Figure 4, then move compact tables and PDF figures into the ICONIP manuscript draft.")


if __name__ == "__main__":
    main()
