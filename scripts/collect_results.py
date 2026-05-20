import argparse
import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def strip_v(run):
    return re.sub(r"_v\d+$", "", str(run))


def infer_model(value):
    text = str(value).lower()
    if "vgg16_bn" in text:
        return "vgg16_bn"
    return "resnet34" if "resnet34" in text else "resnet18"


def parse_args():
    p = argparse.ArgumentParser(description="Collect training, CIFAR-C, Hessian, and top-k results.")
    p.add_argument("--tag-filter", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--output-dir", default="outputs")
    return p.parse_args()


def maybe_filter(df, tag, dataset):
    if df.empty:
        return df
    out = df
    if tag and "run_name" in out.columns:
        out = out[out["run_name"].astype(str).str.contains(tag, na=False)]
    if dataset and "dataset" in out.columns:
        out = out[out["dataset"].astype(str).str.lower() == dataset.lower()]
    return out


def unique_path(path):
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    version = 2
    while True:
        candidate = path.with_name(f"{stem}_v{version}{suffix}")
        if not candidate.exists():
            return candidate
        version += 1


def main():
    args = parse_args()
    root = PROJECT_ROOT / args.output_dir
    logs = root / "logs"
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)

    configs = []
    for path in logs.glob("*_config.json"):
        item = read_json(path)
        item.setdefault("run_name", path.name[: -len("_config.json")])
        item["config_path"] = str(path.resolve())
        configs.append(item)
    config_df = pd.DataFrame(configs)

    train_frames = []
    for path in logs.glob("*_train.csv"):
        df = pd.read_csv(path)
        run = path.name[: -len("_train.csv")]
        if "run_name" not in df.columns:
            df.insert(0, "run_name", run)
        train_frames.append(df)
    training = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame()

    detail_frames = []
    for pattern in ["*_cifarc_details.csv", "*_cifarc_severity*.csv", "*_cifar10c_severity*.csv"]:
        for path in logs.glob(pattern):
            df = pd.read_csv(path)
            if "run_name" not in df.columns:
                run = re.sub(r"_(cifarc|cifar10c)_severity.*$", "", path.stem)
                df.insert(0, "run_name", run)
            if "dataset" not in df.columns:
                df.insert(1, "dataset", "cifar10")
            if "model" not in df.columns:
                df["model"] = df["run_name"].map(infer_model)
            detail_frames.append(df)
    cifarc_details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()

    cifarc_summaries = []
    for pattern in ["*_cifarc_summary.json", "*_cifar10c_summary.json"]:
        for path in logs.glob(pattern):
            item = read_json(path)
            item.setdefault("run_name", re.sub(r"_(cifarc|cifar10c)_summary$", "", path.stem))
            item.setdefault("dataset", "cifar10")
            item.setdefault("model", infer_model(item["run_name"]))
            item["cifarc_summary_path"] = str(path.resolve())
            item["cifarc_csv_path"] = item.get("result_csv_path", "")
            item["cifarc_mean_acc"] = item.get("mean_accuracy_all", item.get("mean_accuracy", ""))
            cifarc_summaries.append(item)
    cifarc_summary = pd.DataFrame(cifarc_summaries)

    geometry = []
    for path in logs.glob("*_hessian_geometry.json"):
        item = read_json(path)
        item.setdefault("run_name", path.name[: -len("_hessian_geometry.json")])
        item.setdefault("dataset", "cifar10")
        item.setdefault("model", infer_model(item["run_name"]))
        item["base_run_name"] = strip_v(item["run_name"])
        item["hessian_json_path"] = str(path.resolve())
        item["modified_time"] = path.stat().st_mtime
        geometry.append(item)
    geometry_df = pd.DataFrame(geometry)

    topk_frames = []
    for path in logs.glob("*_hessian_topk_eigenvalues.csv"):
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if "run_name" not in df.columns:
            df.insert(0, "run_name", path.name[: -len("_hessian_topk_eigenvalues.csv")])
        if "rank" in df.columns and "eigen_index" not in df.columns:
            df["eigen_index"] = df["rank"]
        if "eigenvalue" in df.columns and "raw_eigenvalue" not in df.columns:
            df["raw_eigenvalue"] = df["eigenvalue"]
        if "positive_used_for_metrics" not in df.columns and "raw_eigenvalue" in df.columns:
            df["positive_used_for_metrics"] = pd.to_numeric(df["raw_eigenvalue"], errors="coerce") > 1e-8
        topk_frames.append(df)
    topk = pd.concat(topk_frames, ignore_index=True) if topk_frames else pd.DataFrame()

    run_rows = []
    for _, cfg in config_df.iterrows() if not config_df.empty else []:
        run_rows.append({
            "run_name": cfg.get("run_name", ""), "dataset": str(cfg.get("dataset", "cifar10")).lower(),
            "model": cfg.get("model", infer_model(cfg.get("run_name", ""))), "optimizer": cfg.get("optimizer", ""),
            "seed": cfg.get("seed", ""), "epochs": cfg.get("epochs", ""), "final_train_acc": cfg.get("final_train_acc", ""),
            "final_test_acc": cfg.get("final_test_acc", ""), "best_test_acc": cfg.get("best_test_acc", ""),
            "checkpoint_path": cfg.get("checkpoint_path", ""), "train_log_path": cfg.get("train_log_path", ""),
            "config_path": cfg.get("config_path", ""),
        })
    runs = pd.DataFrame(run_rows)
    if not runs.empty:
        runs["model"] = runs.apply(lambda row: row.get("model") if pd.notna(row.get("model")) and str(row.get("model")) else infer_model(row.get("run_name", "")), axis=1)

    if not runs.empty and not cifarc_summary.empty:
        runs = runs.merge(cifarc_summary[["run_name", "cifarc_mean_acc", "cifarc_csv_path", "cifarc_summary_path"]], on="run_name", how="left")
    else:
        for col in ["cifarc_mean_acc", "cifarc_csv_path", "cifarc_summary_path"]:
            runs[col] = "" if not runs.empty else []
    if not runs.empty:
        runs["ood_gap"] = pd.to_numeric(runs["final_test_acc"], errors="coerce") - pd.to_numeric(runs["cifarc_mean_acc"], errors="coerce")

    geom_cols = [
        "base_run_name",
        "top_eigenvalue",
        "trace_estimate",
        "participation_ratio_approx",
        "lambda_max_over_trace",
        "lambda_max_topk",
        "top_k_sum",
        "num_positive_topk_eigenvalues",
        "top_1_mass_ratio",
        "top_5_mass_ratio",
        "top_10_mass_ratio",
        "participation_ratio_topk",
        "effective_rank_entropy",
        "spectral_entropy",
        "lambda_max_over_topk_sum",
        "top_k_sum_over_trace",
        "hessian_json_path",
        "topk_eigen_csv_path",
    ]
    if not geometry_df.empty:
        valid_geometry = geometry_df.copy()
        for col in ["status", "top_5_mass_ratio", "effective_rank_entropy", "lambda_max_over_topk_sum"]:
            if col not in valid_geometry.columns:
                valid_geometry[col] = pd.NA
        valid_mask = (
            valid_geometry["status"].astype(str).eq("success")
            & valid_geometry["top_5_mass_ratio"].notna()
            & valid_geometry["effective_rank_entropy"].notna()
            & valid_geometry["lambda_max_over_topk_sum"].notna()
            & valid_geometry["top_5_mass_ratio"].astype(str).ne("")
            & valid_geometry["effective_rank_entropy"].astype(str).ne("")
            & valid_geometry["lambda_max_over_topk_sum"].astype(str).ne("")
        )
        geometry_source = valid_geometry[valid_mask].copy()
        if geometry_source.empty:
            geometry_source = geometry_df.copy()
        latest_geom = geometry_source.sort_values("modified_time").drop_duplicates("base_run_name", keep="last").copy()
        if "topk_eigen_csv_path" not in latest_geom.columns:
            latest_geom["topk_eigen_csv_path"] = ""
        final = runs.merge(latest_geom[[c for c in geom_cols if c in latest_geom.columns]], left_on="run_name", right_on="base_run_name", how="left").drop(columns=["base_run_name"], errors="ignore") if not runs.empty else pd.DataFrame()
    else:
        final = runs.copy()
        for col in geom_cols[1:]:
            final[col] = "" if not final.empty else []

    final_cols = [
        "run_name",
        "dataset",
        "model",
        "optimizer",
        "seed",
        "epochs",
        "final_train_acc",
        "final_test_acc",
        "best_test_acc",
        "cifarc_mean_acc",
        "ood_gap",
        "top_eigenvalue",
        "trace_estimate",
        "participation_ratio_approx",
        "lambda_max_over_trace",
        "top_k_sum",
        "lambda_max_topk",
        "num_positive_topk_eigenvalues",
        "top_1_mass_ratio",
        "top_5_mass_ratio",
        "top_10_mass_ratio",
        "participation_ratio_topk",
        "effective_rank_entropy",
        "spectral_entropy",
        "lambda_max_over_topk_sum",
        "top_k_sum_over_trace",
        "checkpoint_path",
        "train_log_path",
        "cifarc_csv_path",
        "hessian_json_path",
        "topk_eigen_csv_path",
    ]
    for name, df in [
        ("all_runs_summary.csv", runs),
        ("all_training_curves.csv", training),
        ("all_cifarc_details.csv", cifarc_details),
        ("all_cifarc_summary.csv", cifarc_summary),
        ("all_geometry_summary.csv", geometry_df.drop(columns=["modified_time", "base_run_name"], errors="ignore")),
        ("all_topk_eigenvalues.csv", topk),
        ("final_analysis_table.csv", final.reindex(columns=final_cols) if not final.empty else pd.DataFrame(columns=final_cols)),
    ]:
        output_path = unique_path(results / name)
        maybe_filter(df, args.tag_filter, args.dataset).to_csv(output_path, index=False)
        print(f"Saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
