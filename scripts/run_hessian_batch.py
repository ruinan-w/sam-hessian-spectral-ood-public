import argparse
import csv
import json
import time
from pathlib import Path

import pandas as pd

from _runner_utils import PROJECT_ROOT, iso, run_child, split_csv, stop_now, timestamp, write_json


def infer_model(value):
    text = str(value).lower()
    if "vgg16_bn" in text:
        return "vgg16_bn"
    return "resnet34" if "resnet34" in text else "resnet18"


DEFAULT_SUMMARY_CSV = "outputs/results/all_runs_summary.csv"
SUMMARY_FIELDS = [
    "run_name",
    "dataset",
    "model",
    "optimizer",
    "seed",
    "checkpoint_path",
    "status",
    "hessian_json_path",
    "eigen_proxy_csv_path",
    "topk_eigen_csv_path",
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
    "start_time",
    "end_time",
    "duration_minutes",
    "error_message",
]


def parse_args():
    p = argparse.ArgumentParser(description="Batch Hessian geometry computation.")
    p.add_argument("--summary-csv", default=DEFAULT_SUMMARY_CSV)
    p.add_argument("--datasets", default="cifar10")
    p.add_argument("--tag-filter", default="ei100")
    p.add_argument("--subset-size", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--power-iters", type=int, default=20)
    p.add_argument("--trace-samples", type=int, default=20)
    p.add_argument("--pr-probes", type=int, default=20)
    p.add_argument("--max-batches", type=int, default=4)
    p.add_argument("--use-lanczos", action="store_true")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--lanczos-steps", type=int, default=30)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--stop-after-hours", type=float, default=None)
    p.add_argument("--continue-on-error", type=lambda x: str(x).lower() != "false", default=True)
    return p.parse_args()


def rows_from_configs(datasets, tag_filter):
    rows = []
    for path in (PROJECT_ROOT / "outputs" / "logs").glob("*_config.json"):
        try:
            cfg = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        run_name = str(cfg.get("run_name", path.name[: -len("_config.json")]))
        dataset = str(cfg.get("dataset", "cifar10")).lower()
        checkpoint = str(cfg.get("checkpoint_path", ""))
        if tag_filter and tag_filter not in run_name:
            continue
        if dataset not in datasets:
            continue
        if checkpoint and Path(checkpoint).exists():
            rows.append(
                {
                    "run_name": run_name,
                    "dataset": dataset,
                    "model": cfg.get("model", infer_model(run_name)),
                    "optimizer": cfg.get("optimizer", cfg.get("optimizer_name", "")),
                    "seed": cfg.get("seed", ""),
                    "checkpoint_path": checkpoint,
                }
            )
    return pd.DataFrame(rows)


def filter_runs(df, datasets, tag_filter):
    if df.empty or "run_name" not in df.columns:
        return pd.DataFrame()
    selected = df.copy()
    if tag_filter:
        selected = selected[selected["run_name"].astype(str).str.contains(tag_filter, na=False)]
    if "dataset" not in selected.columns:
        selected["dataset"] = "cifar10"
    if "checkpoint_path" not in selected.columns:
        selected["checkpoint_path"] = ""
    if "model" not in selected.columns:
        selected["model"] = selected["run_name"].astype(str).map(infer_model)
    selected = selected[selected["dataset"].astype(str).str.lower().isin(datasets)]
    selected = selected[selected["checkpoint_path"].fillna("").astype(str).map(lambda x: bool(x) and Path(x).exists())]
    return selected


def read_csv_or_empty(path):
    try:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def select_runs(summary_csv, datasets, tag_filter):
    requested = PROJECT_ROOT / summary_csv
    selected = filter_runs(read_csv_or_empty(requested), datasets, tag_filter)
    if not selected.empty:
        print(f"USING_SUMMARY_CSV={requested.resolve()}")
        return selected

    if summary_csv == DEFAULT_SUMMARY_CSV:
        results = PROJECT_ROOT / "outputs" / "results"
        candidates = sorted(results.glob("all_runs_summary*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        for candidate in candidates:
            if candidate.resolve() == requested.resolve():
                continue
            selected = filter_runs(read_csv_or_empty(candidate), datasets, tag_filter)
            if not selected.empty:
                print(f"WARNING: no eligible runs in default summary: {requested.resolve()}")
                print(f"USING_FALLBACK_SUMMARY_CSV={candidate.resolve()}")
                return selected

    selected = rows_from_configs(datasets, tag_filter)
    if not selected.empty:
        print("WARNING: no eligible runs found in summary CSV files; using training config files.")
    return selected


def seed_arg(value):
    try:
        return str(int(float(value)))
    except Exception:
        return str(value)


def main():
    args = parse_args()
    logs = PROJECT_ROOT / "outputs" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    ts = timestamp()
    master = logs / f"hessian_batch_master_log_{ts}.txt"
    csv_path = logs / f"hessian_batch_summary_{ts}.csv"
    json_path = logs / f"hessian_batch_summary_{ts}.json"
    datasets = set(split_csv(args.datasets))
    df = select_runs(args.summary_csv, datasets, args.tag_filter)
    rows = []
    start = time.time()
    if df.empty:
        print(f"WARNING: no Hessian runs selected. datasets={sorted(datasets)} tag_filter={args.tag_filter}")
    for _, row in df.iterrows():
        if stop_now(start, args.stop_after_hours):
            break
        if args.tag_filter and args.tag_filter not in str(row.get("run_name", "")):
            continue
        dataset = str(row.get("dataset", "cifar10")).lower()
        if dataset not in datasets:
            continue
        ckpt = str(row.get("checkpoint_path", ""))
        if not ckpt or not Path(ckpt).exists():
            continue
        st = iso()
        wall = time.time()
        model = str(row.get("model", infer_model(row["run_name"])))
        cmd = ["scripts/compute_hessian_geometry.py", "--dataset", dataset, "--checkpoint", ckpt, "--run-name", str(row["run_name"]), "--optimizer-name", str(row.get("optimizer", row.get("optimizer_name", ""))), "--seed", seed_arg(row.get("seed", 0)), "--model", model, "--subset-size", str(args.subset_size), "--batch-size", str(args.batch_size), "--power-iters", str(args.power_iters), "--trace-samples", str(args.trace_samples), "--pr-probes", str(args.pr_probes), "--max-batches", str(args.max_batches), "--top-k", str(args.top_k), "--lanczos-steps", str(args.lanczos_steps), "--num-workers", str(args.num_workers), "--device", args.device]
        if args.use_lanczos:
            cmd.append("--use-lanczos")
        code, _stdout, kv = run_child(cmd, master)
        status = kv.get("STATUS", "failed" if code else "success")
        hjson = kv.get("HESSIAN_JSON_PATH", "")
        metrics = {}
        if hjson and Path(hjson).exists():
            metrics = json.loads(Path(hjson).read_text(encoding="utf-8"))
        rows.append({"run_name": kv.get("RUN_NAME", row["run_name"]), "dataset": dataset, "model": model, "optimizer": row.get("optimizer", row.get("optimizer_name", "")), "seed": row.get("seed", ""), "checkpoint_path": ckpt, "status": status, "hessian_json_path": hjson, "eigen_proxy_csv_path": kv.get("EIGEN_PROXY_CSV_PATH", ""), "topk_eigen_csv_path": kv.get("TOPK_EIGEN_CSV_PATH", ""), "top_eigenvalue": metrics.get("top_eigenvalue", ""), "trace_estimate": metrics.get("trace_estimate", ""), "participation_ratio_approx": metrics.get("participation_ratio_approx", ""), "lambda_max_over_trace": metrics.get("lambda_max_over_trace", ""), "lambda_max_topk": metrics.get("lambda_max_topk", ""), "top_k_sum": metrics.get("top_k_sum", ""), "num_positive_topk_eigenvalues": metrics.get("num_positive_topk_eigenvalues", ""), "top_1_mass_ratio": metrics.get("top_1_mass_ratio", ""), "top_5_mass_ratio": metrics.get("top_5_mass_ratio", ""), "top_10_mass_ratio": metrics.get("top_10_mass_ratio", ""), "participation_ratio_topk": metrics.get("participation_ratio_topk", ""), "effective_rank_entropy": metrics.get("effective_rank_entropy", ""), "spectral_entropy": metrics.get("spectral_entropy", ""), "lambda_max_over_topk_sum": metrics.get("lambda_max_over_topk_sum", ""), "top_k_sum_over_trace": metrics.get("top_k_sum_over_trace", ""), "start_time": st, "end_time": iso(), "duration_minutes": (time.time() - wall) / 60.0, "error_message": metrics.get("error_message", "")})
        if status != "success" and not args.continue_on_error:
            break
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_json(json_path, rows)
    print(f"HESSIAN_BATCH_SUMMARY_CSV={csv_path.resolve()}")
    if not rows:
        print("STATUS=no_runs_selected")
        return 1
    print("STATUS=success")


if __name__ == "__main__":
    main()
