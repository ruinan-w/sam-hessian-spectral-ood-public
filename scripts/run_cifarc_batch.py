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


def parse_args():
    p = argparse.ArgumentParser(description="Batch CIFAR-C evaluation.")
    p.add_argument("--summary-csv", default="outputs/results/all_runs_summary.csv")
    p.add_argument("--datasets", default="cifar10")
    p.add_argument("--tag-filter", default="ei100")
    p.add_argument("--severities", default="1,2,3,4,5")
    p.add_argument("--batch-size", type=int, default=256)
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
            rows.append({
                "run_name": run_name,
                "dataset": dataset,
                "model": cfg.get("model", infer_model(run_name)),
                "optimizer": cfg.get("optimizer", ""),
                "seed": cfg.get("seed", ""),
                "checkpoint_path": checkpoint,
            })
    return pd.DataFrame(rows)


def select_runs(summary_csv, datasets, tag_filter):
    path = PROJECT_ROOT / summary_csv
    if path.exists() and path.stat().st_size > 0:
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame()
    if not df.empty and "run_name" in df.columns:
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
        if not selected.empty:
            return selected
    return rows_from_configs(datasets, tag_filter)


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
    master = logs / f"cifarc_batch_master_log_{ts}.txt"
    csv_path = logs / f"cifarc_batch_summary_{ts}.csv"
    json_path = logs / f"cifarc_batch_summary_{ts}.json"
    datasets = set(split_csv(args.datasets))
    df = select_runs(args.summary_csv, datasets, args.tag_filter)
    rows = []
    start = time.time()
    if df.empty:
        with master.open("a", encoding="utf-8") as f:
            f.write(
                f"No CIFAR-C runs selected. summary_csv={args.summary_csv} "
                f"datasets={sorted(datasets)} tag_filter={args.tag_filter}\n"
            )
    for _, row in df.iterrows():
        if stop_now(start, args.stop_after_hours):
            break
        ckpt = str(row.get("checkpoint_path", ""))
        st = iso()
        wall = time.time()
        model = str(row.get("model", infer_model(row["run_name"])))
        cmd = ["scripts/eval_cifar_c.py", "--dataset", str(row.get("dataset", "cifar10")).lower(), "--checkpoint", ckpt, "--run-name", str(row["run_name"]), "--optimizer-name", str(row.get("optimizer", "")), "--seed", seed_arg(row.get("seed", 0)), "--model", model, "--severities", args.severities, "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers), "--device", args.device]
        code, _stdout, kv = run_child(cmd, master)
        status = kv.get("STATUS", "failed" if code else "success")
        error_message = ""
        summary_path = kv.get("CIFARC_SUMMARY_PATH", "")
        if summary_path and Path(summary_path).exists():
            try:
                error_message = json.loads(Path(summary_path).read_text(encoding="utf-8")).get("error_message", "")
            except Exception:
                error_message = ""
        rows.append({"run_name": kv.get("RUN_NAME", row["run_name"]), "dataset": row.get("dataset", "cifar10"), "model": model, "optimizer": row.get("optimizer", ""), "seed": row.get("seed", ""), "checkpoint_path": ckpt, "status": status, "cifarc_csv_path": kv.get("CIFARC_CSV_PATH", ""), "cifarc_summary_path": summary_path, "mean_accuracy_all": kv.get("MEAN_CIFARC_ACC", ""), "start_time": st, "end_time": iso(), "duration_minutes": (time.time() - wall) / 60.0, "error_message": error_message})
        if status != "success" and not args.continue_on_error:
            break
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run_name"])
        writer.writeheader()
        writer.writerows(rows)
    write_json(json_path, rows)
    print(f"CIFARC_BATCH_SUMMARY_CSV={csv_path.resolve()}")
    print("STATUS=success")


if __name__ == "__main__":
    main()
