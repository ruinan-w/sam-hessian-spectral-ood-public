import argparse
import csv
import json
import time
from pathlib import Path

from _runner_utils import PROJECT_ROOT, iso, run_child, split_csv, stop_now, timestamp, write_json


def parse_args():
    p = argparse.ArgumentParser(description="Run a batch of training jobs.")
    p.add_argument("--datasets", default="cifar10")
    p.add_argument("--models", default="resnet18")
    p.add_argument("--optimizers", default="sgd,sam,asam")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--rho-sam", type=float, default=0.05)
    p.add_argument("--rho-asam", type=float, default=0.5)
    p.add_argument("--asam-eta", type=float, default=0.01)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--scheduler", default="cosine")
    p.add_argument("--stop-after-hours", type=float, default=None)
    p.add_argument("--continue-on-error", type=lambda x: str(x).lower() != "false", default=True)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--tag", default="ei100")
    return p.parse_args()


def main():
    args = parse_args()
    logs = PROJECT_ROOT / args.output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    ts = timestamp()
    master = logs / f"training_batch_master_log_{ts}.txt"
    csv_path = logs / f"training_batch_summary_{ts}.csv"
    json_path = logs / f"training_batch_summary_{ts}.json"
    rows = []
    start = time.time()
    for dataset in split_csv(args.datasets):
        for model in split_csv(args.models):
            for opt in split_csv(args.optimizers):
                for seed in split_csv(args.seeds, int):
                    if stop_now(start, args.stop_after_hours):
                        break
                    run_name = f"{args.tag}_{dataset}_{model}_{opt}_seed{seed}_ep{args.epochs}"
                    cmd = ["scripts/train_experiment.py", "--dataset", dataset, "--model", model, "--optimizer", opt, "--seed", str(seed), "--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--lr", str(args.lr), "--num-workers", str(args.num_workers), "--device", args.device, "--scheduler", args.scheduler, "--output-dir", args.output_dir, "--run-name", run_name]
                    if opt == "sam":
                        cmd += ["--rho", str(args.rho_sam)]
                    if opt == "asam":
                        cmd += ["--rho", str(args.rho_asam), "--asam-eta", str(args.asam_eta)]
                    st = iso()
                    wall = time.time()
                    code, _stdout, kv = run_child(cmd, master)
                    status = kv.get("STATUS", "failed" if code else "success")
                    config_path = kv.get("CONFIG_PATH", "")
                    final_test_acc = best_test_acc = error = ""
                    if config_path and Path(config_path).exists():
                        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
                        final_test_acc = cfg.get("final_test_acc", "")
                        best_test_acc = cfg.get("best_test_acc", "")
                        error = cfg.get("error_message", "")
                    rows.append({"run_name": kv.get("RUN_NAME", run_name), "dataset": dataset, "model": model, "optimizer": opt, "seed": seed, "epochs": args.epochs, "status": status, "checkpoint_path": kv.get("CHECKPOINT_PATH", ""), "train_log_path": kv.get("TRAIN_LOG_PATH", ""), "config_path": config_path, "final_test_acc": final_test_acc, "best_test_acc": best_test_acc, "start_time": st, "end_time": iso(), "duration_minutes": (time.time() - wall) / 60.0, "error_message": error})
                    if status != "success" and not args.continue_on_error:
                        break
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run_name"])
        writer.writeheader()
        writer.writerows(rows)
    write_json(json_path, rows)
    print(f"TRAINING_BATCH_SUMMARY_CSV={csv_path.resolve()}")
    print(f"STATUS=success")


if __name__ == "__main__":
    main()
