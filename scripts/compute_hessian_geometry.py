import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.hessian_geometry import compute_geometry_for_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="Compute Hessian geometry metrics for one checkpoint.")
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--optimizer-name", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--model", choices=["resnet18", "resnet34", "vgg16_bn"], default=None)
    p.add_argument("--data-root", default="data")
    p.add_argument("--subset-size", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--power-iters", type=int, default=20)
    p.add_argument("--trace-samples", type=int, default=20)
    p.add_argument("--pr-probes", type=int, default=20)
    p.add_argument("--max-batches", type=int, default=4)
    p.add_argument("--use-lanczos", action="store_true")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--lanczos-steps", type=int, default=30)
    p.add_argument("--lanczos-damping", type=float, default=0.0)
    p.add_argument("--no-reorthogonalize", action="store_true")
    p.add_argument("--output-dir", default="outputs")
    return p.parse_args()


def iso():
    return datetime.now().isoformat(timespec="seconds")


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def clean_json(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    return value


def reserve(logs_dir, base, use_lanczos):
    version = 1
    while True:
        run = base if version == 1 else f"{base}_v{version}"
        paths = [
            logs_dir / f"{run}_hessian_geometry.json",
            logs_dir / f"{run}_hessian_eigen_proxy.csv",
        ]
        if use_lanczos:
            paths.append(logs_dir / f"{run}_hessian_topk_eigenvalues.csv")
        if not any(path.exists() for path in paths):
            return run, paths[0], paths[1], (paths[2] if use_lanczos else None)
        version += 1


def write_proxy(path, run, optimizer, seed, values):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run_name", "optimizer_name", "seed", "probe_index", "eigen_proxy_value"])
        writer.writeheader()
        for i, value in enumerate(values):
            writer.writerow({"run_name": run, "optimizer_name": optimizer, "seed": seed, "probe_index": i, "eigen_proxy_value": value})


def write_topk(path, run, dataset, optimizer, seed, values):
    if path is None:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "dataset",
                "optimizer_name",
                "seed",
                "eigen_index",
                "raw_eigenvalue",
                "positive_used_for_metrics",
            ],
        )
        writer.writeheader()
        for i, value in enumerate(values, start=1):
            writer.writerow(
                {
                    "run_name": run,
                    "dataset": dataset,
                    "optimizer_name": optimizer,
                    "seed": seed,
                    "eigen_index": i,
                    "raw_eigenvalue": value,
                    "positive_used_for_metrics": bool(value > 1e-8),
                }
            )


def main():
    args = parse_args()
    logs = PROJECT_ROOT / args.output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    run, json_path, proxy_path, topk_path = reserve(logs, args.run_name, args.use_lanczos)
    wall = time.time()
    result = {
        "run_name": run, "dataset": args.dataset, "model": args.model, "optimizer_name": args.optimizer_name, "seed": args.seed,
        "checkpoint_path": str(resolve(args.checkpoint).resolve()), "subset_size": args.subset_size,
        "batch_size": args.batch_size, "max_batches": args.max_batches, "power_iters": args.power_iters,
        "trace_samples": args.trace_samples, "pr_probes": args.pr_probes, "use_lanczos": args.use_lanczos,
        "top_k": args.top_k, "lanczos_steps": args.lanczos_steps, "top_eigenvalue": None,
        "trace_estimate": None, "participation_ratio_approx": None, "lambda_max_over_trace": None,
        "lambda_max_topk": None, "raw_topk_eigenvalues": [], "positive_topk_eigenvalues": [],
        "num_positive_topk_eigenvalues": None,
        "top_1_mass_ratio": None, "top_5_mass_ratio": None, "top_10_mass_ratio": None,
        "top_k_sum": None, "top_k_sum_over_trace": None, "participation_ratio_topk": None,
        "effective_rank_entropy": None, "spectral_entropy": None, "top_k_eigenvalues": [],
        "lambda_max_over_topk_sum": None,
        "start_time": iso(), "end_time": None, "duration_minutes": None, "device": args.device,
        "actual_device": None, "status": "failed", "error_message": "", "warnings": [],
        "hessian_json_path": str(json_path.resolve()), "eigen_proxy_csv_path": str(proxy_path.resolve()),
        "topk_eigen_csv_path": str(topk_path.resolve()) if topk_path else "",
    }
    proxy = []
    try:
        metrics = compute_geometry_for_checkpoint(
            checkpoint=resolve(args.checkpoint), dataset=args.dataset, data_root=resolve(args.data_root),
            subset_size=args.subset_size, batch_size=args.batch_size, seed=args.seed,
            num_workers=args.num_workers, device=args.device, power_iters=args.power_iters,
            trace_samples=args.trace_samples, pr_probes=args.pr_probes, max_batches=args.max_batches,
            use_lanczos=args.use_lanczos,
            top_k=args.top_k,
            lanczos_steps=args.lanczos_steps,
            lanczos_damping=args.lanczos_damping,
            lanczos_reorthogonalize=not args.no_reorthogonalize,
            model_name=args.model,
        )
        proxy = metrics.pop("eigen_proxy_values", [])
        result.update(metrics)
        result["status"] = "success"
    except Exception as exc:
        result["error_message"] = repr(exc)
    finally:
        result["end_time"] = iso()
        result["duration_minutes"] = (time.time() - wall) / 60.0
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(clean_json(result), f, indent=2)
        write_proxy(proxy_path, run, args.optimizer_name, args.seed, proxy)
        write_topk(topk_path, run, args.dataset, args.optimizer_name, args.seed, result.get("raw_topk_eigenvalues", result.get("top_k_eigenvalues", [])))
    print(f"RUN_NAME={run}")
    print(f"HESSIAN_JSON_PATH={json_path.resolve()}")
    print(f"EIGEN_PROXY_CSV_PATH={proxy_path.resolve()}")
    print(f"TOPK_EIGEN_CSV_PATH={topk_path.resolve() if topk_path else 'NONE'}")
    print(f"STATUS={result['status']}")
    if result["status"] != "success":
        print(f"ERROR_MESSAGE={result['error_message']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
