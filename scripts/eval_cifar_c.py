import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.cifar_c import CIFARCDataset
from src.data.datasets import CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD
from src.models.resnet_cifar import get_model
from src.utils.metrics import accuracy


CORRUPTIONS = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate one checkpoint on CIFAR-C.")
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--optimizer-name", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--model", choices=["resnet18", "resnet34", "vgg16_bn"], default=None)
    p.add_argument("--data-root", default="data")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--severities", default="1,2,3,4,5")
    p.add_argument("--corruptions", default="all")
    p.add_argument("--output-dir", default="outputs")
    return p.parse_args()


def iso():
    return datetime.now().isoformat(timespec="seconds")


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def reserve(logs, base):
    version = 1
    while True:
        run = base if version == 1 else f"{base}_v{version}"
        csv_path = logs / f"{run}_cifarc_details.csv"
        json_path = logs / f"{run}_cifarc_summary.json"
        if not csv_path.exists() and not json_path.exists():
            return run, csv_path, json_path
        version += 1


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    total_loss = total_acc = total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(inputs)
        loss = loss_fn(logits, labels)
        bs = labels.size(0)
        total += bs
        total_loss += float(loss.item()) * bs
        total_acc += accuracy(logits, labels) * bs
    return total_acc / total, total_loss / total


def evaluate_cifarc_dataset(model, dataset, batch_size, num_workers, loss_fn, device):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    try:
        return evaluate(model, loader, loss_fn, device), num_workers, ""
    except PermissionError as exc:
        if num_workers <= 0:
            raise
        fallback = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        acc_loss = evaluate(model, fallback, loss_fn, device)
        return acc_loss, 0, f"DataLoader num_workers={num_workers} failed with {exc!r}; retried with num_workers=0."


def main():
    args = parse_args()
    logs = PROJECT_ROOT / args.output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    run, csv_path, summary_path = reserve(logs, args.run_name)
    wall = time.time()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    severities = [int(x) for x in args.severities.split(",") if x]
    corruptions = CORRUPTIONS if args.corruptions == "all" else [x.strip() for x in args.corruptions.split(",") if x.strip()]
    data_root = resolve(args.data_root) / ("CIFAR-10-C" if args.dataset == "cifar10" else "CIFAR-100-C")
    summary = {"run_name": run, "dataset": args.dataset, "model": args.model, "optimizer": args.optimizer_name, "seed": args.seed, "checkpoint_path": str(resolve(args.checkpoint).resolve()), "data_root": str(data_root.resolve()), "severities": severities, "corruptions": corruptions, "mean_accuracy_all": None, "mean_loss_all": None, "mean_accuracy_by_severity": {}, "mean_accuracy_by_corruption": {}, "start_time": iso(), "end_time": None, "duration_minutes": None, "result_csv_path": str(csv_path.resolve()), "status": "failed", "error_message": "", "warnings": []}
    rows = []
    try:
        num_classes = 100 if args.dataset == "cifar100" else 10
        mean, std = (CIFAR100_MEAN, CIFAR100_STD) if args.dataset == "cifar100" else (CIFAR10_MEAN, CIFAR10_STD)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        loaded = torch.load(resolve(args.checkpoint), map_location=device)
        checkpoint_model = loaded.get("model") if isinstance(loaded, dict) else None
        model_name = args.model or checkpoint_model or ("vgg16_bn" if "vgg16_bn" in args.run_name.lower() else ("resnet34" if "resnet34" in args.run_name.lower() else "resnet18"))
        summary["model"] = model_name
        model = get_model(model_name, num_classes=num_classes).to(device)
        model.load_state_dict(loaded["model_state_dict"] if isinstance(loaded, dict) and "model_state_dict" in loaded else loaded)
        model.eval()
        loss_fn = nn.CrossEntropyLoss()
        for severity in severities:
            for corruption in corruptions:
                ds = CIFARCDataset(args.dataset, root=data_root, corruption=corruption, severity=severity, transform=transform)
                (acc, loss), used_workers, warning = evaluate_cifarc_dataset(
                    model, ds, args.batch_size, args.num_workers, loss_fn, device
                )
                if warning:
                    summary["warnings"].append(warning)
                rows.append({"run_name": run, "dataset": args.dataset, "model": summary["model"], "optimizer": args.optimizer_name, "seed": args.seed, "severity": severity, "corruption": corruption, "accuracy": acc, "loss": loss, "checkpoint_path": str(resolve(args.checkpoint).resolve())})
        accs = [r["accuracy"] for r in rows]
        losses = [r["loss"] for r in rows]
        summary["mean_accuracy_all"] = sum(accs) / len(accs)
        summary["mean_loss_all"] = sum(losses) / len(losses)
        for severity in severities:
            vals = [r["accuracy"] for r in rows if r["severity"] == severity]
            summary["mean_accuracy_by_severity"][str(severity)] = sum(vals) / len(vals)
        for corruption in corruptions:
            vals = [r["accuracy"] for r in rows if r["corruption"] == corruption]
            summary["mean_accuracy_by_corruption"][corruption] = sum(vals) / len(vals)
        summary["status"] = "success"
    except Exception as exc:
        summary["error_message"] = repr(exc)
    finally:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["run_name", "dataset", "model", "optimizer", "seed", "severity", "corruption", "accuracy", "loss", "checkpoint_path"])
            writer.writeheader()
            writer.writerows(rows)
        summary["end_time"] = iso()
        summary["duration_minutes"] = (time.time() - wall) / 60.0
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    print(f"RUN_NAME={run}")
    print(f"CIFARC_CSV_PATH={csv_path.resolve()}")
    print(f"CIFARC_SUMMARY_PATH={summary_path.resolve()}")
    print(f"MEAN_CIFARC_ACC={summary['mean_accuracy_all']}")
    print(f"STATUS={summary['status']}")
    if summary["status"] != "success":
        print(f"ERROR_MESSAGE={summary['error_message']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
