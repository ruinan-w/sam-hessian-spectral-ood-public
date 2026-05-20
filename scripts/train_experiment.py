import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from src.data.datasets import get_classification_loaders
from src.models.resnet_cifar import get_model
from src.optim.asam import ASAM
from src.optim.sam import SAM
from src.utils.metrics import accuracy
from src.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train one CIFAR experiment run.")
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    p.add_argument("--model", choices=["resnet18", "resnet34", "vgg16_bn"], default="resnet18")
    p.add_argument("--optimizer", choices=["sgd", "sam", "asam"], default="sgd")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--rho", type=float, default=None)
    p.add_argument("--asam-eta", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--scheduler", choices=["cosine", "multistep"], default="cosine")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--save-every", type=int, default=0)
    return p.parse_args()


def iso():
    return datetime.now().isoformat(timespec="seconds")


def actual_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def reserve_run_paths(output_dir, base):
    logs = output_dir / "logs"
    ckpts = output_dir / "checkpoints"
    logs.mkdir(parents=True, exist_ok=True)
    ckpts.mkdir(parents=True, exist_ok=True)
    version = 1
    while True:
        run = base if version == 1 else f"{base}_v{version}"
        log = logs / f"{run}_train.csv"
        cfg = logs / f"{run}_config.json"
        ckpt = ckpts / f"{run}.pt"
        if not log.exists() and not cfg.exists() and not ckpt.exists():
            return run, log, cfg, ckpt
        version += 1


def train_epoch(model, loader, loss_fn, optimizer, device, two_step, scaler, use_amp):
    model.train()
    total_loss = total_correct = total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if two_step:
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = loss_fn(model(inputs), labels)
            scaler.scale(loss).backward()
            if use_amp:
                scaler.unscale_(optimizer.base_optimizer)
            optimizer.first_step(zero_grad=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss_second = loss_fn(model(inputs), labels)
            scaler.scale(loss_second).backward()
            if use_amp:
                scaler.unscale_(optimizer.base_optimizer)
            optimizer.second_step(zero_grad=True)
            scaler.update()
            logits = model(inputs)
        else:
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(inputs)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        bs = labels.size(0)
        total += bs
        total_loss += float(loss.detach().item()) * bs
        total_correct += (logits.detach().argmax(1) == labels).sum().item()
    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, use_amp):
    model.eval()
    total_loss = total_acc = total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(inputs)
            loss = loss_fn(logits, labels)
        bs = labels.size(0)
        total += bs
        total_loss += float(loss.item()) * bs
        total_acc += accuracy(logits, labels) * bs
    return total_loss / total, total_acc / total


def write_config(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    args = parse_args()
    start = iso()
    wall = time.time()
    output_dir = PROJECT_ROOT / args.output_dir
    base = args.run_name or f"{args.tag + '_' if args.tag else ''}{args.dataset}_{args.model}_{args.optimizer}_seed{args.seed}_ep{args.epochs}"
    run, log_path, config_path, checkpoint_path = reserve_run_paths(output_dir, base)
    config = {**vars(args), "run_name": run, "start_time": start, "status": "failed", "error_message": ""}
    rows = []
    try:
        set_seed(args.seed)
        device = actual_device(args.device)
        config["actual_device"] = str(device)
        train_loader, test_loader, num_classes, _norm = get_classification_loaders(
            args.dataset, PROJECT_ROOT / "data", args.batch_size, args.num_workers, args.seed
        )
        config["num_classes"] = num_classes
        model = get_model(args.model, num_classes=num_classes).to(device)
        loss_fn = nn.CrossEntropyLoss()
        rho = args.rho
        if args.optimizer == "sgd":
            optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer == "sam":
            optimizer = SAM(model.parameters(), SGD, rho=0.05 if rho is None else rho, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            optimizer = ASAM(model.parameters(), SGD, rho=0.5 if rho is None else rho, eta=args.asam_eta, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs) if args.scheduler == "cosine" else MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
        best = 0.0
        final = {}
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["run_name", "dataset", "model", "optimizer", "seed", "epoch", "train_loss", "train_acc", "test_loss", "test_acc", "lr", "epoch_time_seconds"])
            writer.writeheader()
            for epoch in range(1, args.epochs + 1):
                ep_wall = time.time()
                train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, device, args.optimizer in {"sam", "asam"}, scaler, args.amp and device.type == "cuda")
                test_loss, test_acc = evaluate(model, test_loader, loss_fn, device, args.amp and device.type == "cuda")
                lr = optimizer.param_groups[0]["lr"]
                scheduler.step()
                row = {"run_name": run, "dataset": args.dataset, "model": args.model, "optimizer": args.optimizer, "seed": args.seed, "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc, "lr": lr, "epoch_time_seconds": time.time() - ep_wall}
                writer.writerow(row)
                f.flush()
                rows.append(row)
                best = max(best, test_acc)
                final = row
                if args.save_every > 0 and epoch % args.save_every == 0:
                    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "epoch": epoch, "dataset": args.dataset, "num_classes": num_classes, "model": args.model, "optimizer": args.optimizer, "seed": args.seed, "args": vars(args), "best_test_acc": best, "final_test_acc": test_acc, "run_name": run}, output_dir / "checkpoints" / f"{run}_epoch{epoch}.pt")
        torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "epoch": args.epochs, "dataset": args.dataset, "num_classes": num_classes, "model": args.model, "optimizer": args.optimizer, "seed": args.seed, "args": vars(args), "best_test_acc": best, "final_test_acc": final.get("test_acc"), "run_name": run}, checkpoint_path)
        config.update({"status": "success", "best_test_acc": best, "final_train_loss": final.get("train_loss"), "final_train_acc": final.get("train_acc"), "final_test_loss": final.get("test_loss"), "final_test_acc": final.get("test_acc")})
    except Exception as exc:
        config["error_message"] = repr(exc)
    finally:
        config.update({"end_time": iso(), "duration_minutes": (time.time() - wall) / 60.0, "checkpoint_path": str(checkpoint_path.resolve()), "train_log_path": str(log_path.resolve()), "config_path": str(config_path.resolve())})
        write_config(config_path, config)
    print(f"RUN_NAME={run}")
    print(f"CHECKPOINT_PATH={checkpoint_path.resolve()}")
    print(f"TRAIN_LOG_PATH={log_path.resolve()}")
    print(f"CONFIG_PATH={config_path.resolve()}")
    print(f"STATUS={config['status']}")
    if config["status"] != "success":
        print(f"ERROR_MESSAGE={config['error_message']}")
        return 1
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    raise SystemExit(main())
