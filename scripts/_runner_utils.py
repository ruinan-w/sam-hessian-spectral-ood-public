import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def iso():
    return datetime.now().isoformat(timespec="seconds")


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def split_csv(value, cast=str):
    return [cast(x.strip()) for x in str(value).split(",") if x.strip()]


def parse_kv(stdout):
    out = {}
    for line in stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            if re.fullmatch(r"[A-Z0-9_]+", key.strip()):
                out[key.strip()] = value.strip()
    return out


def run_child(args, master_log):
    proc = subprocess.run([sys.executable, *args], cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with master_log.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("COMMAND: " + " ".join([sys.executable, *args]) + "\n")
        f.write(proc.stdout + "\n")
    kv = parse_kv(proc.stdout)
    return proc.returncode, proc.stdout, kv


def stop_now(start_wall, stop_after_hours):
    return stop_after_hours is not None and (time.time() - start_wall) / 3600.0 >= stop_after_hours


def write_json(path, rows):
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
