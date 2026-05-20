import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1"
DEFAULT_MD5 = "11f0ed0f1191edbf9fa23466ae6021d3"
DEFAULT_DATA_ROOT = "data"
DEFAULT_TAR_PATH = r"data\CIFAR-100-C.tar"
DEFAULT_EXTRACT_DIR = "data"
DEFAULT_TARGET_DIR = r"data\CIFAR-100-C"
MIN_TAR_BYTES = 1_000_000_000
REQUIRED_FILES = [
    "labels.npy",
    "gaussian_noise.npy",
    "shot_noise.npy",
    "impulse_noise.npy",
    "defocus_blur.npy",
    "glass_blur.npy",
    "motion_blur.npy",
    "zoom_blur.npy",
    "snow.npy",
    "frost.npy",
    "fog.npy",
    "brightness.npy",
    "contrast.npy",
    "elastic_transform.npy",
    "pixelate.npy",
    "jpeg_compression.npy",
]


def project_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Download, extract, and verify CIFAR-100-C.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--tar-path", default=DEFAULT_TAR_PATH)
    parser.add_argument("--extract-dir", default=DEFAULT_EXTRACT_DIR)
    parser.add_argument("--target-dir", default=DEFAULT_TARGET_DIR)
    parser.add_argument("--expected-md5", default=DEFAULT_MD5)
    parser.add_argument("--chunk-size", type=int, default=1048576)
    parser.add_argument("--skip-md5", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def file_md5(path, chunk_size):
    digest = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def check_md5(path, expected_md5, chunk_size, skip_md5=False):
    if skip_md5:
        print(f"SKIP_MD5: {path}")
        return True
    actual = file_md5(path, chunk_size)
    ok = actual.lower() == expected_md5.lower()
    print(f"MD5: {path}")
    print(f"  expected={expected_md5}")
    print(f"  actual={actual}")
    print(f"  status={'OK' if ok else 'MISMATCH'}")
    return ok


def download(url, destination, chunk_size):
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    downloaded = partial.stat().st_size if partial.exists() else 0
    headers = {}
    mode = "wb"
    if downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"
        mode = "ab"
        print(f"RESUME_PARTIAL={partial.resolve()}")
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request) as response:
            if headers and response.getcode() != 206:
                print("WARNING: server did not resume partial download; restarting partial file.")
                downloaded = 0
                mode = "wb"
            total_header = response.headers.get("Content-Length")
            total = int(total_header) + downloaded if total_header else None
            with partial.open(mode) as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    mb_done = downloaded / (1024 * 1024)
                    if total:
                        mb_total = total / (1024 * 1024)
                        pct = downloaded * 100.0 / total
                        print(f"DOWNLOAD_PROGRESS: {mb_done:.1f} MB / {mb_total:.1f} MB / {pct:.2f}%")
                    else:
                        print(f"DOWNLOAD_PROGRESS: {mb_done:.1f} MB / unknown MB / unknown%")
    except (urllib.error.URLError, OSError) as exc:
        print(f"ERROR: download failed; partial file kept at: {partial.resolve()}")
        print(f"ERROR_DETAIL: {exc!r}")
        raise
    return partial


def validate_tar_path(base_dir, member):
    target = (base_dir / member.name).resolve()
    base = base_dir.resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise RuntimeError(f"Unsafe tar member path: {member.name}") from exc


def extract_tar(tar_path, extract_dir):
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"EXTRACTING_TAR={tar_path.resolve()}")
    print(f"EXTRACT_DIR={extract_dir.resolve()}")
    with tarfile.open(tar_path, "r:*") as archive:
        members = archive.getmembers()
        for member in members:
            validate_tar_path(extract_dir, member)
        archive.extractall(extract_dir, members=members)


def find_cifar100c_dir(root):
    root = root.resolve()
    candidates = []
    for labels in root.rglob("labels.npy"):
        parent = labels.parent
        score = sum((parent / name).exists() for name in REQUIRED_FILES)
        candidates.append((score, parent))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def organize_extracted_dir(extract_dir, target_dir):
    source = find_cifar100c_dir(extract_dir)
    if source is None:
        print(f"ERROR: no directory containing labels.npy found under {extract_dir.resolve()}")
        return False
    if source.resolve() == target_dir.resolve():
        return True
    if target_dir.exists():
        print(f"ERROR: target dir exists but extracted data is elsewhere: {source.resolve()}")
        print(f"ERROR: not overwriting existing target dir: {target_dir.resolve()}")
        return False
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"ORGANIZE_EXTRACTED_DIR: {source.resolve()} -> {target_dir.resolve()}")
    shutil.move(str(source), str(target_dir))
    return True


def shape_of_npy(path):
    try:
        import numpy as np
    except Exception as exc:
        print(f"WARNING: numpy import failed; cannot shape-check {path.name}: {exc!r}")
        return None
    try:
        arr = np.load(path, mmap_mode="r")
        return tuple(arr.shape)
    except Exception as exc:
        print(f"ERROR: failed to read {path}: {exc!r}")
        return None


def check_ready(target_dir):
    target_dir = Path(target_dir)
    print(f"CHECK_DIR={target_dir.resolve()}")
    if not target_dir.exists():
        print(f"MISSING_DIR: {target_dir.resolve()}")
        return False
    ok = True
    for name in REQUIRED_FILES:
        path = target_dir / name
        exists = path.exists()
        print(f"{'OK' if exists else 'MISSING'}: {path.resolve()}")
        ok = ok and exists
    if not ok:
        return False

    labels_shape = shape_of_npy(target_dir / "labels.npy")
    if labels_shape is not None:
        first_dim = labels_shape[0] if labels_shape else 0
        supports_severities = first_dim == 50000 or first_dim % 5 == 0
        labels_ok = first_dim == 50000 or supports_severities
        print(f"SHAPE labels.npy={labels_shape} status={'OK' if labels_ok else 'UNEXPECTED'}")
        ok = ok and labels_ok

    for name in ["gaussian_noise.npy", "jpeg_compression.npy"]:
        shape = shape_of_npy(target_dir / name)
        if shape is not None:
            shape_ok = len(shape) >= 1 and shape[0] == 50000
            print(f"SHAPE {name}={shape} first_dim_50000={'OK' if shape_ok else 'UNEXPECTED'}")
            ok = ok and shape_ok
    return ok


def ensure_tar(args, tar_path):
    if tar_path.exists():
        size = tar_path.stat().st_size
        print(f"TAR_EXISTS={tar_path.resolve()}")
        print(f"TAR_SIZE_BYTES={size}")
        size_ok = size > MIN_TAR_BYTES
        print(f"TAR_SIZE_GT_1GB={'OK' if size_ok else 'FAILED'}")
        if size_ok and check_md5(tar_path, args.expected_md5, args.chunk_size, args.skip_md5):
            print("TAR_ALREADY_VALID")
            return True
        if not args.force:
            print("ERROR: existing tar failed size or md5 check. Re-run with --force to download again.")
            return False
        print("FORCE_DOWNLOAD_ENABLED")

    partial = download(args.url, tar_path, args.chunk_size)
    if partial.stat().st_size <= MIN_TAR_BYTES:
        print(f"ERROR: downloaded partial is too small: {partial.stat().st_size} bytes")
        return False
    if not check_md5(partial, args.expected_md5, args.chunk_size, args.skip_md5):
        print(f"ERROR: downloaded file md5 mismatch; partial kept at: {partial.resolve()}")
        return False
    os.replace(partial, tar_path)
    print(f"TAR_READY={tar_path.resolve()}")
    return True


def main():
    args = parse_args()
    data_root = project_path(args.data_root)
    tar_path = data_root / "CIFAR-100-C.tar" if args.tar_path == DEFAULT_TAR_PATH else project_path(args.tar_path)
    extract_dir = data_root if args.extract_dir == DEFAULT_EXTRACT_DIR else project_path(args.extract_dir)
    target_dir = data_root / "CIFAR-100-C" if args.target_dir == DEFAULT_TARGET_DIR else project_path(args.target_dir)

    if target_dir.exists() and check_ready(target_dir):
        print("CIFAR100C_ALREADY_READY")
        return 0

    if not ensure_tar(args, tar_path):
        return 1
    extract_tar(tar_path, extract_dir)
    if not organize_extracted_dir(extract_dir, target_dir):
        return 1
    if not check_ready(target_dir):
        print("ERROR: CIFAR-100-C verification failed after extraction.")
        return 1

    print("CIFAR100C_DOWNLOAD_SUCCESS")
    print(f"CIFAR100C_READY_DIR={target_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("ERROR: interrupted; any partial download file has been kept.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc!r}")
        raise SystemExit(1)
