from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "CIFAR-10-C"

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


def main():
    if not DATA_ROOT.exists():
        print(f"CIFAR-10-C directory not found: {DATA_ROOT}")
        print("Please manually download CIFAR-10-C and extract it to:")
        print(DATA_ROOT)
        return 1

    missing = [name for name in REQUIRED_FILES if not (DATA_ROOT / name).exists()]
    if missing:
        print(f"CIFAR-10-C directory found: {DATA_ROOT}")
        print("Missing files:")
        for name in missing:
            print(f"- {name}")
        return 1

    print("CIFAR-10-C data check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
