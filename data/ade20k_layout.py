"""Place the ADE20K files this gym needs into their visible / hidden locations.

Standard library only (runs on the login node inside scripts/download.sh).

Visible to methods (data/downloads/ade20k/):
  adapt/images/<id>.jpg        2000 unlabeled training images (adaptation pool)
  dev/images/<id>.jpg          500 training images used as the visible dev set
  dev/annotations/<id>.png     their segmentation labels

Hidden test side (eval/test/data/ade20k/):
  pool/images/<id>.jpg         2000 other unlabeled training images (hidden pool)
  val/images/<id>.jpg          the 2000 ADE20K validation images
  val/annotations/<id>.png     their segmentation labels

The id lists are committed in data/splits/ and eval/test/splits/. No other
ADE20K annotation is kept on disk.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "ADEChallengeData2016"


def _ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _plan() -> list[tuple[str, Path]]:
    """(member path inside the archive, destination path) pairs."""
    vis = ROOT / "data" / "downloads" / "ade20k"
    hid = ROOT / "eval" / "test" / "data" / "ade20k"
    plan: list[tuple[str, Path]] = []
    for i in _ids(ROOT / "data" / "splits" / "ade20k_adapt_ids.txt"):
        plan.append((f"images/training/{i}.jpg", vis / "adapt" / "images" / f"{i}.jpg"))
    for i in _ids(ROOT / "data" / "splits" / "ade20k_dev_ids.txt"):
        plan.append((f"images/training/{i}.jpg", vis / "dev" / "images" / f"{i}.jpg"))
        plan.append((f"annotations/training/{i}.png", vis / "dev" / "annotations" / f"{i}.png"))
    for i in _ids(ROOT / "eval" / "test" / "splits" / "ade20k_pool_ids.txt"):
        plan.append((f"images/training/{i}.jpg", hid / "pool" / "images" / f"{i}.jpg"))
    for i in _ids(ROOT / "eval" / "test" / "splits" / "ade20k_val_ids.txt"):
        plan.append((f"images/validation/{i}.jpg", hid / "val" / "images" / f"{i}.jpg"))
        plan.append((f"annotations/validation/{i}.png", hid / "val" / "annotations" / f"{i}.png"))
    return plan


def place_from_zip(zip_path: Path) -> int:
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member, dest in _plan():
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(f"{PREFIX}/{member}") as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    return n


def place_from_dir(src_root: Path) -> int:
    n = 0
    for member, dest in _plan():
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_root / member, dest)
        n += 1
    return n


def is_complete() -> bool:
    return all(dest.exists() for _, dest in _plan())


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--zip", type=Path)
    group.add_argument("--dir", type=Path, help="an extracted ADEChallengeData2016/ directory")
    parser.add_argument("--check", action="store_true", help="exit 0 iff every file is in place")
    args = parser.parse_args()
    if args.check:
        raise SystemExit(0 if is_complete() else 1)
    if args.zip:
        n = place_from_zip(args.zip)
    elif args.dir:
        n = place_from_dir(args.dir)
    else:
        parser.error("pass --zip or --dir")
    print(f"placed {n} files; complete={is_complete()}")


if __name__ == "__main__":
    main()
