"""Download the SROIE receipt dataset into the project's data/ folder.

Populates:
    data/raw/{img,box,key}/        every receipt (image + word boxes + entities)
    data/train|valid|test/{img,box,key}/   a deterministic 70/15/15 split
    data/preprocess/               (left empty; OCR/crop cache lives here)

Two sources (no Kaggle account needed for the GitHub one):

    # A) GitHub mirror — anonymous, no token
    python scripts/download_dataset.py --source github

    # B) Kaggle — needs ~/.kaggle/kaggle.json (Account -> Create New API Token)
    python scripts/download_dataset.py --source kaggle

Idempotent: re-running re-uses an existing clone/download and re-does the split.
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

GITHUB_REPO = "https://github.com/zzzDavid/ICDAR-2019-SROIE"
KAGGLE_SLUG = "urbikn/sroie-datasetv2"
SUBDIRS = ("img", "box", "key")


def _run(cmd: list[str], **kw) -> None:
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)


def fetch_github(work: Path) -> Path:
    """Clone the mirror (shallow) and return the folder holding img/ box/ key/."""
    clone = work / "_sroie_src"
    if not (clone / ".git").exists():
        _run(["git", "clone", "--depth", "1", GITHUB_REPO, str(clone)],
             env={"GIT_LFS_SKIP_SMUDGE": "1", "PATH": _env_path()})
    data = clone / "data"
    if not (data / "img").is_dir():
        sys.exit(f"Expected {data}/img after clone; repo layout changed?")
    return data


def fetch_kaggle(work: Path) -> Path:
    """Download + unzip via the Kaggle CLI, return the folder with img/ box/ key/."""
    dl = work / "_kaggle_src"
    dl.mkdir(parents=True, exist_ok=True)
    _run(["kaggle", "datasets", "download", "-d", KAGGLE_SLUG,
          "-p", str(dl), "--unzip"], env={"PATH": _env_path()})
    for cand in [dl, *dl.rglob("*")]:
        if cand.is_dir() and (cand / "img").is_dir():
            return cand
    sys.exit(f"Could not find img/ under {dl}; inspect the unzip output.")


def _env_path() -> str:
    import os
    return os.environ.get("PATH", "/usr/bin:/bin")


def copy_raw(src: Path, raw: Path) -> list[str]:
    """Copy img/box/key from the source into data/raw and return the sample ids."""
    for sub in SUBDIRS:
        (raw / sub).mkdir(parents=True, exist_ok=True)
    ids: list[str] = []
    for img in sorted((src / "img").glob("*.jpg")):
        sid = img.stem
        box, key = src / "box" / f"{sid}.csv", src / "key" / f"{sid}.json"
        if not (box.exists() and key.exists()):
            continue  # keep only fully-labelled samples
        shutil.copy2(img, raw / "img" / img.name)
        shutil.copy2(box, raw / "box" / box.name)
        shutil.copy2(key, raw / "key" / key.name)
        ids.append(sid)
    return ids


def split_bucket(sid: str) -> str:
    """Deterministic 70/15/15 bucket from a hash of the id (stable across runs)."""
    h = int(hashlib.md5(sid.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else "valid" if h < 85 else "test"


def materialise_splits(raw: Path, root: Path, ids: list[str]) -> dict[str, int]:
    counts = {"train": 0, "valid": 0, "test": 0}
    for split in counts:
        for sub in SUBDIRS:
            (root / split / sub).mkdir(parents=True, exist_ok=True)
    for sid in ids:
        split = split_bucket(sid)
        counts[split] += 1
        for sub, ext in zip(SUBDIRS, ("jpg", "csv", "json")):
            shutil.copy2(raw / sub / f"{sid}.{ext}",
                         root / split / sub / f"{sid}.{ext}")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data", help="project data/ folder")
    ap.add_argument("--source", choices=["github", "kaggle"], default="github")
    args = ap.parse_args()

    root = Path(args.data_root).expanduser().resolve()
    raw = root / "raw"
    (root / "preprocess").mkdir(parents=True, exist_ok=True)
    work = root / "_work"
    work.mkdir(parents=True, exist_ok=True)

    print(f"Source: {args.source}")
    src = fetch_github(work) if args.source == "github" else fetch_kaggle(work)

    print(f"Copying raw samples -> {raw}")
    ids = copy_raw(src, raw)
    print(f"  {len(ids)} fully-labelled receipts")

    print("Building train/valid/test split (70/15/15, deterministic)")
    counts = materialise_splits(raw, root, ids)
    print(f"  {counts}")

    print(f"\nDone. Point the loader at a split, e.g.:")
    print(f"  export IPP_DATASET_ROOT={root/'train'}")
    print(f"You can delete {work} to reclaim space.")


if __name__ == "__main__":
    main()