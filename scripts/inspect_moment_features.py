from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from highlight_detection.data import SPLIT_FILES, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect extracted Moment-DETR feature folders.")
    parser.add_argument("--features-root", type=Path, default=Path("features"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/qvhighlights"))
    args = parser.parse_args()

    expected_vids = set()
    expected_qids = set()
    for filename in SPLIT_FILES.values():
        for row in load_jsonl(args.data_dir / filename):
            expected_vids.add(row["vid"])
            expected_qids.add(int(row["qid"]))

    for name in ("clip_features", "clip_text_features"):
        path = args.features_root / name
        count = len(list(path.glob("*.npz"))) if path.exists() else 0
        print(f"{name}: {count} .npz files at {path}")

    clip_dir = args.features_root / "clip_features"
    text_dir = args.features_root / "clip_text_features"

    if clip_dir.exists():
        found = {path.stem for path in clip_dir.glob("*.npz")}
        print(f"clip video coverage: {len(found & expected_vids)}/{len(expected_vids)}")
    if text_dir.exists():
        found_qids = {
            int(path.stem.replace("qid", ""))
            for path in text_dir.glob("qid*.npz")
            if path.stem.replace("qid", "").isdigit()
        }
        print(f"text feature coverage: {len(found_qids & expected_qids)}/{len(expected_qids)}")


if __name__ == "__main__":
    main()
