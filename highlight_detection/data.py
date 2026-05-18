from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


SPLIT_FILES = {
    "train": "highlight_train_release.jsonl",
    "val": "highlight_val_release.jsonl",
    "test": "highlight_test_release.jsonl",
}

DEFAULT_CLIP_SECONDS = 2.0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _mean_saliency_by_clip(row: dict[str, Any]) -> dict[int, float]:
    clip_ids = row.get("relevant_clip_ids", [])
    scores = row.get("saliency_scores", [])
    out: dict[int, float] = {}
    for clip_id, triplet in zip(clip_ids, scores):
        if triplet:
            out[int(clip_id)] = float(sum(triplet) / len(triplet))
    return out


def build_clip_records(
    rows: list[dict[str, Any]],
    split: str,
    clip_seconds: float = DEFAULT_CLIP_SECONDS,
) -> list[dict[str, Any]]:
    """Convert query-level QVHighlights records into fixed-length clip examples.

    QVHighlights uses two-second clip ids in the highlight annotations. For
    train/val records the target is 1 when the candidate clip id is annotated
    as relevant. Test records do not include labels, but the same feature schema
    is produced for inference.
    """

    records: list[dict[str, Any]] = []
    for row in rows:
        duration = float(row["duration"])
        n_clips = int(math.ceil(duration / clip_seconds))
        relevant_clip_ids = {int(v) for v in row.get("relevant_clip_ids", [])}
        mean_saliency = _mean_saliency_by_clip(row)
        query = str(row.get("query", ""))
        query_words = query.split()
        gt_windows = row.get("relevant_windows", [])

        for clip_id in range(n_clips):
            clip_start = clip_id * clip_seconds
            clip_end = min((clip_id + 1) * clip_seconds, duration)
            clip_mid = (clip_start + clip_end) / 2.0
            label = int(clip_id in relevant_clip_ids)
            saliency = mean_saliency.get(clip_id, 0.0)

            records.append(
                {
                    "split": split,
                    "qid": int(row["qid"]),
                    "vid": row["vid"],
                    "query": query,
                    "duration": duration,
                    "clip_id": clip_id,
                    "clip_start": clip_start,
                    "clip_end": clip_end,
                    "clip_mid": clip_mid,
                    "clip_len": clip_end - clip_start,
                    "rel_start": clip_start / duration,
                    "rel_mid": clip_mid / duration,
                    "rel_end": clip_end / duration,
                    "query_word_count": len(query_words),
                    "query_char_count": len(query),
                    "label": label,
                    "saliency_mean": saliency,
                    "saliency_norm": saliency / 4.0,
                    "gt_windows": gt_windows,
                }
            )
    return records


def feature_columns() -> list[str]:
    return [
        "duration",
        "clip_id",
        "clip_start",
        "clip_end",
        "clip_mid",
        "clip_len",
        "rel_start",
        "rel_mid",
        "rel_end",
        "query_word_count",
        "query_char_count",
    ]
