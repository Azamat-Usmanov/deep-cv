from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from highlight_detection.data import SPLIT_FILES, load_jsonl
from highlight_detection.metrics import (
    build_prediction_summaries,
    error_analysis,
    evaluate_clip_predictions,
)


META_COLUMNS = {"split", "qid", "vid", "query", "label", "gt_windows"}


try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime.
    tqdm = None


def configure_runtime(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / ".tmp"
    mpl_dir = out_dir / ".mplconfig"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(tmp_dir))
    os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(tmp_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))


def load_npz_array(path: Path, preferred_key: str | None = None) -> np.ndarray:
    values = np.load(path)
    if preferred_key and preferred_key in values:
        return np.asarray(values[preferred_key], dtype=np.float32)
    first_key = values.files[0]
    return np.asarray(values[first_key], dtype=np.float32)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(matrix, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return matrix / denom


def load_video_features(
    vid: str,
    feature_dirs: list[Path],
    max_clips: int,
    normalize: bool,
) -> np.ndarray | None:
    parts = []
    for feature_dir in feature_dirs:
        path = feature_dir / f"{vid}.npz"
        if not path.exists():
            return None
        features = load_npz_array(path, preferred_key="features")[:max_clips]
        if normalize:
            features = l2_normalize(features)
        parts.append(features)
    min_len = min(len(part) for part in parts)
    if min_len == 0:
        return None
    return np.concatenate([part[:min_len] for part in parts], axis=1)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] != b.shape[0]:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def load_text_feature(qid: int, text_feature_dir: Path | None, normalize: bool) -> np.ndarray | None:
    if text_feature_dir is None:
        return None
    path = text_feature_dir / f"qid{qid}.npz"
    if not path.exists():
        return None
    pooled = load_npz_array(path, preferred_key="pooler_output")
    if pooled.ndim > 1:
        pooled = pooled.mean(axis=0)
    pooled = pooled.astype(np.float32)
    if normalize:
        denom = np.linalg.norm(pooled)
        if denom > 0:
            pooled = pooled / denom
    return pooled


def build_records(
    rows: list[dict[str, Any]],
    split: str,
    feature_dirs: list[Path],
    text_feature_dir: Path | None,
    max_clips: int,
    normalize: bool,
) -> list[dict[str, Any]]:
    records = []
    skipped = 0
    iterator = rows
    if tqdm is not None:
        iterator = tqdm(rows, desc=f"loading {split} features", unit="video")
    for row in iterator:
        video_features = load_video_features(row["vid"], feature_dirs, max_clips, normalize)
        if video_features is None:
            skipped += 1
            continue
        clip_features_for_similarity = load_video_features(
            row["vid"],
            [feature_dirs[0]],
            max_clips,
            normalize,
        )
        text_feature = load_text_feature(int(row["qid"]), text_feature_dir, normalize)
        relevant_clip_ids = {int(v) for v in row.get("relevant_clip_ids", [])}
        duration = float(row["duration"])
        query = str(row["query"])

        for clip_id, clip_features in enumerate(video_features):
            clip_start = clip_id * 2.0
            clip_end = min((clip_id + 1) * 2.0, duration)
            item: dict[str, Any] = {
                "split": split,
                "qid": int(row["qid"]),
                "vid": row["vid"],
                "query": query,
                "duration": duration,
                "clip_id": clip_id,
                "clip_start": clip_start,
                "clip_end": clip_end,
                "clip_mid": (clip_start + clip_end) / 2.0,
                "rel_start": clip_start / duration,
                "rel_mid": ((clip_start + clip_end) / 2.0) / duration,
                "rel_end": clip_end / duration,
                "query_word_count": len(query.split()),
                "query_char_count": len(query),
                "clip_text_cosine": cosine_similarity(
                    clip_features_for_similarity[clip_id],
                    text_feature,
                )
                if clip_features_for_similarity is not None and text_feature is not None
                else 0.0,
                "label": int(clip_id in relevant_clip_ids),
                "gt_windows": row.get("relevant_windows", []),
            }
            for idx, value in enumerate(clip_features):
                item[f"v_{idx:04d}"] = float(value)
            if text_feature is not None:
                for idx, value in enumerate(text_feature):
                    item[f"t_{idx:04d}"] = float(value)
            records.append(item)
    print(f"{split}: built {len(records)} clip records; skipped videos without features={skipped}")
    return records


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [
        col
        for col in frame.columns
        if col not in META_COLUMNS and pd.api.types.is_numeric_dtype(frame[col])
    ]


def build_model(feature_columns: list[str], max_text_features: int, c_value: float, max_iter: int):
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "query_tfidf",
                TfidfVectorizer(max_features=max_text_features, ngram_range=(1, 2), min_df=2),
                "query",
            ),
            ("numeric", numeric_pipeline, feature_columns),
        ]
    )
    classifier = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=max_iter,
        random_state=42,
        solver="liblinear",
    )
    return Pipeline(steps=[("features", preprocessor), ("classifier", classifier)])


def build_sgd_model(feature_columns: list[str], max_text_features: int, alpha: float, max_iter: int):
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "query_tfidf",
                TfidfVectorizer(max_features=max_text_features, ngram_range=(1, 2), min_df=2),
                "query",
            ),
            ("numeric", numeric_pipeline, feature_columns),
        ]
    )
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        class_weight="balanced",
        max_iter=max_iter,
        tol=1e-3,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=3,
        random_state=42,
    )
    return Pipeline(steps=[("features", preprocessor), ("classifier", classifier)])


def attach_scores(records: list[dict[str, Any]], scores) -> list[dict[str, Any]]:
    out = []
    for row, score in zip(records, scores):
        item = dict(row)
        item["score"] = float(score)
        out.append(item)
    return out


def compact_for_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = {
        "split",
        "qid",
        "vid",
        "query",
        "duration",
        "clip_id",
        "clip_start",
        "clip_end",
        "clip_mid",
        "label",
        "gt_windows",
        "score",
    }
    return [{key: row[key] for key in keep if key in row} for row in records]


def write_metrics_table(rows: list[dict[str, float]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a QVHighlights model on Moment-DETR features.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/qvhighlights"))
    parser.add_argument("--features-root", type=Path, default=Path("features"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/stage2_moment_features"))
    parser.add_argument("--video-feature-dirs", nargs="+", default=["clip_features"])
    parser.add_argument("--text-feature-dir", default="clip_text_features")
    parser.add_argument("--max-train-videos", type=int, default=None)
    parser.add_argument("--max-val-videos", type=int, default=None)
    parser.add_argument("--max-clips", type=int, default=75)
    parser.add_argument("--max-text-features", type=int, default=8000)
    parser.add_argument("--classifier", choices=["sgd", "logreg"], default="sgd")
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--sgd-alpha", type=float, default=0.0001)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()

    configure_runtime(args.out_dir)

    feature_dirs = [args.features_root / name for name in args.video_feature_dirs]
    text_feature_dir = args.features_root / args.text_feature_dir if args.text_feature_dir else None
    train_rows = load_jsonl(args.data_dir / SPLIT_FILES["train"])
    val_rows = load_jsonl(args.data_dir / SPLIT_FILES["val"])
    if args.max_train_videos is not None:
        train_rows = train_rows[: args.max_train_videos]
    if args.max_val_videos is not None:
        val_rows = val_rows[: args.max_val_videos]

    print(f"Loading train annotations: {len(train_rows)} videos")
    train_records = build_records(
        train_rows,
        "train",
        feature_dirs,
        text_feature_dir,
        args.max_clips,
        not args.no_normalize,
    )
    print(f"Loading val annotations: {len(val_rows)} videos")
    val_records = build_records(
        val_rows,
        "val",
        feature_dirs,
        text_feature_dir,
        args.max_clips,
        not args.no_normalize,
    )
    if not train_records or not val_records:
        raise RuntimeError("No feature records found. Check that moment_detr_features.tar.gz is extracted.")

    print("Building train/val DataFrames")
    train_df = pd.DataFrame(train_records)
    val_df = pd.DataFrame(val_records)
    feature_cols = numeric_columns(train_df)
    print(
        f"Prepared matrices: train clips={len(train_df)}, val clips={len(val_df)}, "
        f"numeric features={len(feature_cols)}"
    )

    if args.classifier == "logreg":
        model = build_model(feature_cols, args.max_text_features, args.c_value, args.max_iter)
    else:
        model = build_sgd_model(feature_cols, args.max_text_features, args.sgd_alpha, args.max_iter)
    print(f"Training {args.classifier} model")
    model.fit(train_df[["query", *feature_cols]], train_df["label"].astype(int))
    print("Predicting validation scores")
    val_scores = model.predict_proba(val_df[["query", *feature_cols]])[:, 1]
    scored_val = attach_scores(val_records, val_scores)
    metric_records = compact_for_metrics(scored_val)

    thresholds = [round(value / 100, 2) for value in range(5, 96, 5)]
    print("Sweeping thresholds")
    threshold_iter = tqdm(thresholds, desc="evaluating thresholds") if tqdm is not None else thresholds
    threshold_metrics = [
        evaluate_clip_predictions(metric_records, threshold) for threshold in threshold_iter
    ]
    best = max(
        threshold_metrics,
        key=lambda row: (row["segment_f1_iou_0_5"], row["clip_f1"], row["mean_best_iou"]),
    )
    best_threshold = float(best["threshold"])
    final_metrics = evaluate_clip_predictions(metric_records, best_threshold)

    print("Saving model and reports")
    joblib.dump(
        {
            "model": model,
            "threshold": best_threshold,
            "feature_columns": feature_cols,
            "video_feature_dirs": [str(path) for path in feature_dirs],
            "text_feature_dir": str(text_feature_dir) if text_feature_dir else None,
            "description": "QVHighlights baseline trained on official Moment-DETR video features",
        },
        args.out_dir / "moment_features_highlight_model.joblib",
    )
    (args.out_dir / "metrics.json").write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_metrics_table(threshold_metrics, args.out_dir / "threshold_sweep.csv")
    (args.out_dir / "val_predictions_examples.json").write_text(
        json.dumps(
            build_prediction_summaries(metric_records, best_threshold, include_gt=True)[:50],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "error_analysis.json").write_text(
        json.dumps(error_analysis(metric_records, best_threshold), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))
    print(f"\nBest threshold: {best_threshold:.2f}")
    print(f"Artifacts written to {args.out_dir}")


if __name__ == "__main__":
    main()
