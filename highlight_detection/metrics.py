from __future__ import annotations

from collections import defaultdict
from typing import Any


def interval_iou(a: list[float] | tuple[float, float], b: list[float] | tuple[float, float]) -> float:
    start = max(float(a[0]), float(b[0]))
    end = min(float(a[1]), float(b[1]))
    inter = max(0.0, end - start)
    union = max(float(a[1]), float(b[1])) - min(float(a[0]), float(b[0]))
    return inter / union if union > 0 else 0.0


def clips_to_windows(records: list[dict[str, Any]], threshold: float) -> list[list[float]]:
    windows: list[list[float]] = []
    active_start: float | None = None
    active_end: float | None = None

    for row in sorted(records, key=lambda item: item["clip_id"]):
        is_active = float(row.get("score", 0.0)) >= threshold
        if not is_active:
            if active_start is not None and active_end is not None:
                windows.append([active_start, active_end])
            active_start = None
            active_end = None
            continue

        start = float(row["clip_start"])
        end = float(row["clip_end"])
        if active_start is None:
            active_start = start
            active_end = end
        elif active_end is not None and start <= active_end + 1e-9:
            active_end = max(active_end, end)
        else:
            windows.append([active_start, active_end])
            active_start = start
            active_end = end

    if active_start is not None and active_end is not None:
        windows.append([active_start, active_end])
    return windows


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def average_precision(y_true: list[int], y_score: list[float]) -> float:
    positives = sum(1 for y in y_true if y)
    if positives == 0:
        return 0.0

    ranked = sorted(zip(y_score, y_true), key=lambda item: item[0], reverse=True)
    tp = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label:
            tp += 1
            precision_sum += tp / rank
    return precision_sum / positives


def segment_counts(
    predicted_windows: list[list[float]],
    gt_windows: list[list[float]],
    iou_threshold: float,
) -> tuple[int, int, int]:
    matched_gt: set[int] = set()
    tp = 0
    fp = 0

    for pred in predicted_windows:
        best_idx = None
        best_iou = 0.0
        for idx, gt in enumerate(gt_windows):
            if idx in matched_gt:
                continue
            score = interval_iou(pred, gt)
            if score > best_iou:
                best_iou = score
                best_idx = idx
        if best_idx is not None and best_iou >= iou_threshold:
            matched_gt.add(best_idx)
            tp += 1
        else:
            fp += 1

    fn = len(gt_windows) - len(matched_gt)
    return tp, fp, fn


def best_iou_for_gt(predicted_windows: list[list[float]], gt_windows: list[list[float]]) -> float:
    if not gt_windows:
        return 0.0
    scores = []
    for gt in gt_windows:
        scores.append(max((interval_iou(pred, gt) for pred in predicted_windows), default=0.0))
    return sum(scores) / len(scores)


def _group_by_query(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[int(row["qid"])].append(row)
    return grouped


def evaluate_clip_predictions(records: list[dict[str, Any]], threshold: float) -> dict[str, float]:
    y_true = [int(row.get("label", 0)) for row in records]
    y_score = [float(row.get("score", 0.0)) for row in records]
    y_pred = [int(score >= threshold) for score in y_score]

    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)

    clip = precision_recall_f1(tp, fp, fn)
    result = {
        "threshold": threshold,
        "clip_precision": clip["precision"],
        "clip_recall": clip["recall"],
        "clip_f1": clip["f1"],
        "clip_average_precision": average_precision(y_true, y_score),
    }

    grouped = _group_by_query(records)
    mean_best_iou_values = []
    for iou_threshold in (0.3, 0.5):
        seg_tp = seg_fp = seg_fn = 0
        for query_records in grouped.values():
            predicted = clips_to_windows(query_records, threshold)
            gt_windows = query_records[0].get("gt_windows", [])
            tp_i, fp_i, fn_i = segment_counts(predicted, gt_windows, iou_threshold)
            seg_tp += tp_i
            seg_fp += fp_i
            seg_fn += fn_i
            if iou_threshold == 0.5:
                mean_best_iou_values.append(best_iou_for_gt(predicted, gt_windows))

        seg = precision_recall_f1(seg_tp, seg_fp, seg_fn)
        suffix = str(iou_threshold).replace(".", "_")
        result[f"segment_precision_iou_{suffix}"] = seg["precision"]
        result[f"segment_recall_iou_{suffix}"] = seg["recall"]
        result[f"segment_f1_iou_{suffix}"] = seg["f1"]

    result["mean_best_iou"] = (
        sum(mean_best_iou_values) / len(mean_best_iou_values) if mean_best_iou_values else 0.0
    )
    return result


def build_prediction_summaries(
    records: list[dict[str, Any]],
    threshold: float,
    include_gt: bool = True,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for qid, query_records in sorted(_group_by_query(records).items()):
        first = query_records[0]
        predicted_windows = clips_to_windows(query_records, threshold)
        top_clips = sorted(query_records, key=lambda item: item.get("score", 0.0), reverse=True)[:5]
        item: dict[str, Any] = {
            "qid": qid,
            "vid": first["vid"],
            "duration": first["duration"],
            "query": first["query"],
            "predicted_windows": predicted_windows,
            "top_clips": [
                {
                    "clip_id": row["clip_id"],
                    "start": row["clip_start"],
                    "end": row["clip_end"],
                    "score": row.get("score", 0.0),
                }
                for row in top_clips
            ],
        }
        if include_gt:
            item["gt_windows"] = first.get("gt_windows", [])
            item["mean_best_iou"] = best_iou_for_gt(predicted_windows, item["gt_windows"])
        summaries.append(item)
    return summaries


def error_analysis(
    records: list[dict[str, Any]],
    threshold: float,
    limit: int = 30,
) -> list[dict[str, Any]]:
    cases = build_prediction_summaries(records, threshold, include_gt=True)
    for item in cases:
        predicted = item["predicted_windows"]
        gt = item["gt_windows"]
        pred_coverage = sum(end - start for start, end in predicted)
        gt_coverage = sum(end - start for start, end in gt)
        if not predicted:
            category = "missed_all"
        elif item["mean_best_iou"] < 0.3:
            category = "low_overlap"
        elif gt_coverage > 0 and pred_coverage > 1.5 * gt_coverage:
            category = "over_prediction"
        elif gt_coverage > 0 and pred_coverage < 0.5 * gt_coverage:
            category = "under_prediction"
        else:
            category = "acceptable_overlap"
        item["error_category"] = category
        item["predicted_coverage"] = pred_coverage
        item["gt_coverage"] = gt_coverage

    cases.sort(key=lambda item: (item["mean_best_iou"], -item["gt_coverage"]))
    return cases[:limit]
