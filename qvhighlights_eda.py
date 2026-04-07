from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path


SPLIT_FILES = {
    "train": "highlight_train_release.jsonl",
    "val": "highlight_val_release.jsonl",
    "test": "highlight_test_release.jsonl",
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p25": None, "median": None, "p75": None, "max": None}
    ordered = sorted(values)

    def q(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        idx = p * (len(ordered) - 1)
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            return ordered[lo]
        return ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo)

    return {
        "min": ordered[0],
        "p25": q(0.25),
        "median": q(0.50),
        "p75": q(0.75),
        "max": ordered[-1],
    }


def merge_windows(windows: list[list[float]]) -> list[list[float]]:
    if not windows:
        return []
    ordered = sorted((float(s), float(e)) for s, e in windows)
    merged: list[list[float]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def summarize_split(name: str, rows: list[dict]) -> tuple[dict, list[dict], list[str]]:
    durations: list[float] = []
    windows_per_video: list[int] = []
    window_lengths: list[float] = []
    coverage_ratios: list[float] = []
    relevant_clip_counts: list[int] = []
    query_lengths: list[int] = []
    saliency_values: list[int] = []
    annotator_spreads: list[float] = []
    issues: list[str] = []

    unique_vids = set()
    unique_qids = set()

    for idx, row in enumerate(rows):
        qid = row.get("qid")
        vid = row.get("vid")
        duration = row.get("duration")
        query = row.get("query", "")
        unique_qids.add(qid)
        unique_vids.add(vid)

        if not isinstance(duration, (int, float)) or duration <= 0:
            issues.append(f"{name}:{idx}: invalid duration={duration!r}")
            continue

        duration = float(duration)
        durations.append(duration)
        query_lengths.append(len(str(query).split()))

        windows = row.get("relevant_windows", [])
        clip_ids = row.get("relevant_clip_ids", [])
        scores = row.get("saliency_scores", [])

        if windows:
            windows_per_video.append(len(windows))
            merged = merge_windows(windows)
            total_relevant = 0.0
            for window_id, (start, end) in enumerate(windows):
                start = float(start)
                end = float(end)
                window_lengths.append(max(0.0, end - start))
                if start < 0 or end <= start or end > duration:
                    issues.append(
                        f"{name}:{idx}: invalid window #{window_id}=[{start}, {end}] duration={duration}"
                    )
            for start, end in merged:
                total_relevant += max(0.0, min(end, duration) - max(start, 0.0))
            coverage_ratios.append(total_relevant / duration)

        if clip_ids:
            relevant_clip_counts.append(len(clip_ids))
            if scores and len(scores) != len(clip_ids):
                issues.append(
                    f"{name}:{idx}: len(saliency_scores)={len(scores)} != len(relevant_clip_ids)={len(clip_ids)}"
                )
        elif name != "test":
            issues.append(f"{name}:{idx}: empty relevant_clip_ids")

        for score_triplet in scores:
            if len(score_triplet) != 3:
                issues.append(f"{name}:{idx}: non-3-rater saliency score={score_triplet!r}")
                continue
            saliency_values.extend(int(v) for v in score_triplet)
            annotator_spreads.append(max(score_triplet) - min(score_triplet))

    summary = {
        "split": name,
        "rows": len(rows),
        "unique_qids": len(unique_qids),
        "unique_videos": len(unique_vids),
        "duration_sec": {
            **quantiles(durations),
            "mean": statistics.fmean(durations) if durations else None,
            "total_hours": sum(durations) / 3600,
        },
        "query_words": {
            **quantiles([float(v) for v in query_lengths]),
            "mean": statistics.fmean(query_lengths) if query_lengths else None,
        },
        "windows_per_annotated_video": {
            **quantiles([float(v) for v in windows_per_video]),
            "mean": statistics.fmean(windows_per_video) if windows_per_video else None,
        },
        "window_length_sec": {
            **quantiles(window_lengths),
            "mean": statistics.fmean(window_lengths) if window_lengths else None,
        },
        "relevant_coverage_ratio": {
            **quantiles(coverage_ratios),
            "mean": statistics.fmean(coverage_ratios) if coverage_ratios else None,
        },
        "relevant_clip_count": {
            **quantiles([float(v) for v in relevant_clip_counts]),
            "mean": statistics.fmean(relevant_clip_counts) if relevant_clip_counts else None,
        },
        "saliency_score_distribution": dict(sorted(Counter(saliency_values).items())),
        "annotator_spread": {
            **quantiles([float(v) for v in annotator_spreads]),
            "mean": statistics.fmean(annotator_spreads) if annotator_spreads else None,
        },
        "issue_count": len(issues),
    }

    examples = []
    for row in rows[:3]:
        examples.append(
            {
                "qid": row.get("qid"),
                "vid": row.get("vid"),
                "duration": row.get("duration"),
                "query": row.get("query"),
                "relevant_windows": row.get("relevant_windows", []),
            }
        )

    return summary, examples, issues


def write_markdown(summary: dict, examples: dict, issues: list[str], out_path: Path) -> None:
    lines = [
        "# QVHighlights EDA Summary",
        "",
        "Generated from local annotation JSONL files in `data/qvhighlights`.",
        "",
        "## Split Summary",
        "",
        "| split | rows | unique videos | hours | median duration, s | median windows | mean coverage | issues |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, stats in summary.items():
        duration = stats["duration_sec"]
        windows = stats["windows_per_annotated_video"]
        coverage = stats["relevant_coverage_ratio"]
        lines.append(
            "| {split} | {rows} | {videos} | {hours:.2f} | {dur_med:.1f} | {win_med} | {cov:.3f} | {issues} |".format(
                split=split,
                rows=stats["rows"],
                videos=stats["unique_videos"],
                hours=duration["total_hours"],
                dur_med=duration["median"] or 0,
                win_med="-" if windows["median"] is None else f"{windows['median']:.1f}",
                cov=coverage["mean"] or 0,
                issues=stats["issue_count"],
            )
        )

    lines += ["", "## Example Records", ""]
    for split, split_examples in examples.items():
        lines.append(f"### {split}")
        for item in split_examples:
            lines.append(
                f"- qid={item['qid']}, duration={item['duration']}, windows={item['relevant_windows']}, query={item['query']}"
            )
        lines.append("")

    lines += ["## Quality Checks", ""]
    if issues:
        lines.append(f"Found {len(issues)} potential issues. First 20:")
        for issue in issues[:20]:
            lines.append(f"- {issue}")
    else:
        lines.append("No annotation consistency issues found in the checked fields.")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_plots(summary: dict, rows_by_split: dict[str, list[dict]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    split_names = list(summary)
    row_counts = [summary[s]["rows"] for s in split_names]
    hours = [summary[s]["duration_sec"]["total_hours"] for s in split_names]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(split_names, row_counts, color="#4C78A8")
    axes[0].set_title("Rows by split")
    axes[0].set_ylabel("videos/queries")
    axes[1].bar(split_names, hours, color="#F58518")
    axes[1].set_title("Total duration by split")
    axes[1].set_ylabel("hours")
    fig.tight_layout()
    fig.savefig(out_dir / "split_sizes.png", dpi=180)
    plt.close(fig)

    train_val = [r for split in ("train", "val") for r in rows_by_split.get(split, [])]
    durations = [r["duration"] for r in train_val]
    window_lengths = [
        float(e) - float(s)
        for r in train_val
        for s, e in r.get("relevant_windows", [])
        if float(e) > float(s)
    ]
    coverage = []
    for r in train_val:
        merged = merge_windows(r.get("relevant_windows", []))
        if merged:
            coverage.append(sum(e - s for s, e in merged) / float(r["duration"]))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].hist(durations, bins=30, color="#4C78A8")
    axes[0].set_title("Video duration")
    axes[0].set_xlabel("seconds")
    axes[1].hist(window_lengths, bins=30, color="#54A24B")
    axes[1].set_title("Relevant window length")
    axes[1].set_xlabel("seconds")
    axes[2].hist(coverage, bins=30, color="#E45756")
    axes[2].set_title("Relevant-time coverage")
    axes[2].set_xlabel("ratio")
    fig.tight_layout()
    fig.savefig(out_dir / "duration_window_coverage.png", dpi=180)
    plt.close(fig)

    score_counter: Counter[int] = Counter()
    spread_counter: Counter[int] = Counter()
    for r in train_val:
        for triplet in r.get("saliency_scores", []):
            score_counter.update(int(v) for v in triplet)
            spread_counter[max(triplet) - min(triplet)] += 1

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    score_keys = list(range(5))
    axes[0].bar(score_keys, [score_counter[k] for k in score_keys], color="#72B7B2")
    axes[0].set_title("Saliency scores")
    axes[0].set_xlabel("score")
    axes[0].set_ylabel("rater labels")
    spread_keys = list(range(5))
    axes[1].bar(spread_keys, [spread_counter[k] for k in spread_keys], color="#B279A2")
    axes[1].set_title("Annotator spread")
    axes[1].set_xlabel("max(score)-min(score)")
    axes[1].set_ylabel("clips")
    fig.tight_layout()
    fig.savefig(out_dir / "saliency_scores.png", dpi=180)
    plt.close(fig)


def write_csv_summary(summary: dict, out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split",
                "rows",
                "unique_videos",
                "total_hours",
                "median_duration_sec",
                "mean_windows",
                "median_window_length_sec",
                "mean_coverage_ratio",
                "issue_count",
            ]
        )
        for split, stats in summary.items():
            writer.writerow(
                [
                    split,
                    stats["rows"],
                    stats["unique_videos"],
                    f"{stats['duration_sec']['total_hours']:.6f}",
                    stats["duration_sec"]["median"],
                    stats["windows_per_annotated_video"]["mean"],
                    stats["window_length_sec"]["median"],
                    stats["relevant_coverage_ratio"]["mean"],
                    stats["issue_count"],
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/qvhighlights"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/qvhighlights_eda"))
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows_by_split = {
        split: load_jsonl(args.data_dir / filename) for split, filename in SPLIT_FILES.items()
    }

    summary: dict[str, dict] = {}
    examples: dict[str, list[dict]] = {}
    all_issues: list[str] = []
    for split, rows in rows_by_split.items():
        split_summary, split_examples, split_issues = summarize_split(split, rows)
        summary[split] = split_summary
        examples[split] = split_examples
        all_issues.extend(split_issues)

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "examples.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "issues.txt").write_text("\n".join(all_issues) + "\n", encoding="utf-8")
    write_csv_summary(summary, args.out_dir / "summary.csv")
    write_markdown(summary, examples, all_issues, args.out_dir / "summary.md")

    if args.plots:
        save_plots(summary, rows_by_split, args.out_dir)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nArtifacts written to {args.out_dir}")


if __name__ == "__main__":
    main()
