# Video Highlight Detection

Проект по поиску хайлайтов в видео на QVHighlights.

## Stage 1: EDA

Артефакты первого этапа:

- `stage1_highlight_detection_report.pdf` - PDF-отчёт.
- `qvhighlights_eda.ipynb` - выполненный EDA notebook.
- `qvhighlights_eda.py` - воспроизводимый EDA-скрипт.
- `outputs/qvhighlights_eda/` - таблицы, JSON summary и PNG-графики.
- `data/qvhighlights/` - JSONL-аннотации QVHighlights.

Воспроизвести EDA:

```bash
python qvhighlights_eda.py --plots
```

## Stage 2: ML Pipeline

Raw-video архив QVHighlights весит около 133 GB, а скачивание исходных YouTube
роликов нестабильно из-за bot-check и удаленных видео. Поэтому основной
pipeline использует официальные Moment-DETR features: готовые CLIP video
features и CLIP text features. Модель обучается на визуальных признаках, но не
требует хранения сырых `.mp4`.

Финальный основной эксперимент:

```bash
python train_moment_features_model.py \
  --max-train-videos 1000 \
  --max-val-videos 300 \
  --classifier sgd \
  --max-iter 30 \
  --video-feature-dirs clip_features
```

Финальные артефакты:

- `outputs/stage2_moment_features/metrics.json`
- `outputs/stage2_moment_features/threshold_sweep.csv`
- `outputs/stage2_moment_features/val_predictions_examples.json`
- `outputs/stage2_moment_features/error_analysis.json`
- `outputs/stage2_moment_features/moment_features_highlight_model.joblib`

Финальные метрики на subset `1000 train / 300 val`:

```text
clip_precision: 0.517
clip_recall:    0.674
clip_f1:        0.585
clip_AP:        0.564
segment_f1@0.3: 0.259
segment_f1@0.5: 0.182
mean_best_iou:  0.423
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data And Features

Аннотации уже лежат в:

```text
data/qvhighlights/
  highlight_train_release.jsonl
  highlight_val_release.jsonl
  highlight_test_release.jsonl
```

Moment-DETR features скачиваются через Hugging Face mirror:

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="dpaul06/QVHighlights_preprocessed",
    filename="moment_detr_features.tar.gz",
    repo_type="dataset",
    local_dir=".",
    local_dir_use_symlinks=False,
)
PY
tar -xf moment_detr_features.tar.gz
rm moment_detr_features.tar.gz
```

После распаковки ожидается структура:

```text
features/
  clip_features/<vid>.npz
  clip_text_features/qid<id>.npz
```

Проверить coverage:

```bash
python scripts/inspect_moment_features.py
```

Ожидаемый результат:

```text
clip video coverage: 10148/10148
text feature coverage: 10310/10310
```

## Demo

Для защиты подготовлен notebook:

```text
demo_highlight_model.ipynb
```

Он загружает сохраненную модель из `outputs/stage2_moment_features/`, считает
scores для нескольких validation-видео, показывает predicted windows, ground
truth windows и timeline-график для выбранного примера.

## Report

Шаблон итогового отчёта:

```text
reports/stage2_report_template.md
```
