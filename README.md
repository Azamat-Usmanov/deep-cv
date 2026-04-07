# Video Highlight Detection - Stage 1

Артефакты для рубежного контроля:

- `report/stage1_highlight_detection_report.pdf` - PDF-отчёт.
- `report/stage1_highlight_detection_report.md` - исходник отчёта.
- `notebooks/qvhighlights_eda.ipynb` - выполненный EDA notebook по QVHighlights.
- `scripts/qvhighlights_eda.py` - воспроизводимый EDA-скрипт.
- `outputs/qvhighlights_eda/` - таблицы, JSON summary и PNG-графики.
- `data/qvhighlights/` - публичные JSONL-аннотации QVHighlights из репозитория Moment-DETR.
- `requirements.txt` - Python-зависимости для воспроизведения в чистом окружении.

Команды воспроизведения в этом окружении:

```bash
MPLCONFIGDIR=.mplconfig TMPDIR=.tmp /Library/Frameworks/Python.framework/Versions/3.9/bin/python3 scripts/qvhighlights_eda.py --plots
cd notebooks
MPLCONFIGDIR=../.mplconfig TMPDIR=../.tmp jupyter nbconvert --to notebook --execute qvhighlights_eda.ipynb --inplace
cd ..
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 scripts/render_report_pdf.py
```

Источник аннотаций QVHighlights:

```bash
curl -L -o data/qvhighlights/highlight_train_release.jsonl https://raw.githubusercontent.com/jayleicn/moment_detr/main/data/highlight_train_release.jsonl
curl -L -o data/qvhighlights/highlight_val_release.jsonl https://raw.githubusercontent.com/jayleicn/moment_detr/main/data/highlight_val_release.jsonl
curl -L -o data/qvhighlights/highlight_test_release.jsonl https://raw.githubusercontent.com/jayleicn/moment_detr/main/data/highlight_test_release.jsonl
```
