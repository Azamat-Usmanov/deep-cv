# QVHighlights EDA Summary

Generated from local annotation JSONL files in `data/qvhighlights`.

## Split Summary

| split | rows | unique videos | hours | median duration, s | median windows | mean coverage | issues |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 7218 | 7100 | 299.84 | 150.0 | 1.0 | 0.298 | 0 |
| val | 1550 | 1519 | 64.42 | 150.0 | 1.0 | 0.281 | 0 |
| test | 1542 | 1529 | 64.06 | 150.0 | - | 0.000 | 0 |

## Example Records

### train
- qid=9769, duration=150, windows=[[72, 82], [84, 94], [96, 106], [108, 118], [120, 130], [136, 142], [144, 146]], query=some military patriots takes us through their safety procedures and measures.
- qid=10016, duration=150, windows=[[96, 114]], query=Man in baseball cap eats before doing his interview.
- qid=10078, duration=150, windows=[[48, 50], [76, 120], [122, 138], [140, 146]], query=A man in a white shirt discusses the right to have and carry firearms.

### val
- qid=2579, duration=150, windows=[[82, 150]], query=A girl and her mother cooked while talking with each other on facetime.
- qid=5071, duration=150, windows=[[118, 136]], query=A woman sitting in front of a desk wearing headphones and using her laptop
- qid=5342, duration=150, windows=[[56, 76], [96, 150]], query=An Asian woman wearing a Boston t-shirt is in her home talking.

### test
- qid=3158, duration=150, windows=[], query=A video covering hill and water from a boat
- qid=7920, duration=150, windows=[], query=Woman talks to the camera out the window of a car.
- qid=8039, duration=150, windows=[], query=Vlogger walks around a large hotel pool.

## Quality Checks

No annotation consistency issues found in the checked fields.
