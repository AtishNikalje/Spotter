# Steps: generate model output and scorer chart

This is the exact sequence used to produce the assessment deliverables:

| Output | Path | What it is |
|--------|------|------------|
| Model output | `validation_predictions.csv` | 12,000 rows: `load_id,predicted_rate` |
| December inputs (filled) | `data/december-chart-inputs.csv` | 31-day Lexington → Fort Wayne scenario with `predicted_rate` |
| Scorer chart | `scorer_results/candidate_december.png` | Official `score.py` plot |
| Technical report | `reports/final_report.md` / `reports/final_report.pdf` | Validation split, model, results, chart |

---

## 1. Environment

From the repo root:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

Docker alternative (no local Python required):

```bash
docker compose run runner
```

The `runner` service trains the model, writes `validation_predictions.csv`, fills December rates, then runs `score.py`.

---

## 2. Train and write the model output file

```bash
uv run python src/train.py
```

What `src/train.py` does:

1. **Load data** — `data/train-test.csv` (48,000 labeled), `data/validation.csv` (12,000 unlabeled), `data/december-chart-inputs.csv` (31 days).
2. **Engineer features** — dates, haversine/bearing, distance and weight transforms, market/quote interactions, cyclical encodings. Categoricals stay as pandas `category` for HistGB.
3. **Time-based holdout** — train Jan–Aug 2025, hold out Sep–Oct 2025. Target is `rate_per_mile = posted_rate / distance`. Target encodings for pickup, delivery, and equipment are fit on the training fold only.
4. **Retrain on all labeled data** (Jan–Oct) with the same recipe.
5. **Infer** on every validation `load_id` and on the 31 December rows.
6. **Write files**
   - Fill `data/validation-predictions-template.csv` → save as `validation_predictions.csv`
   - Write `predicted_rate` onto `data/december-chart-inputs.csv`

Expected console line:

```text
Saved: validation_predictions.csv  (12000 rows)
```

---

## 3. Validate files and generate the December chart

```bash
uv run python score.py \
  --predictions validation_predictions.csv \
  --december-predictions data/december-chart-inputs.csv
```

Expected console:

```text
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results/candidate_december.png
```

`score.py` is the official scorer. It checks column order, 12,000 unique `TE-######` IDs, positive rates, and the fixed December scenario (Lexington → Fort Wayne, 360 mi, Dry Van, 32,000 lb, one row per day 1–31 Dec 2025).

---

## 4. Document results

- Narrative, metrics, and chart: [`reports/final_report.md`](reports/final_report.md)
- Same report as Word (assessment format): [`reports/final_report.docx`](reports/final_report.docx)
- Setup and architecture: [`readme.md`](readme.md)

Regenerate the Word report after the chart exists:

```bash
pandoc reports/final_report.md -o reports/final_report.docx --resource-path=reports:.
```

If you have a LaTeX engine installed, you can also emit PDF:

```bash
pandoc reports/final_report.md -o reports/final_report.pdf --resource-path=reports:.
```

---

## 5. Optional: dashboard

```bash
uv run streamlit run dashboard.py
# http://localhost:8501 (or 8502 if 8501 is already in use)
```

Use the Data tab to browse and download `validation_predictions.csv`, and the December Forecast tab to view the scorer chart.

---

## File format (model output)

```text
load_id,predicted_rate
TE-000001,<positive float>
...
TE-012000,<positive float>
```

Exactly 12,000 rows, IDs `TE-000001` through `TE-012000`, no missing or non-positive rates.
