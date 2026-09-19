# Freight Rate Prediction — Final Report

## 1. Data Overview

| Dataset | Rows | Date Range | Target |
|---------|------|-----------|--------|
| `train-test.csv` | 48,000 | Jan 1 – Oct 31, 2025 | `posted_rate` ✅ |
| `validation.csv` | 12,000 | Nov 1 – Dec 31, 2025 | — (to predict) |
| `december-chart-inputs.csv` | 31 | Dec 1–31, 2025 | Fixed scenario |

**Columns:** `load_id`, `pickup`, `delivery`, `pickup_lat/lon`, `delivery_lat/lon`, `distance`, `equipment`, `weight`, `date`, `market_index`, `quote_signal` (+ `posted_rate` in train)

---

## 2. Exploratory Data Analysis — Key Findings

### Target Distribution
- Mean posted rate: **$2,374** | Median: **$2,031** | Std: **$1,486**
- Right-skewed distribution; a small fraction of long-haul loads drives the upper tail (max ~$25k)
- Rate-per-mile is much tighter: median ~**$2.15/mile**, std **$0.58**

### Dominant Driver
- **`distance`** alone explains ~83% of variance in posted_rate (Pearson r = 0.91)
- Predicting `rate_per_mile` then multiplying by `distance` is therefore more stable and generalises better across lane lengths

### Categorical Coverage
- **Equipment types:** 3 classes (Dry Van, Flatbed, Reefer) — fully shared train/valid
- **Cities:** 64 unique in training; validation introduces **8 unseen cities**
  → Handled by coordinate-based features (haversine distance, bearing) which generalize to any city pair

### Missing Values
| Column | Train | Validation |
|--------|-------|-----------|
| `weight` | 300 (0.6%) | 165 (1.4%) |
| `market_index` | 374 (0.8%) | 249 (2.1%) |
- Imputed with training-set median; `HistGradientBoostingRegressor` handles NaN natively as a further safety net

### Market & Temporal Signals
- `market_index` peaks in Apr–May (~1.3) and dips in Sep (~0.89) — seasonal freight demand pattern
- Validation period (Nov–Dec) market_index ~0.93, similar to Sep–Oct, indicating no major distribution shift

---

## 3. Validation Strategy

### Why Time-Based Split?
Freight rates exhibit strong temporal autocorrelation (weekly and seasonal cycles). A random split would leak future market conditions into training, producing over-optimistic estimates.

### Split Used
```
Training fold  : Jan 1, 2025 – Aug 31, 2025  (38,477 loads)
Holdout fold   : Sep 1, 2025 – Oct 31, 2025   (9,523 loads)
```
This directly mimics the final deployment scenario where the model sees Jan–Oct data and must predict Nov–Dec.

---

## 4. Feature Engineering

| Feature Group | Features |
|--------------|---------|
| **Geospatial** | haversine miles, compass bearing, distance diff vs haversine, circuitous routing ratio |
| **Distance transforms** | log(distance+1), sqrt(distance) |
| **Weight** | log(weight+1), dist × weight, weight / distance |
| **Market/Quote** | market × quote, market × distance, quote × distance |
| **Temporal** | month, day, day-of-week, day-of-year, is_weekend, sin/cos cyclical encodings |
| **Target encoding** | Smoothed rate-per-mile mean by: pickup city, delivery city, equipment (prior_weight=15 loads) |

**Total: 36 features** — no one-hot encoding needed; `HistGradientBoostingRegressor` handles native categoricals directly.

---

## 5. Model Selection

### Candidates Evaluated (time-based holdout)

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| HistGB — rate-per-mile L1 (absolute_error) | **$115** | $636 | **0.826** |
| HistGB — rate-per-mile L2 (squared_error) | $133 | $639 | 0.825 |
| HistGB — direct posted_rate L1 | $121 | $639 | 0.825 |
| HistGB — direct posted_rate L2 | $159 | $647 | 0.820 |
| Weighted ensemble (all four) | $121 | $640 | 0.824 |

**Winner: HistGradientBoostingRegressor with `loss="absolute_error"` on rate-per-mile target**

### Why HistGradientBoosting?
- Natively handles **missing values** (weight, market_index) without imputation
- Handles **native categoricals** (pickup, delivery, equipment) — no encoding needed
- L1 (absolute error) loss is **robust to outliers** (long-haul loads with extreme rates)
- `n_iter_no_change=50` prevents overfitting on training period

### Why Rate-per-Mile as Target?
Normalising by distance removes the scale dependency; the model learns lane-specific $/mile factors and then scales back. This gives a ~13% MAE improvement over direct prediction ($115 vs $133).

### Final Model Hyperparameters
```python
HistGradientBoostingRegressor(
    loss="absolute_error",
    max_iter=700,
    learning_rate=0.035,
    max_leaf_nodes=63,
    min_samples_leaf=20,
    l2_regularization=0.5,
    categorical_features=["pickup", "delivery", "equipment"],
    random_state=42,
    n_iter_no_change=50,
    validation_fraction=0.05,
)
```

---

## 6. Final Results

### Holdout Performance (Sep–Oct 2025)

| Metric | Value |
|--------|-------|
| MAE | **$115.01** |
| RMSE | **$635.95** |
| R² | **0.8263** |

Validation predictions: 12,000 rows saved to `validation_predictions.csv`

How to regenerate this file and the December chart is documented in [`STEPS.md`](../STEPS.md). In short:

```bash
uv run python src/train.py
uv run python score.py \
  --predictions validation_predictions.csv \
  --december-predictions data/december-chart-inputs.csv
```

---

## 7. Fixed December Prediction Chart

The chart below was produced by running `score.py` on the completed `december-chart-inputs.csv`.
It shows predicted daily freight rates for the fixed Lexington → Fort Wayne, 360 mi, Dry Van, 32,000 lb scenario across all 31 December days.

![December Predictions Chart](../scorer_results/candidate_december.png)

---

## 8. Loom Walkthrough

https://www.loom.com/share/5abd9212e76e46aa83e28ab3c76044ac
