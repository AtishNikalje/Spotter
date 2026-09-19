# Freight Rate Prediction Challenge

## Project Structure
- `data/`: Contains datasets and outputs (`validation_predictions.csv`, `december_chart_inputs.csv`)
- `src/`: Source code for feature engineering (`features/`) and model training (`train.py`)
- `notebooks/`: Jupyter notebooks for exploratory data analysis
- `reports/`: Markdown report detailing the approach and findings
- `scorer_results/`: Stores the output chart from the evaluation script

## Setup Instructions

We use `uv` and `Docker` for standard, reproducible environments.

### Option 1: Using Docker (Recommended)

To run the full training pipeline and scoring using Docker:
```bash
docker-compose run runner
```

To explore the data using Jupyter Notebooks:
```bash
docker-compose up jupyter
```
Then navigate to `http://localhost:8888` in your browser.

### Option 2: Using uv locally

1. Install [uv](https://github.com/astral-sh/uv).
2. Sync the environment:
   ```bash
   uv sync
   ```
3. Run the training script:
   ```bash
   uv run python src/train.py
   ```
4. Run the scoring script:
   ```bash
   uv run python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
   ```
