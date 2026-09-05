
# House Price Predictor

A Linear Regression model that predicts median house values in California from
census block-group features, built with scikit-learn. The project covers the full
supervised-learning workflow: data loading, an 80/20 train-test split, model
fitting, evaluation on unseen data, and diagnostic visualisation.

---

## Project Overview

The goal is to establish an honest, reproducible **baseline** for a regression
problem — not to chase the highest possible score. Linear Regression is the right
first model for this: it trains in milliseconds, its coefficients are directly
interpretable, and it sets the bar that any more complex model has to beat.

Every metric reported below is measured on a held-out test set the model never
saw during training.

### Dataset

The **California Housing** dataset (20,640 rows, 8 features), derived from the
1990 U.S. Census and distributed with scikit-learn. Each row is a census block
group, not an individual house.

| Feature | Description |
|---|---|
| `MedInc` | Median income in the block group (tens of thousands of dollars) |
| `HouseAge` | Median age of the houses |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average household occupancy |
| `Latitude` | Block group latitude |
| `Longitude` | Block group longitude |
| **`MedHouseVal`** | **Target** — median house value, in units of $100,000 |

---

## Project Structure

```
House-Price-Predictor/
├── data/
│   └── housing.csv                     # Raw dataset (20,640 rows)
├── notebooks/
│   └── 01_exploratory_analysis.ipynb   # Exploration and prototyping
├── src/
│   ├── make_dataset.py                 # Regenerates data/housing.csv
│   └── predict.py                      # Production training pipeline
├── reports/
│   └── figures/                        # Generated plots
│       ├── correlation_heatmap.png
│       └── actual_vs_predicted.png
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Installation

Requires Python 3.10 or newer.

```bash
# 1. Clone the repository
git clone https://github.com/harshon17fps/House-Price-Predictor.git
cd House-Price-Predictor

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

Run the full pipeline — train, evaluate, and write both figures:

```bash
python src/predict.py
```

Optional flags:

```bash
python src/predict.py --test-size 0.25    # change the holdout fraction
python src/predict.py --random-state 7    # change the split seed
python src/predict.py --no-plots          # metrics only, skip figures
```

Regenerate the dataset if `data/housing.csv` is missing:

```bash
python src/make_dataset.py
```

---

## Features

- **Reproducible 80/20 split** via `train_test_split` with a fixed `random_state`,
  so results are identical on every run and on any machine.
- **Modular, importable functions** — `load_data`, `split_data`, `train_model`,
  `evaluate_model` — each with a single responsibility and type hints. The notebook
  imports the same functions the pipeline uses, so exploration and production never
  drift apart.
- **Three evaluation metrics** (R², RMSE, MAE) computed strictly on held-out data.
- **Ranked coefficient table** showing which features drive the prediction.
- **Two diagnostic plots**, saved to `reports/figures/` at 150 DPI:
  - a correlation heatmap across all features and the target,
  - an actual-vs-predicted scatter with a fitted trendline and a `y = x`
    reference line for a hypothetical perfect model.
- **CLI interface** built with `argparse` for running experiments without editing code.

---

## Evaluation Metrics

Results from the default configuration (`test_size=0.2`, `random_state=42`),
measured on 4,128 unseen test rows:

| Metric | Score |
|---|---|
| **R² Score** | **0.5758** |
| **RMSE** | **0.7456** ($74,560) |
| **MAE** | **0.5332** ($53,320) |

### What R² actually means

**R² (the coefficient of determination) is the proportion of the variance in house
prices that the model successfully explains.** It answers: *how much better is this
model than simply guessing the average price every single time?*

The scale runs as follows:

- **R² = 1.0** — the model explains all variation; every prediction is exact.
- **R² = 0.0** — the model is no better than always predicting the mean price.
- **R² < 0.0** — the model is actively worse than predicting the mean.

An **R² of 0.5758 means the model explains roughly 58% of the variation in
California house prices.** The remaining 42% comes from factors the eight features
simply do not capture — school quality, crime rates, distance to the coast, the
condition of individual properties — plus the genuinely non-linear relationships
that a straight-line model cannot represent.

For a linear baseline on this dataset, ~0.58 is the expected result, and it is a
legitimate benchmark rather than a disappointing one.

### What RMSE means

**Root Mean Squared Error is the model's typical prediction error, expressed in
the same units as the target.** An RMSE of 0.7456 means predictions are off by
about **$74,560** on average. Because errors are squared before averaging, RMSE
penalises large misses more heavily than small ones — so it is the metric to watch
if occasional big errors are costly.

MAE (Mean Absolute Error, $53,320) is the plain average error with no squaring.
**RMSE being noticeably larger than MAE is itself a signal**: it tells you a
minority of predictions are badly wrong and dragging the squared average upward.

---

## Interpreting the Diagnostics

**Correlation heatmap.** `MedInc` is the dominant predictor (r = 0.69 with the
target) — income drives price more than any other single feature. Two pairs are
strongly collinear: `AveRooms`/`AveBedrms` (r = 0.85) and `Latitude`/`Longitude`
(r = -0.92). Multicollinearity does not hurt overall predictive accuracy, but it
does make individual coefficients unstable, which is why `AveBedrms` carries a
large positive weight while `AveRooms` carries a negative one. Neither should be
read as a standalone causal effect.

**Actual vs. predicted.** The trendline is flatter than the `y = x` reference
line, showing the classic regression-to-the-mean behaviour of a linear model: it
**over-predicts cheap homes and under-predicts expensive ones**. The vertical
stripe at 5.0 is a known artefact of the dataset — house values were capped at
$500,000 during collection, so those rows are censored rather than genuine.

---

## Known Limitations

- The $500,000 target cap censors the upper tail and distorts fit at the top end.
- Linear Regression cannot model the interaction between location and income,
  which is where most of the unexplained variance likely sits.
- Features are unscaled. This does not affect ordinary least squares predictions,
  but it does mean coefficient magnitudes are not directly comparable across
  features with different units.
- Data is from the 1990 census — historical, not a current market model.

## Roadmap

- [ ] Drop or explicitly model the capped rows
- [ ] Engineer location features (distance to coast, geographic clustering)
- [ ] Compare Ridge and Lasso to address multicollinearity
- [ ] Benchmark a RandomForest to quantify the non-linear headroom
- [ ] Add `pytest` unit tests for the data and evaluation functions
- [ ] Persist the fitted model with `joblib` for reuse

---

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `seaborn` · `Jupyter`

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 Harsh Lule.
