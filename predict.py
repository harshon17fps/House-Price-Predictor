"""
House Price Predictor - Linear Regression baseline.

Trains a linear regression model on the California Housing dataset,
evaluates it on a held-out test split, and renders diagnostic plots.

Usage:
    python src/predict.py
    python src/predict.py --test-size 0.25 --random-state 7
    python src/predict.py --no-plots
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "housing.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

TARGET_COLUMN = "MedHouseVal"
TEST_SIZE = 0.2
RANDOM_STATE = 42

sns.set_theme(style="whitegrid", palette="deep")


@dataclass(frozen=True)
class EvaluationReport:
    """Container for the metrics produced on the held-out test set."""

    r2: float
    rmse: float
    mae: float

    def __str__(self) -> str:
        return (
            f"R2 Score : {self.r2:.4f}\n"
            f"RMSE     : {self.rmse:.4f}\n"
            f"MAE      : {self.mae:.4f}"
        )


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the housing dataset from disk and drop incomplete rows."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python src/make_dataset.py` first."
        )
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Expected target column '{TARGET_COLUMN}' in {path}.")
    return df.dropna().reset_index(drop=True)


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into 80% training / 20% testing by default."""
    features = df.drop(columns=[TARGET_COLUMN])
    target = df[TARGET_COLUMN]
    return train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Fit an ordinary least squares linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[EvaluationReport, np.ndarray]:
    """Score the fitted model on unseen data and return metrics + predictions."""
    predictions = model.predict(X_test)
    report = EvaluationReport(
        r2=r2_score(y_test, predictions),
        rmse=root_mean_squared_error(y_test, predictions),
        mae=mean_absolute_error(y_test, predictions),
    )
    return report, predictions


def coefficient_table(model: LinearRegression, feature_names: pd.Index) -> pd.DataFrame:
    """Rank features by the magnitude of their learned coefficient."""
    coefficients = pd.DataFrame(
        {"feature": feature_names, "coefficient": model.coef_}
    )
    coefficients["abs_coefficient"] = coefficients["coefficient"].abs()
    return coefficients.sort_values("abs_coefficient", ascending=False).drop(
        columns="abs_coefficient"
    )


# --------------------------------------------------------------------------- #
# Visualisations
# --------------------------------------------------------------------------- #


def plot_correlation_heatmap(
    df: pd.DataFrame, output_dir: Path = FIGURES_DIR
) -> Path:
    """Render a correlation heatmap across every numeric feature and the target."""
    output_dir.mkdir(parents=True, exist_ok=True)
    correlations = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        correlations,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson correlation"},
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, pad=12)
    fig.tight_layout()

    path = output_dir / "correlation_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_actual_vs_predicted(
    y_test: pd.Series,
    predictions: np.ndarray,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Scatter actual against predicted values with a fitted trendline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    actual = np.asarray(y_test, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(x=actual, y=predictions, alpha=0.25, s=18, edgecolor=None, ax=ax)

    # Least-squares trendline through the (actual, predicted) cloud.
    slope, intercept = np.polyfit(actual, predictions, deg=1)
    line_x = np.linspace(actual.min(), actual.max(), 200)
    ax.plot(
        line_x,
        slope * line_x + intercept,
        color="crimson",
        linewidth=2,
        label=f"Trendline (y = {slope:.2f}x + {intercept:.2f})",
    )

    # Reference line for a hypothetical perfect model.
    ax.plot(
        line_x,
        line_x,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Perfect prediction (y = x)",
    )

    ax.set_xlabel("Actual median house value ($100,000s)")
    ax.set_ylabel("Predicted median house value ($100,000s)")
    ax.set_title("Actual vs. Predicted Values", fontsize=14, pad=12)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()

    path = output_dir / "actual_vs_predicted.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=DATA_PATH, help="Path to the housing CSV."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Fraction of the data held out for testing (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Seed for a reproducible split (default: 42).",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip figure generation."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_data(args.data)
    print(f"Loaded {len(df):,} rows and {df.shape[1]} columns from {args.data.name}\n")

    X_train, X_test, y_train, y_test = split_data(
        df, test_size=args.test_size, random_state=args.random_state
    )
    print(f"Training rows: {len(X_train):,}    Testing rows: {len(X_test):,}\n")

    model = train_model(X_train, y_train)
    report, predictions = evaluate_model(model, X_test, y_test)

    print("=" * 46)
    print("MODEL PERFORMANCE (held-out test set)")
    print("=" * 46)
    print(report)
    print("=" * 46)
    print("\nFeature coefficients (ranked by magnitude):")
    print(coefficient_table(model, X_train.columns).to_string(index=False))

    if not args.no_plots:
        heatmap = plot_correlation_heatmap(df)
        scatter = plot_actual_vs_predicted(y_test, predictions)
        print(f"\nSaved figure: {heatmap.relative_to(PROJECT_ROOT)}")
        print(f"Saved figure: {scatter.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
