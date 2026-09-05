"""
Regenerate data/housing.csv from scikit-learn's California Housing dataset.

The CSV is committed to the repository, so this script only needs to be run
if the data file is missing or you want to refresh it.

Usage:
    python src/make_dataset.py
"""

from pathlib import Path

from sklearn.datasets import fetch_california_housing

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "housing.csv"


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = fetch_california_housing(as_frame=True)
    dataset.frame.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(dataset.frame):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
