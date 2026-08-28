"""Load and validate the one-time local WCS analysis table.

The scientific pipelines deliberately read this file only; they never
re-download the WCS.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "wcs_responses.csv.gz"


def load_dataset():
    if not DATASET.exists():
        raise FileNotFoundError("Canonical dataset missing. Run `python setup_dataset.py` once.")
    df = pd.read_csv(DATASET)
    return df


def validate_dataset(df):
    if df.language_id.nunique() != 110:
        raise ValueError(f"Dataset is incomplete: {df.language_id.nunique()} languages, expected 110.")
    if df.chip_id.nunique() != 330:
        raise ValueError(f"Dataset is incomplete: {df.chip_id.nunique()} chips, expected 330.")
    counts = df.groupby("language_id")["chip_id"].nunique()
    if (counts != 330).any():
        raise ValueError(f"At least one language does not contain all 330 chips.")
    return True
