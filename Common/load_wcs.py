from __future__ import annotations

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DIR, WCS_USE_FIXTURE, EXPECTED_WCS_LANGUAGES, EXPECTED_WCS_CHIPS
from download_wcs import ensure_wcs_data


def _find_raw_file(name: str) -> Path:
    for root in [RAW_DIR / "wcs_archive", RAW_DIR]:
        path = root / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not locate {name} under {RAW_DIR}")


def _assert_not_wrong_html(path: Path, allow_html: bool = False) -> None:
    preview = path.read_bytes()[:5000].lstrip().lower()
    if not allow_html and (preview.startswith(b"<!doctype") or preview.startswith(b"<html") or preview.startswith(b"<?xml")):
        raise ValueError(f"{path} contains HTML/XML rather than the expected WCS text data.")


def load_term() -> pd.DataFrame:
    path = _find_raw_file("term.txt")
    _assert_not_wrong_html(path)
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["language_id", "speaker_id", "chip_id", "term"],
        dtype={"language_id": "Int64", "speaker_id": "Int64", "chip_id": "Int64", "term": "string"},
        comment="#",
        keep_default_na=False,
    )
    df["term"] = df["term"].astype("string").str.strip()
    for c in ["language_id", "speaker_id", "chip_id"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["language_id", "speaker_id", "chip_id"])
    df = df[df["term"].notna() & ~df["term"].isin(["", "*", "?"])]
    df[["language_id", "speaker_id", "chip_id"]] = df[["language_id", "speaker_id", "chip_id"]].astype(int)
    df = df[(df["language_id"].between(1, 110)) & (df["chip_id"].between(1, 330))]
    return df.reset_index(drop=True)


def load_speakers() -> pd.DataFrame:
    path = _find_raw_file("spkr.txt")
    _assert_not_wrong_html(path)
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                rows.append(parts[:4])
    df = pd.DataFrame(rows, columns=["language_id", "speaker_id", "age", "sex"])
    for c in ["language_id", "speaker_id"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["age"] = pd.to_numeric(df["age"].replace(["*", "?", ""], np.nan), errors="coerce")
    df["sex"] = df["sex"].astype("string").str.upper()
    df = df.dropna(subset=["language_id", "speaker_id"]).copy()
    df[["language_id", "speaker_id"]] = df[["language_id", "speaker_id"]].astype(int)
    return df


def load_languages() -> pd.DataFrame:
    """Return WCS language IDs, optionally enriched with lang.txt metadata.

    The statistical analyses only require stable language IDs.  The historical
    WCS language table is HTML and may be unreachable on restricted networks,
    so its absence must never block the real-data analysis.
    """
    try:
        path = _find_raw_file("lang.txt")
    except FileNotFoundError:
        term = load_term()
        return pd.DataFrame({"language_id": sorted(term["language_id"].unique())})

    raw = path.read_text(encoding="utf-8", errors="replace")
    records = []
    for m in re.finditer(r"<tr[^>]*>(.*?)</tr>", raw, flags=re.I | re.S):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", m.group(1), flags=re.I | re.S)
        cleaned = [re.sub(r"<[^>]+>", " ", c).strip() for c in cells]
        cleaned = [re.sub(r"\s+", " ", c) for c in cleaned]
        if cleaned and cleaned[0].isdigit():
            records.append(cleaned)

    if records:
        max_len = max(len(r) for r in records)
        rows = [r + [None] * (max_len - len(r)) for r in records]
        cols = ["language_id"] + [f"lang_field_{i}" for i in range(2, max_len + 1)]
        df = pd.DataFrame(rows, columns=cols)
        df["language_id"] = pd.to_numeric(df["language_id"], errors="coerce")
        return df.dropna(subset=["language_id"]).astype({"language_id": int})

    term = load_term()
    return pd.DataFrame({"language_id": sorted(term["language_id"].unique())})


def load_mapping() -> pd.DataFrame:
    path = _find_raw_file("cnum-vhcm-lab-new.txt")
    _assert_not_wrong_html(path)
    raw = path.read_text(encoding="utf-8", errors="replace")
    rows = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = re.split(r"\s+", s)
        if parts and parts[0].lower() == "cnum":
            continue
        if len(parts) >= 9:
            rows.append(parts[:9])

    if not rows:
        raise ValueError("No rows could be parsed from cnum-vhcm-lab-new.txt")

    df = pd.DataFrame(rows, columns=["chip_id", "V", "H", "C", "MunH", "MunV", "L_star", "a_star", "b_star"])
    for c in ["chip_id", "L_star", "a_star", "b_star"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["chip_id", "L_star", "a_star", "b_star"]).copy()
    df["chip_id"] = df["chip_id"].astype(int)
    df = df[["chip_id", "L_star", "a_star", "b_star"]].sort_values("chip_id").reset_index(drop=True)
    if len(df) != 330 or df["chip_id"].nunique() != 330:
        raise ValueError(f"Expected 330 unique WCS chips; found {len(df)} rows and {df['chip_id'].nunique()} IDs.")
    return df


def validate_wcs_coverage(term: pd.DataFrame, mapping: pd.DataFrame) -> dict[str, int]:
    """Validate that a real-data run contains the complete WCS naming task.

    This is deliberately strict for real runs: the analysis must see all 110
    WCS languages and all 330 physical colour chips. A fixture run is allowed
    to be smaller and is validated separately by the tests.
    """
    n_languages = int(term["language_id"].nunique())
    n_chips = int(term["chip_id"].nunique())
    if not WCS_USE_FIXTURE:
        if n_languages != EXPECTED_WCS_LANGUAGES:
            raise ValueError(
                f"Incomplete real WCS term data: expected {EXPECTED_WCS_LANGUAGES} languages, "
                f"found {n_languages}. No sampling is permitted."
            )
        if n_chips != EXPECTED_WCS_CHIPS:
            raise ValueError(
                f"Incomplete real WCS term data: expected {EXPECTED_WCS_CHIPS} chips, "
                f"found {n_chips}. No sampling is permitted."
            )
        chip_counts = term.groupby("language_id")["chip_id"].nunique()
        bad = chip_counts[chip_counts != EXPECTED_WCS_CHIPS]
        if not bad.empty:
            raise ValueError(
                "Incomplete real WCS data: some languages do not contain all 330 chips: "
                + ", ".join(f"{int(k)}->{int(v)}" for k, v in bad.items())
            )
        if len(mapping) != EXPECTED_WCS_CHIPS:
            raise ValueError(
                f"Incomplete colour mapping: expected {EXPECTED_WCS_CHIPS} chips, found {len(mapping)}."
            )
    return {"n_languages": n_languages, "n_chips": n_chips, "n_rows": int(len(term))}


def load_all() -> dict[str, pd.DataFrame]:
    ensure_wcs_data()
    term = load_term()
    mapping = load_mapping()
    coverage = validate_wcs_coverage(term, mapping)
    return {
        "term": term,
        "speakers": load_speakers(),
        "languages": load_languages(),
        "mapping": mapping,
        "coverage": pd.DataFrame([coverage]),
    }


def build_wide_arrays(term: pd.DataFrame):
    pivot = term.pivot_table(index=["language_id", "speaker_id"], columns="chip_id", values="term", aggfunc="first")
    return pivot.sort_index(axis=1)
