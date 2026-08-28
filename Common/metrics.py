from __future__ import annotations

import math
import numpy as np
import pandas as pd

INVALID_TERMS = {"", "*", "?", "nan", "None"}
WCS_N_CHIPS = 330
RAREFIED_TARGET = int(math.floor(0.95 * WCS_N_CHIPS))  # 313 valid responses


def clean_terms(term: pd.DataFrame) -> pd.DataFrame:
    out = term.copy()
    out["term"] = out["term"].astype(str)
    out = out[~out["term"].isin(INVALID_TERMS)].copy()
    out = out.dropna(subset=["language_id", "speaker_id", "chip_id"])
    out["language_id"] = out["language_id"].astype(int)
    out["speaker_id"] = out["speaker_id"].astype(int)
    out["chip_id"] = out["chip_id"].astype(int)
    return out.drop_duplicates(["language_id", "speaker_id", "chip_id"], keep="first")


def entropy(p, base=math.e):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    if not np.isclose(p.sum(), 1.0):
        p = p / p.sum()
    h = float(-(p * np.log(p)).sum())
    return h if base == math.e else h / np.log(base)


def miller_madow_entropy(counts, base=math.e):
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    n = counts.sum()
    if n <= 0:
        return np.nan
    p = counts / n
    h = float(-(p * np.log(p)).sum()) + (len(p) - 1) / (2 * n)
    return h if base == math.e else h / np.log(base)


def miller_madow(h, n, k):
    if n <= 0 or k <= 1:
        return float(h)
    return float(h + (k - 1) / (2 * n))


def miller_madow_mutual_information(counts_2d: np.ndarray) -> float:
    """First-order Miller-Madow bias-corrected mutual information.

    Derivation: apply the Miller-Madow entropy correction to
    I(X;Y)=H(X)+H(Y)-H(X,Y). For r and c non-empty marginals and q
    non-empty joint cells, the correction is (r+c-q-1)/(2n).
    """
    table = np.asarray(counts_2d, dtype=float)
    n = table.sum()
    if n <= 0:
        return np.nan
    pxy = table / n
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    nz = pxy > 0
    denom = px[:, None] * py[None, :]
    mi_mle = float(np.sum(pxy[nz] * np.log(pxy[nz] / denom[nz])))
    r = int(np.count_nonzero(px))
    c = int(np.count_nonzero(py))
    rc = int(np.count_nonzero(pxy))
    correction = (r + c - rc - 1) / (2 * n)
    return float(max(0.0, mi_mle + correction))


def _expected_rarefied_richness(counts: np.ndarray, target_n: int) -> float:
    """Exact expected number of distinct observed categories in a without-replacement sample."""
    counts = np.asarray(counts, dtype=int)
    n = int(counts.sum())
    m = int(target_n)
    if n <= 0 or m <= 0 or m > n:
        return np.nan
    log_den = math.lgamma(n + 1) - math.lgamma(m + 1) - math.lgamma(n - m + 1)
    expected = 0.0
    for c in counts[counts > 0]:
        if n - int(c) < m:
            prob_absent = 0.0
        else:
            a = n - int(c)
            log_num = math.lgamma(a + 1) - math.lgamma(m + 1) - math.lgamma(a - m + 1)
            prob_absent = math.exp(log_num - log_den)
        expected += 1.0 - prob_absent
    return float(expected)


def speaker_statistics(term: pd.DataFrame, rarefaction_target: int = RAREFIED_TARGET) -> pd.DataFrame:
    """Speaker-level repertoire, observation count, Miller-Madow entropy and rarefied richness."""
    valid = clean_terms(term)
    rows = []
    for (lang, spk), g in valid.groupby(["language_id", "speaker_id"], sort=True):
        counts = g["term"].value_counts().to_numpy(dtype=int)
        n_obs = int(g["chip_id"].nunique())
        k = int(len(counts))
        h_mm = float(miller_madow_entropy(counts))
        n_eff = float(np.exp(h_mm))
        k_rare = _expected_rarefied_richness(counts, rarefaction_target)
        rows.append({
            "language_id": int(lang),
            "speaker_id": int(spk),
            "group_id": f"{int(lang)}:{int(spk)}",
            "speaker_K_raw": k,
            "speaker_K_rarefied": k_rare,
            "n_observed_chips": n_obs,
            "n_missing_chips": int(WCS_N_CHIPS - n_obs),
            "n_valid_pairs": int(n_obs * (n_obs - 1) // 2),
            "H_label_MillerMadow": h_mm,
            "effective_colour_categories": n_eff,
            "rarefaction_target": int(rarefaction_target),
            "rarefaction_eligible": bool(n_obs >= rarefaction_target),
        })
    return pd.DataFrame(rows)


def consensus_vocabulary(term, min_speaker_fraction=0.20, min_chips_per_term=3):
    valid = clean_terms(term)
    n_speakers = valid.groupby("language_id")["speaker_id"].nunique().rename("n_speakers")
    usage = (valid.groupby(["language_id", "term", "chip_id"])["speaker_id"]
             .nunique().rename("n_speakers_using").reset_index()
             .merge(n_speakers.reset_index(), on="language_id"))
    usage["fraction_speakers"] = usage["n_speakers_using"] / usage["n_speakers"]
    consensus = usage[usage["fraction_speakers"] >= min_speaker_fraction]
    summary = (consensus.groupby(["language_id", "term"])
               .agg(n_consensus_chips=("chip_id", "nunique"),
                    max_fraction=("fraction_speakers", "max"))
               .reset_index())
    summary = summary[summary["n_consensus_chips"] >= min_chips_per_term].copy()
    vocab = summary.groupby("language_id")["term"].nunique().rename("K").reset_index()
    return vocab, summary


def speaker_vocabulary(term):
    return speaker_statistics(term)[[
        "language_id", "speaker_id", "speaker_K_raw", "speaker_K_rarefied",
        "n_observed_chips", "n_missing_chips", "rarefaction_eligible"
    ]].rename(columns={"speaker_K_raw": "speaker_K"})


def language_vocabulary(term):
    """Raw distinct-label count; retained for diagnostics only."""
    valid = clean_terms(term)
    return (valid.groupby("language_id")["term"]
            .nunique().rename("language_K").reset_index())


def language_prevalence_vocabulary(term, min_speaker_fraction=0.20, min_chips_per_term=3):
    """Speaker-prevalence-normalized language vocabulary used as the primary predictor."""
    vocab, _ = consensus_vocabulary(
        term,
        min_speaker_fraction=min_speaker_fraction,
        min_chips_per_term=min_chips_per_term,
    )
    return vocab.rename(columns={"K": "language_K"})


def bootstrap_mean_ci(values, n_boot=2000, alpha=0.05, random_state=20260822):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(random_state)
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        means[b] = rng.choice(x, size=len(x), replace=True).mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def _pair_setup(mapping):
    chipmeta = mapping[["chip_id", "L_star", "a_star", "b_star"]].drop_duplicates("chip_id").sort_values("chip_id")
    ids = chipmeta.chip_id.to_numpy(dtype=int)
    X = chipmeta[["L_star", "a_star", "b_star"]].to_numpy(float)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    iu, ju = np.triu_indices(len(ids), k=1)
    return ids, iu, ju, D[iu, ju]


def speaker_resolution_curves(term, mapping, deltas, min_valid_pairs=1):
    valid = clean_terms(term)
    ids, iu, ju, pair_dist = _pair_setup(mapping)
    rows = []
    for (lang, spk), g in valid.groupby(["language_id", "speaker_id"], sort=True):
        labels = g.set_index("chip_id")["term"].reindex(ids).to_numpy()
        ok = pd.notna(labels[iu]) & pd.notna(labels[ju])
        d = pair_dist[ok]
        different = labels[iu[ok]] != labels[ju[ok]]
        stats = g["term"].value_counts().to_numpy(dtype=int)
        k_rare = _expected_rarefied_richness(stats, RAREFIED_TARGET)
        for delta in np.asarray(deltas, dtype=float):
            keep = d <= delta
            total = int(keep.sum())
            if total < min_valid_pairs:
                continue
            rows.append({
                "language_id": int(lang), "speaker_id": int(spk), "delta": float(delta),
                "different_pairs": int(different[keep].sum()), "total_pairs": total,
                "R": float(different[keep].mean()), "speaker_K_raw": int(len(stats)),
                "speaker_K_rarefied": k_rare, "n_observed_chips": int(g.chip_id.nunique()),
                "n_valid_pairs": int(total), "group_id": f"{int(lang)}:{int(spk)}",
            })
    return pd.DataFrame(rows)


def speaker_pair_distance_bins(term, mapping, n_bins=40, min_bin_pairs=1):
    valid = clean_terms(term)
    ids, iu, ju, pair_dist = _pair_setup(mapping)
    edges = np.linspace(float(pair_dist.min()), float(pair_dist.max()) + 1e-12, n_bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    rows = []
    for (lang, spk), g in valid.groupby(["language_id", "speaker_id"], sort=True):
        labels = g.set_index("chip_id")["term"].reindex(ids).to_numpy()
        ok = pd.notna(labels[iu]) & pd.notna(labels[ju])
        d = pair_dist[ok]
        different = (labels[iu[ok]] != labels[ju[ok]]).astype(int)
        b = np.searchsorted(edges, d, side="right") - 1
        keep = (b >= 0) & (b < n_bins)
        agg = (pd.DataFrame({"bin": b[keep], "different": different[keep]})
               .groupby("bin")["different"].agg(["sum", "count"]).reset_index())
        counts = g["term"].value_counts().to_numpy(dtype=int)
        k_rare = _expected_rarefied_richness(counts, RAREFIED_TARGET)
        for _, r in agg.iterrows():
            total = int(r["count"])
            if total >= min_bin_pairs:
                rows.append({
                    "language_id": int(lang), "speaker_id": int(spk),
                    "speaker_K_raw": int(len(counts)), "speaker_K_rarefied": k_rare,
                    "distance_bin": int(r["bin"]), "distance": float(mids[int(r["bin"])]),
                    "different_count": int(r["sum"]), "total_count": total,
                    "n_observed_chips": int(g.chip_id.nunique()),
                    "group_id": f"{int(lang)}:{int(spk)}",
                })
    return pd.DataFrame(rows)