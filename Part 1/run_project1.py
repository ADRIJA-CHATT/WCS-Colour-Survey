"""Project 1: full-data hypothesis test for linguistic colour resolution.

The analysis is inferential rather than predictive. All available speakers and
all valid colour-chip pairs are used. No train/test split or cross-validation
is performed. The hypothesis is tested with a language-clustered generalized
estimating equation.
"""

from __future__ import annotations

import shutil
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.preprocessing import SplineTransformer
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "project1"
N_KNOTS = 8  # Pre-specified smoothness; no model-selection step is used.
N_BINS = 30
LANG_MIN_SPEAKER_FRACTION = 0.20
LANG_MIN_CHIPS_PER_TERM = 3


def clean_output() -> None:
    """Remove the previous generated Project 1 directory before writing new results."""
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)


def build_speaker_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute speaker-level repertoire, entropy and observation-count statistics."""
    from common.metrics import speaker_statistics
    return speaker_statistics(df).copy()


def build_pair_info(df: pd.DataFrame):
    """Construct the complete physical pair geometry for all 330 WCS chips."""
    chipmeta = (
        df[["chip_id", "L_star", "a_star", "b_star"]]
        .drop_duplicates("chip_id")
        .sort_values("chip_id")
    )
    if len(chipmeta) != 330:
        raise ValueError(f"Expected 330 unique WCS chips; found {len(chipmeta)}.")
    ids = chipmeta["chip_id"].to_numpy(dtype=int)
    X = chipmeta[["L_star", "a_star", "b_star"]].to_numpy(float)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    iu, ju = np.triu_indices(len(ids), k=1)
    return ids, iu, ju, D[iu, ju]


def pair_bins_for_all_speakers(df: pd.DataFrame, bins: np.ndarray, pair_info):
    """Aggregate every valid unordered pair for every speaker into distance bins."""
    ids, iu, ju, dist = pair_info
    bin_id = np.digitize(dist, bins, right=False) - 1

    work = df[["language_id", "speaker_id", "chip_id", "term"]].dropna().copy()
    work = work.drop_duplicates(["language_id", "speaker_id", "chip_id"], keep="first")
    work["language_id"] = work["language_id"].astype(int)
    work["speaker_id"] = work["speaker_id"].astype(int)
    work["chip_id"] = work["chip_id"].astype(int)
    work["term"] = work["term"].astype(str)

    records = []
    positions = {int(chip): idx for idx, chip in enumerate(ids)}
    for (lang, spk), g in work.groupby(["language_id", "speaker_id"], sort=False):
        labels = pd.Series(g["term"].to_numpy(), index=g["chip_id"].to_numpy(), dtype="object")
        observed = np.asarray(pd.Index(ids).isin(labels.index), dtype=bool)
        if observed.sum() < 2:
            continue

        label_arr = np.empty(len(ids), dtype=object)
        label_arr[:] = None
        for chip, term in labels.items():
            idx = positions.get(int(chip))
            if idx is not None:
                label_arr[idx] = term

        pair_ok = observed[iu] & observed[ju]
        if not np.any(pair_ok):
            continue
        diff = (label_arr[iu[pair_ok]] != label_arr[ju[pair_ok]]).astype(np.int8)
        bins_used = bin_id[pair_ok]
        tmp = pd.DataFrame({"bin": bins_used, "different": diff})
        agg = (
            tmp.groupby("bin", as_index=False)
            .agg(different=("different", "sum"), total=("different", "size"))
        )
        agg["language_id"] = int(lang)
        agg["speaker_id"] = int(spk)
        agg["group_id"] = f"{int(lang)}:{int(spk)}"
        agg["n_observed_chips"] = int(observed.sum())
        agg["n_missing_chips"] = int(330 - observed.sum())
        agg["n_valid_pairs"] = int(observed.sum() * (observed.sum() - 1) // 2)
        records.append(agg)

    if not records:
        return pd.DataFrame(columns=[
            "bin", "different", "total", "language_id", "speaker_id", "group_id",
            "n_observed_chips", "n_missing_chips", "n_valid_pairs"
        ])
    return pd.concat(records, ignore_index=True)


def add_distance_centers(pair_bins: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    """Attach the physical-distance centre of each bin."""
    centers = (edges[:-1] + edges[1:]) / 2
    out = pair_bins.copy()
    out["distance"] = out["bin"].map(dict(enumerate(centers)))
    return out.dropna(subset=["distance"]).copy()


def attach_language_vocabulary(pair_bins: pd.DataFrame, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct the language vocabulary from the full observed language sample."""
    from common.metrics import language_prevalence_vocabulary
    k = language_prevalence_vocabulary(
        df,
        min_speaker_fraction=LANG_MIN_SPEAKER_FRACTION,
        min_chips_per_term=LANG_MIN_CHIPS_PER_TERM,
    )
    merged = pair_bins.merge(k, on="language_id", how="left", validate="many_to_one")
    if merged["language_K"].isna().any():
        raise ValueError("Some speakers belong to languages with no estimated language vocabulary.")
    return merged, k


def fit_full_model(pair_bins: pd.DataFrame):
    """Fit the pre-specified cubic-spline GEE to all available speakers."""
    trans = SplineTransformer(n_knots=N_KNOTS, degree=3, include_bias=False)
    Z = trans.fit_transform(pair_bins[["distance"]])
    logk = np.log(pair_bins["language_K"].to_numpy(float))
    # The main distance curve is f(d); its interaction with log K captures
    # whether the entire linguistic-resolution curve shifts with vocabulary.
    X = sm.add_constant(np.column_stack([Z, Z * logk[:, None]]), has_constant="add")
    y = pair_bins["different"] / pair_bins["total"]
    model = sm.GEE(
        y,
        X,
        groups=pair_bins["language_id"],
        weights=pair_bins["total"],
        family=sm.families.Binomial(),
    ).fit()
    return model, trans


def hypothesis_test_language_effect(model) -> dict:
    """Joint Wald test of H0: language vocabulary does not alter the resolution curve."""
    # Nine spline-by-log(K) coefficients are the only terms that let K modify
    # the distance-response curve, so H0 sets all nine to zero jointly.
    n_params = len(model.params)
    n_basis = (n_params - 1) // 2
    C = np.zeros((n_basis, n_params))
    C[:, 1 + n_basis :] = np.eye(n_basis)
    beta = C @ model.params
    cov = C @ model.cov_params() @ C.T
    statistic = float(beta @ np.linalg.pinv(cov) @ beta)
    p_value = float(chi2.sf(statistic, n_basis))
    return {
        "null_hypothesis": "All language-vocabulary-by-distance interaction coefficients are zero",
        "alternative": "At least one interaction coefficient is nonzero",
        "test": "Language-clustered joint Wald chi-square test",
        "statistic": statistic,
        "df": n_basis,
        "p_value": p_value,
        "alpha_0_05_reject": p_value < 0.05,
    }


def make_fitted_curve_plot(model, trans, pair_bins: pd.DataFrame) -> None:
    """Plot the fitted linguistic-distinction probability at representative vocabularies."""
    cmap = plt.colormaps["viridis"]
    distances = np.linspace(pair_bins["distance"].min(), pair_bins["distance"].max(), 350)
    Z = trans.transform(pd.DataFrame({"distance": distances}))

    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    for frac, K in zip([0.15, 0.50, 0.85], [3, 8, 17]):
        logk = np.log(K)
        X = sm.add_constant(np.column_stack([Z, Z * logk]), has_constant="add")
        pred = model.predict(X)
        ax.plot(distances, pred, color=cmap(frac), linewidth=2.6, label=fr"$K_l={K}$")

    ax.set_xlabel(r"CIELAB distance $d$")
    ax.set_ylabel(r"Predicted probability $P(Y_i\neq Y_j\mid d,K_l)$")
    ax.set_title("Fitted linguistic colour-resolution curves", fontweight="bold", fontsize=13, pad=12)
    ax.grid(True, color="#D9D9D9", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, title="Language colour inventory")
    fig.subplots_adjust(bottom=0.23, top=0.88, left=0.12, right=0.96)
    fig.text(
        0.12, 0.035,
        "Fitted probabilities use all available speakers. The language-vocabulary effect is tested "
        "jointly through the spline-by-log-inventory interaction; no held-out prediction is involved.",
        ha="left", va="bottom", fontsize=9.5, fontweight="bold", color="#333333", wrap=True,
    )
    fig.savefig(OUT / "fitted_distinction_curves.png", dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_resolution_plot(df: pd.DataFrame) -> None:
    """Plot the all-speaker median linguistic resolution curve."""
    from common.metrics import speaker_resolution_curves

    chipmap = df[["chip_id", "L_star", "a_star", "b_star"]].drop_duplicates()
    distances = np.linspace(0, float(np.sqrt(((chipmap[['L_star','a_star','b_star']].to_numpy(float)[:,None,:] - chipmap[['L_star','a_star','b_star']].to_numpy(float)[None,:,:])**2).sum(axis=2)).max()), 30)[1:]
    curves = speaker_resolution_curves(df, chipmap, distances)
    med = curves.groupby("delta")["R"].median()

    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    ax.plot(med.index, med.values, color=plt.colormaps["viridis"](0.5), marker="o", markersize=4.5, linewidth=2.4)
    ax.set_xlabel(r"CIELAB distance threshold $\delta$")
    ax.set_ylabel(r"Median linguistic resolution $R_s(\delta)$")
    ax.set_title("Speaker-level linguistic colour resolution", fontweight="bold", fontsize=13, pad=12)
    ax.grid(True, color="#D9D9D9", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(bottom=0.23, top=0.88, left=0.12, right=0.96)
    fig.text(
        0.12, 0.035,
        "Median across all speakers. Every valid pair contributes to the curve; no train/test partition or pair sampling is used.",
        ha="left", va="bottom", fontsize=9.5, fontweight="bold", color="#333333", wrap=True,
    )
    fig.savefig(OUT / "resolution_curve_all_speakers.png", dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    from common.data import load_dataset, validate_dataset
    from project_1_geometry.plotting import language_summary_plots

    clean_output()
    df = load_dataset()
    validate_dataset(df)

    stats = build_speaker_stats(df)
    stats.to_csv(OUT / "speaker_statistics.csv", index=False)

    pair_info = build_pair_info(df)
    max_dist = float(pair_info[3].max())
    edges = np.linspace(0.0, max_dist + 1e-9, N_BINS + 1)
    pair_bins = add_distance_centers(pair_bins_for_all_speakers(df, edges, pair_info), edges)
    pair_bins, language_k = attach_language_vocabulary(pair_bins, df)
    pair_bins.to_csv(OUT / "pair_bins.csv", index=False)
    language_k.to_csv(OUT / "language_vocabulary.csv", index=False)

    model, trans = fit_full_model(pair_bins)
    model_summary = model.summary().as_text()
    (OUT / "model_summary.txt").write_text(model_summary, encoding="utf-8")

    coef_names = ["intercept"] + [f"distance_spline_{i+1}" for i in range(9)] + [f"language_interaction_{i+1}" for i in range(9)]
    pd.DataFrame({
        "term": coef_names,
        "estimate": np.asarray(model.params, dtype=float),
        "std_error": np.asarray(model.bse, dtype=float),
        "z": np.asarray(model.tvalues, dtype=float),
        "p_value": np.asarray(model.pvalues, dtype=float),
    }).to_csv(OUT / "model_coefficients.csv", index=False)

    test = hypothesis_test_language_effect(model)
    pd.DataFrame([test]).to_csv(OUT / "hypothesis_test.csv", index=False)

    # Speaker-level descriptive outputs are generated from the same full dataset.
    from common.metrics import language_prevalence_vocabulary
    lang_k = language_prevalence_vocabulary(
        df, min_speaker_fraction=LANG_MIN_SPEAKER_FRACTION, min_chips_per_term=LANG_MIN_CHIPS_PER_TERM
    )
    exact = []
    clean = df.dropna(subset=["term"]).drop_duplicates(["language_id", "speaker_id", "chip_id"])
    # Reuse the complete resolution summary routine from this script's pair geometry.
    chipmeta = df[["chip_id", "L_star", "a_star", "b_star"]].drop_duplicates("chip_id").sort_values("chip_id")
    ids = chipmeta.chip_id.to_numpy(int)
    X = chipmeta[["L_star", "a_star", "b_star"]].to_numpy(float)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    iu, ju = np.triu_indices(len(ids), k=1)
    pair_dist = D[iu, ju]
    for (lang, spk), g in clean.groupby(["language_id", "speaker_id"], sort=True):
        lab = pd.Series(g.term.to_numpy(), index=g.chip_id.to_numpy(), dtype="object").reindex(ids)
        obs = lab.notna().to_numpy()
        if obs.sum() < 2:
            continue
        ok = obs[iu] & obs[ju]
        d = pair_dist[ok]
        diff = (lab.to_numpy()[iu[ok]] != lab.to_numpy()[ju[ok]]).astype(int)
        order = np.argsort(d); d = d[order]; diff = diff[order]
        cum = np.cumsum(diff) / np.arange(1, len(diff) + 1)
        hits = np.flatnonzero(cum >= 0.5)
        j = int(hits[0]) if len(hits) else len(cum) - 1
        st = stats.set_index(["language_id", "speaker_id"]).loc[(lang, spk)]
        exact.append({
            "language_id": int(lang), "speaker_id": int(spk), "delta50": float(d[j]),
            "speaker_K_rarefied": float(st["speaker_K_rarefied"]),
            "effective_colour_categories": float(st["effective_colour_categories"]),
            "n_observed_chips": int(st["n_observed_chips"]),
            "rarefaction_eligible": bool(st["rarefaction_eligible"]),
        })
    exact = pd.DataFrame(exact)
    rows=[]
    for lang,g in exact.groupby("language_id"):
        from common.metrics import bootstrap_mean_ci
        rare=g.loc[g.rarefaction_eligible,"speaker_K_rarefied"]
        eff=g.effective_colour_categories
        d50=g.delta50
        mk,klo,khi=bootstrap_mean_ci(rare); me,elo,ehi=bootstrap_mean_ci(eff); md,dlo,dhi=bootstrap_mean_ci(d50)
        rows.append({"language_id":int(lang),"mean_speaker_K_rarefied":mk,"boot_K_lo":klo,"boot_K_hi":khi,
                     "mean_effective_categories":me,"boot_eff_lo":elo,"boot_eff_hi":ehi,
                     "mean_delta50":md,"boot_delta_lo":dlo,"boot_delta_hi":dhi,
                     "mean_valid_chips":float(g.n_observed_chips.mean()),"median_valid_chips":float(g.n_observed_chips.median()),
                     "n_speakers":int(len(g)),"n_rarefaction_eligible":int(g.rarefaction_eligible.sum())})
    summary=pd.DataFrame(rows).merge(lang_k,on="language_id",how="left",validate="one_to_one")
    summary.to_csv(OUT / "language_resolution_summary.csv", index=False)

    language_summary_plots(summary, OUT)
    make_resolution_plot(df)
    make_fitted_curve_plot(model, trans, pair_bins)

    print(
        f"Project 1 complete. All {df.language_id.nunique()} languages and {df[['language_id','speaker_id']].drop_duplicates().shape[0]} speakers used. "
        f"Primary hypothesis Wald chi-square={test['statistic']:.3f}, df={test['df']}, p={test['p_value']:.3e}."
    )


if __name__ == "__main__":
    main()
