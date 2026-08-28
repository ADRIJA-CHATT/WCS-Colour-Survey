"""Project 2: full-data hypothesis test for physical-colour information.

All available speakers are used. The primary regression is a pre-specified
linear fractional-response model with language-clustered robust covariance.
There is no train/test split, cross-validation, or predictive evaluation.
"""

from __future__ import annotations

import shutil
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.cluster import KMeans
from matplotlib import colors

from common.data import load_dataset, validate_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "project2"
RANDOM_STATE = 20260822
N_PHYSICAL_BINS = 20
LANG_MIN_SPEAKER_FRACTION = 0.20
LANG_MIN_CHIPS_PER_TERM = 3
RAREFIED_TARGET = 313


def clean_output() -> None:
    """Remove the previous generated Project 2 directory before writing new results."""
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)


def mutual_information_mm(table):
    from common.metrics import miller_madow_mutual_information
    return miller_madow_mutual_information(table)


def fit_physical_bins(df, n_bins=N_PHYSICAL_BINS):
    """Partition physical stimuli from CIELAB coordinates only."""
    chips = (df[["chip_id", "L_star", "a_star", "b_star"]]
             .drop_duplicates("chip_id")
             .sort_values("chip_id"))
    X = chips[["L_star", "a_star", "b_star"]].to_numpy(float)
    if len(X) < n_bins:
        raise ValueError(f"Need at least {n_bins} chips; found {len(X)}.")
    km = KMeans(n_clusters=n_bins, random_state=RANDOM_STATE, n_init=50)
    labels = km.fit_predict(X)
    return pd.DataFrame({"chip_id": chips.chip_id.to_numpy(), "physical_bin": labels}), km


def speaker_information(df, chip_bins):
    """Compute information efficiency for every speaker from all valid responses."""
    from common.metrics import clean_terms, miller_madow_entropy, speaker_statistics, entropy
    d = clean_terms(df).merge(chip_bins, on="chip_id", how="left", validate="many_to_one")
    full_physical = d[["chip_id", "physical_bin"]].drop_duplicates().sort_values("chip_id")
    full_counts = full_physical["physical_bin"].value_counts().to_numpy(dtype=float)
    h_b_full = float(entropy(full_counts / full_counts.sum()))
    sp_stats = speaker_statistics(df).set_index(["language_id", "speaker_id"])

    rows = []
    for (lang, spk), g in d.groupby(["language_id", "speaker_id"], sort=True):
        table = pd.crosstab(g["physical_bin"], g["term"]).to_numpy(dtype=float)
        n_obs = int(g["chip_id"].nunique())
        if n_obs < 2:
            continue
        h_b_obs = float(miller_madow_entropy(table.sum(axis=1)))
        h_y_mm = float(miller_madow_entropy(table.sum(axis=0)))
        mi_mm = float(mutual_information_mm(table))
        efficiency = float(np.clip(mi_mm / h_b_full, 0.0, 1.0)) if h_b_full > 0 else np.nan
        st = sp_stats.loc[(lang, spk)]
        rows.append({
            "language_id": int(lang), "speaker_id": int(spk), "group_id": f"{int(lang)}:{int(spk)}",
            "speaker_K_raw": int(st["speaker_K_raw"]), "speaker_K_rarefied": float(st["speaker_K_rarefied"]),
            "H_physical_bin_observed_MillerMadow": h_b_obs, "H_physical_bin_full": h_b_full,
            "H_label_MillerMadow": h_y_mm, "I_physical_label_MillerMadow": mi_mm,
            "information_efficiency": efficiency, "effective_colour_categories": float(np.exp(h_y_mm)),
            "n_chips": 330, "n_observed_chips": n_obs, "n_missing_chips": int(330 - n_obs),
            "n_rarefaction_eligible": bool(st["rarefaction_eligible"]),
        })
    return pd.DataFrame(rows).sort_values(["language_id", "speaker_id"]).reset_index(drop=True)


def language_k_from_all_speakers(df):
    from common.metrics import language_prevalence_vocabulary
    return language_prevalence_vocabulary(
        df,
        min_speaker_fraction=LANG_MIN_SPEAKER_FRACTION,
        min_chips_per_term=LANG_MIN_CHIPS_PER_TERM,
    ).rename(columns={"language_K": "language_K"})


def fit_linear_fractional_logit(stats, language_k):
    """Fit the pre-specified linear fractional-response model to all speakers."""
    d = stats.merge(language_k, on="language_id", how="left", validate="many_to_one")
    x = np.log(d["language_K"].to_numpy(float))
    y = np.clip(d["information_efficiency"].to_numpy(float), 1e-6, 1 - 1e-6)
    X = sm.add_constant(x, has_constant="add")
    model = sm.GLM(y, X, family=sm.families.Binomial()).fit(
        cov_type="cluster",
        cov_kwds={"groups": d["language_id"]},
    )
    return model, d


def fit_quadratic_sensitivity(stats, language_k):
    """Fit a fixed quadratic sensitivity model; no model-selection step is involved."""
    d = stats.merge(language_k, on="language_id", how="left", validate="many_to_one")
    x = np.log(d["language_K"].to_numpy(float))
    y = np.clip(d["information_efficiency"].to_numpy(float), 1e-6, 1 - 1e-6)
    X = sm.add_constant(np.column_stack([x, x * x]), has_constant="add")
    model = sm.GLM(y, X, family=sm.families.Binomial()).fit(
        cov_type="cluster",
        cov_kwds={"groups": d["language_id"]},
    )
    return model, d


def make_primary_plots(stats, language_k, outdir):
    """Create the Project 2 language-level repertoire, information, and completeness plots."""
    from common.metrics import bootstrap_mean_ci
    from matplotlib import colors

    d = stats.merge(language_k, on="language_id", how="left", validate="many_to_one")
    rows = []
    for lang, g in d.groupby("language_id", sort=True):
        rare = g.loc[g["n_rarefaction_eligible"], "speaker_K_rarefied"]
        eff = g["effective_colour_categories"]
        info = g["information_efficiency"]
        mk, klo, khi = bootstrap_mean_ci(rare)
        me, elo, ehi = bootstrap_mean_ci(eff)
        mi, ilo, ihi = bootstrap_mean_ci(info)
        rows.append({
            "language_id": int(lang), "language_K": float(g["language_K"].iloc[0]),
            "mean_rarefied_speaker_K": mk, "boot_K_lo": klo, "boot_K_hi": khi,
            "mean_effective_categories": me, "boot_eff_lo": elo, "boot_eff_hi": ehi,
            "mean_information_efficiency": mi, "boot_info_lo": ilo, "boot_info_hi": ihi,
            "mean_valid_chips": float(g["n_observed_chips"].mean()),
            "median_valid_chips": float(g["n_observed_chips"].median()),
            "n_speakers": int(len(g)), "n_rarefaction_eligible": int(g["n_rarefaction_eligible"].sum()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "language_summary.csv", index=False)

    cmap = plt.colormaps["viridis"]
    grid = "#D9D9D9"
    err = "#444444"

    def style(ax):
        ax.grid(True, color=grid, linewidth=.8, alpha=.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def save(fig, filename, caption):
        fig.subplots_adjust(bottom=.25, top=.88, left=.12, right=.94)
        fig.text(.12,.035,caption,ha="left",va="bottom",fontsize=9.5,fontweight="bold",color="#333333",wrap=True)
        fig.savefig(outdir/filename,dpi=320,bbox_inches="tight",facecolor="white")
        plt.close(fig)

    def scatter(q, y, lo, hi, ylabel, title, filename, caption):
        q=q[np.isfinite(q[y])].copy()
        norm=colors.Normalize(vmin=max(1,float(q.n_speakers.min())),vmax=max(1,float(q.n_speakers.max())))
        fig,ax=plt.subplots(figsize=(8.6,6.0))
        ax.scatter(q.language_K,q[y],c=cmap(norm(q.n_speakers.to_numpy(float))),s=56,alpha=.9,edgecolors="white",linewidths=.7,zorder=3)
        vals=q[y].to_numpy(float); lo_v=q[lo].to_numpy(float); hi_v=q[hi].to_numpy(float)
        ax.errorbar(q.language_K,vals,yerr=np.vstack([vals-lo_v,hi_v-vals]),fmt="none",ecolor=err,alpha=.55,capsize=2.5,linewidth=.9,zorder=2)
        ax.set_xlabel("Primary-language colour inventory, $K_l$")
        ax.set_ylabel(ylabel); ax.set_title(title,fontweight="bold",fontsize=13,pad=12); style(ax)
        smap=plt.cm.ScalarMappable(norm=norm,cmap=cmap); smap.set_array([])
        cb=fig.colorbar(smap,ax=ax,pad=.02); cb.set_label("Number of speakers contributing",fontsize=10)
        save(fig,filename,caption)

    scatter(summary,"mean_rarefied_speaker_K","boot_K_lo","boot_K_hi",
            "Mean speaker colour-label richness at 313 valid responses",
            "Language vocabulary and standardized speaker repertoire",
            "language_K_vs_rarefied_speaker_K.png",
            "Figure 1. Each point is a language; colour encodes the number of contributing speakers. The response is exact hypergeometric-rarefied speaker label richness at a common target of 313 valid responses.")

    scatter(summary,"mean_effective_categories","boot_eff_lo","boot_eff_hi",
            r"Mean effective speaker repertoire, $\exp\{\widehat H_{\mathrm{MM}}(Y_s)\}$",
            "Language vocabulary and effective speaker repertoire",
            "language_K_vs_effective_categories.png",
            "Figure 2. Effective repertoire is the exponential of Miller--Madow-corrected speaker-label entropy; error bars are 95% within-language speaker-bootstrap intervals.")

    scatter(summary,"mean_information_efficiency","boot_info_lo","boot_info_hi",
            r"Mean information efficiency, $I(B;Y)/H(B)$",
            "Language vocabulary and physical-colour information",
            "language_K_vs_information_efficiency.png",
            "Figure 3. Information efficiency is corrected mutual information between a speaker's linguistic labels and a fixed CIELAB-only physical representation, normalized by physical-bin entropy.")

    q=summary[np.isfinite(summary.median_valid_chips)].copy()
    norm=colors.Normalize(vmin=max(1,float(q.n_speakers.min())),vmax=max(1,float(q.n_speakers.max())))
    fig,ax=plt.subplots(figsize=(8.6,6.0))
    ax.scatter(q.language_K,q.median_valid_chips,c=cmap(norm(q.n_speakers.to_numpy(float))),s=56,alpha=.9,edgecolors="white",linewidths=.7,zorder=3)
    ax.axhline(RAREFIED_TARGET,color=cmap(.82),linestyle="--",linewidth=1.7,label=f"Rarefaction target = {RAREFIED_TARGET}")
    ax.set_xlabel("Primary-language colour inventory, $K_l$"); ax.set_ylabel("Median valid colour responses per speaker")
    ax.set_title("Response completeness by language colour vocabulary",fontweight="bold",fontsize=13,pad=12); style(ax); ax.legend(frameon=False,fontsize=9.5)
    smap=plt.cm.ScalarMappable(norm=norm,cmap=cmap); smap.set_array([]); cb=fig.colorbar(smap,ax=ax,pad=.02); cb.set_label("Number of speakers contributing",fontsize=10)
    save(fig,"language_K_vs_response_completeness.png","Figure 4. Median valid responses per speaker make response completeness explicit. The 313-response line is the standardization target for lexical richness; information calculations retain all valid responses.")

def make_full_data_fit_plot(model, data, outdir):
    """Plot all speaker outcomes and the fitted population-level fractional-logit curve."""
    cmap = plt.colormaps["viridis"]
    x_obs = data["language_K"].to_numpy(float)
    y_obs = data["information_efficiency"].to_numpy(float)
    norm = colors.Normalize(vmin=float(np.min(data["language_K"])), vmax=float(np.max(data["language_K"])))

    grid_x = np.linspace(float(np.min(x_obs)), float(np.max(x_obs)), 250)
    Xg = sm.add_constant(np.log(grid_x), has_constant="add")
    pred = model.predict(Xg)

    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    ax.scatter(x_obs, y_obs, c=cmap(norm(x_obs)), s=22, alpha=0.27, edgecolors="none")
    ax.plot(grid_x, pred, color=cmap(0.50), linewidth=2.8, label="Full-data fractional-logit mean")
    ax.set_xlabel("Primary-language colour inventory, $K_l$")
    ax.set_ylabel(r"Speaker information efficiency, $I(B;Y)/H(B)$")
    ax.set_title("Speaker information efficiency and language colour vocabulary", fontweight="bold", fontsize=13, pad=12)
    ax.grid(True, color="#D9D9D9", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    smap = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    smap.set_array([])
    cb = fig.colorbar(smap, ax=ax, pad=0.02)
    cb.set_label("Primary-language colour inventory", fontsize=10)
    fig.subplots_adjust(bottom=0.23, top=0.88, left=0.12, right=0.94)
    fig.text(
        0.12, 0.035,
        "All speakers are used for estimation. The curve represents the conditional population mean; "
        "it is not a test-set prediction curve.",
        ha="left", va="bottom", fontsize=9.5, fontweight="bold", color="#333333", wrap=True,
    )
    fig.savefig(outdir / "speaker_information_efficiency_vs_language_K.png", dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def hypothesis_test(model) -> dict:
    """Wald test of H0: beta_1 = 0 in the primary linear fractional model."""
    estimate = float(model.params[1])
    se = float(model.bse[1])
    statistic = (estimate / se) ** 2
    p_value = float(chi2.sf(statistic, 1))
    lo, hi = map(float, model.conf_int().iloc[1] if hasattr(model.conf_int(), "iloc") else model.conf_int()[1])
    return {
        "null_hypothesis": "Language colour vocabulary is unrelated to speaker-level information efficiency",
        "alternative": "Language colour vocabulary is related to speaker-level information efficiency",
        "test": "Language-clustered Wald chi-square test on log(K_l) coefficient",
        "statistic": statistic,
        "df": 1,
        "p_value": p_value,
        "beta_log_K": estimate,
        "se_beta_log_K": se,
        "ci95_lower": lo,
        "ci95_upper": hi,
        "alpha_0_05_reject": p_value < 0.05,
    }


def main():
    clean_output()
    df = load_dataset()
    validate_dataset(df)

    chip_bins, km = fit_physical_bins(df, N_PHYSICAL_BINS)
    chip_bins.to_csv(OUT / "physical_colour_bins_20.csv", index=False)
    np.save(OUT / "physical_bin_centres_20.npy", km.cluster_centers_)

    stats = speaker_information(df, chip_bins)
    stats.to_csv(OUT / "speaker_information.csv", index=False)

    language_k = language_k_from_all_speakers(df)
    language_k.to_csv(OUT / "language_vocabulary.csv", index=False)

    model, model_data = fit_linear_fractional_logit(stats, language_k)
    (OUT / "model_summary.txt").write_text(model.summary().as_text(), encoding="utf-8")
    pd.DataFrame({
        "term": ["intercept", "log_K_language"],
        "estimate": np.asarray(model.params, dtype=float),
        "std_error": np.asarray(model.bse, dtype=float),
        "z": np.asarray(model.tvalues, dtype=float),
        "p_value": np.asarray(model.pvalues, dtype=float),
    }).to_csv(OUT / "model_coefficients.csv", index=False)

    hyp = hypothesis_test(model)
    pd.DataFrame([hyp]).to_csv(OUT / "hypothesis_test.csv", index=False)

    # A fixed quadratic sensitivity model is reported only as a robustness check;
    # there is no cross-validation or model-selection step.
    quad_model, _ = fit_quadratic_sensitivity(stats, language_k)
    pd.DataFrame({
        "term": ["intercept", "log_K_language", "log_K_language_squared"],
        "estimate": np.asarray(quad_model.params, dtype=float),
        "std_error": np.asarray(quad_model.bse, dtype=float),
        "z": np.asarray(quad_model.tvalues, dtype=float),
        "p_value": np.asarray(quad_model.pvalues, dtype=float),
    }).to_csv(OUT / "quadratic_sensitivity_coefficients.csv", index=False)

    make_primary_plots(stats, language_k, OUT)
    make_full_data_fit_plot(model, model_data, OUT)

    sensitivity_rows = []
    for n_bins in [10, 15, 20, 25, 30]:
        bins, _ = fit_physical_bins(df, n_bins)
        info = speaker_information(df, bins)
        summ = info.merge(language_k, on="language_id", how="left", validate="many_to_one").groupby("language_id")["information_efficiency"].mean().rename("mean_E").reset_index()
        summ = summ.merge(language_k, on="language_id", how="inner")
        sensitivity_rows.append({
            "physical_bins": n_bins,
            "language_level_corr_mean_E_vs_K": float(summ["mean_E"].corr(summ["language_K"])),
            "mean_E": float(info["information_efficiency"].mean()),
        })
    sens = pd.DataFrame(sensitivity_rows)
    sens.to_csv(OUT / "physical_partition_sensitivity.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    ax.plot(sens["physical_bins"], sens["language_level_corr_mean_E_vs_K"], marker="o", linewidth=2.4, color=plt.colormaps["viridis"](0.5))
    ax.axhline(0, color="#888888", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Number of physical CIELAB clusters")
    ax.set_ylabel(r"Language-level correlation, $\mathrm{corr}(\bar E_l,K_l)$")
    ax.set_title("Information-efficiency association across physical resolutions", fontweight="bold", fontsize=13, pad=12)
    ax.grid(True, color="#D9D9D9", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.subplots_adjust(bottom=0.23, top=0.88, left=0.12, right=0.96)
    fig.text(0.12, 0.035, "The positive cross-language association is evaluated under five fixed physical discretizations; no predictive model selection is involved.", ha="left", va="bottom", fontsize=9.5, fontweight="bold", color="#333333", wrap=True)
    fig.savefig(OUT / "physical_partition_sensitivity.png", dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(
        f"Project 2 complete. All {df.language_id.nunique()} languages and {stats.shape[0]} speakers used. "
        f"Primary hypothesis Wald chi-square={hyp['statistic']:.3f}, df=1, p={hyp['p_value']:.3e}."
    )


if __name__ == "__main__":
    main()
