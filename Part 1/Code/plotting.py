from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors

from common.metrics import bootstrap_mean_ci


# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------
# Use a restrained, publication-oriented style without depending on seaborn.
# Viridis is perceptually uniform and remains readable for many forms of
# colour-vision deficiency.
VIRIDIS = plt.colormaps["viridis"]
CAPTION_COLOR = "#333333"
GRID_COLOR = "#D9D9D9"
ERROR_COLOR = "#444444"


def _style_axes(ax):
    """Apply a consistent, clean style to an individual Matplotlib axis."""
    ax.grid(True, which="major", color=GRID_COLOR, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)
    ax.xaxis.label.set_size(11)
    ax.yaxis.label.set_size(11)


def _save(fig, path: Path, caption: str):
    """Save a figure with a compact bold caption beneath the plotting area."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(bottom=0.24, top=0.88, left=0.12, right=0.96)
    fig.text(
        0.12,
        0.035,
        caption,
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color=CAPTION_COLOR,
        wrap=True,
    )
    fig.savefig(
        path,
        dpi=320,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def _add_language_points(
    ax,
    q: pd.DataFrame,
    ycol: str,
    lo_col: str,
    hi_col: str,
    *,
    size: float = 54,
):
    """Scatter languages, colouring by the number of contributing speakers."""
    q = q.copy()
    q = q[np.isfinite(q["language_K"]) & np.isfinite(q[ycol])].copy()

    if q.empty:
        return

    norm = colors.Normalize(
        vmin=max(1, float(q["n_speakers"].min())),
        vmax=max(1, float(q["n_speakers"].max())),
    )

    point_colors = VIRIDIS(norm(q["n_speakers"].to_numpy(float)))

    ax.scatter(
        q["language_K"],
        q[ycol],
        c=point_colors,
        s=size,
        alpha=0.88,
        edgecolors="white",
        linewidths=0.7,
        zorder=3,
    )

    if {lo_col, hi_col}.issubset(q.columns):
        y = q[ycol].to_numpy(float)
        lo = q[lo_col].to_numpy(float)
        hi = q[hi_col].to_numpy(float)
        ax.errorbar(
            q["language_K"],
            y,
            yerr=np.vstack([y - lo, hi - y]),
            fmt="none",
            ecolor=ERROR_COLOR,
            alpha=0.55,
            capsize=2.5,
            linewidth=0.85,
            zorder=2,
        )

    sm = cm.ScalarMappable(norm=norm, cmap=VIRIDIS)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Number of speakers contributing", fontsize=10)
    cbar.ax.tick_params(labelsize=9)


def language_summary_plots(summary: pd.DataFrame, outdir: Path) -> None:
    """Create publication-style language-level descriptive figures."""
    outdir.mkdir(parents=True, exist_ok=True)
    s = summary.copy()

    # -----------------------------------------------------------------------
    # Figure 1: standardized speaker repertoire
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    q = s[np.isfinite(s["mean_speaker_K_rarefied"])].copy()

    _add_language_points(
        ax,
        q,
        "mean_speaker_K_rarefied",
        "boot_K_lo",
        "boot_K_hi",
    )

    ax.set_xlabel(
        "Primary-language colour inventory, $K_l$"
        "\n(prevalence-normalized)"
    )
    ax.set_ylabel(
        "Mean speaker colour-label richness\n"
        "after rarefaction to 313 valid responses"
    )
    ax.set_title(
        "Primary-language colour inventory and standardized speaker repertoire",
        fontweight="bold",
        fontsize=13,
        pad=12,
    )
    _style_axes(ax)
    _save(
        fig,
        outdir / "language_K_vs_rarefied_speaker_K.png",
        "Figure 1. Each point is a language; error bars are 95% speaker-bootstrap CIs. "
        "The speaker richness measure is exactly rarefied to a common target of 313 valid "
        "responses, preventing unequal response counts from mechanically inflating the "
        "number of observed labels. Point colour encodes the number of speakers contributing.",
    )

    # -----------------------------------------------------------------------
    # Figure 2: effective repertoire
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    q = s[np.isfinite(s["mean_effective_categories"])].copy()

    _add_language_points(
        ax,
        q,
        "mean_effective_categories",
        "boot_eff_lo",
        "boot_eff_hi",
    )

    ax.set_xlabel(
        "Primary-language colour inventory, $K_l$"
        "\n(prevalence-normalized)"
    )
    ax.set_ylabel(
        r"Mean effective speaker repertoire, "
        r"$\exp\{\widehat H_{\mathrm{MM}}(Y_s)\}$"
    )
    ax.set_title(
        "Primary-language colour inventory and effective speaker repertoire",
        fontweight="bold",
        fontsize=13,
        pad=12,
    )
    _style_axes(ax)
    _save(
        fig,
        outdir / "language_K_vs_effective_speaker_colours.png",
        "Figure 2. Effective repertoire is the exponential of the Miller–Madow-corrected "
        "speaker label entropy. The quantity therefore reflects how evenly the speaker "
        "uses the available colour labels, rather than merely counting rare labels. "
        "Error bars are 95% speaker-bootstrap CIs; point colour encodes the number of speakers.",
    )

    # -----------------------------------------------------------------------
    # Figure 3: linguistic resolution threshold
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    q = s[np.isfinite(s["mean_delta50"])].copy()

    _add_language_points(
        ax,
        q,
        "mean_delta50",
        "boot_delta_lo",
        "boot_delta_hi",
    )

    ax.set_xlabel(
        "Primary-language colour inventory, $K_l$"
        "\n(prevalence-normalized)"
    )
    ax.set_ylabel(r"Mean speaker linguistic resolution threshold, $\delta_{50}$")
    ax.set_title(
        "Primary-language colour inventory and linguistic colour resolution",
        fontweight="bold",
        fontsize=13,
        pad=12,
    )
    _style_axes(ax)
    _save(
        fig,
        outdir / "language_K_vs_delta50.png",
        "Figure 3. Lower $\\delta_{50}$ means that a speaker tends to assign different "
        "labels to physically closer CIELAB stimuli. This is a linguistic categorization "
        "threshold, not an independent psychophysical visual-discrimination threshold. "
        "Error bars are 95% speaker-bootstrap CIs.",
    )

    # -----------------------------------------------------------------------
    # Figure 4: response completeness
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    q = s[
        np.isfinite(s["median_valid_chips"])
        & np.isfinite(s["language_K"])
    ].copy()

    norm = colors.Normalize(
        vmin=max(1, float(q["n_speakers"].min())),
        vmax=max(1, float(q["n_speakers"].max())),
    )
    ax.scatter(
        q["language_K"],
        q["median_valid_chips"],
        c=VIRIDIS(norm(q["n_speakers"].to_numpy(float))),
        s=54,
        alpha=0.88,
        edgecolors="white",
        linewidths=0.7,
        zorder=3,
    )

    ax.axhline(
        313,
        color=VIRIDIS(0.82),
        linestyle="--",
        linewidth=1.6,
        label="Rarefaction target = 313",
        zorder=2,
    )

    ax.set_xlabel(
        "Primary-language colour inventory, $K_l$"
        "\n(prevalence-normalized)"
    )
    ax.set_ylabel("Median valid colour responses per speaker")
    ax.set_title(
        "Response completeness by primary-language colour inventory",
        fontweight="bold",
        fontsize=13,
        pad=12,
    )
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9.5, loc="best")

    sm = cm.ScalarMappable(norm=norm, cmap=VIRIDIS)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Number of speakers contributing", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    _save(
        fig,
        outdir / "language_K_vs_response_completeness.png",
        "Figure 4. Median number of valid colour responses per speaker is shown to make "
        "response completeness explicit. The standardized speaker-richness analysis "
        "uses a common 313-response target; the resolution and information analyses use "
        "all valid responses. Point colour encodes the number of speakers contributing.",
    )
