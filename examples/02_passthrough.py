"""
Full application: exchange-rate pass-through to U.S. prices, four regimes.

    python examples/02_passthrough.py [--B 999] [--quick]

Produces, in ``output/``:

  fig1_regimes.png        the dollar index with regime windows shaded
  fig2_series.png         all nine series
  fig3_bracket.png        the trend-absorption bracket
  fig4_null_<regime>.png  the simulated null for each regime, full control set
  fig5_blocks.png         the h-block partition
  fig6_diagnostic.png     full vs reduced p-values across regimes
  tab1_regimes.tex/.csv   the main results table
  tab2_diagnostic.tex     the trend-absorption diagnostic

On replication
--------------
The paper does not publish its FRED series codes, data vintage, per-control
log/level treatment, or its cross-fitting settings. The mapping used here is
inferred from the variable descriptions, so these numbers are **not** a
replication of its Table 11. What does reproduce exactly is the sample design:
156, 156, 108 and 156 observations across the four regimes.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import pandas as pd

import ardldml as ad
from ardldml import (
    CONTROLS,
    DEFAULT_INTEGRATED,
    PASSTHROUGH_REGIMES,
    REDUCED_DROP,
    DMLBounds,
)

SEED = 20260625
OUT = Path("output")


def fit_one(df: pd.DataFrame, cols, B: int, seed: int):
    return (
        DMLBounds(
            df["cpi"], df["neer"], df[cols],
            lags=4, n_blocks=5, buffer=6,
            integrated=[c for c in DEFAULT_INTEGRATED if c in cols],
        )
        .fit()
        .bootstrap(B=B, seed=seed)
    )


def main(B: int = 999, quick: bool = False) -> None:
    OUT.mkdir(exist_ok=True)
    ad.use_journal_style()
    if quick:
        B = 199

    full_sample = ad.load_passthrough()
    reduced_cols = [c for c in CONTROLS if c not in REDUCED_DROP]

    print(f"sample 1973-01 to 2020-12, {len(full_sample)} months")
    print(f"full control set    ({len(CONTROLS)}): {', '.join(CONTROLS)}")
    print(f"reduced control set ({len(reduced_cols)}): {', '.join(reduced_cols)}")
    print(f"dropped: {', '.join(REDUCED_DROP)}  (most likely to share the tested trend)\n")

    ad.plot_regimes(full_sample, PASSTHROUGH_REGIMES, "neer").savefig(OUT / "fig1_regimes.png")
    ad.plot_series(full_sample, title="Pass-through data, 1973-2020").savefig(OUT / "fig2_series.png")
    ad.plot_bracket(k=10, k_tilde=6).savefig(OUT / "fig3_bracket.png")

    results = {}
    for i, regime in enumerate(PASSTHROUGH_REGIMES):
        df = ad.load_passthrough(regime=regime)
        fits = {}
        for name, cols in (("full", CONTROLS), ("reduced", reduced_cols)):
            r = fit_one(df, cols, B=B, seed=SEED + i)
            fits[name] = r
            print(
                f"{regime}  {name:8s} n={r.nobs:3d}  F={r.stat:7.3f}  "
                f"cv={r.critical_value:7.3f}  p={r.pvalue:.3f}  "
                f"theta={r.theta:+7.3f} ({r.theta_se:.3f})  -> {r.decision()}"
            )
        results[regime] = fits

        ad.plot_bootstrap_null(fits["full"]).savefig(
            OUT / f"fig4_null_{regime.replace('-', '_')}.png"
        )

    ad.plot_block_structure(results["1999-2007"]["full"].first_stage.folds).savefig(
        OUT / "fig5_blocks.png"
    )

    table = ad.regime_table(results)
    print("\n" + table.to_string(index=False))
    table.to_csv(OUT / "tab1_regimes.csv", index=False)
    ad.to_latex(
        table,
        caption="Exchange-rate pass-through to U.S. prices across monetary regimes",
        label="tab:passthrough",
        notes=(
            f"Critical values are the 95th percentile of the restricted system wild "
            f"bootstrap, B = {B}. Verdicts are bootstrap decisions, not comparisons "
            "with tabulated bounds. 'full' conditions on all seven controls; "
            "'reduced' drops money and the oil price."
        ),
        path=OUT / "tab1_regimes.tex",
        index=False,
    )

    # The diagnostic figure: full vs reduced p-values across regimes.
    labels = list(PASSTHROUGH_REGIMES)
    frames = {
        "full controls (7)": pd.DataFrame(
            {"p": [results[r]["full"].pvalue for r in labels]}, index=labels
        ),
        "drop M2, oil (5)": pd.DataFrame(
            {"p": [results[r]["reduced"].pvalue for r in labels]}, index=labels
        ),
    }
    ad.plot_diagnostic(frames, xlabel="monetary regime").savefig(OUT / "fig6_diagnostic.png")

    # The formal four-fit diagnostic, on the regime the paper singles out.
    print("\nTrend-absorption diagnostic, 1999-2007:")
    df99 = ad.load_passthrough(regime="1999-2007")
    diag = ad.trend_absorption(
        df99["cpi"], df99["neer"], df99[CONTROLS],
        drop=REDUCED_DROP,
        lags=4, n_blocks=5, buffer=6,
        integrated=[c for c in DEFAULT_INTEGRATED if c in CONTROLS],
        B=B, seed=SEED, progress=True,
    )
    print(diag.summary())
    ad.to_latex(
        ad.diagnostic_table(diag),
        caption="Trend-absorption diagnostic, 1999--2007",
        label="tab:diagnostic",
        notes=(
            "Four fits: full and reduced control sets crossed with the adaptive and "
            "unpenalised $m_Z$ projection. $\\Delta_W$ is the gap between the full- "
            "and reduced-set bootstrap p-values."
        ),
        path=OUT / "tab2_diagnostic.tex",
        index=False,
    )

    print(f"\nwrote figures and tables to {OUT.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=999)
    ap.add_argument("--quick", action="store_true", help="B=199, for a fast pass")
    a = ap.parse_args()
    main(B=a.B, quick=a.quick)
