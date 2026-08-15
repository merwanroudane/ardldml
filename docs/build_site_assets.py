"""
Generate every figure and table shown on the project website.

    python docs/build_site_assets.py [--B 999] [--R 60] [--quick]

Writes PNGs to ``docs/img/`` and a JSON blob of every table to
``docs/_results.json``, which ``index.html`` embeds. Re-run it after any change
that could move the numbers, so the site never drifts from the code.

Everything here is real output: the empirical results come from the bundled
FRED-MD series, the Monte Carlo from the paper's own designs.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import numpy as np
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
HERE = Path(__file__).resolve().parent
IMG = HERE / "img"


def _fit(df, cols, B, seed):
    return (
        DMLBounds(
            df["cpi"], df["neer"], df[cols],
            lags=4, n_blocks=5, buffer=6,
            integrated=[c for c in DEFAULT_INTEGRATED if c in cols],
        )
        .fit()
        .bootstrap(B=B, seed=seed)
    )


def main(B: int = 999, R: int = 60, quick: bool = False) -> None:
    if quick:
        B, R = 199, 20
    IMG.mkdir(parents=True, exist_ok=True)
    # Transparent ground: the page's CSS decides what sits behind a figure, and
    # examples/04_full_analysis.py --style sunny writes its figures the same way.
    ad.use_sunny_style(transparent=True)
    results: dict = {"settings": {"B": B, "R": R, "seed": SEED}}
    t_start = time.time()

    # ---------------------------------------------------------------- data --
    print("[1/7] data figures")
    full = ad.load_passthrough()
    ad.plot_regimes(full, PASSTHROUGH_REGIMES, "neer").savefig(IMG / "regimes.png")
    ad.plot_series(full, title=None).savefig(IMG / "series.png")
    results["data"] = {
        "start": str(full.index.min().date()),
        "end": str(full.index.max().date()),
        "n_months": int(len(full)),
        "n_series": int(full.shape[1]),
        "regimes": {k: int(len(ad.load_passthrough(regime=k))) for k in PASSTHROUGH_REGIMES},
    }

    # ------------------------------------------------------------- bracket --
    print("[2/7] trend-absorption bracket")
    ad.plot_bracket(k=10, k_tilde=6).savefig(IMG / "bracket.png")

    # ------------------------------------------- classical bound validation --
    print("[3/7] regenerating the classical bracket")
    rows = []
    for k in (1, 2, 3, 4):
        sim = ad.simulate_pss_bounds(k=k, case=3, T=1000, nsim=8000, seed=0)
        try:
            pub = ad.pss_reference(k, 3)
            pub5 = [round(float(pub.loc[0.05, "I(0)"]), 2), round(float(pub.loc[0.05, "I(1)"]), 2)]
        except KeyError:
            pub5 = None
        rows.append({
            "k": k,
            "sim_lower": round(float(sim.loc[0.05, "I(0)"]), 3),
            "sim_upper": round(float(sim.loc[0.05, "I(1)"]), 3),
            "pub_lower": pub5[0] if pub5 else None,
            "pub_upper": pub5[1] if pub5 else None,
        })
    results["bounds_validation"] = rows

    off = ad.statsmodels_offset(k=1, case=3)
    results["statsmodels_offset"] = {
        "correct": [round(float(off.iloc[0, 0]), 2), round(float(off.iloc[0, 1]), 2)],
        "reported": [round(float(off.iloc[1, 0]), 2), round(float(off.iloc[1, 1]), 2)],
    }

    # --------------------------------------------------------- application --
    print(f"[4/7] pass-through application (B={B})")
    reduced_cols = [c for c in CONTROLS if c not in REDUCED_DROP]
    fits, app_rows = {}, []
    for i, regime in enumerate(PASSTHROUGH_REGIMES):
        df = ad.load_passthrough(regime=regime)
        fits[regime] = {}
        for name, cols in (("full", CONTROLS), ("reduced", reduced_cols)):
            r = _fit(df, cols, B=B, seed=SEED + i)
            fits[regime][name] = r
            app_rows.append({
                "regime": regime, "set": name, "n": r.nobs, "controls": len(cols),
                "F": round(r.stat, 3), "cv": round(r.critical_value, 3),
                "p": round(r.pvalue, 3), "verdict": r.decision(),
                "alpha": round(r.alpha, 4),
                "theta": round(r.theta, 3), "se": round(r.theta_se, 3),
            })
            print(f"      {regime} {name:8s} F={r.stat:7.3f} p={r.pvalue:.3f}")
        ad.plot_bootstrap_null(fits[regime]["full"]).savefig(
            IMG / f"null_{regime.replace('-', '_')}.png"
        )
    results["application"] = app_rows

    ad.plot_block_structure(fits["1999-2007"]["full"].first_stage.folds).savefig(
        IMG / "blocks.png"
    )

    labels = list(PASSTHROUGH_REGIMES)
    ad.plot_diagnostic(
        {
            "full controls (7)": pd.DataFrame(
                {"p": [fits[r]["full"].pvalue for r in labels]}, index=labels),
            "drop M2, oil (5)": pd.DataFrame(
                {"p": [fits[r]["reduced"].pvalue for r in labels]}, index=labels),
        },
        xlabel="monetary regime",
    ).savefig(IMG / "diagnostic.png")

    # --------------------------------------------------------- diagnostics --
    print("[5/7] trend-absorption diagnostic + penalty sweep")
    df99 = ad.load_passthrough(regime="1999-2007")
    integ99 = [c for c in DEFAULT_INTEGRATED if c in CONTROLS]
    diag = ad.trend_absorption(
        df99["cpi"], df99["neer"], df99[CONTROLS], drop=REDUCED_DROP,
        lags=4, n_blocks=5, buffer=6, integrated=integ99, B=B, seed=SEED,
    )
    results["diagnostic"] = {
        "table": json.loads(diag.to_frame().round(4).to_json(orient="records")),
        "delta_m": round(diag.delta_m, 4),
        "delta_W": round(diag.delta_W, 4),
        "theta_spread": round(diag.theta_spread, 4),
        "verdict": diag.verdict(),
        "dropped": diag.dropped,
    }

    sweep = ad.penalty_sensitivity(
        df99["cpi"], df99["neer"], df99[CONTROLS],
        lags_grid=(4,), integrated=integ99, n_blocks=5, buffer=6,
    )
    results["penalty_sweep"] = json.loads(sweep.round(4).to_json(orient="records"))
    results["theta_sign_flips"] = bool(sweep.attrs.get("theta_sign_flips", False))

    # -------------------------------------------------------- monte carlo --
    print(f"[6/7] Monte Carlo (R={R})")
    cells = [
        ad.run_design(design=g, T=200, R=R, B=max(B // 5, 99), d=40,
                      borrowed_bound=5.73, seed=SEED,
                      lags=2, n_blocks=5, buffer=3)
        for g in "ABCDE"
    ]
    mc = ad.montecarlo_table(cells)
    results["montecarlo"] = json.loads(mc.to_json(orient="records"))

    plot_frame = mc.rename(columns={"size rej @ 5.73": "rej @ borrowed",
                                    "size @ boot": "rej @ boot"})
    ad.plot_size_comparison(
        plot_frame, borrowed_col="rej @ borrowed", boot_col="rej @ boot",
        labels={k: k for k in "ABCDE"},
    ).savefig(IMG / "size.png")

    ultra = ad.run_ultra_check(T=100, d=150, R=max(R // 3, 8), seed=SEED)
    results["ultra"] = json.loads(ultra.to_json(orient="records"))

    # ------------------------------------------------------------- sample --
    print("[7/7] sample-use grid")
    su = ad.sample_use_table(n=108)
    results["sample_use"] = {
        "n": 108,
        "K": [int(k) for k in su.index],
        "h": [int(c) for c in su.columns],
        "values": [[round(float(v), 2) for v in row] for row in su.to_numpy()],
    }

    results["settings"]["elapsed_sec"] = round(time.time() - t_start, 1)
    (HERE / "_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nwrote {len(list(IMG.glob('*.png')))} figures to {IMG}")
    print(f"wrote tables to {HERE / '_results.json'}")
    print(f"elapsed {results['settings']['elapsed_sec']:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=999)
    ap.add_argument("--R", type=int, default=60)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    main(B=a.B, R=a.R, quick=a.quick)
