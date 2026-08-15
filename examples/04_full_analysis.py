"""
The whole analysis, end to end, on the bundled real data.

    python examples/04_full_analysis.py                     # everything, ~15-40 min
    python examples/04_full_analysis.py --quick             # same shape, small B and R
    python examples/04_full_analysis.py --skip-montecarlo   # empirical results only
    python examples/04_full_analysis.py --outdir docs/example --style sunny

This is the long-form companion to ``01_quickstart.py``. It runs every public
entry point of the package against the bundled FRED-MD series and writes every
table and every figure the package can produce, in order, with the reasoning in
between: what the classical test says, why its critical values cannot be
borrowed here, what cross-fitting costs, what the bootstrap gives back, and
which of those conclusions survive a change of conditioning set.

Nothing is downloaded and nothing is synthetic except the Monte Carlo section,
which is synthetic by definition. The data ship inside the package.

Output
------
Each step writes into ``--outdir`` (default ``output/full``):

    fig01..fig12  *.png      every figure, one file each
    tab01..tab09  *.csv      the same tables as data
                  *.tex      booktabs versions, caption and notes attached
                  *.md       markdown versions
    _manifest.json           the transcript, plus the source of every step

``_manifest.json`` is what the project website renders, so the page and this
script can never disagree: the page shows the source of the function that ran
and the output it actually printed.

On replication
--------------
The source paper does not publish its FRED series codes, data vintage,
per-control log/level treatment, or its cross-fitting settings, so the data
mapping here is inferred from the variable descriptions. These numbers are
**not** a replication of its published table and should not be cited as one.
What does reproduce exactly is the sample design: 156, 156, 108 and 156
observations across the four monetary regimes.
"""

from __future__ import annotations

import argparse
import inspect
import io
import json
import textwrap
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
FOCUS = "1999-2007"          # the regime carried through the single-fit steps
LAGS, N_BLOCKS, BUFFER = 4, 5, 6


# ---------------------------------------------------------------------------
# Plumbing: a step registry that records what each step ran and what it printed
# ---------------------------------------------------------------------------
STEPS: list = []


def step(title: str, blurb: str = ""):
    """Register a function as a numbered step of the walkthrough."""

    def deco(fn):
        STEPS.append((title, blurb, fn))
        return fn

    return deco


class _Tee(io.TextIOBase):
    """Write to the console and to the transcript at the same time."""

    def __init__(self, *targets):
        self.targets = targets

    def write(self, s):
        for t in self.targets:
            t.write(s)
        return len(s)

    def flush(self):
        for t in self.targets:
            t.flush()


class Run:
    """Shared state, and the two recorders every step uses."""

    def __init__(self, outdir: Path, B: int, R: int, seed: int):
        self.outdir = outdir
        self.B, self.R, self.seed = B, R, seed
        self.figures: list = []
        self.tables: list = []
        self.data: pd.DataFrame | None = None
        self.fits: dict = {}

    # -- recorders ---------------------------------------------------------
    def fig(self, name: str, figure, caption: str):
        """Save a figure and register it for the manifest."""
        path = self.outdir / f"{name}.png"
        figure.savefig(path)
        plt.close(figure)
        self.figures.append({"file": path.name, "caption": caption})
        print(f"    -> {path.name}")

    def table(self, name: str, frame: pd.DataFrame, caption: str,
              notes: str = "", show: bool = True, index: bool = False):
        """Save a table as csv, LaTeX and markdown, print it, and register it."""
        frame.to_csv(self.outdir / f"{name}.csv", index=index)
        ad.to_latex(frame, caption=caption, label=f"tab:{name}", notes=notes,
                    path=self.outdir / f"{name}.tex", index=index)
        md = ad.to_markdown(frame, index=index)
        (self.outdir / f"{name}.md").write_text(md, encoding="utf-8")
        if show:
            print(md)
        self.tables.append({
            "file": f"{name}.csv",
            "caption": caption,
            "notes": notes,
            "columns": [str(c) for c in frame.columns],
            "rows": json.loads(frame.to_json(orient="records")),
        })
        print(f"    -> {name}.csv / .tex / .md")

    # -- shared helper -----------------------------------------------------
    def fit(self, df: pd.DataFrame, cols, seed_offset: int = 0):
        """The one specification used throughout: 4 lags, 5 blocks, 6-month buffer."""
        return (
            DMLBounds(
                df["cpi"], df["neer"], df[cols],
                lags=LAGS, n_blocks=N_BLOCKS, buffer=BUFFER,
                integrated=[c for c in DEFAULT_INTEGRATED if c in cols],
            )
            .fit()
            .bootstrap(B=self.B, seed=self.seed + seed_offset)
        )


# ---------------------------------------------------------------------------
# 1. The data
# ---------------------------------------------------------------------------
@step(
    "The data",
    "Nine monthly FRED-MD series, 1973-2020, bundled with the package. "
    "Nothing is downloaded.",
)
def step_data(run: Run):
    full = ad.load_passthrough()
    run.data = full

    print(f"{len(full)} months, {full.shape[1]} series, "
          f"{full.index.min():%Y-%m} to {full.index.max():%Y-%m}")
    print(f"outcome   cpi   (log US consumer price index)")
    print(f"regressor neer  (log trade-weighted dollar)")
    print(f"controls  {', '.join(CONTROLS)}")
    print(f"of which integrated: {', '.join(DEFAULT_INTEGRATED)}")
    print(f"reduced set drops:   {', '.join(REDUCED_DROP)}\n")

    desc = (full.describe().T[["mean", "std", "min", "max"]]
            .round(3).reset_index().rename(columns={"index": "series"}))
    run.table("tab01_data_summary", desc,
              "The bundled pass-through data, 1973-2020.",
              notes="Series in logs except the rates, which are in percent.")

    spans = pd.DataFrame([
        {"regime": r, "start": a, "end": b,
         "n": len(ad.load_passthrough(regime=r))}
        for r, (a, b) in PASSTHROUGH_REGIMES.items()
    ])
    run.table("tab02_regimes", spans,
              "The four monetary regimes and their sample sizes.",
              notes="Testing within regimes avoids conflating a structural "
                    "break with a long-run relationship.")

    run.fig("fig01_series", ad.plot_series(full, title=None),
            "All nine series over the full 1973-2020 window.")
    run.fig("fig02_regimes", ad.plot_regimes(full, PASSTHROUGH_REGIMES, "neer"),
            "The trade-weighted dollar with the four regime windows shaded.")


# ---------------------------------------------------------------------------
# 2. Where the critical values come from
# ---------------------------------------------------------------------------
@step(
    "Where the critical values come from",
    "No bounds table is stored in the package. The classical bracket is "
    "regenerated from the data-generating process Pesaran, Shin and Smith "
    "printed in the notes to their Table CI, which makes it checkable against "
    "print and extends it past where the published tables stop.",
)
def step_critvals(run: Run):
    tab = ad.critical_value_table(k_values=(1, 2, 3, 4), case=3, T=1000,
                                  nsim=8000, level=0.05, seed=0)
    print(f"simulated, {tab.attrs['note']}\n")

    rows = []
    for k in (1, 2, 3, 4):
        try:
            pub = ad.pss_reference(k, 3)
            lo = round(float(pub.loc[0.05, "I(0)"]), 2)
            hi = round(float(pub.loc[0.05, "I(1)"]), 2)
        except KeyError:
            lo = hi = None          # not in the published table at all
        rows.append({
            "k": k,
            "simulated I(0)": float(tab.loc[k, "I(0)"]),
            "published I(0)": lo,
            "simulated I(1)": float(tab.loc[k, "I(1)"]),
            "published I(1)": hi,
        })
    run.table("tab03_bounds_validation", pd.DataFrame(rows),
              "Regenerated 5% bounds against the published table, case III, T = 1000.",
              notes="Agreement is to Monte Carlo error. The empty published "
                    "cells are values the printed table does not carry at all; "
                    "the simulator has no such gaps, which is the practical "
                    "argument for generating bounds rather than storing them.")

    off = ad.statsmodels_offset(k=1, case=3)
    print("\nthe statsmodels off-by-one, k = 1, case III:")
    print(off.round(3).to_string())
    print("\n  UECMResults.bounds_test indexes its table with k+1 instead of k,")
    print("  so the bounds it reports are one row too far down, hence too small,")
    print("  and the test over-rejects. ardldml never calls it.")

    run.fig("fig03_bracket", ad.plot_bracket(k=10, k_tilde=6),
            "The trend-absorption bracket: as the effective integrated count "
            "falls from k to 0, the limiting null slides from the I(1) endpoint "
            "to the I(0) endpoint. Classical bounds testing is the right-hand end.")


# ---------------------------------------------------------------------------
# 3. The classical test, as a benchmark
# ---------------------------------------------------------------------------
@step(
    "The classical test first",
    "Run the ordinary three-step procedure before anything else, so there is "
    "something to compare against. Once with the dollar alone, once holding "
    "the seven controls fixed.",
)
def step_classical(run: Run):
    df = ad.load_passthrough(regime=FOCUS)

    plain = ad.classical_bounds_test(df["cpi"], df[["neer"]], lags=LAGS, case=3)
    print(f"[dollar alone]      {plain.summary()}")

    withw = ad.classical_bounds_test(df["cpi"], df[["neer"]], lags=LAGS, case=3,
                                     fixed=df[CONTROLS])
    print(f"\n[controls fixed]    {withw.summary()}")

    rows = []
    for label, cb in (("dollar alone", plain), ("+ 7 controls fixed", withw)):
        br = cb.bounds(nsim=8000, seed=0)
        lo, hi = float(br.loc[0.05, "I(0)"]), float(br.loc[0.05, "I(1)"])
        rows.append({
            "specification": label, "n": cb.nobs, "k": cb.k,
            "F": round(cb.f_stat, 3), "t": round(cb.t_stat, 3),
            "I(0) 5%": round(lo, 3), "I(1) 5%": round(hi, 3),
            # bounds_stars returns "" below the lower bound; spell it out here
            "verdict": ad.bounds_stars(cb.f_stat, lo, hi) or "fail to reject",
            "theta(neer)": round(float(cb.long_run["neer"]), 3),
        })
    run.table("tab04_classical", pd.DataFrame(rows),
              f"The classical bounds test on the {FOCUS} regime.",
              notes="Bounds are simulated at this sample size, not read off a "
                    "table calibrated at T = 1000. Holding controls fixed is "
                    "the best the classical procedure can do with them; it "
                    "cannot account for their trend content.")


# ---------------------------------------------------------------------------
# 4. What cross-fitting costs
# ---------------------------------------------------------------------------
@step(
    "What cross-fitting costs",
    "The h-block buffer is what decouples first-stage error from the "
    "evaluation-fold innovations. It is not free, and the cost should be "
    "visible before a configuration is chosen.",
)
def step_folds(run: Run):
    n = len(ad.load_passthrough(regime=FOCUS))
    su = ad.sample_use_table(n=n)
    grid = su.round(3).reset_index()
    grid.columns = ["K"] + [f"h = {c}" for c in su.columns]
    run.table("tab05_sample_use", grid,
              f"Average share of the sample available for training, n = {n}.",
              notes="Rows are the number of blocks K, columns the buffer h. "
                    "This is why power is weak at small T.")

    folds = ad.hblock_folds(n=n, n_blocks=N_BLOCKS, buffer=BUFFER)
    print(f"\nchosen: K = {N_BLOCKS}, h = {BUFFER}, n = {n}")
    print(folds.to_frame().to_string(index=False))

    run.fig("fig04_blocks", ad.plot_block_structure(folds),
            "The h-block partition actually used: evaluation window, training "
            "set, and the discarded buffer between them.")


# ---------------------------------------------------------------------------
# 5. One fit, in full
# ---------------------------------------------------------------------------
@step(
    "One fit, in full",
    "The headline specification on a single regime, with the whole summary "
    "printed and the simulated null drawn. Everything after this is the same "
    "call repeated across regimes and settings.",
)
def step_fit(run: Run):
    df = ad.load_passthrough(regime=FOCUS)
    res = run.fit(df, CONTROLS)
    run.fits[(FOCUS, "full")] = res

    print(res.summary())

    fs = res.first_stage
    print(f"\ncontrol levels retained by the adaptive projection: "
          f"{int(fs.supports['Z'].sum())} of {len(CONTROLS)}")
    print(f"that is the empirical counterpart of k-tilde: keep none and the "
          f"null sits at the classical I(1) corner.")

    run.table("tab06_single_fit",
              ad.result_table(res, labels=[f"{FOCUS}, full controls"]).reset_index()
                .rename(columns={"index": ""}),
              f"DML-Bounds on the {FOCUS} regime, all seven controls.",
              notes=f"Restricted system wild bootstrap, B = {run.B}, "
                    f"lags = {LAGS}, K = {N_BLOCKS}, h = {BUFFER}.")

    run.fig("fig05_null_focus", ad.plot_bootstrap_null(res),
            f"The bootstrap null for {FOCUS}, with the observed statistic and "
            f"the borrowed classical bound marked. The gap between the "
            f"bootstrap critical value and the borrowed bound is the whole "
            f"argument for not using a table.")


# ---------------------------------------------------------------------------
# 6. Every regime, both conditioning sets
# ---------------------------------------------------------------------------
@step(
    "Every regime, both conditioning sets",
    "Four regimes crossed with the full and reduced control sets: eight fits, "
    "each with its own bootstrap. The reduced set drops money and the oil "
    "price, the two controls most likely to share a trend with the "
    "pass-through relation itself.",
)
def step_regimes(run: Run):
    reduced = [c for c in CONTROLS if c not in REDUCED_DROP]
    per_regime: dict = {}

    for i, regime in enumerate(PASSTHROUGH_REGIMES):
        df = ad.load_passthrough(regime=regime)
        per_regime[regime] = {}
        for name, cols in (("full", CONTROLS), ("reduced", reduced)):
            key = (regime, name)
            res = run.fits.get(key) or run.fit(df, cols, seed_offset=i)
            run.fits[key] = res
            per_regime[regime][name] = res
            print(f"  {regime}  {name:8s}  n={res.nobs:4d}  F={res.stat:8.3f}  "
                  f"cv={res.critical_value:8.3f}  p={res.pvalue:.3f}  "
                  f"{res.decision()}")
        run.fig(f"fig06_null_{regime.replace('-', '_')}",
                ad.plot_bootstrap_null(per_regime[regime]["full"]),
                f"Bootstrap null, {regime}, full control set.")

    print()
    run.table("tab07_regimes", ad.regime_table(per_regime),
              "Pass-through by monetary regime and conditioning set.",
              notes=f"Critical values are the 95th percentile of the restricted "
                    f"system wild bootstrap, B = {run.B}. Verdicts are bootstrap "
                    f"decisions, not comparisons with a tabulated bound.")

    labels = list(PASSTHROUGH_REGIMES)
    run.fig("fig07_diagnostic", ad.plot_diagnostic(
        {
            "full controls (7)": pd.DataFrame(
                {"p": [run.fits[(r, "full")].pvalue for r in labels]}, index=labels),
            f"drop {', '.join(REDUCED_DROP)} ({len(reduced)})": pd.DataFrame(
                {"p": [run.fits[(r, "reduced")].pvalue for r in labels]}, index=labels),
        },
        xlabel="monetary regime",
    ), "Full against reduced conditioning set across the four regimes. A "
       "verdict that flips when trend-sharing controls are dropped is the "
       "signature of over-absorption.")


# ---------------------------------------------------------------------------
# 7. The diagnostic that has to be run
# ---------------------------------------------------------------------------
@step(
    "The over-absorption diagnostic",
    "The estimand is conditional. If a control is itself part of the "
    "equilibrium system, residualising removes the relation rather than the "
    "confounding, and a non-rejection then means nothing. Four fits: full and "
    "reduced control sets crossed with the adaptive and unpenalised level "
    "projection.",
)
def step_diagnostic(run: Run):
    df = ad.load_passthrough(regime=FOCUS)
    diag = ad.trend_absorption(
        df["cpi"], df["neer"], df[CONTROLS], drop=REDUCED_DROP,
        lags=LAGS, n_blocks=N_BLOCKS, buffer=BUFFER,
        integrated=[c for c in DEFAULT_INTEGRATED if c in CONTROLS],
        B=run.B, seed=run.seed,
    )
    run.table("tab08_trend_absorption", ad.diagnostic_table(diag),
              f"The four-fit trend-absorption diagnostic, {FOCUS} regime.",
              notes=f"Dropped from the reduced set: {', '.join(diag.dropped)}.")

    print(f"\ndelta_m (ols - adaptive) = {diag.delta_m:+.4f}")
    print(f"delta_W (full - reduced) = {diag.delta_W:+.4f}")
    print(f"theta spread across fits = {diag.theta_spread:.4f}")
    print(f"\nreading: {diag.verdict()}")


# ---------------------------------------------------------------------------
# 8. Penalty sensitivity
# ---------------------------------------------------------------------------
@step(
    "Penalty sensitivity",
    "A verdict can turn on the penalty and the projection. Reporting one cell "
    "is specification search; reporting the grid is the method. The column to "
    "watch is the number of control levels retained, the empirical "
    "counterpart of k-tilde.",
)
def step_penalty(run: Run):
    df = ad.load_passthrough(regime=FOCUS)
    sweep = ad.penalty_sensitivity(
        df["cpi"], df["neer"], df[CONTROLS],
        lags_grid=(LAGS,), n_blocks=N_BLOCKS, buffer=BUFFER,
        integrated=[c for c in DEFAULT_INTEGRATED if c in CONTROLS],
    )
    run.table("tab09_penalty_sweep", sweep.round(4),
              f"Estimators against penalty rules, {FOCUS} regime.",
              notes="Rules are lambda-min, the geometric midpoint and "
                    "lambda-1se. The unpenalised projection ignores the rule.")

    if sweep.attrs.get("theta_sign_flips"):
        print("\n  theta changes sign across this grid: the conditioning "
              "choice, not the data, is driving it.")
    else:
        print("\n  theta keeps its sign across the grid.")


# ---------------------------------------------------------------------------
# 9. Monte Carlo
# ---------------------------------------------------------------------------
@step(
    "Monte Carlo: does the borrowed bound over-reject?",
    "The empirical sections cannot answer whether the procedure is correctly "
    "sized, because the truth is unknown there. These designs are the paper's "
    "own, and the truth is imposed.",
)
def step_montecarlo(run: Run):
    cells = [
        ad.run_design(design=g, T=200, R=run.R, B=max(run.B // 5, 99), d=40,
                      borrowed_bound=5.73, seed=run.seed,
                      lags=2, n_blocks=5, buffer=3)
        for g in ad.DESIGNS
    ]
    mc = ad.montecarlo_table(cells)
    run.table("tab10_montecarlo", mc,
              f"Empirical size and power, R = {run.R} replications, nominal 5%.",
              notes="'size rej @ 5.73' is the rejection rate of a true null "
                    "against the borrowed classical bound; '@ boot' against "
                    "the bootstrap critical value.")

    frame = mc.rename(columns={"size rej @ 5.73": "rej @ borrowed",
                               "size @ boot": "rej @ boot"})
    run.fig("fig08_size", ad.plot_size_comparison(
        frame, borrowed_col="rej @ borrowed", boot_col="rej @ boot",
        labels={g: g for g in ad.DESIGNS},
    ), "Empirical size under the null. The borrowed classical bound "
       "over-rejects once the nuisance space is integrated; the restricted "
       "system wild bootstrap holds size near the nominal 5%.")

    print("\nwhen the classical test cannot run at all:")
    ultra = ad.run_ultra_check(T=100, d=150, R=max(run.R // 3, 8), seed=run.seed)
    run.table("tab11_ultra", ultra,
              "T = 100 with d = 150 controls.",
              notes="The unpenalised conditional ECM has a singular Gram "
                    "matrix in every replication, so no statistic exists. "
                    "This is implementability, not a size comparison.")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(B: int = 999, R: int = 60, quick: bool = False,
         outdir: str = "output/full", style: str = "journal",
         skip_montecarlo: bool = False) -> None:
    if quick:
        B, R = 199, 12

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    if style == "sunny":
        ad.use_sunny_style(transparent=True)
    else:
        ad.use_journal_style()

    run = Run(out, B=B, R=R, seed=SEED)
    steps = [s for s in STEPS if not (skip_montecarlo and s[2] is step_montecarlo)]

    print("=" * 78)
    print("ardldml -- the complete analysis on the bundled FRED-MD data")
    print(f"B = {B} bootstrap draws, R = {R} Monte Carlo replications, seed {SEED}")
    print(f"writing to {out.resolve()}")
    print("=" * 78)

    t0 = time.time()
    records = []
    for i, (title, blurb, fn) in enumerate(steps, start=1):
        head = f"[{i}/{len(steps)}] {title}"
        print(f"\n\n{head}\n{'-' * len(head)}")
        if blurb:
            print(textwrap.fill(blurb, 78) + "\n")

        n_fig, n_tab = len(run.figures), len(run.tables)
        buf = io.StringIO()
        t_step = time.time()
        with _redirect(buf):
            fn(run)
        records.append({
            "n": i,
            "title": title,
            "blurb": blurb,
            "code": _clean_source(fn),
            "output": buf.getvalue().rstrip(),
            "seconds": round(time.time() - t_step, 1),
            "figures": run.figures[n_fig:],
            "tables": run.tables[n_tab:],
        })

    elapsed = round(time.time() - t0, 1)
    manifest = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "version": ad.__version__,
        "elapsed_sec": elapsed,
        "settings": {"B": B, "R": R, "seed": SEED, "lags": LAGS,
                     "n_blocks": N_BLOCKS, "buffer": BUFFER, "focus": FOCUS,
                     "style": style, "montecarlo": not skip_montecarlo},
        "script": Path(__file__).name,
        "steps": records,
    }
    (out / "_manifest.json").write_text(json.dumps(manifest, indent=1),
                                        encoding="utf-8")

    print("\n\n" + "=" * 78)
    print(f"done in {elapsed:.0f}s -- {len(run.figures)} figures, "
          f"{len(run.tables)} tables")
    print(f"everything is in {out.resolve()}")
    print("=" * 78)


def _clean_source(fn) -> str:
    """The body of a step function, without the registration decorator."""
    src = inspect.getsource(fn)
    lines = src.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("def "))
    return "\n".join(lines[start:])


class _redirect:
    """Tee stdout into a buffer while leaving the console output intact."""

    def __init__(self, buf):
        self.buf = buf

    def __enter__(self):
        import sys

        self._old = sys.stdout
        sys.stdout = _Tee(self._old, self.buf)
        return self

    def __exit__(self, *exc):
        import sys

        sys.stdout = self._old
        return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--B", type=int, default=999, help="bootstrap draws")
    ap.add_argument("--R", type=int, default=60, help="Monte Carlo replications")
    ap.add_argument("--quick", action="store_true", help="B=199, R=12")
    ap.add_argument("--outdir", default="output/full")
    ap.add_argument("--style", choices=("journal", "sunny"), default="journal")
    ap.add_argument("--skip-montecarlo", action="store_true")
    a = ap.parse_args()
    main(B=a.B, R=a.R, quick=a.quick, outdir=a.outdir, style=a.style,
         skip_montecarlo=a.skip_montecarlo)
