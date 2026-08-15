"""
Monte Carlo: does the borrowed bound over-reject, and does the bootstrap fix it?

    python examples/03_montecarlo.py [--R 100] [--B 199] [--T 200] [--jobs 4]

Reproduces the structure of the paper's Table 3 and Figure 2 across the five
designs, plus the endogeneity comparison of its Table 5.

Cost warning
------------
A full cell at R=1000, B=999 is a million statistic evaluations. Defaults here
are small enough to finish in minutes. Scale up with --R, --B and --jobs.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import pandas as pd

import ardldml as ad
from ardldml import DESIGNS, run_design, run_endogeneity_grid

SEED = 20260625
OUT = Path("output")
BORROWED = 5.73  # PSS case III, k=1, 5% upper bound


def main(R: int = 100, B: int = 199, T: int = 200, d: int = 40, jobs: int = 1) -> None:
    OUT.mkdir(exist_ok=True)
    ad.use_journal_style()

    print(f"Monte Carlo: R={R}, B={B}, T={T}, d={d}")
    print(f"borrowed bound = {BORROWED} (Pesaran-Shin-Smith case III, k=1, 5%)\n")
    for key, spec in DESIGNS.items():
        print(f"  {key}: {spec.description} (frac I(1) = {spec.frac_i1}, "
              f"common trend = {spec.common_trend})")
    print()

    cells = []
    for design in DESIGNS:
        t0 = time.time()
        cell = run_design(
            design=design, T=T, R=R, B=B, d=d,
            borrowed_bound=BORROWED, seed=SEED,
            lags=2, n_blocks=5, buffer=3,
        )
        cells.append(cell)
        size = cell[cell["kind"] == "size"].iloc[0]
        power = cell[cell["kind"] == "power"].iloc[0]
        print(
            f"design {design}:  size @ {BORROWED} = {size[f'rej @ {BORROWED}']:.3f}   "
            f"size @ boot = {size['rej @ boot']:.3f}   "
            f"power @ boot = {power['rej @ boot']:.3f}   [{time.time() - t0:.0f}s]"
        )

    table = ad.montecarlo_table(cells)
    print("\n" + table.to_string(index=False))
    table.to_csv(OUT / "tab3_montecarlo.csv", index=False)
    ad.to_latex(
        table,
        caption="Empirical size and power across the five designs",
        label="tab:montecarlo",
        notes=(
            f"R = {R} replications, B = {B} bootstrap draws, nominal level 5\\%. "
            f"'size @ {BORROWED}' is the rejection rate of a true null against the "
            "borrowed classical bound; 'size @ boot' against the restricted system "
            "wild bootstrap critical value."
        ),
        path=OUT / "tab3_montecarlo.tex",
        index=False,
    )

    plot_frame = table.rename(
        columns={f"size rej @ {BORROWED}": "rej @ borrowed", "size @ boot": "rej @ boot"}
    )
    ad.plot_size_comparison(
        plot_frame, borrowed_col="rej @ borrowed", boot_col="rej @ boot",
        labels={k: f"{k}\n{DESIGNS[k].description.split(' / ')[0][:18]}" for k in DESIGNS},
    ).savefig(OUT / "fig7_size.png")

    print("\nEndogeneity: system vs fixed-regressor bootstrap")
    endo = run_endogeneity_grid(
        deltas=(0.0, 0.4, 0.8), T=T, R=max(R // 2, 20), B=B, d=d,
        seed=SEED, lags=2, n_blocks=5, buffer=3,
    )
    print(endo.to_string(index=False))
    endo.to_csv(OUT / "tab4_endogeneity.csv", index=False)
    print(
        "\nAt delta = 0 the two schemes should agree -- that is the check that joint "
        "regeneration introduces no distortion of its own."
    )
    print(f"\nwrote results to {OUT.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=100)
    ap.add_argument("--B", type=int, default=199)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--d", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    main(R=a.R, B=a.B, T=a.T, d=a.d, jobs=a.jobs)
