"""
Quickstart: the shortest complete DML-Bounds run.

    python examples/01_quickstart.py

Fits one regime of the exchange-rate pass-through data, attaches a bootstrap
critical value, and prints the summary. Runs in well under a minute.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import ardldml as ad
from ardldml import CONTROLS, DEFAULT_INTEGRATED, DMLBounds

SEED = 20260625


def main() -> None:
    df = ad.load_passthrough(regime="1999-2007")
    print(f"loaded {len(df)} monthly observations, {df.shape[1]} series")
    print(f"controls: {', '.join(CONTROLS)}\n")

    # 1. The classical benchmark, on two variables only.
    classical = ad.classical_bounds_test(df["cpi"], df[["neer"]], lags=4, order=4, case=3)
    print(classical.summary(T=len(df), nsim=5_000))
    print()

    # 2. DML-Bounds, conditioning on all seven controls.
    res = (
        DMLBounds(
            df["cpi"], df["neer"], df[CONTROLS],
            lags=4, n_blocks=5, buffer=6,
            integrated=[c for c in DEFAULT_INTEGRATED if c in CONTROLS],
        )
        .fit()
        .bootstrap(B=999, seed=SEED)
    )
    print(res.summary())

    print("\nOne-row frame:")
    print(res.to_frame().to_string(index=False))

    print(
        "\nNote: the statistic has no tabulated reference. The bootstrap critical "
        f"value is {res.critical_value:.3f}, against a borrowed classical bound of "
        "5.73 -- comparing with the latter is what this method exists to avoid."
    )


if __name__ == "__main__":
    main()
