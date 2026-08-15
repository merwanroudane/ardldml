"""
Bundled data: exchange-rate pass-through to U.S. prices.

The package ships the monthly series behind the paper's main application, so
every example runs offline on real macroeconomic data.

Source
------
FRED-MD, the standard monthly macroeconomic database of McCracken and Ng
(2016), vintage 2025-06. Nine series are extracted and stored in
``data/passthrough.csv`` over 1973-01 to 2020-12.

======================  ==============  ===========================================
FRED-MD code            column          description
======================  ==============  ===========================================
``CPIAUCSL``            ``cpi``         Consumer price index, all urban consumers
``TWEXAFEGSMTHx``       ``neer``        Trade-weighted U.S. dollar index
``M2SL``                ``m2``          M2 money stock
``FEDFUNDS``            ``ffr``         Effective federal funds rate
``INDPRO``              ``ip``          Industrial production index
``UNRATE``              ``unrate``      Civilian unemployment rate
``OILPRICEx``           ``oil``         Crude oil spot price, West Texas Intermediate
``GS10``                ``gs10``        10-year Treasury constant maturity yield
``BAA``                 ``baa``         Moody's Baa corporate bond yield
======================  ==============  ===========================================

The sample starts in 1973-01 because that is the first observation of the
trade-weighted dollar index, which is also why the paper's first monetary
regime starts there.

Why this reproduces the paper's setup
-------------------------------------
The paper takes the log CPI as :math:`Y`, the log trade-weighted dollar as the
focal regressor :math:`D`, and conditions on seven macroeconomic and financial
controls: M2, the federal funds rate, industrial production, the unemployment
rate, the WTI oil price, the ten-year Treasury yield and the Baa corporate
yield. That is exactly the set above. The four monetary regimes give complete
samples of 156, 156, 108 and 156 observations, matching the ``n`` column of the
paper's Table 11 exactly.

Transformations
---------------
``log=True`` takes logs of the quantity and price series -- ``cpi``, ``neer``,
``m2``, ``ip``, ``oil`` -- and leaves the four interest and unemployment rates
in levels, since they are already in percentage-point units and can approach
zero. This is the conventional treatment and the one the linearisation of the
pass-through relation requires.

References
----------
McCracken, M. W. and Ng, S. (2016). FRED-MD: A monthly database for
macroeconomic research. *Journal of Business & Economic Statistics*, 34(4),
574-589.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "PASSTHROUGH_REGIMES",
    "CONTROLS",
    "REDUCED_DROP",
    "LOG_SERIES",
    "load_passthrough",
    "passthrough_regimes",
    "data_path",
]

#: The four monetary-policy regimes of the paper's Table 11.
PASSTHROUGH_REGIMES: Dict[str, Tuple[str, str]] = {
    "1973-1985": ("1973-01", "1985-12"),
    "1986-1998": ("1986-01", "1998-12"),
    "1999-2007": ("1999-01", "2007-12"),
    "2008-2020": ("2008-01", "2020-12"),
}

#: The seven controls of the full conditioning set.
CONTROLS: List[str] = ["m2", "ffr", "ip", "unrate", "oil", "gs10", "baa"]

#: Controls dropped to form the reduced set, per the paper: the two most likely
#: to share a stochastic trend with the pass-through relation itself.
REDUCED_DROP: List[str] = ["m2", "oil"]

#: Series transformed to logs when ``log=True``.
LOG_SERIES: List[str] = ["cpi", "neer", "m2", "ip", "oil"]

#: Controls treated as I(1) in the balanced design, on economic grounds.
DEFAULT_INTEGRATED: List[str] = ["m2", "ip", "oil", "gs10", "baa", "ffr"]


def data_path(name: str = "passthrough.csv") -> Path:
    """Absolute path to a bundled data file."""
    return Path(__file__).resolve().parent / "data" / name


def load_passthrough(
    regime: Optional[str] = None,
    log: bool = True,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load the exchange-rate pass-through dataset.

    Parameters
    ----------
    regime : str, optional
        One of the keys of :data:`PASSTHROUGH_REGIMES`. If ``None``, the full
        1973-2020 sample is returned.
    log : bool
        Take logs of the quantity and price series; see the module docstring.
    start, end : str, optional
        Explicit date bounds, applied after ``regime``.

    Returns
    -------
    pandas.DataFrame
        Monthly, indexed by period-start timestamps, with columns ``cpi``,
        ``neer`` and the seven controls.

    Examples
    --------
    >>> df = load_passthrough(regime="1999-2007")          # doctest: +SKIP
    >>> len(df)                                             # doctest: +SKIP
    108
    """
    df = pd.read_csv(data_path(), parse_dates=["date"]).set_index("date")
    df.index.freq = None

    if regime is not None:
        if regime not in PASSTHROUGH_REGIMES:
            raise ValueError(
                f"regime must be one of {list(PASSTHROUGH_REGIMES)}; got {regime!r}"
            )
        a, b = PASSTHROUGH_REGIMES[regime]
        df = df.loc[a:b]
    if start is not None:
        df = df.loc[start:]
    if end is not None:
        df = df.loc[:end]

    if log:
        out = df.copy()
        for c in LOG_SERIES:
            if c in out.columns:
                if (out[c] <= 0).any():
                    raise ValueError(f"cannot take logs of {c!r}: non-positive values present")
                out[c] = np.log(out[c])
        df = out
    return df


def passthrough_regimes(log: bool = True) -> Dict[str, pd.DataFrame]:
    """Load all four regimes at once, keyed by regime label."""
    return {k: load_passthrough(regime=k, log=log) for k in PASSTHROUGH_REGIMES}
