"""
h-block cross-fitting for time series.

Cross-fitting removes the bias that arises when the same observations are used
both to estimate a nuisance function and to evaluate the score built from it.
In cross-sectional DML the folds are random; in a time series they cannot be,
because randomly held-out observations are adjacent in time to their own
training data and the first-stage error is then correlated with the evaluation
innovations.

DML-Bounds uses the *h-block* scheme: the sample is cut into ``K`` contiguous
chronological blocks, and each block is evaluated using a model trained on the
other blocks *minus an ``h``-observation buffer* on either side of the
evaluation window. The buffer is what buys the decoupling -- Lemma 2 of the
paper turns Neyman orthogonality from an assumption into a consequence of the
construction, provided :math:`h \\to \\infty` and :math:`h = o(T)`.

The cost is sample. The buffer discards at most :math:`h(K-1)` observations
from the training sets, which is second-order asymptotically (Remark 7 notes
:math:`h(K-1)/T \\to 0`) but very much first-order at the sample sizes applied
ARDL work uses. :func:`sample_use` reports exactly how much is lost, so the
trade-off is visible rather than implicit.

Choosing ``h`` and ``K``
------------------------
``h`` should cover the memory of the process: with monthly data and an ARDL of
order 4, ``h`` of 6-12 is reasonable. ``K`` trades bias against the size of
each training set; ``K = 5`` is the default here and ``K = 6`` is used in the
paper's applications. Larger ``K`` means more, smaller, training sets and more
buffer loss in total.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd

__all__ = ["BlockStructure", "hblock_folds", "sample_use", "sample_use_table"]


@dataclass
class BlockStructure:
    """
    The evaluation/training partition produced by :func:`hblock_folds`.

    Attributes
    ----------
    n : int
        Number of observations.
    n_blocks : int
        Number of chronological blocks ``K``.
    buffer : int
        Buffer width ``h`` in observations.
    eval_blocks : list of numpy.ndarray
        Index arrays of the evaluation windows. These partition ``range(n)``.
    train_blocks : list of numpy.ndarray
        Index arrays of the corresponding training samples, each excluding its
        own evaluation window and the ``h`` observations either side of it.
    """

    n: int
    n_blocks: int
    buffer: int
    eval_blocks: List[np.ndarray] = field(default_factory=list)
    train_blocks: List[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        return self.n_blocks

    def __iter__(self):
        return iter(zip(self.train_blocks, self.eval_blocks))

    def to_frame(self) -> pd.DataFrame:
        """One row per block, with its extent and training size."""
        rows = []
        for b, (tr, ev) in enumerate(zip(self.train_blocks, self.eval_blocks), start=1):
            rows.append(
                {
                    "block": b,
                    "eval_start": int(ev[0]),
                    "eval_end": int(ev[-1]),
                    "n_eval": int(ev.size),
                    "n_train": int(tr.size),
                    "share_train": tr.size / self.n,
                }
            )
        return pd.DataFrame(rows).set_index("block")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<BlockStructure n={self.n} K={self.n_blocks} h={self.buffer} "
            f"mean train share={np.mean([t.size for t in self.train_blocks]) / self.n:.2f}>"
        )


def hblock_folds(n: int, n_blocks: int = 5, buffer: int = 0) -> BlockStructure:
    """
    Build contiguous evaluation blocks with buffered training sets.

    Parameters
    ----------
    n : int
        Number of observations.
    n_blocks : int
        Number of chronological blocks ``K``. Must be at least 2.
    buffer : int
        Buffer width ``h``. Observations within ``h`` of an evaluation window
        are dropped from that window's training set.

    Returns
    -------
    BlockStructure

    Raises
    ------
    ValueError
        If the buffer is so wide that some block has an empty training set,
        which happens when ``K`` is small and ``h`` is large relative to ``n``.

    Examples
    --------
    >>> bs = hblock_folds(100, n_blocks=4, buffer=5)
    >>> bs.to_frame()["n_train"].tolist()
    [70, 65, 65, 70]
    """
    if n_blocks < 2:
        raise ValueError(f"n_blocks must be at least 2; got {n_blocks!r}")
    if buffer < 0:
        raise ValueError(f"buffer must be non-negative; got {buffer!r}")
    if n_blocks > n:
        raise ValueError(f"n_blocks={n_blocks} exceeds n={n}")

    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    struct = BlockStructure(n=int(n), n_blocks=int(n_blocks), buffer=int(buffer))
    all_idx = np.arange(n)

    for b in range(n_blocks):
        lo, hi = edges[b], edges[b + 1]
        ev = all_idx[lo:hi]
        keep = np.ones(n, dtype=bool)
        keep[max(0, lo - buffer):min(n, hi + buffer)] = False
        tr = all_idx[keep]
        if tr.size == 0:
            raise ValueError(
                f"block {b + 1} has an empty training set: buffer={buffer} is too wide "
                f"for n={n} with n_blocks={n_blocks}"
            )
        struct.eval_blocks.append(ev)
        struct.train_blocks.append(tr)
    return struct


def sample_use(n: int, n_blocks: int = 5, buffer: int = 0) -> float:
    """
    Average share of the sample available for training, across blocks.

    This is the quantity that makes the cost of buffering concrete. With no
    buffer it is ``1 - 1/K``; each unit of ``h`` removes roughly ``2h/n`` more.
    """
    bs = hblock_folds(n, n_blocks=n_blocks, buffer=buffer)
    return float(np.mean([t.size for t in bs.train_blocks]) / n)


def sample_use_table(
    n: int,
    n_blocks: Tuple[int, ...] = (4, 5, 6, 8),
    buffers: Tuple[int, ...] = (0, 2, 5, 10),
) -> pd.DataFrame:
    """
    Training-sample share for a grid of ``K`` and ``h``.

    Useful before committing to a configuration on a short sample: it shows at
    a glance when the buffer starts to cost more than the cross-fitting buys.

    Returns
    -------
    pandas.DataFrame
        Rows indexed by ``K``, columns by ``h``, entries the average training
        share. ``NaN`` marks configurations that are infeasible.
    """
    out = pd.DataFrame(index=pd.Index(n_blocks, name="K"), columns=list(buffers), dtype=float)
    out.columns.name = "h"
    for k in n_blocks:
        for h in buffers:
            try:
                out.loc[k, h] = sample_use(n, n_blocks=k, buffer=h)
            except ValueError:
                out.loc[k, h] = np.nan
    return out
