"""Shared fixtures: the Appendix B data-generating process, in miniature."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_system(T=180, rho=0.5, d_ctl=12, delta=0.0, seed=1, burn=60):
    """
    A small instance of the paper's core system.

    ``Y_t = D_t + u_t`` with ``D_t`` a random walk and ``u_t = rho u_{t-1} + e_t``;
    the nuisance matrix is half I(1) random walks loading on a latent common
    trend and half stationary AR(1) with coefficient 0.5. Endogeneity enters
    through ``v_t = delta e_t + sqrt(1 - delta^2) xi_t``.
    """
    rng = np.random.default_rng(seed)
    n = T + burn
    trend = np.cumsum(rng.standard_normal(n))

    cols, integ = {}, []
    for j in range(d_ctl // 2):
        name = f"w1_{j}"
        cols[name] = np.cumsum(rng.standard_normal(n)) + 0.5 * trend
        integ.append(name)
    for j in range(d_ctl // 2):
        s = np.zeros(n)
        e = rng.standard_normal(n)
        for t in range(1, n):
            s[t] = 0.5 * s[t - 1] + e[t]
        cols[f"w0_{j}"] = s

    e = rng.standard_normal(n)
    xi = rng.standard_normal(n)
    v = delta * e + np.sqrt(max(1 - delta**2, 0.0)) * xi
    D = np.cumsum(v) + 0.3 * trend

    u = np.zeros(n)
    for t in range(1, n):
        u[t] = rho * u[t - 1] + e[t]

    idx = pd.RangeIndex(n)
    return (
        pd.Series(D + u, index=idx, name="Y").iloc[burn:],
        pd.Series(D, index=idx, name="D").iloc[burn:],
        pd.DataFrame(cols, index=idx).iloc[burn:],
        integ,
    )


@pytest.fixture(scope="module")
def system():
    return make_system()
