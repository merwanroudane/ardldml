# ardldml

**ARDL bounds testing for cointegration when you have many persistent controls.**

`ardldml` implements **DML-Bounds**: a test for a long-run (cointegrating)
relationship in which the lagged levels are first orthogonalised against a
high-dimensional control set, and inference comes from a restricted system wild
bootstrap rather than from a table.

```bash
pip install ardldml
```

```python
from ardldml import DMLBounds, load_passthrough, CONTROLS

df = load_passthrough(regime="1999-2007")           # real monthly FRED-MD data
res = (
    DMLBounds(df["cpi"], df["neer"], df[CONTROLS],
              lags=4, n_blocks=5, buffer=6,
              integrated=["m2", "ip", "oil", "gs10", "baa", "ffr"])
    .fit()
    .bootstrap(B=999, seed=20260625)
)
print(res.summary())
```

---

## Table of contents

- [Why this exists](#why-this-exists)
- [The idea in five minutes](#the-idea-in-five-minutes)
- [Installation](#installation)
- [The one thing you must not skip](#the-one-thing-you-must-not-skip)
- [Syntax reference](#syntax-reference)
- [Worked example: exchange-rate pass-through](#worked-example-exchange-rate-pass-through)
- [Monte Carlo](#monte-carlo)
- [Figures and tables](#figures-and-tables)
- [Two bugs and a convention](#two-bugs-and-a-convention)
- [Honest limitations](#honest-limitations)
- [API index](#api-index)
- [References](#references)

---

## Why this exists

The classical ARDL bounds test of Pesaran, Shin and Smith (2001) is popular
because it avoids pre-testing the integration order of the regressors. It was
not designed for a large conditioning set. When you condition on forty
macroeconomic controls, two things break:

1. **The projection becomes unstable or infeasible.** With `d > T` the
   unpenalised conditional ECM has a singular Gram matrix and no statistic
   exists at all.
2. **Residualising can absorb the thing you are testing for.** Partialling
   lagged levels out against controls that themselves carry stochastic trends
   removes part of the long-run variation that identifies the error-correction
   relation.

The second is the subtle one, and it is what this package is really about.

## The idea in five minutes

**Classical bounds testing brackets one unknown.** You do not know whether the
regressors are `I(0)` or `I(1)`, so you compute two critical values — one for
each polar case — and read your statistic against the interval. Outside the
interval the verdict is conclusive; inside it, inconclusive.

**With many persistent controls a second unknown appears.** Some controls may
share stochastic trends with the variables under test. Projecting on them can
absorb those trends. What then governs the null distribution is not the
integration order of the original regressors but the **effective integrated
count**

$$\tilde{k} = k - r$$

where `r` is the number of cointegrating relations between the tested levels
and the integrated control block. This is the paper's central object.

**The bracket reappears, over a different quantity.**

| effective integrated count | where the null sits |
| --- | --- |
| `k̃ = k` (nothing absorbed) | the classical `I(1)` upper endpoint |
| `0 < k̃ < k` | strictly inside the classical bracket |
| `k̃ = 0` (fully absorbed) | the classical `I(0)` lower endpoint |

A purely stationary control set forces `k̃ = k`: a stationary regressor cannot
track the stochastic trend of an integrated one, so the upper-bound logic
survives no matter how many stationary controls you add. Only **integrated**
controls make the bracket live.

**`k̃` is never estimated.** It indexes the limiting experiment; it is not a
tuning parameter. The bootstrap conditions on its realised value implicitly, by
holding the control path fixed and regenerating the focal regressor from a
marginal model that conditions on the differenced controls — so any common
trend is inherited rather than broken.

### Three moving parts

**1. A balanced first stage.** The two projections use *different* regressors,
because their targets have different integration orders:

| target | integration order | projected on |
| --- | --- | --- |
| `ΔY_t` | `I(0)` | stationary controls in **levels**, integrated controls in **differences**, lagged differences of `Y` and `D` |
| `Z_{t-1} = (Y_{t-1}, D_{t-1})` | `I(1)` | control **levels** |

Regressing a stationary target on integrated levels would be unbalanced and
spurious. Absorption lives entirely in the second row.

**2. h-block cross-fitting.** `K` contiguous chronological blocks, each
evaluated by a model trained on the others *minus an `h`-observation buffer*.
The buffer is what decouples first-stage error from evaluation-fold
innovations. It costs sample — `sample_use_table()` shows exactly how much.

**3. A restricted system wild bootstrap.** One Rademacher weight is applied to
the **stacked** pair of conditional and marginal residuals, and `Y*` and `D*`
are regenerated jointly. This preserves the contemporaneous correlation between
them — the endogeneity channel the whole Pesaran–Shin–Smith setup exists for. A
scheme that holds `D` fixed and reweights only the equation error simulates a
world with zero correlation whatever the data say.

### Why there is no bounds table in this package

Because tabulated critical values are not valid here. The generated-regressor
remainder is `O_p(s·log d/√T)`, which at `s=3, d=40, T=200` is about **0.78** —
order one, not negligible. The asymptotics have not taken hold at the sample
sizes applied work uses. Every critical value is computed:

| route | function | use |
| --- | --- | --- |
| restricted system wild bootstrap | `restricted_system_wild_bootstrap` | **inference on real data** |
| simulated null of the DGP | `empirical_critical_value` | Monte Carlo only (infeasible on real data) |
| classical bracket, regenerated | `simulate_pss_bounds` | benchmark and teaching |

Even the classical bounds are *simulated*, from the data-generating process
Pesaran, Shin and Smith printed in the notes to Table CI, rather than typed in
from the table. That makes them verifiable, extensible to any `T` (so the
small-sample bounds Narayan tabulated for `n = 30…80` come free), and
extensible past `k = 10` where the published tables stop.

```python
from ardldml import simulate_pss_bounds, pss_reference
simulate_pss_bounds(k=1, case=3, T=1000, nsim=40_000, seed=0)
#         I(0)   I(1)
# level
# 0.10   4.036  4.771
# 0.05   4.933  5.744     <- published: 4.94, 5.73
# 0.01   6.822  7.902
```

## Installation

```bash
pip install ardldml
```

From source:

```bash
git clone https://github.com/merwanroudane/ardldml
cd ardldml
pip install -e ".[dev]"
pytest
```

Requires Python ≥ 3.9 and `numpy`, `pandas`, `scipy`, `scikit-learn`,
`statsmodels`, `matplotlib`, `joblib`.

## The one thing you must not skip

**The estimand is conditional.** DML-Bounds does not ask whether `Y` and `D`
cointegrate. It asks whether they cointegrate *given `W`*. If a control is
itself part of the equilibrium system, residualising removes the relation
rather than the confounding, and a non-rejection reflects **over-absorption**,
not the absence of a long-run relationship.

This cannot be detected from a single fit. Always run the diagnostic:

```python
from ardldml import trend_absorption

diag = trend_absorption(
    df["cpi"], df["neer"], df[CONTROLS],
    drop=["m2", "oil"],          # controls most likely to share the tested trend
    lags=4, B=999, seed=20260625,
    integrated=["m2", "ip", "oil", "gs10", "baa", "ffr"],
)
print(diag.summary())
```

It fits four models — full/reduced control set × adaptive/unpenalised
projection — and reports two gaps:

- `Δ_m = p_ols − p_ad`
- `Δ_W = p_full − p_red`

**Reading it.** A large positive `Δ_W` — rejecting under the reduced set but
not the full set — combined with a long-run coefficient that is *more sharply
estimated* under the reduced set, says the nuisance space is eating the
relation. The reduced-set verdict is then the credible one. Concordant verdicts
across all four fits say your conclusion is not an artefact.

It is a hypothesis-generating device, not a formal test. It has no size and no
power. It tells you where to look.

## Syntax reference

### `DMLBounds(y, d, W, **spec)`

| argument | type | default | meaning |
| --- | --- | --- | --- |
| `y` | `Series` | — | outcome, in **levels** |
| `d` | `Series` | — | focal regressor, in **levels** |
| `W` | `DataFrame` | — | controls, in **levels**; may have more columns than rows |
| `lags` | `int` | `4` | short-run lag order `p` |
| `case` | `int` | `3` | deterministics of the *restricted bootstrap model*; does not enter the statistic |
| `n_blocks` | `int` | `5` | cross-fitting blocks `K` |
| `buffer` | `int` | `0` | buffer `h`; set it to cover the memory of the process |
| `adaptive` | `bool` | `True` | adaptive weights on the `m_Z` projection |
| `adaptive_integrated_only` | `bool` | `True` | restrict those weights to the **integrated block**, per §4.1 |
| `penalised` | `bool` | `True` | `False` gives unpenalised OLS projections (low-dimensional corner) |
| `penalty` | `str`/`float` | `"plugin"` | `"plugin"`, `"low"`/`"medium"`/`"high"`, or a fixed λ |
| `c` | `float` | `1.1` | constant in `λ = c·√(log d/n)·σ̂` |
| `integrated` | `list[str]` | `None` | controls to treat as `I(1)`; `None` triggers an ADF fallback |
| `include_constant` | `bool` | `False` | add an intercept to stage 3; `False` is equation (10) as written |

Returns a `DMLBoundsResults` after `.fit()`.

### `DMLBoundsResults`

| member | meaning |
| --- | --- |
| `.stat` | the `F`-form statistic |
| `.alpha` | speed of adjustment, `α = −π_y` |
| `.theta`, `.theta_se` | long-run coefficient and delta-method standard error |
| `.nobs` | effective sample size |
| `.first_stage` | `FirstStage` — residuals, folds, selected supports |
| `.estimable` | `False` if a projection exhausted its degrees of freedom |
| `.bootstrap(B, level, seed, scheme, n_jobs)` | attach a critical value; returns `self` |
| `.critical_value`, `.pvalue` | available after `.bootstrap()` |
| `.decision(level)` | `"reject"` / `"fail to reject"` |
| `.summary()`, `.to_frame()` | text and one-row frame |

`.bootstrap(scheme=...)` takes `"system"` (Algorithm 1, the default) or
`"fixed"` (holds `D` at its realised path — valid only under strong exogeneity,
retained because Algorithm 1 nests it).

### Choosing `K` and `h`

`h` should cover the memory of the process — with monthly data and `lags=4`,
`h` of 6–12 is reasonable. `K` trades bias against training-set size. Look
before you leap:

```python
from ardldml import sample_use_table
sample_use_table(n=108)
# h      0     2     5     10
# K
# 4   0.75  0.71  0.66  0.57
# 5   0.80  0.76  0.71  0.62
# 6   0.83  0.80  0.74  0.65
# 8   0.88  0.84  0.79  0.70
```

On a 108-observation regime, `K=5, h=10` leaves each fold training on 62% of
the sample. That is the cost of decoupling, and it is why power is modest at
small `T`.

## Worked example: exchange-rate pass-through

The bundled dataset is real: FRED-MD vintage 2025-06, monthly 1973-01 to
2020-12, nine series. `Y` is the log CPI, `D` the log trade-weighted dollar,
and the seven controls are M2, the federal funds rate, industrial production,
the unemployment rate, the WTI oil price, the ten-year Treasury yield and the
Baa corporate yield.

The four monetary regimes give **156, 156, 108 and 156** complete observations,
matching the paper's Table 11 exactly.

```python
from ardldml import (DMLBounds, load_passthrough, passthrough_regimes,
                     CONTROLS, REDUCED_DROP, DEFAULT_INTEGRATED, regime_table)

results = {}
for regime in passthrough_regimes():
    df = load_passthrough(regime=regime)
    fits = {}
    for name, cols in [("full", CONTROLS),
                       ("reduced", [c for c in CONTROLS if c not in REDUCED_DROP])]:
        fits[name] = (
            DMLBounds(df["cpi"], df["neer"], df[cols], lags=4, n_blocks=5, buffer=6,
                      integrated=[c for c in DEFAULT_INTEGRATED if c in cols])
            .fit().bootstrap(B=999, seed=20260625)
        )
    results[regime] = fits

print(regime_table(results).to_string(index=False))
```

Run it end to end with `python examples/02_passthrough.py`.

> **On replication.** The paper does not publish its FRED series codes, data
> vintage, per-control log/level treatment, or its `K` and `h` settings. The
> mapping above is inferred, so the numbers here are **not** a replication of
> its Table 11 and should not be cited as one. What does reproduce is the
> structure: sample sizes match exactly, and the full-versus-reduced contrast
> moves in the direction the diagnostic predicts.

## Monte Carlo

The paper's five designs are built in.

| design | nuisance space | `k̃` | purpose |
| --- | --- | --- | --- |
| A | low-dimensional stationary | `= k` | reproduce classical behaviour |
| B | high-dimensional `I(0)` | `= k` | residualisation helps |
| C | high-dimensional with `I(1)` block | `≤ k` | classical bounds distorted |
| D | cointegrated `I(1)` controls | `< k` | trend absorption reduces dimension |
| E | weak signal, near unit root, `d = T` | varies | robustness |

```python
from ardldml import run_design, montecarlo_table, plot_size_comparison

cells = [run_design(design=dg, T=200, R=100, B=199, d=40, seed=20260625)
         for dg in "ABCDE"]
tab = montecarlo_table(cells)
fig = plot_size_comparison(tab.assign(**{"rej @ 5.73": tab["size rej @ 5.73"],
                                         "rej @ boot": tab["size @ boot"]}))
```

A full cell at `R=1000, B=999` is a million statistic evaluations. The defaults
are deliberately small; scale up with `n_jobs`.

To compare the two bootstrap schemes under endogeneity:

```python
from ardldml import run_endogeneity_grid
run_endogeneity_grid(deltas=(0.0, 0.4, 0.8), T=200, R=100, B=199)
```

At `δ = 0` the schemes coincide — the sanity check that joint regeneration
introduces no distortion of its own.

### Robustness under oracle critical values (§7.5–7.6)

Fixes `d = 40` and traces size and power across sample sizes, penalties and
estimators using *method-specific empirical* critical values instead of the
bootstrap, which isolates the estimator from the bootstrap approximation:

```python
from ardldml import run_robustness_grid
run_robustness_grid(T_grid=(100, 250), R=50, mixed=True)   # mixed I(0)/I(1)
```

Watch the `cv95` column. Under mixed integrated nuisance it climbs far above
the borrowed 5.73 and keeps climbing with `T` — that is the inference problem
made numerical. Size-adjusted power flatters the unpenalised benchmark only
because it is recentred with a critical value nobody has in practice.

### Implementability (§7.7)

```python
from ardldml import run_ultra_check
run_ultra_check(T=100, d=150, R=40)
#               method  implementable  statistic (median, IQR)
#      Unpenalised ECM           0.0%              not defined
# DML-Bounds (h-block)         100.0%              2.45 (2.99)
```

At `d > T` the unpenalised Gram matrix is singular in every draw, so no
classical statistic exists at all.

## Figures and tables

All figures apply a journal style (serif, no top/right spine, Wong
colourblind-safe palette) and return a `matplotlib` `Figure`.

| function | figure |
| --- | --- |
| `plot_bracket` | the trend-absorption bracket |
| `plot_bootstrap_null` | simulated null with observed `F`, bootstrap cv, and the borrowed bound |
| `plot_size_comparison` | borrowed bound vs bootstrap, by design |
| `plot_diagnostic` | full vs reduced p-values across specifications |
| `plot_block_structure` | what cross-fitting did to your sample |
| `plot_series`, `plot_regimes` | the data |

Tables return `DataFrame`s; `to_latex` wraps any of them in `booktabs` with a
caption, label and notes block.

```python
from ardldml import result_table, to_latex
to_latex(result_table(res, labels=["(1)"]),
         caption="DML-Bounds, 1999--2007",
         label="tab:pt",
         notes="Restricted system wild bootstrap, B = 999.",
         path="output/tab_passthrough.tex")
```

## Two bugs and a convention

**The `k` convention.** `k` is the number of long-run forcing regressors, as in
Pesaran, Shin and Smith. It does **not** count the dependent variable. Two
neighbouring conventions exist and mixing them shifts every bound by one row.

**`statsmodels` gets it wrong.** `UECMResults.bounds_test` sets
`k = len(model.ardl_order)`, which is `k + 1`, and uses it to index a table that
is itself indexed by the PSS `k`. Every reported bound is one row too far down,
hence too small, so the test **over-rejects**. With one regressor it reports
`(3.80, 4.81)` where PSS give `(4.94, 5.73)`:

```python
from ardldml import statsmodels_offset
statsmodels_offset(k=1, case=3)
#                                  I(0)  I(1)
# correct (k=1)                    4.94  5.73
# statsmodels reports (k=2 row)    3.79  4.85
```

`ardldml` never calls it. `classical_bounds_test` computes the Wald statistic
itself and reads it against a simulated bracket.

**The paper's own `k` is different again.** It writes
`Z_{t-1} = (Y_{t-1}, D_{t-1})' ∈ ℝ^k`, so *its* `k` counts level terms. This
package uses the PSS convention throughout and converts at the boundary.

## Honest limitations

Stated plainly, because they affect how you should read a result.

- **Power is modest at small `T`.** Under integrated nuisance the paper reports
  bootstrap power of 0.366 at `T = 120`, reaching useful levels only from
  `T ≈ 250`. Most applied ARDL work runs at `n = 30–80`. There is no evidence,
  in the paper or here, that DML-Bounds is usable at `n = 40`. If your sample
  is short, the classical test with simulated finite-sample bounds is the more
  honest tool.
- **Over-absorption is not testable.** Assumption 5 — that the controls span
  confounding trends but not the cointegrating relation — cannot be verified.
  The diagnostic is a signal, not a guarantee.
- **The theory keeps the integrated control block fixed-dimensional.** Growing
  it is a sparse-cointegration selection problem the paper explicitly leaves
  open. Adaptive weighting here is a stabilisation device, not a
  selection-consistency theorem. The package warns when you classify more than
  `max(10, 0.1·n)` controls as `I(1)`.
- **The penalty can change the verdict, and sometimes the sign of θ.** On the
  bundled pass-through data the adaptive projection selects zero level controls
  at `c = 1.1` — no residualisation at all, the `k̃ = k` corner — while looser
  penalties absorb controls and flip θ. Run `penalty_sensitivity` and report
  the grid, not a cell.
- **Bootstrap validity with estimated first stages rests on a high-level
  condition.** The paper proves consistency under *oracle* projections; the
  step to the feasible procedure is an assumption, not a theorem.
- **The `I(0)`/`I(1)` classification of controls is a specification choice.**
  Passing `integrated=` explicitly on economic grounds is better than the ADF
  fallback, whose pre-test error is not propagated into any inference.

## API index

**Testing** — `DMLBounds`, `DMLBoundsResults`, `DMLBoundsSpec`,
`compute_statistic`

**Inference** — `restricted_system_wild_bootstrap`, `rademacher`

**Diagnostics** — `trend_absorption`, `TrendAbsorption`, `penalty_sensitivity`

**Classical benchmark** — `classical_bounds_test`, `ClassicalBounds`,
`conditional_ecm`, `restricted_null_model`

**Critical values** — `simulate_pss_bounds`, `pss_reference`,
`statsmodels_offset`, `n_restrictions`

**First stage** — `build_balanced_design`, `BalancedDesign`, `classify_controls`,
`adaptive_post_lasso`, `cross_fit_projection`, `plugin_penalty`, `tscv_penalty`,
`PENALTY_RULES`

**Cross-fitting** — `hblock_folds`, `BlockStructure`, `sample_use`,
`sample_use_table`

**Simulation** — `simulate_design`, `run_design`, `run_endogeneity_grid`,
`run_robustness_grid`, `run_ultra_check`, `empirical_critical_value`,
`DESIGNS`, `default_d`, `RHO_ALTERNATIVES`

**Data** — `load_passthrough`, `passthrough_regimes`, `CONTROLS`,
`REDUCED_DROP`, `DEFAULT_INTEGRATED`

**Output** — `plot_*`, `result_table`, `regime_table`, `montecarlo_table`,
`diagnostic_table`, `critical_value_table`, `to_latex`, `to_markdown`,
`use_journal_style`

## References

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey,
W. and Robins, J. (2018). Double/debiased machine learning for treatment and
structural parameters. *The Econometrics Journal*, 21(1), C1–C68.

McCracken, M. W. and Ng, S. (2016). FRED-MD: A monthly database for
macroeconomic research. *Journal of Business & Economic Statistics*, 34(4),
574–589.

Narayan, P. K. (2004). Reformulating critical values for the bounds
F-statistics approach to cointegration. Monash University Discussion Paper
02/04.

Pesaran, M. H., Shin, Y. and Smith, R. J. (2001). Bounds testing approaches to
the analysis of level relationships. *Journal of Applied Econometrics*, 16(3),
289–326.

Zou, H. (2006). The adaptive lasso and its oracle properties. *Journal of the
American Statistical Association*, 101(476), 1418–1429.

## Licence

MIT. See [LICENSE](LICENSE).
