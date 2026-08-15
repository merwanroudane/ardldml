# DML-Bounds, step by step

A tutorial for someone who has never used this package, and possibly never run
a bounds test. Every code block is runnable as written. Work through it in
order and by the end you will have produced a full set of results, figures and
LaTeX tables from real macroeconomic data, and — more importantly — you will
know which numbers you are allowed to believe.

**Contents**

1. [Setup](#step-0--setup)
2. [What question are we asking?](#step-1--what-question-are-we-asking)
3. [The classical test first](#step-2--do-the-classical-test-first)
4. [Where classical bounds come from](#step-3--where-do-bounds-actually-come-from)
5. [Adding controls: the problem](#step-4--adding-controls-and-the-problem-that-creates)
6. [Your first DML-Bounds fit](#step-5--your-first-dml-bounds-fit)
7. [Getting a critical value](#step-6--getting-a-critical-value)
8. [Reading the output](#step-7--reading-the-output)
9. [The diagnostic you must run](#step-8--the-diagnostic-you-must-run)
10. [Tuning K, h and the penalty](#step-9--tuning-k-h-and-the-penalty)
11. [All four regimes](#step-10--the-full-application)
12. [Figures and LaTeX](#step-11--figures-and-latex)
13. [Checking it on data you control](#step-12--check-it-on-data-you-control)
14. [Common mistakes](#common-mistakes)

---

## Step 0 — Setup

```bash
pip install ardldml
```

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import ardldml as ad

print(ad.__version__)
```

Everything below runs offline. The data ships with the package.

---

## Step 1 — What question are we asking?

We want to know whether two variables move together **in the long run**, not
whether they happen to be correlated month to month.

Concretely: does the exchange rate pass through to consumer prices? If a
long-run relationship exists, then when the dollar and the price level drift
apart, some force pulls them back. That force is the **speed of adjustment**,
`α`. If `α = 0` there is no such force and no long-run relationship, whatever
the correlations look like.

Load the data:

```python
df = ad.load_passthrough(regime="1999-2007")
print(df.shape)          # (108, 9)
print(df.columns.tolist())
# ['cpi', 'neer', 'm2', 'ffr', 'ip', 'unrate', 'oil', 'gs10', 'baa']
```

`cpi` and `neer` are already in logs. The other seven are our controls.

Look at it before modelling anything:

```python
fig = ad.plot_series(df, title="Pass-through data, 1999–2007")
fig.savefig("output/data.png")
```

---

## Step 2 — Do the classical test first

Always start here. If the classical test on two variables gives a clean answer,
you may not need any of the machinery in this package.

```python
res_c = ad.classical_bounds_test(
    y=df["cpi"],
    x=df[["neer"]],       # note: a DataFrame, even with one column
    lags=4, order=4, case=3,
)
print(res_c.summary())
```

You get three numbers, corresponding to the three steps of the procedure:

| step | statistic | what it rules out |
| --- | --- | --- |
| 1 | `F` on all level terms | no level effects at all |
| 2 | `t` on `π_y` | `Y` is `I(1)` and cointegrated with nothing |
| 3 | Wald on `θ` | `Y` is stationary but unrelated to `D` |

**All three must reject** before you may claim a long-run relationship.
Rejecting step 1 alone is not enough — two degenerate cases survive it. Step 3
uses ordinary χ² critical values because `θ̂` is asymptotically normal; steps 1
and 2 do not.

### The five deterministic cases

`case` controls the intercept and trend, and it changes the critical values:

| case | intercept | trend | use when |
| --- | --- | --- | --- |
| 1 | none | none | series fluctuate around zero |
| 2 | restricted | none | non-zero means, no trending |
| 3 | unrestricted | none | **the common default** |
| 4 | unrestricted | restricted | `Y` trends, source unclear — the safe trending choice |
| 5 | unrestricted | unrestricted | rarely needed |

Case 3 is the default here. If your series visibly trend, case 4 is the safer
option.

---

## Step 3 — Where do bounds actually come from?

This is worth understanding before you trust any of them.

Pesaran, Shin and Smith obtained their tables by **simulation** — `T = 1000`,
40,000 replications — from a data-generating process they printed in the notes
to Table CI. They ran it twice: once with regressors purely `I(1)` (giving the
upper bound) and once purely `I(0)` (the lower bound).

You can do the same thing yourself, right now:

```python
cv = ad.simulate_pss_bounds(k=1, case=3, T=1000, nsim=20_000, seed=0)
print(cv)
#         I(0)   I(1)
# level
# 0.10   4.03   4.76
# 0.05   4.85   5.70
# 0.01   6.78   7.88
```

Compare with print:

```python
print(ad.pss_reference(k=1, case=3))
#         I(0)  I(1)
# 0.10    4.04  4.78
# 0.05    4.94  5.73
# 0.01    6.84  7.84
```

Two consequences worth internalising:

**You are not stuck with `T = 1000`.** Pass your own sample size and you get
finite-sample bounds — which is what Narayan (2004) tabulated for `n = 30…80`,
except you generate them instead of looking them up:

```python
ad.simulate_pss_bounds(k=1, case=3, T=108, nsim=20_000, seed=0)
#         I(0)   I(1)
# level
# 0.10   4.06   4.87
# 0.05   4.99   5.88
# 0.01   7.09   8.17
```

Compare the 5% upper bound: **5.88** at `T = 108` against **5.70**
asymptotically. Using `T = 1000` bounds on 108 observations over-rejects, which
is Narayan's point.

**Watch out for `statsmodels`.** Its `UECMResults.bounds_test` indexes the
critical-value table with `k + 1` instead of `k`, so it reports bounds that are
too small:

```python
print(ad.statsmodels_offset(k=1, case=3))
#                                  I(0)  I(1)
# correct (k=1)                    4.94  5.73
# statsmodels reports (k=2 row)    3.79  4.85
```

`ardldml` never calls it.

---

## Step 4 — Adding controls, and the problem that creates

Now suppose the bivariate relation is confounded. Monetary policy, oil, output
— all plausibly drive both prices and the dollar. So condition on them.

Here is the trap. Some of those controls are **themselves integrated**, and
they may share stochastic trends with `cpi` and `neer`. When you project the
lagged levels onto them, you can absorb the very trend that identifies the
relation.

The quantity that matters becomes the **effective integrated count**:

```
k̃ = k − r
```

`r` counts the cointegrating relations between the tested levels and the
integrated controls. Where the null sits depends on it:

```python
fig = ad.plot_bracket(k=10, k_tilde=6)
fig.savefig("output/bracket.png")
```

- `k̃ = k` — nothing absorbed — the classical `I(1)` endpoint, 5.73
- `k̃ = 0` — everything absorbed — the `I(0)` endpoint, 4.94
- in between — somewhere inside, and you do not know where

**Stationary controls are harmless.** A stationary regressor cannot track a
stochastic trend, so it cannot absorb one. Add a hundred and `k̃ = k` still.
Only integrated controls make the bracket live. This is why the next step asks
you which controls are `I(1)`.

---

## Step 5 — Your first DML-Bounds fit

```python
from ardldml import DMLBounds, CONTROLS

integrated = ["m2", "ip", "oil", "gs10", "baa", "ffr"]

model = DMLBounds(
    y=df["cpi"],
    d=df["neer"],
    W=df[CONTROLS],
    lags=4,             # short-run lag order p
    n_blocks=5,         # cross-fitting blocks K
    buffer=6,           # buffer h — covers ~6 months of memory
    integrated=integrated,
)
res = model.fit()
print(res)
```

### What just happened

**The balanced first stage.** Two projections with *different* regressors:

| target | order | projected on |
| --- | --- | --- |
| `ΔY_t` | `I(0)` | stationary controls in levels, integrated controls **differenced**, lagged `ΔY`, contemporaneous `ΔD` |
| `Z_{t−1} = (Y_{t−1}, D_{t−1})` | `I(1)` | control **levels** |

The short-run block is equation (3) verbatim: `δΔD_t + Σᵢ γᵢΔY_{t−i}`, with no
lagged `ΔD`. If you want the general ARDL(p,q) structure instead, pass
`dlags=True` — but that is a departure from the paper, so say so if you use it.

Regressing a stationary target on integrated levels would be spurious. All
absorption happens in the second row.

Inspect it:

```python
print(res.first_stage.design.summary())
```

**Cross-fitting.** Five contiguous blocks, each predicted by a model trained on
the others minus a six-observation buffer:

```python
fig = ad.plot_block_structure(res.first_stage.folds)
fig.savefig("output/blocks.png")
```

The white gaps are the buffer. That is sample you paid to decouple first-stage
error from evaluation innovations.

**The statistic.** Equation (10) of the paper: an unpenalised, **no-intercept**
regression of the residualised `ΔY` on the residualised levels, and the `F`
form of the Wald test that both coefficients are zero.

### Why you cannot interpret `res.stat` yet

```python
print(res.summary())
```

The summary says so explicitly. There is no table to read it against. Which
brings us to the point of the whole exercise.

---

## Step 6 — Getting a critical value

```python
res = res.bootstrap(B=999, seed=20260625)
```

### What the bootstrap does

1. Fit two auxiliary models under the null: a **restricted conditional model**
   for `ΔY` with the level terms removed (residuals `ε̂`), and a **marginal
   model** for `ΔD` on its own lags and the selected differenced controls
   (residuals `v̂`).
2. For each of `B` draws: pick one Rademacher sequence `η_t ∈ {−1,+1}` and
   apply it to the **stacked pair** `(ε̂_t η_t, v̂_t η_t)`.
3. Regenerate `ΔD*` from the marginal dynamics with the control path **fixed**,
   cumulate to `D*`; regenerate `ΔY*` from the restricted dynamics driven by
   `ΔD*` and `ε*`, cumulate to `Y*`.
4. Recompute the **entire** residualised statistic on `(Y*, D*, W)` —
   re-running first-stage selection — so the simulated null carries the same
   generated-regressor error the observed statistic does.

**Why one shared weight?** Because the focal regressor need not be exogenous.
Reweighting `ε̂` alone while holding `D` fixed makes the simulated innovations
independent of the regressor path by construction — a world with
`corr(ε, v) = 0` regardless of the data. The shared weight carries the
empirical cross-covariance into every draw. You can see it:

```python
print(f"corr(eps, v) = {res.boot['corr_eps_v']:.3f}")
```

**Why hold `W` fixed?** That conditions on the realised trend content — the
object the bracket is built on. And since the marginal model conditions on the
differenced controls, any trend `D` shares with `W` is inherited through the
fixed path rather than destroyed.

Look at the null you just generated:

```python
fig = ad.plot_bootstrap_null(res, borrowed=5.73)
fig.savefig("output/null.png")
```

The gap between the bootstrap critical value and the dashed 5.73 line is the
entire argument for not using a table.

---

## Step 7 — Reading the output

```python
print(res.summary())
```

| field | meaning | what to check |
| --- | --- | --- |
| `F` | the statistic | meaningless alone |
| `bootstrap cv (95%)` | generated critical value | compare with 5.73 to see the distortion |
| `bootstrap p` | `B⁻¹ Σ 1{F* ≥ F}` | **this is your answer** |
| `alpha` | speed of adjustment | should be in `(0, 1]`; outside `[0, 2)` signals misspecification |
| `theta` | long-run coefficient | check the sign against theory |
| `theta_se` | delta-method SE | compare across specifications |

**A significant `θ` means nothing if the test fails to reject.** Under the null
`α = 0`, the error-correction term drops out of the model entirely, so the
long-run coefficients are not identified. Report the test first.

**Economic vs statistical significance.** A speed of adjustment of 0.005 means
half-life measured in centuries. The paper makes this point about Bitcoin: with
enough observations you can detect an economically irrelevant effect. Look at
`alpha` before celebrating a small p-value.

---

## Step 8 — The diagnostic you must run

Everything so far tests whether `cpi` and `neer` cointegrate **given** those
seven controls. If one of the controls is part of the equilibrium system,
you have partialled out the relation itself, and your non-rejection means
nothing.

```python
diag = ad.trend_absorption(
    df["cpi"], df["neer"], df[CONTROLS],
    drop=["m2", "oil"],        # chosen on economic grounds
    lags=4, n_blocks=5, buffer=6,
    integrated=integrated,
    B=999, seed=20260625, progress=True,
)
print(diag.summary())
```

Four fits, two gaps:

- `Δ_m = p_ols − p_ad` — does penalisation matter?
- `Δ_W = p_full − p_red` — does dropping trend-sharing controls matter?

**How to choose `drop`.** Not by trying combinations until one rejects. Pick
the controls most likely to be *cointegrated with the relation under test* on
economic grounds, and commit before you look. For pass-through, money and oil
both plausibly share trends with the dollar and the price level.

**How to read it.**

| pattern | reading |
| --- | --- |
| concordant verdicts across all four | conclusion is robust |
| reduced rejects, full does not, `Δ_W > 0`, and `θ` sharper under reduced | over-absorption; **trust the reduced set** |
| discordant in the other direction | both verdicts fragile |

`diag.verdict()` says which it thinks you have, conservatively.

---

## Step 9 — Tuning K, h and the penalty

### Sample cost

```python
print(ad.sample_use_table(n=len(df)))
```

On 108 observations, `K=5, h=10` leaves each fold training on about 62% of the
sample. Buffering is not free, and this is why power is weak at small `T`.

### Rules of thumb

| parameter | guidance |
| --- | --- |
| `lags` | data frequency: 4 or 8 for quarterly, 12 for monthly if you can afford it |
| `n_blocks` | 5–6; more blocks means smaller training sets |
| `buffer` | cover the memory of the process; 6–12 for monthly |
| `penalty` | `"plugin"` is fast and principled; `"tscv"` is slower and data-driven |
| `adaptive` | leave `True` — plain `ℓ₁` over-selects integrated regressors and induces spurious absorption |

### Does the penalty change your answer?

It should not. If it does, say so.

```python
for pen in ["plugin", "tscv"]:
    r = (DMLBounds(df["cpi"], df["neer"], df[CONTROLS], lags=4, n_blocks=5,
                   buffer=6, integrated=integrated, penalty=pen)
         .fit().bootstrap(B=299, seed=20260625))
    print(f"{pen:8s}  F={r.stat:6.3f}  p={r.pvalue:.3f}  theta={r.theta:+.3f}")
```

The paper's own appendix shows a case rejecting at one penalty and not another.
Report the sensitivity rather than the most convenient cell.

---

## Step 10 — The full application

```python
from ardldml import passthrough_regimes, REDUCED_DROP, DEFAULT_INTEGRATED, regime_table

results = {}
for regime in passthrough_regimes():
    d_r = ad.load_passthrough(regime=regime)
    fits = {}
    for name, cols in [("full", CONTROLS),
                       ("reduced", [c for c in CONTROLS if c not in REDUCED_DROP])]:
        fits[name] = (
            DMLBounds(d_r["cpi"], d_r["neer"], d_r[cols], lags=4, n_blocks=5, buffer=6,
                      integrated=[c for c in DEFAULT_INTEGRATED if c in cols])
            .fit().bootstrap(B=999, seed=20260625)
        )
    results[regime] = fits
    print(f"{regime} done")

table = regime_table(results)
print(table.to_string(index=False))
```

Testing within regimes rather than across them avoids conflating a structural
break with a long-run relationship.

---

## Step 11 — Figures and LaTeX

```python
ad.use_journal_style()

ad.plot_regimes(ad.load_passthrough(), ad.PASSTHROUGH_REGIMES, "neer") \
  .savefig("output/fig1_regimes.png")
ad.plot_bracket(k=10, k_tilde=6).savefig("output/fig2_bracket.png")
ad.plot_bootstrap_null(results["1999-2007"]["full"]).savefig("output/fig3_null.png")

ad.to_latex(
    table, caption="Exchange-rate pass-through across monetary regimes",
    label="tab:passthrough",
    notes="Critical values are the 95th percentile of the restricted system "
          "wild bootstrap, B = 999. Verdicts are bootstrap decisions, not "
          "comparisons with tabulated bounds.",
    path="output/tab_passthrough.tex", index=False,
)
```

Write that note, or something like it, in any paper using this method. A reader
who assumes you compared with 5.73 will misread your table.

---

## Step 12 — Check it on data you control

Before trusting the method on real data, watch it behave where you know the
answer.

```python
from ardldml import simulate_design

for rho, label in [(1.0, "null (no cointegration)"), (0.5, "alternative")]:
    y, d, W, integ = simulate_design("D", T=200, rho=rho, d=40, seed=20260625)
    r = (DMLBounds(y, d, W, lags=2, n_blocks=5, buffer=3, integrated=integ)
         .fit().bootstrap(B=299, seed=20260625))
    print(f"{label:28s} F={r.stat:6.3f}  p={r.pvalue:.3f}  -> {r.decision()}")
```

Under the null it should fail to reject about 95% of the time. Under the
alternative it should reject — but **not always**: power at `T = 200` under
integrated nuisance is roughly two-thirds. That is a real property of the
method, not a bug.

To see the over-rejection the bootstrap fixes:

```python
from ardldml import run_design
print(run_design(design="D", T=200, R=50, B=199, d=40, seed=20260625))
```

Compare `rej @ 5.73` with `rej @ boot` in the size row.

---

## Common mistakes

**Passing differenced data.** `y`, `d` and `W` all go in as **levels**. The
package differences internally. Passing differences silently tests something
else.

**Skipping the diagnostic.** A non-rejection with a rich control set is
ambiguous between "no long-run relationship" and "the controls ate it".

**Comparing with 5.73.** The whole point is that you cannot.

**Reading `θ` after a non-rejection.** Under the null the error-correction term
drops out and `θ` is not identified.

**Using it on 40 observations.** Power is not there. Use the classical test
with finite-sample bounds:
`simulate_pss_bounds(k=1, case=3, T=40, nsim=40_000)`.

**Letting the ADF fallback classify your controls.** Pass `integrated=`
explicitly. The fallback's pre-test error is not propagated into the inference.

**Choosing `drop` after seeing results.** That turns the diagnostic into
specification search. Commit first.

---

## Where to go next

- `examples/01_quickstart.py` — the shortest complete run
- `examples/02_passthrough.py` — all four regimes with figures and LaTeX
- `examples/03_montecarlo.py` — size and power across designs
- [README](../README.md) — full syntax reference and limitations
