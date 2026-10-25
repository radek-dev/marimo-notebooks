# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "matplotlib", "numpy", "pandas"]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    return mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Correlation Measures
    ## Kendall's tau: agreement between pairs

    Kendall's correlation compares the ordering of **pairs of observations**.
    A pair is concordant if both variables order it the same way, discordant
    if they order it oppositely, and tied if either variable has equal values.
    Tau-b adjusts for ties and ranges from -1 to +1 when defined.

    All data is synthetic.
    <a href="https://molab.marimo.io/github/radek-dev/marimo-notebooks/blob/main/correlation_measures/summary.py/wasm?mode=read&amp;show-code=false" target="_blank" rel="noopener noreferrer">Back to Correlation Measures Summary</a>

    [Back to the Notebook Gallery](https://radek-dev.github.io/notebook-gallery/)

    ## 1. Explore ordering and ties

    Increase noise to scramble the ordering. Enable rounding to introduce
    ties in y; the random seed stays fixed.
    """)
    return


@app.cell
def _(mo):
    slope = mo.ui.slider(-2, 2, value=1, step=0.5, label="Slope")
    noise = mo.ui.slider(0, 3, value=0.5, step=0.25, label="Noise")
    rounding = mo.ui.checkbox(value=False, label="Round y to create ties")
    mo.hstack([slope, noise, rounding])
    return noise, rounding, slope


@app.cell
def _(noise, np, pd, rounding, slope):
    _rng = np.random.default_rng(42)
    x = np.arange(12, dtype=float)
    y = slope.value * x + noise.value * _rng.normal(size=len(x))
    if rounding.value:
        y = np.round(y)
    _i, _j = np.triu_indices(len(x), k=1)
    _dx = np.sign(x[_j] - x[_i])
    _dy = np.sign(y[_j] - y[_i])
    concordant = int(np.sum(_dx * _dy > 0))
    discordant = int(np.sum(_dx * _dy < 0))
    ties_x = int(np.sum((_dx == 0) & (_dy != 0)))
    ties_y = int(np.sum((_dy == 0) & (_dx != 0)))
    _denom = np.sqrt(
        (concordant + discordant + ties_x) * (concordant + discordant + ties_y)
    )
    tau = None if _denom == 0 else (concordant - discordant) / _denom
    pairs = pd.DataFrame(
        {
            "First observation": _i + 1,
            "Second observation": _j + 1,
            "Classification": np.where(
                _dx * _dy > 0,
                "Concordant",
                np.where(_dx * _dy < 0, "Discordant", "Tied"),
            ),
        }
    )
    return concordant, discordant, pairs, tau, ties_x, ties_y, x, y


@app.cell
def _(concordant, discordant, mo, pairs, plt, tau, ties_x, ties_y, x, y):
    _fig, _ax = plt.subplots(figsize=(8, 3.5))
    _ax.scatter(x, y)
    _ax.set(xlabel="x", ylabel="y", title="Compare the ordering of any two points")
    _fig.tight_layout()
    _result = "undefined (zero denominator)" if tau is None else f"{tau:.3f}"
    mo.vstack(
        [
            _fig,
            mo.md(
                f"**Tau-b = {_result}**. Concordant: **{concordant}**; "
                f"discordant: **{discordant}**; tied only in x: **{ties_x}**; "
                f"tied only in y: **{ties_y}**."
            ),
            mo.ui.table(pairs, selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Formula and worked calculation

    Let C and D count concordant and discordant pairs. Let $T_x$ count pairs
    tied **only in x**, and $T_y$ pairs tied **only in y**. Then

    $$
    \tau_b=\frac{C-D}{\sqrt{(C+D+T_x)(C+D+T_y)}}.
    $$

    Pairs tied in both variables do not contribute. Without ties, the
    denominator is $n(n-1)/2$, giving tau-a.

    For x = `[1, 2, 3]`, y = `[1, 3, 2]`, there are three pairs:
    two concordant and one discordant. Thus **tau = (2 - 1) / 3 = 1/3**.
    Without ties, tau is the proportion concordant minus the proportion
    discordant; tau-b with ties has an adjusted normalization.

    ## 3. Interpretation and limitations

    Kendall's tau is a **nonparametric rank statistic**, not generally a
    maximum likelihood estimate of Pearson's population correlation.
    No normality assumption is needed for the descriptive calculation.
    Standard inference needs appropriate sampling assumptions, commonly
    independent observation pairs.

    Strictly increasing transformations preserve ordering and tau. Extreme
    magnitudes matter less than for Pearson's r, but changed ranks still
    matter. Constant variables make tau-b undefined. Tau-b and Spearman's
    rho have different scales: do not interpret equal numeric values as
    equivalent association strengths.

    A U-shaped relationship can have tau near zero despite strong dependence.
    Correlation does not establish causation.

    ### Try it yourself

    1. Set noise to zero and reverse the slope.
    2. Enable rounding and inspect the tied pairs.
    3. Set slope and noise to zero. Why is tau undefined?
    4. Verify the three-pair calculation above by hand.
    """)
    return


if __name__ == "__main__":
    app.run()
