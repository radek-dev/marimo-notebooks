# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "pandas",
# ]
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

    ## Pearson's r: exploring linear relationships

    **Pearson's correlation coefficient** measures the direction and strength
    of a linear relationship between two numeric variables.

    - **+1:** all points lie on an upward-sloping straight line.
    - **-1:** all points lie on a downward-sloping straight line.
    - **Near 0:** little linear association, but a nonlinear relationship may remain.

    We will explore synthetic data, calculate r step by step, and examine its
    limitations. No real or confidential data is used.

    [Back to the Notebook Gallery](https://radek-dev.github.io/notebook-gallery/)

    ## 1. Explore the relationship

    Change the slope, noise, and sample size. The random seed stays fixed so
    changing a control does not create an entirely unrelated random sample.
    The slope is **not** the correlation: a shallow, noise-free line still has
    r = +1 or -1.
    """)
    return


@app.cell
def _(mo):
    slope = mo.ui.slider(-3, 3, value=1, step=0.25, label="Slope")
    noise = mo.ui.slider(0, 5, value=1, step=0.25, label="Noise standard deviation")
    sample_size = mo.ui.slider(20, 200, value=80, step=20, label="Sample size")
    mo.hstack([slope, noise, sample_size])
    return noise, sample_size, slope


@app.cell
def _(noise, np, sample_size, slope):
    _rng = np.random.default_rng(42)
    _x_pool = _rng.normal(size=200)
    _noise_pool = _rng.normal(size=200)
    x = _x_pool[: sample_size.value]
    y = 5 + slope.value * x + noise.value * _noise_pool[: sample_size.value]
    linear_r = None if np.ptp(y) == 0 else float(np.corrcoef(x, y)[0, 1])
    return linear_r, x, y


@app.cell
def _(linear_r, mo, plt, x, y):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.scatter(x, y, alpha=0.7)
    _ax.axvline(x.mean(), color="gray", linestyle=":", label="Mean of x")
    _ax.axhline(y.mean(), color="gray", linestyle="--", label="Mean of y")
    _ax.set(xlabel="x", ylabel="y", title="Synthetic paired observations")
    _ax.legend()
    _fig.tight_layout()
    _result = (
        "**r is undefined:** y is constant, so its standard deviation is zero."
        if linear_r is None
        else f"**Sample Pearson's r = {linear_r:.3f}**"
    )
    mo.vstack([_fig, mo.md(_result)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Calculate r step by step

    For paired observations $(x_i, y_i)$, the sample correlation is

    $$
    r =
    \frac{\sum_i (x_i-\bar{x})(y_i-\bar{y})}
    {\sqrt{\sum_i (x_i-\bar{x})^2 \sum_i (y_i-\bar{y})^2}}.
    $$

    Subtracting the means gives **deviations**. When both deviations have the
    same sign, their product is positive; opposite signs give a negative
    product. The denominator scales the result into the interval [-1, 1].
    This is also sample covariance divided by the two sample standard deviations.

    Here is a small example you can calculate by hand:
    """)
    return


@app.cell
def _(mo, np, pd):
    _x = np.array([1, 2, 3, 4, 5], dtype=float)
    _y = np.array([2, 1, 4, 3, 5], dtype=float)
    _dx = _x - _x.mean()
    _dy = _y - _y.mean()
    _products = _dx * _dy
    _numerator = _products.sum()
    _denominator = np.sqrt(np.sum(_dx**2) * np.sum(_dy**2))
    _table = pd.DataFrame(
        {
            "x": _x,
            "y": _y,
            "x - mean(x)": _dx,
            "y - mean(y)": _dy,
            "Deviation product": _products,
            "(x - mean(x))^2": _dx**2,
            "(y - mean(y))^2": _dy**2,
        }
    )
    mo.vstack(
        [
            mo.ui.table(_table, selection=None),
            mo.md(
                f"Both means are **3**. The deviation products sum to "
                f"**{_numerator:.0f}**, and each column's deviations^2 "
                f"sum to **10**. Therefore, **r = {_numerator:.0f} / "
                f"{_denominator:.0f} = {_numerator / _denominator:.2f}**."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Is sample r a maximum likelihood estimator?

    **Yes, under a particular probability model.** If the pairs $(x_i, y_i)$
    are independent and identically distributed observations from a
    **bivariate normal distribution**, with unknown means, variances, and
    population correlation $\rho$, then for a nondegenerate sample:

    $$
    \hat{\rho}_{\mathrm{MLE}} = r.
    $$

    The maximum likelihood estimator (MLE) chooses the parameter values that
    maximize the likelihood of the observed data under that model.

    - **Descriptive statistic:** r summarizes linear association without
      requiring normality.
    - **MLE:** interpreting r this way depends on the assumed probability model.
    - **Bias:** r is generally not an unbiased estimator of $\rho$ in finite
      samples, even under the bivariate normal model.

    Using n or n - 1 consistently as the divisor in the covariance and both
    variances gives the same correlation: those factors cancel in the ratio.
    Using n - 1 does **not** make the resulting correlation estimator unbiased.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Zero correlation does not mean no relationship

    Compare a straight line with a symmetric U-shaped relationship below.
    In the second example, $y=x^2$ determines y exactly, yet its sample
    Pearson correlation is zero (up to floating-point rounding).

    The rising and falling parts cancel in the correlation calculation.
    **Always inspect a scatter plot alongside r.**
    """)
    return


@app.cell
def _(np, plt):
    _x = np.linspace(-2, 2, 101)
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for _ax, _y, _label in zip(_axes, [_x, _x**2], ["y = x", "y = x^2"]):
        _r = float(np.corrcoef(_x, _y)[0, 1])
        _ax.scatter(_x, _y, s=15)
        _ax.set(xlabel="x", ylabel="y", title=f"{_label}: r = {_r:.2f}")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Interpretation and limitations

    - **Use paired numeric observations.** Both variables must have nonzero
      variance. Setting both slope and noise to zero above shows the undefined case.
    - **Outliers can change r substantially.** A single extreme observation can
      create or conceal an apparent linear association.
    - **Correlation is not causation.** Confounding or a shared trend can make
      two variables correlated without either causing the other.
    - **Units do not affect r.** Adding a constant or multiplying by a positive
      constant leaves r unchanged; multiplying one variable by a negative
      constant reverses its sign.
    - **Normality is not required to calculate r.** Standard hypothesis tests
      and confidence intervals need additional assumptions about sampling and
      distributions. Repeated or time-dependent observations require particular care.
    - **Sample r is an estimate.** Its population counterpart is called rho
      ($\rho$). Small samples can give unstable estimates; labels such as
      "strong" or "weak" depend on context.

    ### Try it yourself

    1. Set noise to zero and compare slopes 0.25 and 3. Why is r the same?
    2. Keep a positive slope and increase noise. What happens to r?
    3. Make the slope negative. What changes about the sign of r?
    4. Set the slope to zero with nonzero noise. Why isn't sample r exactly zero?
    5. Explain why the U-shaped example has r near zero despite a perfect
       deterministic relationship.
    """)
    return


if __name__ == "__main__":
    app.run()
