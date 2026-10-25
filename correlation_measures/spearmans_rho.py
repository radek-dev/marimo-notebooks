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

    ## Spearman's rho: exploring ranked relationships

    **Spearman's rank correlation** measures the direction and strength of a
    monotonic relationship: as one variable increases, the other tends to
    increase or tends to decrease. Unlike Pearson's r, the relationship
    does not have to follow a straight line.

    - **+1:** the two variables have identical ranks.
    - **-1:** the ranks are exactly reversed.
    - **Near 0:** little monotonic association, but other dependence may remain.

    Spearman's correlation is **Pearson's correlation applied to ranks**.
    It works with paired numeric or ordinal observations whose ordering is
    meaningful. All data here is synthetic.

    <a href="https://molab.marimo.io/github/radek-dev/marimo-notebooks/blob/main/correlation_measures/summary.py/wasm?mode=read&amp;show-code=false" target="_blank" rel="noopener noreferrer">Back to Correlation Measures Summary</a>

    [Back to the Notebook Gallery](https://radek-dev.github.io/notebook-gallery/)

    ## 1. Explore a monotonic relationship

    Start with no noise and increase the curvature. The relationship becomes
    less linear but remains strictly increasing, so Spearman's correlation
    stays at +1. Reverse the direction to get -1.

    Increase noise to introduce rank reversals. The random seed stays fixed
    when changing controls.
    """)
    return


@app.cell
def _(mo):
    curvature = mo.ui.slider(1, 8, value=3, label="Curvature (power)")
    noise = mo.ui.slider(0, 1, value=0, step=0.05, label="Noise standard deviation")
    direction = mo.ui.dropdown(
        {"Increasing": 1, "Decreasing": -1},
        value="Increasing",
        label="Direction",
    )
    sample_size = mo.ui.slider(20, 200, value=80, step=20, label="Sample size")
    mo.hstack([curvature, noise, direction, sample_size])
    return curvature, direction, noise, sample_size


@app.cell
def _(curvature, direction, noise, np, pd, sample_size):
    _rng = np.random.default_rng(42)
    _x_pool = _rng.uniform(0, 1, size=200)
    _noise_pool = _rng.normal(size=200)
    x = _x_pool[: sample_size.value]
    y = (
        direction.value * x**curvature.value
        + noise.value * _noise_pool[: sample_size.value]
    )
    rank_x = pd.Series(x).rank(method="average").to_numpy()
    rank_y = pd.Series(y).rank(method="average").to_numpy()
    pearson_r = float(np.corrcoef(x, y)[0, 1])
    spearman_rho = float(np.corrcoef(rank_x, rank_y)[0, 1])
    return pearson_r, rank_x, rank_y, spearman_rho, x, y


@app.cell
def _(mo, pearson_r, plt, rank_x, rank_y, spearman_rho, x, y):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))
    _axes[0].scatter(x, y, alpha=0.7)
    _axes[0].set(xlabel="x", ylabel="y", title="Original values")
    _axes[1].scatter(rank_x, rank_y, alpha=0.7)
    _axes[1].set(xlabel="Rank of x", ylabel="Rank of y", title="Ranked values")
    _fig.tight_layout()
    mo.vstack(
        [
            _fig,
            mo.md(
                f"**Pearson's r = {pearson_r:.3f}** on the original values. "
                f"**Spearman's rho = {spearman_rho:.3f}** "
                "(Pearson's r on the ranks)."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Calculate Spearman's correlation step by step

    Assign rank 1 to the smallest observation in each variable. Write
    $R_i=\operatorname{rank}(x_i)$ and $S_i=\operatorname{rank}(y_i)$. Then

    $$
    r_s =
    \frac{\sum_i(R_i-\bar{R})(S_i-\bar{S})}
    {\sqrt{\sum_i(R_i-\bar{R})^2\sum_i(S_i-\bar{S})^2}}.
    $$

    **When there are no ties in either variable**, this simplifies to

    $$
    r_s = 1 - \frac{6\sum_i d_i^2}{n(n^2-1)},
    \qquad d_i=R_i-S_i.
    $$

    Here is a five-observation example without ties:
    """)
    return


@app.cell
def _(mo, np, pd):
    _x = pd.Series([10, 20, 30, 40, 50])
    _y = pd.Series([20, 10, 40, 30, 50])
    _rx = _x.rank(method="average")
    _ry = _y.rank(method="average")
    _d = _rx - _ry
    _n = len(_x)
    _sum_d2 = float((_d**2).sum())
    _rho = 1 - 6 * _sum_d2 / (_n * (_n**2 - 1))
    _rank_correlation = float(np.corrcoef(_rx, _ry)[0, 1])
    mo.vstack(
        [
            mo.ui.table(
                pd.DataFrame(
                    {
                        "x": _x,
                        "y": _y,
                        "Rank of x": _rx,
                        "Rank of y": _ry,
                        "Rank difference": _d,
                        "(Rank difference)^2": _d**2,
                    }
                ),
                selection=None,
            ),
            mo.md(
                f"The (rank differences)^2 sum to **{_sum_d2:.0f}**. "
                f"With n = {_n}, **r_s = 1 - 6 x {_sum_d2:.0f} / "
                f"({_n} x ({_n}^2 - 1)) = {_rho:.2f}**. "
                f"Pearson's correlation on the ranks agrees: "
                f"**{_rank_correlation:.2f}**."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What about ties?

    Equal observations receive their **average rank**. For example,
    `[10, 20, 20, 40]` becomes `[1, 2.5, 2.5, 4]`: the two 20s share
    positions 2 and 3.

    Compute Pearson's correlation on these average ranks. **Do not use the
    simplified rank-difference formula above without tie corrections.**
    If either variable is constant, all its ranks are equal, so Spearman's
    correlation is undefined.

    ### Is this a maximum likelihood estimator?

    Spearman's correlation is a **nonparametric rank statistic**; calculating
    it does not require a normal distribution or a likelihood model.
    Unlike sample Pearson's r under a bivariate normal model, it is not
    generally the MLE of that model's correlation parameter.

    The notation can be confusing: population Spearman correlation is often
    written $\rho_s$, while the sample statistic is written $r_s$ or
    $\hat{\rho}_s$. It is a different population quantity from Pearson's
    $\rho$ and is not generally unbiased in finite samples.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Zero rank correlation does not mean independence

    For a symmetric sample with $y=x^2$, the relationship first decreases and
    then increases. It is deterministic but **not monotonic**. Both Pearson's
    and Spearman's correlations are zero, up to floating-point rounding.
    """)
    return


@app.cell
def _(mo, np, pd, plt):
    _x = np.arange(-20, 21, dtype=float)
    _y = _x**2
    _rx = pd.Series(_x).rank(method="average")
    _ry = pd.Series(_y).rank(method="average")
    _r = float(np.corrcoef(_x, _y)[0, 1])
    _rho = float(np.corrcoef(_rx, _ry)[0, 1])
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))
    _axes[0].scatter(_x, _y)
    _axes[0].set(xlabel="x", ylabel="y", title="y = x^2")
    _axes[1].scatter(_rx, _ry)
    _axes[1].set(xlabel="Rank of x", ylabel="Rank of y", title="Average ranks")
    _fig.tight_layout()
    mo.vstack(
        [
            _fig,
            mo.md(f"**Pearson's r = {_r:.3f}; Spearman's rho = {_rho:.3f}.**"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Interpretation and limitations

    - **Ranks discard distances.** Spearman's correlation measures ordering,
      not the size of differences or the slope of a relationship.
    - **Strictly increasing transformations preserve ranks.** Applying one
      to either variable leaves Spearman's correlation unchanged. A strictly
      decreasing transformation of one variable reverses its sign.
    - **Outlier magnitudes have less influence than with Pearson's r.**
      Once an observation is the largest, moving it farther out does not
      change its rank. Outliers can still affect the result through ordering.
    - **Ties matter.** Use average ranks, particularly for ordinal scores
      with only a few possible values.
    - **Correlation is not causation.** Confounding can produce rank association.
    - **Inference needs assumptions.** A descriptive calculation does not
      establish statistical significance. Standard tests typically assume
      independent observation pairs; time dependence and repeated measures
      require appropriate methods.
    - **Inspect the scatter plot.** A near-zero value can hide a strong
      non-monotonic relationship, and small samples can give unstable estimates.

    ### Try it yourself

    1. With no noise, increase curvature. Why does Pearson's r change while
       Spearman's rho stays at +1?
    2. Reverse the direction. What happens to both coefficients?
    3. Add noise. Identify points whose ordering changes in the rank plot.
    4. Calculate the average ranks of `[3, 3, 7, 9, 9]`.
    5. Explain why neither coefficient captures the U-shaped relationship.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References and further reading

    - **Research article:** Spearman, C. (1904), [The Proof and Measurement
      of Association between Two Things](https://doi.org/10.2307/1412159).
      The foundational rank-correlation article.
    - **Book:** Hollander, M., Wolfe, D. A., and Chicken, E.,
      [Nonparametric Statistical Methods, 3rd edition](https://doi.org/10.1002/9781119196037).
      A reference for rank-based methods and inference.
    - **Practical reference:** [SciPy's Spearman correlation documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)
      covers calculation, missing values, and inference limitations.

    DOI links lead to publisher pages; full-text access may require a subscription.
    """)
    return


if __name__ == "__main__":
    app.run()
