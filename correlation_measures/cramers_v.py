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
    ## Cramer's V: association between categories

    Cramer's V measures association between two categorical variables.
    It ranges from **0 to 1** and has **no positive or negative direction**.
    Categories do not need an ordering.

    All counts are synthetic.
    [Back to the Notebook Gallery](https://radek-dev.github.io/notebook-gallery/)

    ## 1. Explore a categorical table

    This example cross-tabulates three regions and three product choices.
    Add diagonal counts to make each region favor a different product.
    Counts are constructed directly, not randomly sampled.
    """)
    return


@app.cell
def _(mo):
    diagonal = mo.ui.slider(0, 100, value=30, step=5, label="Extra diagonal counts")
    baseline = mo.ui.slider(1, 20, value=10, label="Baseline count in every cell")
    mo.hstack([diagonal, baseline])
    return baseline, diagonal


@app.cell
def _(baseline, diagonal, np):
    observed = np.full((3, 3), baseline.value) + diagonal.value * np.eye(3, dtype=int)
    n = observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / n
    chi2 = float(np.sum((observed - expected)**2 / expected))
    v = np.sqrt(chi2 / (n * min(observed.shape[0] - 1, observed.shape[1] - 1)))
    return chi2, expected, n, observed, v


@app.cell(hide_code=True)
def _(chi2, expected, mo, n, observed, pd, plt, v):
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for _ax, _values, _title in zip(
        _axes, [observed, expected], ["Observed counts", "Expected under independence"]
    ):
        _ax.imshow(_values, cmap="Blues", vmin=0, vmax=observed.max())
        for _i in range(3):
            for _j in range(3):
                _ax.text(_j, _i, f"{_values[_i, _j]:.1f}", ha="center", va="center")
        _ax.set(xticks=[0, 1, 2], xticklabels=["A", "B", "C"],
                yticks=[0, 1, 2], yticklabels=["North", "South", "West"],
                xlabel="Product", title=_title)
    _fig.tight_layout()
    mo.vstack(
        [
            _fig,
            mo.md(f"**V = {v:.3f}**, chi-square = **{chi2:.2f}**, n = **{n}**."),
            mo.ui.table(pd.DataFrame((observed - expected)**2 / expected,
                                    columns=["A contribution", "B contribution", "C contribution"]),
                        selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Formula and worked calculation

    For an R-row, C-column table, expected counts under independence are
    $E_{ij}=O_{i+}O_{+j}/n$. Then

    $$
    \chi^2=\sum_{ij}\frac{(O_{ij}-E_{ij})^2}{E_{ij}},
    \qquad V=\sqrt{\frac{\chi^2}{n\min(R-1,C-1)}}.
    $$

    At the default settings, each diagonal count is 40 and each off-diagonal
    count is 10. Every row and column totals 60, so n = 180 and every expected
    count is 20. Thus chi-square = 3 x 20 + 6 x 5 = 90 and
    **V = sqrt(90 / (180 x 2)) = 0.5**.

    ## 3. Interpretation and limitations

    V summarizes departures from independence, not which categories drive
    them. Inspect the table and individual contributions. Reordering labels
    does not change V. For a 2 x 2 table, V equals the absolute value of phi.

    This is the **uncorrected sample V**: it is upward biased in small samples,
    particularly with many categories. Bias-corrected alternatives exist.
    It is a descriptive transformation of chi-square, not generally an MLE
    of a universal correlation parameter.

    All expected counts must be positive; remove empty marginal categories
    before calculating, and require at least two occupied levels per variable.
    No normality assumption is needed. Standard chi-square inference assumes
    an appropriate independent sampling design and adequate expected counts.
    Statistical significance is not the same as association strength.

    There is no causal interpretation and no universally useful "strong"
    threshold for every table shape or subject area.

    ### Try it yourself

    1. Set extra diagonal counts to zero. Why is V zero?
    2. Increase baseline counts with the diagonal addition fixed.
    3. Why would swapping product labels leave V unchanged?
    4. Multiply every count by 2 on paper: what happens to chi-square and V?
    """)
    return


if __name__ == "__main__":
    app.run()
