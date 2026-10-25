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
    ## Point-biserial correlation: binary meets numeric

    Point-biserial correlation is **Pearson's r between a 0/1 variable and
    a numeric variable**. It ranges from -1 to +1. A positive value means
    group 1 has a higher numeric mean than group 0.

    All data is synthetic.
    [Back to the Notebook Gallery](https://radek-dev.github.io/notebook-gallery/)

    ## 1. Explore group separation

    Change the group mean difference, within-group noise, and group balance.
    A mean difference alone does not determine correlation.
    """)
    return


@app.cell
def _(mo):
    difference = mo.ui.slider(-5, 5, value=2, step=0.5, label="Mean shift for group 1")
    noise = mo.ui.slider(0, 4, value=1, step=0.25, label="Within-group noise")
    group_one = mo.ui.slider(10, 90, value=50, step=10, label="Group 1 count (of 100)")
    mo.hstack([difference, noise, group_one])
    return difference, group_one, noise


@app.cell(hide_code=True)
def _(difference, group_one, noise, np):
    _rng = np.random.default_rng(42)
    binary = np.r_[np.zeros(100 - group_one.value), np.ones(group_one.value)]
    numeric = difference.value * binary + noise.value * _rng.normal(size=100)
    _p = binary.mean()
    mean_zero = numeric[binary == 0].mean()
    mean_one = numeric[binary == 1].mean()
    sd = numeric.std(ddof=0)
    point_r = None if sd == 0 else (mean_one - mean_zero) / sd * np.sqrt(_p * (1 - _p))
    pearson_r = None if sd == 0 else float(np.corrcoef(binary, numeric)[0, 1])
    return binary, mean_one, mean_zero, numeric, pearson_r, point_r, sd


@app.cell(hide_code=True)
def _(
    binary,
    mean_one,
    mean_zero,
    mo,
    numeric,
    pd,
    pearson_r,
    plt,
    point_r,
    sd,
):
    _fig, _ax = plt.subplots(figsize=(8, 3.5))
    _ax.boxplot([numeric[binary == 0], numeric[binary == 1]], tick_labels=["0", "1"])
    _ax.set(xlabel="Binary group", ylabel="Numeric outcome", title="Compare group distributions")
    _fig.tight_layout()
    _result = (
        "**Correlation is undefined:** the numeric variable is constant."
        if point_r is None
        else f"**Point-biserial r = {point_r:.3f}; Pearson's r = {pearson_r:.3f}.**"
    )
    mo.vstack(
        [
            _fig,
            mo.md(_result),
            mo.ui.table(
                pd.DataFrame(
                    {"Quantity": ["Mean, group 0", "Mean, group 1", "Overall SD (ddof=0)"],
                     "Value": [mean_zero, mean_one, sd]}
                ),
                selection=None,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Formula and worked calculation

    With group proportions p and q = 1 - p, group means $\bar{y}_1$ and
    $\bar{y}_0$, and overall standard deviation $s_{y,0}$ using divisor n:

    $$
    r_{pb}=\frac{\bar{y}_1-\bar{y}_0}{s_{y,0}}\sqrt{pq}.
    $$

    If using sample SD with divisor n - 1, replace $\sqrt{pq}$ with
    $\sqrt{npq/(n-1)}$ to obtain exactly the same coefficient.

    For groups `[0, 0, 1, 1]` and outcomes `[1, 3, 3, 5]`, the group
    means are 2 and 4, the overall mean is 3, and the overall variance
    with divisor n is 2. Therefore **r = (4 - 2) / sqrt(2) x 0.5 = 0.707**.

    ## 3. Interpretation and limitations

    Both groups must be present, and the numeric variable must vary.
    Reversing the group coding reverses the sign, not the strength.
    Outliers and group imbalance affect the result.

    **No normality is needed to calculate the coefficient.** Group-normal
    assumptions can be relevant to particular tests or models, but a binary
    variable and a numeric variable are not jointly bivariate normal.
    Do not transfer Pearson's bivariate-normal MLE interpretation here
    automatically.

    Do not confuse point-biserial with **biserial correlation**, which assumes
    the binary variable arises by thresholding a latent continuous variable.
    Artificially splitting a numeric variable into two groups loses information.
    Association does not establish a treatment's causal effect.

    ### Try it yourself

    1. Reverse the mean shift. What changes about the sign?
    2. Keep the shift fixed and increase noise.
    3. Change the group balance while keeping the shift fixed.
    4. Set shift and noise to zero. Why is r undefined?
    """)
    return


if __name__ == "__main__":
    app.run()
