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
    ## Partial correlation: adjusting for another variable

    **Linear partial correlation** measures the Pearson correlation between
    the residuals of x and y after regressing each on the same control variables.
    It ranges from -1 to +1 when both residual variances are nonzero.

    All data is synthetic.
    <a href="https://molab.marimo.io/github/radek-dev/marimo-notebooks/blob/main/correlation_measures/summary.py/wasm?mode=read&amp;show-code=false" target="_blank" rel="noopener noreferrer">Back to Correlation Measures Summary</a>

    [Back to the Notebook Gallery](https://radek-dev.github.io/notebook-gallery/)

    ## 1. Explore a shared influence

    Generate z and independent noise terms, then set
    $x=az+\epsilon_x$ and $y=bx+az+\epsilon_y$.
    With b = 0, any population association between x and y comes from z.
    Compare the original observations with the residuals after adjusting for z.
    """)
    return


@app.cell
def _(mo):
    shared = mo.ui.slider(0, 4, value=2, step=0.5, label="Shared influence a")
    direct = mo.ui.slider(-2, 2, value=0, step=0.25, label="Direct coefficient b")
    mo.hstack([shared, direct])
    return direct, shared


@app.cell
def _(direct, np, shared):
    _rng = np.random.default_rng(42)
    z, _ex, _ey = _rng.normal(size=(3, 200))
    x = shared.value * z + _ex
    y = direct.value * x + shared.value * z + _ey
    _design = np.column_stack([np.ones(len(z)), z])
    residual_x = x - _design @ np.linalg.lstsq(_design, x, rcond=None)[0]
    residual_y = y - _design @ np.linalg.lstsq(_design, y, rcond=None)[0]
    correlations = np.corrcoef([x, y, z])
    partial_r = float(np.corrcoef(residual_x, residual_y)[0, 1])
    return correlations, partial_r, residual_x, residual_y, x, y


@app.cell
def _(correlations, mo, partial_r, pd, plt, residual_x, residual_y, x, y):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))
    _axes[0].scatter(x, y, alpha=0.6)
    _axes[0].set(xlabel="x", ylabel="y", title="Before adjustment")
    _axes[1].scatter(residual_x, residual_y, alpha=0.6)
    _axes[1].set(xlabel="Residual x", ylabel="Residual y", title="After linear adjustment for z")
    _fig.tight_layout()
    mo.vstack(
        [
            _fig,
            mo.md(f"**Raw r = {correlations[0, 1]:.3f}; partial r = {partial_r:.3f}.**"),
            mo.ui.table(pd.DataFrame(correlations, index=["x", "y", "z"], columns=["x", "y", "z"]),
                        selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Formula and worked calculation

    For one control variable z:

    $$
    r_{xy\cdot z}=
    \frac{r_{xy}-r_{xz}r_{yz}}
    {\sqrt{(1-r_{xz}^2)(1-r_{yz}^2)}}.
    $$

    If all three pairwise correlations equal 0.8, the numerator is
    0.8 - 0.64 = 0.16 and the denominator is sqrt(0.36 x 0.36) = 0.36.
    Thus **partial r = 0.444**, lower than the raw correlation of 0.8.

    For multiple controls, use the same residual approach with an intercept
    and all control columns in each regression. Perfectly explained x or y
    leaves zero residual variance and makes partial correlation undefined.

    ## 3. Estimation and interpretation

    The residual calculation needs no normality assumption. Under an
    independent multivariate normal model with unknown mean and covariance,
    and a positive-definite sample covariance, sample partial correlation is
    the plug-in **MLE of the population partial correlation**.
    It is not generally unbiased in finite samples.

    **Zero partial correlation is not generally conditional independence.**
    That equivalence holds for multivariate normal variables, but nonlinear
    conditional relationships may remain in other distributions.

    ## 4. Limitations

    Linear adjustment removes fitted linear components, not every possible
    influence of z. Nonlinearities, measurement error, and unmeasured variables
    can leave misleading results. Outliers and highly correlated controls
    can destabilize estimates.

    **Controlling for a variable does not automatically remove causal bias.**
    Conditioning on a collider can create association; adjusting for a mediator
    changes the causal question. Choose controls using subject knowledge,
    not just their observed correlations.

    ### Try it yourself

    1. With b = 0, increase the shared influence. Compare raw and partial r.
    2. Set a = 0 and vary b. Why are the two correlations similar?
    3. Set b negative while a stays positive. Can adjustment reverse the sign?
    4. Explain why the direct coefficient b and partial r are not the same quantity.
    """)
    return


if __name__ == "__main__":
    app.run()
