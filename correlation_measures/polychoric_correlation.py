# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "matplotlib", "numpy", "pandas", "scipy"]
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
    from scipy.integrate import quad
    from scipy.optimize import minimize_scalar
    from scipy.special import ndtr
    from scipy.stats import norm

    return minimize_scalar, mo, ndtr, norm, np, pd, plt, quad


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Correlation Measures
    ## Polychoric correlation: ordinal observations, latent numeric variables

    Polychoric correlation estimates the correlation of two **unobserved,
    standard bivariate-normal variables** from their ordered categories.
    Thresholds turn latent values into observed categories, such as low,
    medium, and high. The model's correlation parameter ranges from -1 to +1.

    This is a model-based estimate, not simply a correlation of category codes.
    All data is synthetic.
    <a href="https://molab.marimo.io/github/radek-dev/marimo-notebooks/blob/main/correlation_measures/summary.py/wasm?mode=edit&amp;show-code=true" target="_blank" rel="noopener noreferrer">Back to Correlation Measures Summary</a>

    [Back to the Notebook Gallery](https://radek-dev.github.io/notebook-gallery/)

    ## 1. Generate latent values and categorize them

    The simulation uses thresholds -0.5 and 0.5 for both variables. The estimator
    does not know those thresholds: it estimates them from marginal counts.
    The random seed stays fixed.
    """)
    return


@app.cell
def _(mo):
    latent_rho = mo.ui.slider(-0.8, 0.8, value=0.6, step=0.1, label="True latent correlation")
    sample_size = mo.ui.slider(200, 1000, value=600, step=200, label="Sample size")
    mo.hstack([latent_rho, sample_size])
    return latent_rho, sample_size


@app.cell
def _(latent_rho, np, sample_size):
    _rng = np.random.default_rng(42)
    _u, _e = _rng.normal(size=(2, 1000))
    latent_x = _u[: sample_size.value]
    latent_y = (
        latent_rho.value * latent_x
        + np.sqrt(1 - latent_rho.value**2) * _e[: sample_size.value]
    )
    ordinal_x = np.digitize(latent_x, [-0.5, 0.5])
    ordinal_y = np.digitize(latent_y, [-0.5, 0.5])
    counts = np.zeros((3, 3), dtype=int)
    np.add.at(counts, (ordinal_x, ordinal_y), 1)
    return counts, latent_x, latent_y, ordinal_x, ordinal_y


@app.cell
def _(counts, mo, norm, np):
    mo.stop(
        bool(np.any(counts.sum(axis=0) == 0) or np.any(counts.sum(axis=1) == 0)),
        mo.md("**Cannot estimate:** an ordinal category has no observations. Increase the sample size."),
    )
    thresholds_x = np.r_[-np.inf, norm.ppf(np.cumsum(counts.sum(axis=1))[:-1] / counts.sum()), np.inf]
    thresholds_y = np.r_[-np.inf, norm.ppf(np.cumsum(counts.sum(axis=0))[:-1] / counts.sum()), np.inf]
    return thresholds_x, thresholds_y


@app.cell(hide_code=True)
def _(counts, ndtr, np, quad, thresholds_x, thresholds_y):
    def cell_probabilities(rho):
        probabilities = np.empty((3, 3))
        scale = np.sqrt(1 - rho**2)
        for i in range(3):
            for j in range(3):
                def integrand(t):
                    upper = ndtr((thresholds_y[j + 1] - rho * t) / scale)
                    lower = ndtr((thresholds_y[j] - rho * t) / scale)
                    return np.exp(-t**2 / 2) / np.sqrt(2 * np.pi) * (upper - lower)

                probabilities[i, j] = quad(
                    integrand, thresholds_x[i], thresholds_x[i + 1], epsabs=1e-9
                )[0]
        return probabilities

    def negative_log_likelihood(rho):
        probabilities = cell_probabilities(rho)
        occupied = counts > 0
        if np.any(probabilities[occupied] <= 0):
            return np.inf
        return -float(np.sum(counts[occupied] * np.log(probabilities[occupied])))

    return cell_probabilities, negative_log_likelihood


@app.cell(hide_code=True)
def _(cell_probabilities, minimize_scalar, negative_log_likelihood, np):
    fit = minimize_scalar(
        negative_log_likelihood, bounds=(-0.98, 0.98), method="bounded",
        options={"xatol": 1e-5},
    )
    if not fit.success or not np.isfinite(fit.fun):
        raise RuntimeError(f"Polychoric estimation failed: {fit.message}")
    estimate = float(fit.x)
    fitted_probabilities = cell_probabilities(estimate)
    return estimate, fitted_probabilities


@app.cell(hide_code=True)
def _(
    counts,
    estimate,
    fitted_probabilities,
    latent_rho,
    latent_x,
    latent_y,
    mo,
    np,
    ordinal_x,
    ordinal_y,
    pd,
    plt,
    thresholds_x,
    thresholds_y,
):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))
    _axes[0].scatter(latent_x, latent_y, s=8, alpha=0.4)
    for _threshold in [-0.5, 0.5]:
        _axes[0].axvline(_threshold, color="gray", linestyle=":")
        _axes[0].axhline(_threshold, color="gray", linestyle=":")
    _axes[0].set(xlabel="Latent x", ylabel="Latent y", title="Latent simulation (normally unobserved)")
    _axes[1].imshow(counts, cmap="Blues")
    for _i in range(3):
        for _j in range(3):
            _axes[1].text(_j, _i, str(counts[_i, _j]), ha="center", va="center")
    _axes[1].set(xticks=[0, 1, 2], yticks=[0, 1, 2],
                 xlabel="Ordinal y", ylabel="Ordinal x", title="Observed category counts")
    _fig.tight_layout()
    _code_r = float(np.corrcoef(ordinal_x, ordinal_y)[0, 1])
    _boundary = (
        " **Warning:** the fit is near the search boundary; do not treat it as an unconstrained optimum."
        if abs(estimate) > 0.975 else ""
    )
    mo.vstack(
        [
            _fig,
            mo.md(
                f"**True latent rho = {latent_rho.value:.2f}; estimated polychoric rho = {estimate:.3f}.** "
                f"Pearson's r on category codes = **{_code_r:.3f}**.{_boundary}"
            ),
            mo.md("**Estimated thresholds** (not the simulation's known thresholds):"),
            mo.ui.table(pd.DataFrame(
                {"Variable": ["x", "y"], "First threshold": [thresholds_x[1], thresholds_y[1]],
                 "Second threshold": [thresholds_x[2], thresholds_y[2]]}), selection=None),
            mo.md("**Fitted expected counts** (category probabilities x sample size):"),
            mo.ui.table(pd.DataFrame(fitted_probabilities * counts.sum(),
                                    columns=["y=0", "y=1", "y=2"]), selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Estimation step by step

    1. Count observations in each ordinal category.
    2. Estimate thresholds from cumulative marginal proportions:
       $t_k=\Phi^{-1}(P(X\leq k))$.
    3. Compute bivariate-normal rectangle probabilities $p_{ij}(\rho)$.
    4. With thresholds fixed, maximize the multinomial log-likelihood:

    $$
    \ell(\rho)=\sum_{ij}n_{ij}\log p_{ij}(\rho).
    $$

    For example, marginal category proportions `[0.25, 0.50, 0.25]` give
    cumulative proportions 0.25 and 0.75, hence thresholds approximately
    **-0.674 and +0.674**. Under rho = 0, the probability of both variables
    falling in their middle category is **0.5 x 0.5 = 0.25**.
    A nonzero rho changes the joint probabilities without changing the margins.

    The rectangle probabilities here integrate the conditional normal
    distribution, whose mean is $\rho x$ and variance is $1-\rho^2$.

    ### Is this maximum likelihood?

    This notebook uses a **two-step estimator**: marginal thresholds first,
    then likelihood optimization for rho with thresholds fixed.
    It is **not full joint MLE** of thresholds and rho. Full maximum likelihood
    optimizes all those parameters together. For numerical stability, this
    demonstration restricts rho to [-0.98, 0.98] and reports near-boundary fits.

    ## 3. Interpretation and limitations

    The latent normality and threshold assumptions are substantive: ordinal
    data alone do not guarantee them. Misspecification can distort the estimate.
    Category order matters; arbitrary nominal categories are inappropriate.
    Reversing one category order reverses the sign.

    Sparse cells, empty categories, and near-perfect association can make
    estimation unstable. Finite-sample estimates are not generally unbiased.
    This educational fit does not supply standard errors; uncertainty needs
    a method appropriate to the two-step procedure and sampling design.
    Association is not a causal effect.

    ### Try it yourself

    1. Reverse the true latent correlation.
    2. Change sample size. Does the estimate exactly match the true parameter?
    3. Compare category-code Pearson's r with the latent estimate.
    4. Why would equal-width numeric scores impose a different assumption?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References and further reading

    - **Research article:** Olsson, U. (1979), [Maximum Likelihood Estimation
      of the Polychoric Correlation Coefficient](https://doi.org/10.1007/BF02296207).
      The central reference for likelihood-based estimation; read alongside
      this notebook's distinction between two-step and joint estimation.
    - **Book:** Agresti, A. (2010),
      [Analysis of Ordinal Categorical Data, 2nd edition: author resources](https://alanagresti.com/ordinal/ord.html).
      Background on ordinal models, ordering, and latent-variable interpretations.

    DOI links lead to publisher pages; full-text access may require a subscription.
    """)
    return


if __name__ == "__main__":
    app.run()
