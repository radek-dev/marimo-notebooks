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
    ## Distance correlation: beyond linear and monotonic relationships

    Distance correlation compares patterns of **pairwise distances**.
    The standard population measure ranges from 0 to 1 and is zero
    **if and only if the variables are independent**, for Euclidean random
    vectors with finite first moments. It has no positive or negative direction.

    The finite-sample value is an estimate, not proof of independence.
    All data is synthetic.
    [Back to the Notebook Gallery](https://radek-dev.github.io/notebook-gallery/)

    ## 1. Explore different relationships

    Compare a linear relationship, a U-shaped relationship, and independent
    variables. The displayed estimate uses the standard **biased,
    double-centered sample estimator**.
    """)
    return


@app.cell
def _(mo):
    relationship = mo.ui.dropdown(
        ["Linear", "U-shaped", "Independent"], value="U-shaped", label="Relationship"
    )
    noise = mo.ui.slider(0, 2, value=0, step=0.1, label="Noise")
    sample_size = mo.ui.slider(20, 200, value=80, step=20, label="Sample size")
    mo.hstack([relationship, noise, sample_size])
    return noise, relationship, sample_size


@app.cell
def _(noise, np, relationship, sample_size):
    _rng = np.random.default_rng(42)
    x = (np.arange(sample_size.value) - (sample_size.value - 1) / 2) * 4 / (sample_size.value - 1)
    _error = _rng.normal(size=len(x))
    if relationship.value == "Linear":
        y = x + noise.value * _error
    elif relationship.value == "U-shaped":
        y = x**2 + noise.value * _error
    else:
        y = _rng.normal(size=len(x)) + noise.value * _error
    return x, y


@app.cell
def _(np):
    def distance_components(values):
        distances = np.abs(values[:, None] - values[None, :])
        centered = (
            distances - distances.mean(axis=0, keepdims=True)
            - distances.mean(axis=1, keepdims=True) + distances.mean()
        )
        return distances, centered

    return (distance_components,)


@app.cell
def _(distance_components, np, pd, x, y):
    distances_x, centered_x = distance_components(x)
    _, centered_y = distance_components(y)
    covariance2 = float(np.mean(centered_x * centered_y))
    variance_x2 = float(np.mean(centered_x**2))
    variance_y2 = float(np.mean(centered_y**2))
    _denom = np.sqrt(variance_x2 * variance_y2)
    distance_r = 0.0 if _denom == 0 else float(np.sqrt(max(covariance2, 0.0) / _denom))
    pearson_r = float(np.corrcoef(x, y)[0, 1])
    spearman_r = float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1])
    return centered_x, covariance2, distance_r, distances_x, pearson_r, spearman_r, variance_x2, variance_y2


@app.cell
def _(centered_x, distance_r, distances_x, mo, pearson_r, plt, spearman_r, x, y):
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 3.5))
    _axes[0].scatter(x, y, s=15)
    _axes[0].set(xlabel="x", ylabel="y", title="Observations")
    _axes[1].imshow(distances_x, cmap="Blues")
    _axes[1].set(title="Pairwise distances in x", xlabel="Observation", ylabel="Observation")
    _axes[2].imshow(centered_x, cmap="coolwarm")
    _axes[2].set(title="Double-centered distances", xlabel="Observation", ylabel="Observation")
    _fig.tight_layout()
    mo.vstack(
        [_fig, mo.md(
            f"**Distance correlation = {distance_r:.3f}; "
            f"Pearson's r = {pearson_r:.3f}; Spearman's rho = {spearman_r:.3f}.**"
        )]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Calculation

    Start with distances $a_{ij}=|x_i-x_j|$ and $b_{ij}=|y_i-y_j|$.
    Double-center each distance matrix:

    $$
    A_{ij}=a_{ij}-\bar{a}_{i\cdot}-\bar{a}_{\cdot j}+\bar{a}_{\cdot\cdot},
    \qquad
    B_{ij}=b_{ij}-\bar{b}_{i\cdot}-\bar{b}_{\cdot j}+\bar{b}_{\cdot\cdot}.
    $$

    Then calculate

    $$
    \mathcal{V}_n^2(X,Y)=\frac{1}{n^2}\sum_{ij}A_{ij}B_{ij},
    \qquad
    \mathcal{R}_n^2=
    \frac{\mathcal{V}_n^2(X,Y)}
    {\sqrt{\mathcal{V}_n^2(X,X)\mathcal{V}_n^2(Y,Y)}}.
    $$

    **Take one more square root to get distance correlation**, rather than
    distance correlation^2. If a distance variance is zero, the conventional
    distance correlation is defined as zero. Tiny negative roundoff in the
    theoretically nonnegative sample distance covariance^2 is clipped to zero.
    """)
    return


@app.cell
def _(covariance2, distance_r, mo, variance_x2, variance_y2):
    mo.md(
        f"For the current data, distance covariance^2 is **{covariance2:.4f}**. "
        f"The two distance variance^2 terms are **{variance_x2:.4f}** and "
        f"**{variance_y2:.4f}**. Their normalization gives distance "
        f"correlation^2 = **{distance_r**2:.4f}**, hence "
        f"distance correlation = **{distance_r:.4f}**."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Interpretation and limitations

    Unlike Pearson or Spearman, population distance correlation detects
    non-monotonic dependence. However, the **biased sample estimate is
    generally positive even for independent data**, especially in small samples.
    Independence testing typically uses a permutation procedure with
    exchangeable observations, not a universal numerical cutoff.

    This is a nonparametric dependence statistic, not generally an MLE.
    Normality is not required. Bias-corrected estimators use different
    centering; an unbiased estimate of distance covariance^2 can be negative.

    Distance correlation is not a slope or a causal effect. Outliers and the
    choice of distances or multivariate scaling can influence it. This direct
    implementation stores n x n matrices, requiring O(n^2) memory.
    Multivariate versions use Euclidean vector distances; this explorer
    uses one-dimensional variables.

    ### Try it yourself

    1. Compare Pearson, Spearman, and distance correlation for the noise-free U.
    2. Select Independent. Why is sample distance correlation still positive?
    3. Increase sample size under independence. Is any one sample definitive?
    4. Why does a high distance correlation not tell you the relationship's direction?
    """)
    return


if __name__ == "__main__":
    app.run()
