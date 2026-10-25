# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo"]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Correlation Measures: a Learning Guide

    Nine ways to describe association, with interactive synthetic examples,
    formulas, worked calculations, and interpretation pitfalls.

    **Follow the notebook links below to open interactive browser versions.**
    No local Python installation is needed. To explore the collection locally:

    ```sh
    cd marimo-notebooks/correlation_measures
    uv run marimo edit
    ```

    Open `summary.py` or any of the individual notebooks in the local
    notebook browser. Public links below open the hosted versions.

    ## Choose a measure

    | Notebook | Data / question | Range | Main caution |
    | :-- | :-- | :-- | :-- |
    | [Pearson's r](https://molab.marimo.io/github/radek-dev/marimo-notebooks/blob/main/correlation_measures/pearsons_r.py/wasm) | Linear association between numeric variables | -1 to +1 | Can miss nonlinear dependence; sensitive to outliers |
    | [Spearman's rho](https://molab.marimo.io/github/radek-dev/marimo-notebooks/blob/main/correlation_measures/spearmans_rho.py/wasm) | Monotonic association using ranks | -1 to +1 | Can miss non-monotonic dependence; use average ranks for ties |
    | [Kendall's tau](https://molab.marimo.io/github/radek-dev/marimo-notebooks/blob/main/correlation_measures/kendalls_tau.py/wasm) | Agreement in pairwise ordering | -1 to +1 | Tau-b adjusts for ties; its scale differs from Spearman's |
    | [Point-biserial correlation](https://molab.marimo.io/github/radek-dev/marimo-notebooks/blob/main/correlation_measures/point_biserial.py/wasm) | One binary variable and one numeric variable | -1 to +1 | Group coding determines the sign; imbalance affects the value |
    | [Phi coefficient](https://molab.marimo.io/github/radek-dev/marimo-notebooks/blob/main/correlation_measures/phi_coefficient.py/wasm) | Two binary variables | -1 to +1 | Fixed margins can restrict attainable values |
    | [Cramer's V](https://molab.marimo.io/github/radek-dev/marimo-notebooks/blob/main/correlation_measures/cramers_v.py/wasm) | Two categorical variables, without requiring an order | 0 to 1 | No direction; uncorrected sample V is upward biased |
    | [Polychoric correlation](https://molab.marimo.io/github/radek-dev/marimo-notebooks/blob/main/correlation_measures/polychoric_correlation.py/wasm) | Latent normal correlation underlying ordinal categories | -1 to +1 | Depends on the latent normal and threshold model |
    | [Partial correlation](https://molab.marimo.io/github/radek-dev/marimo-notebooks/blob/main/correlation_measures/partial_correlation.py/wasm) | Linear association after adjusting both variables for controls | -1 to +1 | Adjustment is not automatically causal or sufficient |
    | [Distance correlation](https://molab.marimo.io/github/radek-dev/marimo-notebooks/blob/main/correlation_measures/distance_correlation.py/wasm) | General dependence, including non-monotonic relationships | 0 to 1 | Sample estimates can be positive under independence |

    Ranges describe defined coefficients or the model's parameter space;
    finite samples and fixed margins can limit which values are attainable.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Suggested learning order

    Start with **Pearson**, then compare **Spearman** and **Kendall** to see
    what ranks preserve and discard. Move to **point-biserial**, **phi**, and
    **Cramer's V** for binary and categorical data. Finally, explore
    **polychoric**, **partial**, and **distance correlation** for latent
    variables, adjustment, and more general dependence.

    ## Connections to remember

    - **Spearman = Pearson on ranks**, with average ranks for tied values.
    - **Point-biserial = Pearson** with one variable coded 0/1.
    - **Phi = Pearson** with both variables coded 0/1.
    - **Cramer's V = absolute phi** for a 2 x 2 table.
    - **Partial correlation = Pearson on regression residuals**, using the
      same controls and an intercept in both regressions.
    - **Polychoric estimates a latent correlation**, not the correlation
      of arbitrary numeric category codes.

    ## Estimation is not the same as description

    Pearson's sample r is the MLE of population correlation under an
    independent bivariate-normal model with unknown means and variances,
    for a nondegenerate sample. Sample partial correlation has a related
    multivariate-normal MLE interpretation under suitable conditions.

    The polychoric notebook uses **two-step estimation**, not full joint MLE:
    it estimates thresholds first, then optimizes the correlation with those
    thresholds fixed. The other descriptive coefficients should not
    automatically be called MLEs. Being an MLE does not imply being unbiased.

    ## What no single coefficient tells you

    **Association is not causation.** Inspect plots or contingency tables,
    consider sample size and uncertainty, and check the assumptions of any
    inferential procedure.

    Zero Pearson, Spearman, or Kendall correlation need not mean independence.
    Population distance correlation is zero exactly under independence for
    Euclidean random vectors with finite first moments, but a finite-sample
    estimate alone cannot establish that conclusion.

    **Practice:** for each notebook, explain what is measured, calculate the
    worked example, change the controls, and identify a situation in which
    the coefficient could mislead.

    [Back to the Notebook Gallery](https://radek-dev.github.io/notebook-gallery/)
    """)
    return


if __name__ == "__main__":
    app.run()
