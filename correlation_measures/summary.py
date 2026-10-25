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

    **Each notebook link opens an editable browser version in a new tab.**
    Change code and rerun cells without a local Python installation. Browser
    preview edits are not saved back to GitHub automatically; save a copy
    to keep your changes. To explore the collection locally:

    ```sh
    cd marimo-notebooks/correlation_measures
    uv run marimo edit
    ```

    Open `summary.py` or any of the individual notebooks in the local
    notebook browser. Public links below open the hosted versions.

    ## Choose a measure

    | Notebook | Data / question | Range | Main caution |
    | :-- | :-- | :-- | :-- |
    | <a href="https://molab.marimo.io/github/radek-dev/marimo-notebooks/blob/main/correlation_measures/pearsons_r.py/wasm?mode=edit&amp;show-code=true" target="_blank" rel="noopener noreferrer">Pearson's r</a> | Linear association between numeric variables | -1 to +1 | Can miss nonlinear dependence; sensitive to outliers |
    | <a href="https://molab.marimo.io/github/radek-dev/marimo-notebooks/blob/main/correlation_measures/spearmans_rho.py/wasm?mode=edit&amp;show-code=true" target="_blank" rel="noopener noreferrer">Spearman's rho</a> | Monotonic association using ranks | -1 to +1 | Can miss non-monotonic dependence; use average ranks for ties |
    | <a href="https://molab.marimo.io/github/radek-dev/marimo-notebooks/blob/main/correlation_measures/kendalls_tau.py/wasm?mode=edit&amp;show-code=true" target="_blank" rel="noopener noreferrer">Kendall's tau</a> | Agreement in pairwise ordering | -1 to +1 | Tau-b adjusts for ties; its scale differs from Spearman's |
    | <a href="https://molab.marimo.io/github/radek-dev/marimo-notebooks/blob/main/correlation_measures/point_biserial.py/wasm?mode=edit&amp;show-code=true" target="_blank" rel="noopener noreferrer">Point-biserial correlation</a> | One binary variable and one numeric variable | -1 to +1 | Group coding determines the sign; imbalance affects the value |
    | <a href="https://molab.marimo.io/github/radek-dev/marimo-notebooks/blob/main/correlation_measures/phi_coefficient.py/wasm?mode=edit&amp;show-code=true" target="_blank" rel="noopener noreferrer">Phi coefficient</a> | Two binary variables | -1 to +1 | Fixed margins can restrict attainable values |
    | <a href="https://molab.marimo.io/github/radek-dev/marimo-notebooks/blob/main/correlation_measures/cramers_v.py/wasm?mode=edit&amp;show-code=true" target="_blank" rel="noopener noreferrer">Cramer's V</a> | Two categorical variables, without requiring an order | 0 to 1 | No direction; uncorrected sample V is upward biased |
    | <a href="https://molab.marimo.io/github/radek-dev/marimo-notebooks/blob/main/correlation_measures/polychoric_correlation.py/wasm?mode=edit&amp;show-code=true" target="_blank" rel="noopener noreferrer">Polychoric correlation</a> | Latent normal correlation underlying ordinal categories | -1 to +1 | Depends on the latent normal and threshold model |
    | <a href="https://molab.marimo.io/github/radek-dev/marimo-notebooks/blob/main/correlation_measures/partial_correlation.py/wasm?mode=edit&amp;show-code=true" target="_blank" rel="noopener noreferrer">Partial correlation</a> | Linear association after adjusting both variables for controls | -1 to +1 | Adjustment is not automatically causal or sufficient |
    | <a href="https://molab.marimo.io/github/radek-dev/marimo-notebooks/blob/main/correlation_measures/distance_correlation.py/wasm?mode=edit&amp;show-code=true" target="_blank" rel="noopener noreferrer">Distance correlation</a> | General dependence, including non-monotonic relationships | 0 to 1 | Sample estimates can be positive under independence |

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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References and reading guide

    Each measure's notebook ends with references and a short explanation of
    what to read. These links point to original research, books, and author
    resources rather than unrelated gallery topics.

    ### Research articles by measure

    | Measure | Reading | Why it matters |
    | :-- | :-- | :-- |
    | Pearson | Pearson (1895), [Note on regression and inheritance in the case of two parents](https://doi.org/10.1098/rspl.1895.0041) | Historical regression and correlation foundation |
    | Spearman | Spearman (1904), [The Proof and Measurement of Association between Two Things](https://doi.org/10.2307/1412159) | Rank correlation foundation |
    | Kendall | Kendall (1938), [A New Measure of Rank Correlation](https://doi.org/10.1093/biomet/30.1-2.81) | Pairwise concordance foundation; modern tau-b adjusts for ties |
    | Point-biserial | Tate (1954), [Correlation Between a Discrete and a Continuous Variable. Point-Biserial Correlation](https://doi.org/10.1214/aoms/1177728730) | Binary-continuous association |
    | Phi / Cramer's V | Bergsma (2013), [A bias-correction for Cramer's and Tschuprow's association measures](https://doi.org/10.1016/j.jkss.2012.10.002) | Bias of related unsigned categorical measures, not the historical origin of signed phi |
    | Polychoric | Olsson (1979), [Maximum Likelihood Estimation of the Polychoric Correlation Coefficient](https://doi.org/10.1007/BF02296207) | Likelihood-based latent correlation estimation |
    | Partial | Baba, Shibata, and Sibuya (2004), [Partial Correlation and Conditional Correlation as Measures of Conditional Independence](https://doi.org/10.1111/j.1467-842X.2004.00360.x) | Distinguishing linear adjustment from conditional independence |
    | Distance | Szekely, Rizzo, and Bakirov (2007), [Measuring and testing dependence by correlation of distances](https://doi.org/10.1214/009053607000000505) | General dependence and independence testing; [open-access version](https://arxiv.org/abs/0803.4101) |

    ### Books

    - **Linear and multivariate background:** Venables, W. N., and Ripley,
      B. D. (2002), [Modern Applied Statistics with S, 4th edition](https://doi.org/10.1007/978-0-387-21706-2).
      Useful alongside Pearson, point-biserial, and partial correlation.
    - **Rank-based inference:** Hollander, M., Wolfe, D. A., and Chicken, E.,
      [Nonparametric Statistical Methods, 3rd edition](https://doi.org/10.1002/9781119196037).
      Useful alongside Spearman and Kendall.
    - **Categorical association:** Agresti, A. (2013),
      [Categorical Data Analysis, 3rd edition: author resources](https://alanagresti.com/cda/cda.html).
      Useful alongside phi and Cramer's V.
    - **Ordinal models:** Agresti, A. (2010),
      [Analysis of Ordinal Categorical Data, 2nd edition: author resources](https://alanagresti.com/ordinal/ord.html).
      Useful alongside polychoric correlation and threshold assumptions.
    - **Accessible regression background:** James, G., Witten, D., Hastie, T.,
      Tibshirani, R., and Taylor, J. (2023),
      [An Introduction to Statistical Learning with Applications in Python](https://www.statlearning.com/).
      The authors provide a free edition; use the regression chapter for
      background on linear association and adjustment.

    DOI links may require a subscription for full text. Author-resource pages
    provide supporting material, not necessarily the complete book. For
    distance correlation, start with the open-access research articles rather
    than expecting a general introductory statistics book to cover it.
    """)
    return


if __name__ == "__main__":
    app.run()
