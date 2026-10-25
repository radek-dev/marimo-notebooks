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
    ## Phi: association between two binary variables

    Phi is **Pearson's r for two 0/1 variables**. It ranges from -1 to +1:
    matching outcomes produce positive association, opposite outcomes
    negative association. The sign depends on which outcomes are coded 1.

    All counts are synthetic.
    <a href="https://molab.marimo.io/github/radek-dev/marimo-notebooks/blob/main/correlation_measures/summary.py/wasm?mode=read&amp;show-code=false" target="_blank" rel="noopener noreferrer">Back to Correlation Measures Summary</a>

    [Back to the Notebook Gallery](https://radek-dev.github.io/notebook-gallery/)

    ## 1. Build a 2 x 2 table

    Adjust the four observed counts. Rows represent x, columns represent y.
    """)
    return


@app.cell
def _(mo):
    a = mo.ui.slider(0, 50, value=30, label="a: x=0, y=0")
    b = mo.ui.slider(0, 50, value=10, label="b: x=0, y=1")
    c = mo.ui.slider(0, 50, value=10, label="c: x=1, y=0")
    d = mo.ui.slider(0, 50, value=30, label="d: x=1, y=1")
    mo.hstack([a, b, c, d])
    return a, b, c, d


@app.cell
def _(a, b, c, d, np):
    counts = np.array([[a.value, b.value], [c.value, d.value]])
    denominator = np.sqrt(float(
        (a.value + b.value) * (c.value + d.value)
        * (a.value + c.value) * (b.value + d.value)
    ))
    phi = None if denominator == 0 else (a.value * d.value - b.value * c.value) / denominator
    return counts, denominator, phi


@app.cell
def _(counts, denominator, mo, pd, phi, plt):
    _fig, _ax = plt.subplots(figsize=(5, 3.5))
    _ax.imshow(counts, cmap="Blues", vmin=0, vmax=50)
    for _i in range(2):
        for _j in range(2):
            _ax.text(_j, _i, str(counts[_i, _j]), ha="center", va="center")
    _ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="y", ylabel="x", title="Observed counts")
    _fig.tight_layout()
    _result = (
        "**Phi is undefined:** an empty marginal category or no observations."
        if phi is None
        else f"**Phi = {phi:.3f}**, with denominator **{denominator:.1f}**."
    )
    mo.vstack(
        [_fig, mo.md(_result),
         mo.ui.table(pd.DataFrame(counts, index=["x=0", "x=1"], columns=["y=0", "y=1"]), selection=None)]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Formula and worked calculation

    For the table $\begin{pmatrix}a&b\\c&d\end{pmatrix}$:

    $$
    \phi=\frac{ad-bc}{\sqrt{(a+b)(c+d)(a+c)(b+d)}}.
    $$

    With a = d = 30 and b = c = 10, the numerator is 900 - 100 = 800
    and the denominator is sqrt(40 x 40 x 40 x 40) = 1600.
    Thus **phi = 0.5**.

    For the **uncorrected Pearson chi-square statistic** on a 2 x 2 table:

    $$
    \phi^2=\chi^2/n.
    $$

    Taking the square root loses the sign. Cramer's V for this table is
    $|\phi|$, not signed phi.

    ## 3. Interpretation and limitations

    All four marginal totals must be positive. Fixed unequal marginal
    proportions can prevent phi from reaching -1 or +1. Zero phi corresponds
    to independence in the empirical binary table when margins are nonzero,
    but does not establish population independence from a finite sample.

    No normality is needed for the descriptive coefficient. Inference from
    counts needs an appropriate sampling design; sparse counts can make
    chi-square approximations unreliable. Phi is not generally an MLE of
    a latent normal correlation; that is a different model.

    Relabeling one variable reverses the sign. Relabeling both preserves it.
    Association alone cannot establish causation.

    ### Try it yourself

    1. Set all four counts equal. Why is phi zero?
    2. Set b = c = 0 with a and d positive.
    3. Set a = d = 0 with b and c positive.
    4. Empty an entire row. Why is phi undefined?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References and further reading

    - **Book:** Agresti, A. (2013),
      [Categorical Data Analysis, 3rd edition: author resources](https://alanagresti.com/cda/cda.html).
      Background on 2 x 2 tables, independence, and categorical association.
    - **Research article:** Bergsma, W. (2013), [A bias-correction for
      Cramer's and Tschuprow's association measures](https://doi.org/10.1016/j.jkss.2012.10.002).
      Further reading on finite-sample bias in related unsigned measures;
      for a 2 x 2 table, Cramer's V equals absolute phi.
    - **Practical reference:** [SciPy's contingency association documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.association.html)
      explains the relation between phi and larger-table association measures.

    DOI links lead to publisher pages; full-text access may require a subscription.
    """)
    return


if __name__ == "__main__":
    app.run()
