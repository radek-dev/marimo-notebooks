# /// script
# requires-python = ">=3.11"
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
    import numpy as np
    import pandas as pd

    return mo, np, pd


@app.cell
def _(mo):
    mo.md("""
    # Sample: Seasonal Sales Explorer

    A small, self-contained notebook using **synthetic data** (no real or
    confidential data). Adjust the controls to see how trend and seasonality
    change the series.

    [← Back to the Notebook Gallery](https://radek-dev.github.io/notebook-gallery/)
    """)
    return


@app.cell
def _(mo):
    trend = mo.ui.slider(0, 5, value=2, step=0.5, label="Monthly trend")
    amplitude = mo.ui.slider(0, 50, value=20, label="Seasonal amplitude")
    noise = mo.ui.slider(0, 20, value=5, label="Noise")
    mo.hstack([trend, amplitude, noise])
    return amplitude, noise, trend


@app.cell
def _(amplitude, noise, np, pd, trend):
    rng = np.random.default_rng(42)
    months = pd.date_range("2020-01-01", periods=48, freq="MS")
    t = np.arange(len(months))
    sales = (
        100
        + trend.value * t
        + amplitude.value * np.sin(2 * np.pi * t / 12)
        + rng.normal(0, noise.value, len(months))
    )
    df = pd.DataFrame({"month": months, "sales": sales}).set_index("month")
    return (df,)


@app.cell
def _(df):
    df["sales"].plot(title="Monthly sales (synthetic)", figsize=(8, 3.5))
    return


@app.cell
def _(df, mo):
    by_month = df.groupby(df.index.month)["sales"].mean().round(1)
    mo.md(
        f"Average sales peak in month **{by_month.idxmax()}** "
        f"and bottom out in month **{by_month.idxmin()}**."
    )
    return


if __name__ == "__main__":
    app.run()
