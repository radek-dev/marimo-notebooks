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
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    return mo, np, pd, plt


@app.cell
def _(mo):
    mo.md("""
    # Sample: Forecasting Walkthrough

    How much does a slightly smarter model improve on a simple baseline? This
    notebook compares three forecasts on **synthetic monthly data** (no real or
    confidential data), scored on a held-out period.

    [← Back to the Notebook Gallery](https://radek-dev.github.io/notebook-gallery/)
    """)
    return


@app.cell
def _(mo):
    horizon = mo.ui.slider(6, 18, value=12, label="Forecast horizon (months)")
    noise = mo.ui.slider(1, 20, value=6, label="Noise")
    mo.hstack([horizon, noise])
    return horizon, noise


@app.cell
def _(noise, np, pd):
    rng = np.random.default_rng(7)
    idx = pd.date_range("2018-01-01", periods=72, freq="MS")
    t = np.arange(len(idx))
    y = (
        100
        + 1.5 * t
        + 18 * np.sin(2 * np.pi * t / 12)
        + rng.normal(0, noise.value, len(idx))
    )
    series = pd.Series(y, index=idx)
    return series, t


@app.cell
def _(horizon, np, series, t):
    h = horizon.value
    train, test = series.iloc[:-h], series.iloc[-h:]
    n = len(train)

    # Baseline 1: repeat the last observed value
    naive = np.full(h, train.iloc[-1])

    # Baseline 2: repeat the value from the same month last year
    seasonal_naive = np.array([train.iloc[n - 12 + (i % 12)] for i in range(h)])

    # Model: linear trend plus month-of-year effects, fitted by least squares
    def design(tt):
        month = tt % 12
        return np.column_stack([tt] + [(month == m).astype(float) for m in range(12)])

    coef, *_ = np.linalg.lstsq(design(t[:n]), train.values, rcond=None)
    regression = design(t[n:]) @ coef
    return h, naive, regression, seasonal_naive, test, train


@app.cell
def _(h, mo, naive, np, pd, regression, seasonal_naive, test):
    def mae(pred):
        return float(np.mean(np.abs(test.values - pred)))

    scores = pd.DataFrame(
        {
            "Model": ["Naive", "Seasonal naive", "Trend + seasonality"],
            "MAE": [mae(naive), mae(seasonal_naive), mae(regression)],
        }
    ).round(2)
    best = scores.loc[scores["MAE"].idxmin(), "Model"]
    mo.vstack(
        [
            mo.md(f"**Backtest on the last {h} months** (lower MAE is better)"),
            mo.ui.table(scores, selection=None),
            mo.md(f"Best model here: **{best}**."),
        ]
    )
    return


@app.cell
def _(naive, plt, regression, seasonal_naive, test, train):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(train.index, train.values, color="#14213d", label="History")
    ax.plot(test.index, test.values, color="#14213d", linestyle=":", label="Actual")
    ax.plot(test.index, naive, label="Naive")
    ax.plot(test.index, seasonal_naive, label="Seasonal naive")
    ax.plot(test.index, regression, label="Trend + seasonality", linewidth=2.5)
    ax.set_title("Forecasts vs. actuals (synthetic)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig
    return


if __name__ == "__main__":
    app.run()
