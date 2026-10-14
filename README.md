# marimo-notebooks

Public [marimo](https://marimo.io) notebooks, showcased in the [Notebook Gallery](https://radek-dev.github.io/notebook-gallery/).

| Notebook | Description | Preview |
| :-- | :-- | :-- |
| `sales_trends.py` | Seasonal sales explorer on synthetic data | [Static](https://molab.marimo.io/github/radek-dev/marimo-notebooks/blob/main/sales_trends.py) · [Interactive](https://molab.marimo.io/github/radek-dev/marimo-notebooks/blob/main/sales_trends.py/wasm) |
| `forecasting.py` | Baseline vs. trend + seasonality forecasts with a backtest, on synthetic data | [Static](https://molab.marimo.io/github/radek-dev/marimo-notebooks/blob/main/forecasting.py) · [Interactive](https://molab.marimo.io/github/radek-dev/marimo-notebooks/blob/main/forecasting.py/wasm) |

## Run locally

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/). The environment is locked in `uv.lock`.

```sh
uv sync                              # create .venv from uv.lock
uv run marimo edit sales_trends.py   # open a notebook
```

Each notebook also declares its dependencies as inline script metadata, so it runs on its own with `uvx marimo edit --sandbox sales_trends.py`.

Add packages with `uv add <package>`. After changing a notebook, refresh its saved static outputs with `uv run marimo export session <notebook>.py` and commit `__marimo__/`.
