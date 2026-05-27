# -*- coding: utf-8 -*-
"""


@author: ibhan
"""


import pandas as pd
import numpy as np
import requests
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Callable, Optional
from cvxopt import matrix, spdiag


from model_cvx import mv_opt


_CACHE_DIR = Path(__file__).parent / "currency"


def _ecb_cache_path(currencies, start, end, freq, agg):
    key = "_".join(sorted(c.upper() for c in currencies))
    name = f"ecb_{key}__{start}__{end or 'open'}__{freq}__{agg}.pkl"
    return _CACHE_DIR / name


def ecb_fx_eur(
    currencies,
    start="2000-01-01",
    end=None,
    freq="Y",          # 'D', 'M', 'Q', 'A'
    agg="last",        # 'mean', 'last', 'first'
    refresh=False,
):
    """
    Download ECB FX rates and return data at user-chosen frequency.

    Results are cached on disk in `optimization/currency/`, keyed by the
    request parameters. With `refresh=False` (default) a cached file is
    returned if present; otherwise the data is fetched from the ECB and
    written to cache. With `refresh=True` the cache is always overwritten.

    Parameters
    ----------
    currencies : list[str]
        Currency codes, e.g. ['USD', 'GBP']
    start, end : str
        Date range (YYYY-MM-DD)
    freq : str
        Output frequency: 'D', 'M', 'QE', 'A'
    agg : str
        Aggregation: 'mean', 'last', 'first'
    refresh : bool
        If True, bypass the on-disk cache and re-download from the ECB.

    Returns
    -------
    pd.DataFrame
        FX rates with PeriodIndex(freq)
    """

    cache = _ecb_cache_path(currencies, start, end, freq, agg)

    if not refresh and cache.exists():
        return pd.read_pickle(cache)

    currency_str = "+".join(currencies).upper()

    url = (
        "https://data-api.ecb.europa.eu/service/data/EXR/"
        f"D.{currency_str}.EUR.SP00.A"
    )

    params = {
        "startPeriod": start,
        "endPeriod": end,
        "format": "csvdata"
    }

    r = requests.get(
        url,
        params=params,
        headers={"Accept": "text/csv"},
        timeout=30
    )
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text))
    df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"])

    wide = df.pivot(
        index="TIME_PERIOD",
        columns="CURRENCY",
        values="OBS_VALUE"
    ).sort_index()

    # --- resample ---
    if freq == "D":
        out = wide
    else:
        if agg == "mean":
            out = wide.resample(freq).mean()
        elif agg == "last":
            out = wide.resample(freq).last()
        elif agg == "first":
            out = wide.resample(freq).first()
        else:
            raise ValueError("agg must be 'mean', 'last', or 'first'")

    # PeriodIndex (macro-friendly)
    out.index = out.index.to_period(freq)

    # EUR-based names
    out = out.rename(columns={c: f"EUR_{c}" for c in out.columns})

    out.loc[:,'EUR_EUR']=1.0

    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_pickle(cache)

    return out




def convert_base_currency(
    fx,
    base,
    drop_base=True
):
    """
    Convert EUR-based FX rates to a single base currency.

    Parameters
    ----------
    fx : pd.DataFrame
        FX rates with columns like EUR_USD, EUR_GBP
        Index can be DatetimeIndex or PeriodIndex (any freq).
    base : str
        Base currency (e.g. 'USD')
    drop_base : bool
        If True, drop BASE_BASE column.

    Returns
    -------
    pd.DataFrame
        FX rates expressed as BASE_X
    """
    ubase= base.upper() 
    base_col = f"EUR_{ubase}"
    if base_col not in fx.columns:
        raise ValueError(f"{base_col} not found in DataFrame")

    out = pd.DataFrame(index=fx.index)

    for col in fx.columns:
        _, ccy = col.split("_", 1)

        if ccy == base:
            out[f"{ubase}_{ccy}"] = 1.0
        else:
            out[f"{ubase}_{ccy}"] = fx[col] / fx[base_col]

    if drop_base:
        out = out.drop(columns=f"{ubase}_{ubase}", errors="ignore")
    
        

    return out



def fx_from_xlsx(
    path=None,
    freq="Q",
    agg="mean",
    base="ZAR",
):
    """
    Load daily FX rates from an Excel file and resample to a chosen frequency.

    Expected Excel layout: one **Date** column (datetime-parseable) and one or
    more rate columns named ``{QUOTE}{BASE}`` or ``{BASE}{QUOTE}`` (e.g.
    ``'CHFZAR'``, ``'USDZAR '``). Trailing/leading whitespace in column names is
    stripped automatically.

    The function:

    1. Strips whitespace from column names.
    2. Detects columns whose names end in ``{BASE}`` (e.g. ``'…ZAR'``).
    3. Renames them to ``{BASE}_{QUOTE}`` (e.g. ``'ZAR_CHF'``).
    4. **Inverts** the values so each column represents *units of quote per one
       unit of base* — the same convention produced by
       ``ecb_fx_eur`` + ``convert_base_currency``.
       Example: CHFZAR = 15.30 → ZAR_CHF = 1/15.30 ≈ 0.065.
    5. Columns already in ``{BASE}{QUOTE}`` form (base as prefix) are renamed
       to ``{BASE}_{QUOTE}`` without inversion.
    6. Resamples from daily to the requested *freq* using *agg*.
    7. Returns a ``pd.PeriodIndex`` DataFrame, plug-in compatible with
       ``get_fx_returns`` / ``get_fx_covariance`` / ``mv_from_dataframes``.

    Parameters
    ----------
    path : str or Path, optional
        Excel file to read. Defaults to
        ``<this_file_parent>/currency/FX Data.xlsx``.
    freq : str
        Output frequency: ``'D'`` (daily), ``'M'`` (monthly), ``'Q'``
        (quarterly), ``'Y'`` (annual). Default ``'Q'``.
    agg : str
        Aggregation used when resampling: ``'mean'`` (default), ``'last'``,
        ``'first'``.
    base : str
        Base currency code that appears as a suffix in the raw column names.
        Default ``'ZAR'``.

    Returns
    -------
    pd.DataFrame
        Columns named ``{BASE}_{QUOTE}`` (e.g. ``'ZAR_CHF'``, ``'ZAR_USD'``),
        values = units of quote per 1 base unit.
        Index: ``pd.PeriodIndex`` at the requested frequency.

    Examples
    --------
    >>> import exchangerates_get as er
    >>> fx_q = er.fx_from_xlsx(freq='Q', agg='mean')   # quarterly means
    >>> fx_returns = er.get_fx_returns(fx_q)
    >>> fx_cov = er.get_fx_covariance(fx_returns)
    """
    if path is None:
        path = Path(__file__).parent / "currency" / "FX Data.xlsx"

    df = pd.read_excel(path)

    # 1. strip whitespace from column names
    df.columns = df.columns.str.strip()

    # 2. ensure Date column is datetime and use it as the index
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    # 3. detect, rename and (where needed) invert columns
    base_up = base.upper()
    rename_map = {}
    invert_cols = []  # columns to invert (XXXZAR -> ZAR_XXX)

    for col in df.columns:
        up = col.upper()
        if up.endswith(base_up):
            # e.g. 'CHFZAR' → quote='CHF', new='ZAR_CHF', must invert
            quote = up[: -len(base_up)]
            new_name = f"{base_up}_{quote}"
            rename_map[col] = new_name
            invert_cols.append(new_name)
        elif up.startswith(base_up):
            # e.g. 'ZARUSD' → quote='USD', new='ZAR_USD', already correct units
            quote = up[len(base_up):]
            rename_map[col] = f"{base_up}_{quote}"
        # else: leave unchanged (non-rate columns, if any)

    df = df.rename(columns=rename_map)

    # 4. invert XXXBASE columns: CHFZAR=15.30 → ZAR_CHF = 1/15.30 ≈ 0.065
    for col in invert_cols:
        df[col] = 1.0 / df[col]

    # 5. resample from daily
    if freq == "D":
        out = df
    else:
        if agg == "mean":
            out = df.resample(freq).mean()
        elif agg == "last":
            out = df.resample(freq).last()
        elif agg == "first":
            out = df.resample(freq).first()
        else:
            raise ValueError(f"agg must be 'mean', 'last', or 'first'; got {agg!r}")

    # 6. convert to PeriodIndex (macro-friendly, consistent with ecb_fx_eur)
    out.index = out.index.to_period(freq)

    return out


def get_fx_returns(
    fx,
    log_returns=True
):
    """
    Compute FX returns at the native frequency of the DataFrame.
    No missing-data policy is enforced here.
    """

    if log_returns:
        r = np.log(fx).diff()
    else:
        r = fx.pct_change()

    # first row always NaN
    return r.iloc[1:]

def get_fx_covariance(
    returns,
    allow_missing=False,
    correlation=False
):
    """
    Compute covariance (and optionally correlation) from FX returns.

    Parameters
    ----------
    returns : pd.DataFrame
        FX returns at native frequency
    allow_missing : bool
        If False, raise error on missing data
        If True, drop rows with any missing values
    correlation : bool
        If True, also return correlation matrix
    """

    r = returns.copy()

    if r.isna().any().any():

        if not allow_missing:
            mask = r.isna()

            missing = (
                mask.stack()
                    .loc[lambda x: x]
                    .reset_index()
            )
            missing.columns = ["period", "variable", "is_missing"]

            details = (
                missing.groupby("variable")["period"]
                       .apply(lambda x: ", ".join(map(str, x)))
                       .to_string()
            )

            raise ValueError(
                "Missing return data detected.\n\n"
                "Missing observations by variable:\n"
                f"{details}"
            )

        # explicit policy if missing allowed
        r = r.dropna(how="any")

    cov = r.cov()

    if not correlation:
        return cov

    corr = r.corr()
    return corr



import matplotlib.pyplot as plt


def plot_corr_with_std(
    returns,
    title="Correlation matrix (std on diagonal)",
    cmap="coolwarm",
    fmt_corr=".2f",
    fmt_std=".2f"
):
    """
    Plot correlation heatmap with standard deviation (%) on diagonal.

    Parameters
    ----------
    returns : pd.DataFrame
        FX returns at native frequency
    title : str
        Plot title
    cmap : str
        Matplotlib colormap
    fmt_corr : str
        Format for correlations
    fmt_std : str
        Format for std deviation (%)
    """

    # --- statistics ---
    corr = returns.corr()
    std_pct = 100 * returns.std()

    labels = corr.columns
    n = len(labels)

    fig, ax = plt.subplots(figsize=(1.2*n, 1.1*n))

    # --- heatmap ---
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)

    # --- ticks ---
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # --- annotations ---
    for i in range(n):
        for j in range(n):
            if i == j:
                text = f"{std_pct.iloc[i]:{fmt_std}}%"
                ax.text(j, i, text,
                        ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="black")
            else:
                ax.text(j, i, f"{corr.iloc[i, j]:{fmt_corr}}",
                        ha="center", va="center",
                        fontsize=9,
                        color="black")

    # --- colorbar ---
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Correlation")

    ax.set_title(title)
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt


def plot_return_scatter_matrix(
    returns,
    title="FX return scatterplot matrix"
):
    """
    Plot a scatterplot matrix of FX returns.
    Diagonal is left empty.

    Parameters
    ----------
    returns : pd.DataFrame
        FX returns at native frequency
    title : str
        Overall figure title
    """

    cols = returns.columns
    n = len(cols)

    fig, axes = plt.subplots(n, n, figsize=(2.2*n, 2.2*n))

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]

            if i == j:
                # diagonal intentionally empty
                ax.axis("off")
                continue

            ax.scatter(
                returns.iloc[:, j],
                returns.iloc[:, i],
                s=8,
                alpha=0.6
            )

            # Axis labels only on outer edges
            if i == n - 1:
                ax.set_xlabel(cols[j], fontsize=9)
            else:
                ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel(cols[i], fontsize=9)
            else:
                ax.set_yticklabels([])

            ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

import seaborn as sns


def plot_return_scatter_matrix_seaborn(
    returns,
    title="FX return scatterplot matrix"
):
    """
    Scatterplot matrix of FX returns using seaborn.
    Diagonal is left empty.
    """

    g = sns.PairGrid(
        returns,
        diag_sharey=False
    )

    # Off-diagonal scatterplots
    g.map_offdiag(
        sns.scatterplot,
        s=15,
        alpha=0.6
    )

    # Diagonal left empty
    def empty_diag(x, **kwargs):
        pass

    g.map_diag(empty_diag)

    # Improve layout
    for ax in g.axes.flatten():
        if ax is not None:
            ax.grid(True, alpha=0.3)

    g.fig.suptitle(title, y=1.02)
    plt.show()


import seaborn as sns
import matplotlib.pyplot as plt


def plot_return_scatter_matrix_with_marginals(
    returns,
    title="FX return scatterplot matrix with marginal distributions"
):
    """
    Scatterplot matrix of FX returns using seaborn.
    Diagonal shows histograms + KDE (marginal distributions).
    """

    g = sns.PairGrid(
        returns,
        diag_sharey=False,
        corner=False
    )

    # Off-diagonal: scatterplots
    g.map_offdiag(
        sns.scatterplot,
        s=15,
        alpha=0.6
    )

    # Diagonal: histogram + KDE
    g.map_diag(
        sns.histplot,
        kde=True,
        stat="density",
        bins=30
    )

    # Grid and aesthetics
    for ax in g.axes.flatten():
        if ax is not None:
            ax.grid(True, alpha=0.3)

    g.fig.suptitle(title, y=1.02)
    plt.show()





def plot_indexed_fx(
    fx,
    base_year=None,
    show_legend=False,
    title="FX indices (base = 100)",
    min_label_gap=3.0
):
    """
    Plot FX series indexed to 100 with collision-aware end labels.

    - Base currency inferred from column names (before '_')
    - Quote currency (after '_') used for labels
    - End labels are spread vertically if too close
    - Arrows point from label to line end
    """

    # --- infer base currency ---
    base_currency = fx.columns[0].split("_", 1)[0]

    # --- choose base period ---
    if base_year is None:
        base = fx.iloc[0]
        base_label = "first observation"
    else:
        base = fx.loc[fx.index.year == base_year].iloc[0]
        base_label = str(base_year)

    indexed = 100 * fx / base

    # --- x-axis for matplotlib ---
    if isinstance(indexed.index, pd.PeriodIndex):
        x = indexed.index.to_timestamp()
    else:
        x = indexed.index

    fig, ax = plt.subplots(figsize=(10, 6))

    # store end points for label placement
    end_points = []

    for col in indexed.columns:
        base_ccy, quote_ccy = col.split("_", 1)

        # skip BASE_BASE
        if base_ccy == quote_ccy:
            continue

        y = indexed[col]
        ax.plot(x, y, linewidth=3)

        end_points.append({
            "label": quote_ccy,
            "x": x[-1],
            "y": y.iloc[-1]
        })

    # --- label collision handling ---
    if not show_legend:
        # sort by y-value
        end_points = sorted(end_points, key=lambda d: d["y"])

        adjusted = []
        for p in end_points:
            y_adj = p["y"]
            if adjusted:
                y_adj = max(y_adj, adjusted[-1]["y_adj"] + min_label_gap)
            p["y_adj"] = y_adj
            adjusted.append(p)

        # draw labels and arrows
            # horizontal offset (2% of x-range)
            x_offset = (x[-1] - x[0]) * 0.02
            
            for p in adjusted:
                ax.annotate(
                    p["label"],
                    xy=(p["x"], p["y"]),                    # arrow target (line end)
                    xytext=(p["x"] + x_offset, p["y_adj"]), # label moved right
                    textcoords="data",
                    ha="left",
                    va="center",
                    fontsize=9,
                    arrowprops=dict(
                        arrowstyle="-",
                        lw=0.8,
                        color="black"
                    )
                )

    else:
        ax.legend([p["label"] for p in end_points], loc="best")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    
    ax.tick_params(axis="both", which="both", length=4)


    ax.set_title(f"{title} (base currency: {base_currency}, {base_label} = 100)")
    ax.set_ylabel("Index")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def mv_frontier_from_dataframes(
    cov_df: pd.DataFrame,
    ret_s: pd.Series,
    min_s: pd.Series,
    max_s: pd.Series,
    n_points: int = 101
):
    """
    Compute the full mean–variance efficient frontier.

    Parameters
    ----------
    cov_df : pd.DataFrame
        Covariance matrix (assets × assets)
    ret_s : pd.Series
        Expected returns
    min_s : pd.Series
        Minimum weights
    max_s : pd.Series
        Maximum weights
    n_points : int
        Number of points on the frontier (default 101)

    Returns
    -------
    pd.DataFrame
        Columns:
          - risk
          - return
          - one column per asset with portfolio weights
    """

    # --------------------------------------------------
    # Input validation and alignment
    # --------------------------------------------------
    assets = cov_df.index

    if not cov_df.columns.equals(assets):
        raise ValueError("Covariance matrix must be square with matching index/columns.")

    for s, name in [(ret_s, "ret_s"), (min_s, "min_s"), (max_s, "max_s")]:
        if not s.index.equals(assets):
            raise ValueError(f"{name} index must match covariance matrix.")

    n = len(assets)

    # --------------------------------------------------
    # Convert static inputs to CVXOPT matrices
    # --------------------------------------------------
    P = matrix(cov_df.values)
    q = matrix(ret_s.values)

    bounds_min = matrix(min_s.values)
    bounds_max = matrix(max_s.values)

    # --------------------------------------------------
    # Risk aversion grid
    # --------------------------------------------------
    risk_aversions = np.linspace(0.0, 1.0, n_points)

    rows = []

    # --------------------------------------------------
    # Solve one QP per risk aversion
    # --------------------------------------------------
    for lam in risk_aversions:

        x = mv_opt(
            PP=P,
            qq=q,
            riskaversion=float(lam),
            bsum=1.0,
            weights=[],          # no weighted constraints
            weigthtedsum=[],
            boundsmin=bounds_min,
            boundsmax=bounds_max,
            maximize=False 
        )

        x = np.array(x).ravel()

        # portfolio statistics
        port_return = float(x @ ret_s.values)
        port_risk   = float(np.sqrt(x @ cov_df.values @ x))

        rows.append(
            [port_risk, port_return, *x]
        )

    # --------------------------------------------------
    # Assemble DataFrame
    # --------------------------------------------------
    columns = ["risk", "return"] + list(assets)

    frontier = pd.DataFrame(rows, columns=columns)

    return frontier


def mv_from_dataframes(
    cov_df: pd.DataFrame,
    assumptions: pd.DataFrame,
    n_points: int = 101,
):
    """
    Compute the MV debt cost-risk frontier with the current composition as row 0.

    Convenience wrapper around `mv_frontier_from_dataframes` that takes a single
    `assumptions` DataFrame instead of separate series, and prepends the current
    portfolio composition as the first row of the result.

    Parameters
    ----------
    cov_df : pd.DataFrame
        Covariance matrix of currency returns (assets x assets).
    assumptions : pd.DataFrame
        One row per currency. Index must equal `cov_df.index`/`cov_df.columns`.
        Required columns:
          - interest_rate          : coupon / yield per currency
          - expected_appreciation  : expected appreciation of the foreign currency
                                     vs. the base (positive = base weakens =
                                     debt becomes more expensive)
          - min_share              : lower bound on portfolio share
          - max_share              : upper bound on portfolio share
          - current_share          : current portfolio composition
        Expected cost per currency = interest_rate + expected_appreciation.
    n_points : int
        Number of points on the frontier (default 101).

    Returns
    -------
    pd.DataFrame
        Columns: ['risk', 'return', *assets].
        Row 0 is the current composition (`risk` and `return` evaluated at
        `current_share`); rows 1..n_points are the frontier sweeping
        risk-aversion from 0 to 1. `return` is expected debt cost.
    """

    required = [
        'interest_rate',
        'expected_appreciation',
        'min_share',
        'max_share',
        'current_share',
    ]
    missing = [c for c in required if c not in assumptions.columns]
    if missing:
        raise ValueError(
            f"assumptions DataFrame is missing required columns: {missing}"
        )

    if not assumptions.index.equals(cov_df.index):
        raise ValueError(
            "assumptions index must match covariance matrix index."
        )

    expected_cost = assumptions['interest_rate'] + assumptions['expected_appreciation']

    frontier = mv_frontier_from_dataframes(
        cov_df=cov_df,
        ret_s=expected_cost,
        min_s=assumptions['min_share'],
        max_s=assumptions['max_share'],
        n_points=n_points,
    )

    current = assumptions['current_share'].values
    current_cost = float(current @ expected_cost.values)
    current_risk = float(np.sqrt(current @ cov_df.values @ current))

    current_row = pd.DataFrame(
        [[current_risk, current_cost, *current]],
        columns=frontier.columns,
    )

    return pd.concat([current_row, frontier], ignore_index=True)


# ============================================================
# Debt cost frontier plotting – FINAL, collision-safe version
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# Public API
# ============================================================

def plot_debt_frontier_labeled(
    res: pd.DataFrame,
    risk_col="risk",
    cost_col="return",
    asset_cols=None,
    figsize=(9, 12),
    title="Debt cost–risk frontier",
    min_label_gap=0.03,
    label_pos="start",              # "start" or "end"
    label_xpad_frac=0.08,
    yaxis_clearance_frac=0.14,      # <<< HARD GUARANTEE AGAINST Y-AXIS OVERLAP
    show_currency_points=True,
    cost_s: pd.Series | None = None,
    cov_df: pd.DataFrame | None = None,
    point_min_gap_frac=0.04,
    cmap="tab10",
    current: pd.Series | None = None,
    current_label: str = "Current portfolio",
    export_path=None,
    export_formats=("png", "pdf", "svg"),
):
    """
    Debt cost frontier with:
      Panel 1: frontier + standalone currency points (+ optional current point)
      Panel 2: funding weights (lines + labels)
      Panel 3: funding weights (stacked area + labels)

    Designed for cost minimisation (maximize=False).

    If `current` is provided (a Series with at least `risk_col` and `cost_col`
    entries, e.g. `res.iloc[0]` when `res` came from `mv_from_dataframes`), the
    current portfolio is drawn on panel 1 as a labelled marker.
    """

    if asset_cols is None:
        asset_cols = [c for c in res.columns if c not in (risk_col, cost_col)]

    x = res[risk_col]
    colors = _make_color_map(asset_cols, cmap)

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=False)

    # --------------------------------------------------
    # >>> STRUCTURAL FIX: MOVE AXES RIGHT <<<
    # --------------------------------------------------
    fig.subplots_adjust(left=yaxis_clearance_frac)

    # ==================================================
    # 1. Cost–risk frontier
    # ==================================================
    axes[0].plot(x, res[cost_col], lw=4, color="black")
    axes[0].set_ylabel("Expected debt cost")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    if show_currency_points:
        if cost_s is None or cov_df is None:
            raise ValueError("cost_s and cov_df must be provided")

        _annotate_currency_points_panel1(
            ax=axes[0],
            cost_s=cost_s,
            cov_df=cov_df,
            colors=colors,
            min_gap_frac=point_min_gap_frac
        )

    if current is not None:
        _annotate_current_point_panel1(
            ax=axes[0],
            current=current,
            risk_col=risk_col,
            cost_col=cost_col,
            label=current_label,
        )

    # ==================================================
    # 2. Funding weights — lines
    # ==================================================
    for c in asset_cols:
        axes[1].plot(x, res[c], lw=2, color=colors[c])

    axes[1].set_ylabel("Funding share")
    axes[1].set_title("Funding composition (lines)")
    axes[1].grid(True, alpha=0.3)

    _expand_xlim_for_labels(axes[1], x, label_pos, label_xpad_frac)

    _annotate_line_labels(
        ax=axes[1],
        x=x,
        data=res[asset_cols],
        colors=colors,
        min_gap=min_label_gap,
        pos=label_pos,
        xpad_frac=label_xpad_frac
    )

    # ==================================================
    # 3. Funding weights — stacked area
    # ==================================================
    axes[2].stackplot(
        x,
        *[res[c] for c in asset_cols],
        colors=[colors[c] for c in asset_cols],
        alpha=0.85
    )

    axes[2].set_xlabel("Risk (cost volatility)")
    axes[2].set_ylabel("Funding share")
    axes[2].set_title("Funding composition (stacked)")
    axes[2].grid(True, alpha=0.3)

    _expand_xlim_for_labels(axes[2], x, label_pos, label_xpad_frac)

    _annotate_area_labels(
        ax=axes[2],
        x=x,
        data=res[asset_cols],
        colors=colors,
        min_gap=min_label_gap,
        pos=label_pos,
        xpad_frac=label_xpad_frac
    )

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if export_path:
        export_figure(fig, export_path, export_formats)

    plt.show()


# ============================================================
# Export
# ============================================================

def export_figure(fig, path, formats=("png", "pdf"), dpi=300):
    base = Path(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(base.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")


# ============================================================
# Colour handling
# ============================================================

def _make_color_map(assets, cmap):
    cmap = plt.get_cmap(cmap)
    return {a: cmap(i % cmap.N) for i, a in enumerate(assets)}


# ============================================================
# Axis helpers
# ============================================================

def _expand_xlim_for_labels(ax, x, pos, frac):
    xmin, xmax = float(x.min()), float(x.max())
    xrng = xmax - xmin if xmax != xmin else 1.0

    if pos == "start":
        ax.set_xlim(xmin - frac * xrng, xmax)
    else:
        ax.set_xlim(xmin, xmax + frac * xrng)


# ============================================================
# Annotation helpers
# ============================================================

def _annotate_line_labels(ax, x, data, colors, min_gap, pos, xpad_frac):
    xmin, xmax = ax.get_xlim()
    xrng = xmax - xmin if xmax != xmin else 1.0

    if pos == "start":
        idx = 0
        x_anchor = x.iloc[0]

        # place label BETWEEN y-axis and line start
        x_text = xmin + 0.5 * (x_anchor - xmin)
        ha = "left"

    else:  # "end"
        idx = -1
        x_anchor = x.iloc[-1]
        x_text = x_anchor + xrng * (xpad_frac * 0.8)
        ha = "left"

    points = [{"c": c, "y": float(data[c].iloc[idx])} for c in data.columns]
    points = sorted(points, key=lambda d: d["y"])

    for i, p in enumerate(points):
        y_adj = p["y"] if i == 0 else max(
            p["y"], points[i-1]["y_adj"] + min_gap
        )
        p["y_adj"] = y_adj

        ax.annotate(
            p["c"],
            xy=(x_anchor, p["y"]),
            xytext=(x_text, y_adj),
            ha=ha,
            va="center",
            fontsize=9,
            color=colors[p["c"]],
            arrowprops=dict(
                arrowstyle="-",
                lw=0.8,
                color=colors[p["c"]],
            ),
        )


def _annotate_area_labels(ax, x, data, colors, min_gap, pos, xpad_frac):
    cumulative = data.cumsum(axis=1)
    xmin, xmax = ax.get_xlim()
    xrng = xmax - xmin if xmax != xmin else 1.0

    if pos == "start":
        idx = 0
        x_anchor = x.iloc[0]

        # label BETWEEN y-axis and area start
        x_text = xmin + 0.5 * (x_anchor - xmin)
        ha = "left"

    else:  # "end"
        idx = -1
        x_anchor = x.iloc[-1]
        x_text = x_anchor + xrng * (xpad_frac * 0.8)
        ha = "left"

    mids = cumulative.iloc[idx] - data.iloc[idx] / 2
    points = [{"c": c, "y": float(mids[c])} for c in data.columns]
    points = sorted(points, key=lambda d: d["y"])

    for i, p in enumerate(points):
        y_adj = p["y"] if i == 0 else max(
            p["y"], points[i-1]["y_adj"] + min_gap
        )
        p["y_adj"] = y_adj

        ax.annotate(
            p["c"],
            xy=(x_anchor, p["y"]),
            xytext=(x_text, y_adj),
            ha=ha,
            va="center",
            fontsize=9,
            color=colors[p["c"]],
            arrowprops=dict(
                arrowstyle="-",
                lw=0.8,
                color=colors[p["c"]],
            ),
        )


def _annotate_current_point_panel1(ax, current, risk_col, cost_col, label):
    x = float(current[risk_col])
    y = float(current[cost_col])

    ax.scatter(
        [x], [y],
        marker="*",
        s=260,
        color="black",
        edgecolor="white",
        linewidth=1.2,
        zorder=5,
    )

    xmin, xmax = ax.get_xlim()
    xrng = xmax - xmin if xmax != xmin else 1.0

    ax.annotate(
        label,
        xy=(x, y),
        xytext=(x + xrng * 0.05, y),
        ha="left",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="black",
        arrowprops=dict(arrowstyle="-", lw=1.0, color="black"),
    )


def _annotate_currency_points_panel1(ax, cost_s, cov_df, colors, min_gap_frac):
    assets = cost_s.index
    risk = np.sqrt(np.diag(cov_df.loc[assets, assets].values))
    pts = pd.DataFrame({"risk": risk, "cost": cost_s.values}, index=assets)

    ax.scatter(pts["risk"], pts["cost"], c=[colors[c] for c in assets], s=35)

    xmin, xmax = ax.get_xlim()
    xrng = xmax - xmin if xmax != xmin else 1.0
    x_offset = xrng * 0.04

    ymin, ymax = ax.get_ylim()
    yrng = ymax - ymin if ymax != ymin else 1.0
    min_gap = abs(yrng) * min_gap_frac

    points = [
        {"c": c, "x": pts.loc[c, "risk"], "y": pts.loc[c, "cost"]}
        for c in assets
    ]
    points = sorted(points, key=lambda d: d["y"])

    for i, p in enumerate(points):
        y_adj = p["y"] if i == 0 else max(p["y"], points[i-1]["y_adj"] + min_gap)
        p["y_adj"] = y_adj

        ax.annotate(
            p["c"],
            xy=(p["x"], p["y"]),
            xytext=(p["x"] + x_offset, y_adj),
            ha="left",
            va="center",
            fontsize=9,
            color=colors[p["c"]],
            arrowprops=dict(arrowstyle="-", lw=0.8, color=colors[p["c"]])
        )


# ============================================================
# Interactive input widget for the debt-cost frontier
# ============================================================

_FRONTIER_COLS = [
    'interest_rate',
    'expected_appreciation',
    'min_share',
    'max_share',
    'current_share',
]

_FRONTIER_COL_LABELS = {
    'interest_rate':         'Interest rate',
    'expected_appreciation': 'Expected FX appr.',
    'min_share':             'Min share',
    'max_share':             'Max share',
    'current_share':         'Current share',
}

_FRONTIER_COL_STEP = {
    'interest_rate':         0.001,
    'expected_appreciation': 0.001,
    'min_share':             0.05,
    'max_share':             0.05,
    'current_share':         0.05,
}


@dataclass
class DebtFrontierInputs:
    """User-editable inputs for the debt-cost frontier.

    Wraps a covariance matrix plus the per-currency assumptions DataFrame
    used by `mv_from_dataframes`. Provides an ipywidgets grid editor and
    a `plot()` shortcut.

    Parameters
    ----------
    cov_df : pd.DataFrame
        Covariance matrix indexed by currency code (square, matching rows/cols).
        Treated as read-only — provided once at construction.
    assumptions : pd.DataFrame
        One row per currency. Index must cover `cov_df.index`. Required columns:
        `interest_rate`, `expected_appreciation`, `min_share`, `max_share`,
        `current_share`. A copy is stored on the instance so the caller's
        frame is not mutated.
    n_points : int
        Frontier resolution forwarded to `mv_from_dataframes`.
    """

    cov_df: pd.DataFrame
    assumptions: pd.DataFrame
    n_points: int = 101
    name: str = 'basis'
    chartfolder: str = 'graph/'

    def __post_init__(self):
        if not self.cov_df.index.equals(self.cov_df.columns):
            raise ValueError("cov_df must be square with matching index/columns.")

        missing = [c for c in _FRONTIER_COLS if c not in self.assumptions.columns]
        if missing:
            raise ValueError(
                f"assumptions DataFrame is missing required columns: {missing}"
            )

        if not set(self.cov_df.index).issubset(self.assumptions.index):
            raise ValueError(
                "assumptions index must cover every currency in cov_df.index."
            )

        # store an aligned, editable copy
        self.assumptions = (
            self.assumptions.loc[self.cov_df.index, _FRONTIER_COLS].astype(float).copy()
        )

    # ----- defaults --------------------------------------------------
    @classmethod
    def from_cov(
        cls,
        cov_df: pd.DataFrame,
        interest_rate: float = 0.02,
        expected_appreciation: float = 0.0,
        min_share: float = 0.0,
        max_share: float = 1.0,
        equal_weights: bool = True,
        n_points: int = 101,
        name: str = 'basis',
        chartfolder: str = 'graph/',
    ):
        """Build an instance with sensible defaults for every currency."""
        n = len(cov_df.index)
        current = (1.0 / n) if equal_weights else 0.0

        assumptions = pd.DataFrame(
            {
                'interest_rate':         interest_rate,
                'expected_appreciation': expected_appreciation,
                'min_share':             min_share,
                'max_share':             max_share,
                'current_share':         current,
            },
            index=cov_df.index,
        )
        return cls(cov_df=cov_df, assumptions=assumptions, n_points=n_points, name=name, chartfolder=chartfolder)

    # ----- compute ---------------------------------------------------
    def solve(self):
        """Run `mv_from_dataframes` against the current assumptions."""
        return mv_from_dataframes(
            cov_df=self.cov_df,
            assumptions=self.assumptions,
            n_points=self.n_points,
        )

    def plot(self, **kwargs):
        """Solve and draw the labelled debt-cost frontier."""
        res = self.solve()
        expected_cost = (
            self.assumptions['interest_rate']
            + self.assumptions['expected_appreciation']
        )

        export_path = str(Path(self.chartfolder) / self.name)

        defaults = dict(
            label_pos="start",
            cost_col="return",
            cost_s=expected_cost,
            cov_df=self.cov_df,
            current=res.iloc[0],
            export_path=export_path,
            export_formats=('svg',),
        )
        defaults.update(kwargs)

        plot_debt_frontier_labeled(
            res.iloc[1:].reset_index(drop=True),
            **defaults,
        )
        return res

    # ----- widget ----------------------------------------------------
    def widget(self, on_change: Optional[Callable] = None):
        """Build an ipywidgets grid editor.

        Rows are currencies; columns are the five input fields. Editing a
        cell writes through to `self.assumptions` immediately. The Run
        button calls `self.plot()` into an Output area below.

        Parameters
        ----------
        on_change : callable, optional
            Called as `on_change(self)` after every cell edit. Useful for
            wiring up custom live displays.

        Returns
        -------
        ipywidgets.VBox
            Drop into a notebook cell to render.
        """
        import ipywidgets as widgets
        from IPython.display import clear_output

        ccys = list(self.assumptions.index)
        cells = {}

        # header row: blank corner + column titles
        header = [widgets.HTML("")]
        for col in _FRONTIER_COLS:
            header.append(
                widgets.HTML(
                    f"<div style='text-align:center;font-weight:600'>"
                    f"{_FRONTIER_COL_LABELS[col]}</div>"
                )
            )

        body = []
        for ccy in ccys:
            body.append(
                widgets.HTML(f"<b style='padding-right:6px'>{ccy}</b>")
            )
            for col in _FRONTIER_COLS:
                w = widgets.FloatText(
                    value=float(self.assumptions.at[ccy, col]),
                    step=_FRONTIER_COL_STEP[col],
                    layout=widgets.Layout(width='110px'),
                )
                cells[(ccy, col)] = w
                body.append(w)

        grid = widgets.GridBox(
            children=header + body,
            layout=widgets.Layout(
                grid_template_columns='80px ' + 'repeat({}, 120px)'.format(
                    len(_FRONTIER_COLS)
                ),
                grid_gap='4px 8px',
                align_items='center',
            ),
        )

        name_box = widgets.Text(
            value=self.name,
            description='Name:',
            layout=widgets.Layout(width='260px'),
        )

        sum_label = widgets.HTML()
        run_btn = widgets.Button(
            description='Run frontier',
            button_style='primary',
            icon='play',
        )
        out = widgets.Output()

        def refresh_sum():
            s = float(self.assumptions['current_share'].sum())
            ok = abs(s - 1.0) < 1e-6
            color = '#1a7f37' if ok else '#b42318'
            sum_label.value = (
                f"<span style='color:{color};margin-left:12px'>"
                f"Σ current_share = {s:.4f}</span>"
            )

        def on_name_change(change):
            self.name = change['new']

        def on_cell_change(change, ccy, col):
            self.assumptions.at[ccy, col] = float(change['new'])
            if col == 'current_share':
                refresh_sum()
            if on_change is not None:
                on_change(self)

        name_box.observe(on_name_change, names='value')

        for (ccy, col), w in cells.items():
            w.observe(
                lambda change, ccy=ccy, col=col: on_cell_change(change, ccy, col),
                names='value',
            )

        def on_run(_):
            with out:
                clear_output(wait=True)
                self.plot()

        run_btn.on_click(on_run)
        refresh_sum()

        return widgets.VBox([
            widgets.HTML(
                "<h3 style='margin:4px 0'>Debt-cost frontier inputs</h3>"
                "<div style='color:#555;margin-bottom:6px'>"
                "Edit any cell, then press <b>Run frontier</b>. "
                f"Chart saved to <code>{self.chartfolder}&lt;name&gt;.svg</code>."
                "</div>"
            ),
            name_box,
            grid,
            widgets.HBox([run_btn, sum_label]),
            out,
        ])


if __name__ == '__main__':

    #%% Step 1: download
    fx_eur = ecb_fx_eur(
        currencies=["USD", "GBP", "JPY", "CHF","EUR","ZAR"],
        start="2010-01-01",
        freq='Q'
    )
    
    # Step 2: express everything in base currency 
    fx_ccy = convert_base_currency(fx_eur, base="zar")
    
    fx_returns = get_fx_returns(fx_ccy)
    # Step 3: yearly covariance matrices
    fx_cov = get_fx_covariance(fx_returns)
    if 0:    
        fx_corr = get_fx_covariance(fx_returns,correlation=True)
        # Example: covariance matrix for 2022
        print(fx_cov)
        std = np.sqrt(np.diag(fx_cov))
        std = pd.Series(std, index=fx_cov.index, name="std")
        
        print(std*100)
        
        plot_corr_with_std(
            fx_returns,
            title="FX correlation (std on diagonal, %)"
        )
        
        
        plot_return_scatter_matrix_with_marginals(
            fx_returns,
            title="FX return scatterplot matrix (native frequency)"
        )
        #%%
        plot_indexed_fx(
            fx_ccy,
            min_label_gap=10.0
            
        )
    #%%
    cov_df = fx_cov.rename(
        index=lambda x: x.split('_')[1],
        columns=lambda x: x.split('_')[1],
    )
    names = cov_df.index

    assumptions = pd.DataFrame(
        {
            'interest_rate':         [0.010, 0.020, 0.023, 0.013, 0.034],
            'expected_appreciation': [0.000, 0.000, 0.000, 0.000, 0.000],
            'min_share':             [0.0, 0.0, 0.0, 0.0, 0.0],
            'max_share':             [1.0, 1.0, 1.0, 1.0, 1.0],
            'current_share':         [0.2, 0.2, 0.2, 0.2, 0.2],
        },
        index=names,
    )

    res = mv_from_dataframes(cov_df=cov_df, assumptions=assumptions)

    expected_cost = assumptions['interest_rate'] + assumptions['expected_appreciation']

    plot_debt_frontier_labeled(
        res.iloc[1:].reset_index(drop=True),
        label_pos="start",
        cost_col="return",
        cost_s=expected_cost,
        cov_df=cov_df,
        current=res.iloc[0],
        export_path="zar_debt_frontier",
        export_formats=("png", "pdf", "svg"),
    )

