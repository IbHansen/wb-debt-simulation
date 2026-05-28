# -*- coding: utf-8 -*-
"""
get_kes.py – reader for the CBK (Central Bank of Kenya) FX rate file.

The file ``currency/kes.csv`` contains daily KES exchange-rate quotes
published by the Central Bank of Kenya.  It is a headerless, tall-format
CSV with five columns:

    date       publication date  (two formats: DD/MM/YYYY or YYYY-MM-DD)
    currency   currency description, e.g. 'US DOLLAR', 'STG POUND',
               'KES / USHS', 'JPY (100)'
    mid        mid-market rate  (KES per unit of foreign currency)
    bid        bid rate
    ask        ask rate

Public API
----------
read_kes_fx(path=DEFAULT_KES_PATH) -> pd.DataFrame
    Load the file, normalise dates, drop duplicate rows, sort, and return
    a tidy DataFrame with columns [date, currency, mid, bid, ask].

Usage
-----
    from get_kes import read_kes_fx

    df = read_kes_fx()                          # uses bundled kes.csv
    df = read_kes_fx("path/to/other/kes.csv")   # custom path

    # pivot to wide: one column per currency, date as index
    wide = df.pivot_table(values="mid", index="date", columns="currency")

    # filter to a single currency
    usd = df[df["currency"] == "US DOLLAR"].set_index("date")["mid"]

    # monthly average per currency
    monthly = (
        df.set_index("date")
        .groupby("currency")["mid"]
        .resample("ME")
        .mean()
        .reset_index()
    )

Notes
-----
- JPY is quoted per 100 units ("JPY (100)"); divide mid/bid/ask by 100
  if you need per-unit rates.
- Some dates have duplicate rows in the source file; they are silently
  removed via drop_duplicates().
- Date parsing uses ``format="mixed", errors="coerce", dayfirst=True``
  (pandas ≥ 2.0).  On older pandas it falls back to
  ``infer_datetime_format=True``.  Rows whose date cannot be parsed
  (e.g. embedded header lines like ``Date,"US DOLLAR",...``) are
  silently dropped.

@author: ibhan
"""

from pathlib import Path

import pandas as pd
from datetime import date as _date

# ---------------------------------------------------------------------------
# Default data paths
# ---------------------------------------------------------------------------
DEFAULT_KES_PATH: Path = Path(__file__).parent / "currency" / "kes.csv"
DEFAULT_KES_LATEST_PATH: Path = Path(__file__).parent / "currency" / "kes_latest.xlsx"
DEFAULT_KES_WIDE_PATH: Path = Path(__file__).parent / "currency" / "kes_wide.xlsx"

# ---------------------------------------------------------------------------
# CBK currency-name → ISO 4217 code
# Covers every name that appears in both kes.csv and kes_latest.xlsx
# ---------------------------------------------------------------------------
CBK_TO_ISO: dict[str, str] = {
    "AE DIRHAM":        "AED",
    "AUSTRALIAN $":     "AUD",
    "CAN $":            "CAD",
    "CHINESE YUAN":     "CNY",
    "DAN KRONER":       "DKK",
    "EURO":             "EUR",
    "HONGKONG DOLLAR":  "HKD",
    "IND RUPEE":        "INR",
    "JPY (100)":        "JPY",
    "NOR KRONER":       "NOK",
    "S FRANC":          "CHF",
    "SA RAND":          "ZAR",
    "SAUDI RIYAL":      "SAR",
    "SINGAPORE $":      "SGD",
    "SINGAPORE DOLLAR": "SGD",
    "STG POUND":        "GBP",
    "SW KRONER":        "SEK",
    "US DOLLAR":        "USD",
}


def read_kes_fx(path: str | Path = DEFAULT_KES_PATH) -> pd.DataFrame:
    """
    Read the CBK KES FX rate file into a tidy DataFrame.

    Parameters
    ----------
    path : str or Path, optional
        Path to the kes.csv file.  Defaults to ``currency/kes.csv``
        next to this module.

    Returns
    -------
    pd.DataFrame
        Columns:
            date      (datetime64[ns])  – publication date
            currency  (object/str)      – currency description
            mid       (float64)         – mid-market rate  (KES / foreign unit)
            bid       (float64)         – bid rate
            ask       (float64)         – ask rate

        Sorted by (date, currency); exact duplicate rows removed.

    Raises
    ------
    FileNotFoundError
        If *path* does not point to an existing file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"KES data file not found: {path}")

    # --- load raw CSV (no header) ----------------------------------------
    df = pd.read_csv(
        path,
        header=None,
        names=["date", "currency", "mid", "bid", "ask"],
    )

    # --- clean currency labels -------------------------------------------
    # Some names arrive wrapped in double-quotes from the source; strip them.
    df["currency"] = df["currency"].str.strip().str.strip('"')

    # --- parse mixed date formats ----------------------------------------
    # Source mixes DD/MM/YYYY (older rows) and YYYY-MM-DD (most rows).
    # It also contains stray embedded header/wide-format rows (e.g. the
    # literal string "Date" in the date column).  errors="coerce" turns
    # those into NaT so they can be filtered out cleanly.
    # pandas ≥ 2.0 supports format="mixed"; older versions fall back to
    # infer_datetime_format which handles both patterns with dayfirst=True.
    try:
        df["date"] = pd.to_datetime(
            df["date"], format="mixed", dayfirst=True, errors="coerce"
        )
    except TypeError:
        # pandas < 2.0
        df["date"] = pd.to_datetime(
            df["date"], infer_datetime_format=True, dayfirst=True, errors="coerce"
        )

    # --- tidy up ---------------------------------------------------------
    # Drop rows whose date could not be parsed (e.g. embedded header lines)
    n_bad_date = df["date"].isna().sum()
    if n_bad_date:
        print(f"[get_kes] dropped {n_bad_date} row(s) with unparseable dates")
    df = df.dropna(subset=["date"])

    # Drop rows whose currency is blank (e.g. empty wide-format rows)
    n_bad_cur = df["currency"].isna().sum()
    if n_bad_cur:
        print(f"[get_kes] dropped {n_bad_cur} row(s) with missing currency")
    df = df.dropna(subset=["currency"])

    # Fix implausible future years (e.g. 2038 typo for 2018):
    # If the year is more than 1 year ahead of today, assume the tens digit
    # was mis-typed and subtract 20 years (covers 20xx -> 20(xx-2)0 typos).
    _max_year = _date.today().year + 1
    future_mask = df["date"].dt.year > _max_year
    if future_mask.any():
        corrected = df.loc[future_mask, "date"] - pd.DateOffset(years=20)
        print(
            f"[get_kes] corrected {future_mask.sum()} row(s) with implausible "
            f"future year (e.g. {df.loc[future_mask, 'date'].dt.year.iloc[0]} "
            f"-> {corrected.dt.year.iloc[0]})"
        )
        df.loc[future_mask, "date"] = corrected

    # Coerce numeric columns — stray header rows may have left them as object
    for col in ("mid", "bid", "ask"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df
        .drop_duplicates()
        .sort_values(["date", "currency"])
        .reset_index(drop=True)
    )

    return df



def read_kes_latest(
    path: str | Path = DEFAULT_KES_LATEST_PATH,
) -> pd.DataFrame:
    """
    Read the Summary sheet of *kes_latest.xlsx* into a tidy wide DataFrame.

    The sheet stores its column names in row 0 (not in the Excel header row),
    so this function promotes that row to the header, parses the Date column,
    and renames every CBK currency description to its ISO 4217 code using
    :data:`CBK_TO_ISO`.

    Parameters
    ----------
    path : str or Path, optional
        Path to *kes_latest.xlsx*.  Defaults to
        ``currency/kes_latest.xlsx`` next to this module.

    Returns
    -------
    pd.DataFrame
        Index  : DatetimeIndex named ``date``
        Columns: AED, AUD, …, USD  (float) – mid-rates (KES per foreign unit)

        Sorted by date; rows with unparseable dates dropped.
        Use ``df.resample('QE').mean()`` / ``df.resample('YE').mean()``
        directly on the result for quarterly / annual aggregation.

    Notes
    -----
    JPY is quoted per 100 units ("JPY (100)") — the column is renamed to
    ``JPY`` but the values are **not** rescaled here.  Divide by 100 if you
    need a per-unit rate.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"kes_latest file not found: {path}")

    # The Summary sheet has a blank first row (row 0) that pandas picks up as
    # the unnamed header.  The real column names ("Date", "AE DIRHAM", …) live
    # in row 1.  Using header=1 skips the blank row and uses row 1 as columns.
    raw = pd.read_excel(path, sheet_name="Summary", header=1)

    # Parse the date column (CBK uses DD/MM/YYYY)
    raw["Date"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce")
    n_bad = raw["Date"].isna().sum()
    if n_bad:
        print(f"[read_kes_latest] dropped {n_bad} row(s) with unparseable dates")
    raw = raw.dropna(subset=["Date"]).rename(columns={"Date": "date"})

    # Rename CBK descriptions → bare ISO 4217 codes
    iso_rename = {cbk: iso for cbk, iso in CBK_TO_ISO.items() if cbk in raw.columns}
    raw = raw.rename(columns=iso_rename)

    # Keep only date + successfully-mapped ISO columns
    # Deduplicate (e.g. SINGAPORE $ and SINGAPORE DOLLAR both → SGD)
    iso_bare = list(dict.fromkeys(iso_rename.values()))
    available = [c for c in iso_bare if c in raw.columns]
    raw = raw[["date"] + available].copy()

    # Rename bare ISO → {ISO}_KES  (e.g. USD → USD_KES).
    # Values are KES per 1 unit of the foreign currency — the CBK convention.
    kes_cols = [f"{iso}_KES" for iso in available]
    raw = raw.rename(columns=dict(zip(available, kes_cols)))

    # Coerce rate columns to float
    for col in kes_cols:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    return raw.sort_values("date").set_index("date")


def read_kes_wide(
    path_csv: str | Path = DEFAULT_KES_PATH,
    path_xlsx: str | Path = DEFAULT_KES_LATEST_PATH,
) -> pd.DataFrame:
    """
    Return a wide DataFrame of **daily KES mid-rates** with ISO 4217 column names.

    Combines two sources:

    1. **kes_latest.xlsx** – recent data already in wide format
       (read via :func:`read_kes_latest`).
    2. **kes.csv** – historical tall-format file (read via :func:`read_kes_fx`,
       then pivoted to wide) filtered to the same currencies that appear in
       *kes_latest*.

    The two frames are concatenated so that *kes_latest* rows take priority
    on any date that appears in both sources.

    Parameters
    ----------
    path_csv : str or Path, optional
        Path to *kes.csv*.  Defaults to ``currency/kes.csv`` next to this module.
    path_xlsx : str or Path, optional
        Path to *kes_latest.xlsx*.  Defaults to ``currency/kes_latest.xlsx``
        next to this module.

    Returns
    -------
    pd.DataFrame
        Index  : DatetimeIndex named ``date``
        Columns: USD_KES, EUR_KES, GBP_KES, …  (float) – KES per foreign unit

        Sorted by date; duplicate dates de-duplicated (xlsx wins).
    """
    # 1. Recent wide data – already has DatetimeIndex from read_kes_latest()
    #    Columns are USD_KES, EUR_KES, … (KES per foreign unit)
    df_latest = read_kes_latest(path_xlsx)
    kes_cols = df_latest.columns.tolist()          # e.g. ["AED_KES", "AUD_KES", …]
    plain_isos = [c.removesuffix("_KES") for c in kes_cols]   # ["AED", "AUD", …]

    # 2. Historical tall data → wide with DatetimeIndex, filtered to same currencies
    df_tall = read_kes_fx(path_csv)
    df_tall["iso"] = df_tall["currency"].map(CBK_TO_ISO)
    df_tall = df_tall[df_tall["iso"].isin(plain_isos)].copy()

    df_hist_wide = (
        df_tall
        .pivot_table(values="mid", index="date", columns="iso", aggfunc="mean")
    )
    df_hist_wide.columns.name = None
    df_hist_wide.index.name = "date"

    # Rename plain ISO → {ISO}_KES to match df_latest column names
    df_hist_wide = df_hist_wide.rename(
        columns={iso: f"{iso}_KES" for iso in df_hist_wide.columns}
    )

    # Ensure every {ISO}_KES column is present (fill missing ones with NaN)
    for col in kes_cols:
        if col not in df_hist_wide.columns:
            df_hist_wide[col] = float("nan")
    df_hist_wide = df_hist_wide[kes_cols]

    # 3. Concatenate – historical first, then latest; xlsx wins on duplicates
    combined = (
        pd.concat([df_hist_wide, df_latest])
        .pipe(lambda df: df[~df.index.duplicated(keep="last")])
        .sort_index()
    )
    combined.index.name = "date"
    return combined


def save_kes_wide(
    df: "pd.DataFrame",
    path: str | Path = DEFAULT_KES_WIDE_PATH,
) -> None:
    """
    Save a wide KES rate DataFrame to *kes_wide.xlsx* (sheet ``"rates"``).

    The DatetimeIndex is written as the first column labelled ``date`` in
    ``YYYY-MM-DD`` format so the file is easy to read and extend manually in
    Excel — just append rows at the bottom and call :func:`load_kes_wide` to
    reload.

    Parameters
    ----------
    df : pd.DataFrame
        Wide DataFrame as returned by :func:`read_kes_wide` or
        :func:`read_kes_latest` (DatetimeIndex + ISO currency columns).
    path : str or Path, optional
        Destination file.  Defaults to ``currency/kes_wide.xlsx``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl", datetime_format="YYYY-MM-DD") as writer:
        df.to_excel(writer, sheet_name="rates", index=True)
    print(f"[save_kes_wide] saved {len(df):,} rows → {path}")


def load_kes_wide(
    path: str | Path = DEFAULT_KES_WIDE_PATH,
) -> "pd.DataFrame":
    """
    Load a wide KES rate DataFrame previously saved by :func:`save_kes_wide`.

    Reads ``currency/kes_wide.xlsx`` (sheet ``"rates"``), uses the first
    column as a :class:`~pandas.DatetimeIndex`, and sorts by date so
    manually-appended rows end up in the right chronological position.

    Parameters
    ----------
    path : str or Path, optional
        Source file.  Defaults to ``currency/kes_wide.xlsx``.

    Returns
    -------
    pd.DataFrame
        Index  : DatetimeIndex named ``date``
        Columns: ISO currency codes (AED, AUD, …)

    Example
    -------
    >>> df = load_kes_wide()
    >>> df.resample("QE").mean()          # quarterly averages
    >>> df.resample("YE").mean()          # annual averages
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"kes_wide file not found: {path}")
    df = pd.read_excel(path, sheet_name="rates", index_col=0, parse_dates=True)
    df.index.name = "date"
    return df.sort_index()


def get_kes_iso(
    currencies=None,
    start: str = "2000-01-01",
    end: str | None = None,
    freq: str = "D",
    agg: str = "last",
    path: str | Path = DEFAULT_KES_WIDE_PATH,
) -> "pd.DataFrame":
    """
    Load *kes_wide.xlsx* and return rates in **KES-as-base** format,
    mirroring the signature of :func:`ecb_fx_eur` in *exchangerates_get.py*.

    The saved file stores CBK rates as ``USD_KES``, ``EUR_KES``, … (KES per
    foreign unit).  This function inverts them and renames columns so the
    output matches ``ecb_fx_eur`` + ``convert_base_currency(fx, base="KES")``
    — i.e. ``KES_USD``, ``KES_EUR``, … where each value is *units of foreign
    per 1 KES*.

    Parameters
    ----------
    currencies : list[str] or None
        ISO codes to keep, e.g. ``["USD", "EUR", "GBP"]``.
        ``None`` (default) returns all currencies in *kes_wide.xlsx*.
    start : str
        Start date, ``"YYYY-MM-DD"``.  Rows before this date are dropped.
    end : str or None
        End date, ``"YYYY-MM-DD"``.  ``None`` means no upper cutoff.
    freq : str
        Output frequency.  ``"D"`` (default) returns daily rows with a
        DatetimeIndex.  Any other pandas offset alias (``"M"``, ``"QE"``,
        ``"YE"``, …) resamples to that frequency and returns a PeriodIndex
        — identical behaviour to :func:`ecb_fx_eur`.
    agg : str
        Aggregation to use when resampling: ``"last"`` (default),
        ``"mean"``, or ``"first"``.
    path : str or Path
        Source xlsx file.  Defaults to ``currency/kes_wide.xlsx``.

    Returns
    -------
    pd.DataFrame
        Daily (``freq="D"``)  → DatetimeIndex
        Other frequencies      → PeriodIndex(freq)
        Columns: KES_USD, KES_EUR, … (float)
                 Values = units of foreign per 1 KES

    Examples
    --------
    Drop-in replacement for the ECB + convert path used for ZAR:

    >>> # ZAR path:
    >>> # fx  = er.ecb_fx_eur(["USD","GBP","EUR"], start="2010-01-01", freq="Q")
    >>> # fx  = er.convert_base_currency(fx, base="zar")
    >>> # KES path:
    >>> kes = get_kes_iso(["USD", "GBP", "EUR"], start="2010-01-01", freq="QE")
    >>> returns = er.get_fx_returns(kes)
    >>> cov     = er.get_fx_covariance(returns)
    """
    df = load_kes_wide(path)               # USD_KES, EUR_KES, … (KES per foreign)

    # --- filter currencies -------------------------------------------------
    if currencies is not None:
        wanted = [f"{c.upper()}_KES" for c in currencies]
        missing = [c for c in wanted if c not in df.columns]
        if missing:
            raise ValueError(
                f"Currencies not found in {path}: {missing}\n"
                f"Available: {df.columns.tolist()}"
            )
        df = df[wanted]

    # --- filter date range -------------------------------------------------
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]

    # --- invert CBK direction: KES-per-foreign → foreign-per-KES -----------
    result = 1.0 / df

    # --- rename USD_KES → KES_USD, EUR_KES → KES_EUR, … -------------------
    result.columns = [
        f"KES_{col.removesuffix('_KES')}" for col in result.columns
    ]

    # --- resample (mirrors ecb_fx_eur logic exactly) -----------------------
    # to_period() does not accept the newer *-end / *-start aliases that
    # resample() accepts (e.g. 'QE', 'YE', 'ME').  Map them back to the
    # Period-compatible forms before calling to_period().
    _RESAMPLE_TO_PERIOD = {
        "QE": "Q",  "QS": "Q",
        "YE": "A",  "YS": "A",
        "ME": "M",  "MS": "M",
    }
    period_freq = _RESAMPLE_TO_PERIOD.get(freq.upper(), freq)

    if freq != "D":
        if agg == "mean":
            result = result.resample(freq).mean()
        elif agg == "last":
            result = result.resample(freq).last()
        elif agg == "first":
            result = result.resample(freq).first()
        else:
            raise ValueError("agg must be 'mean', 'last', or 'first'")
        result.index = result.index.to_period(period_freq)

    result.index.name = "date"
    return result


# ---------------------------------------------------------------------------
# Module self-test  (python get_kes.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = read_kes_fx()

    print("dtypes")
    print("------")
    print(df.dtypes)

    print(f"\n{len(df):,} rows  |  {df['currency'].nunique()} distinct currencies")
    print(f"Date range : {df['date'].min().date()}  ->  {df['date'].max().date()}")

    print("\nAll currencies found:")
    for c in sorted(df["currency"].unique()):
        print(f"  {c}")

    print("\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))

    # --- quick wide pivot (US DOLLAR, EURO, STG POUND) -------------------
    key_currencies = ["US DOLLAR", "EURO", "STG POUND"]
    wide = (
        df[df["currency"].isin(key_currencies)]
        .pivot_table(values="mid", index="date", columns="currency")
    )
    print(f"\nWide pivot (last 5 dates, selected currencies):")
    print(wide.tail().to_string())

    # --- test read_kes_latest -------------------------------------------
    print("\n" + "=" * 60)
    print("read_kes_latest()  →  DatetimeIndex")
    print("=" * 60)
    df_lat = read_kes_latest()
    print(f"{len(df_lat):,} rows  |  {len(df_lat.columns)} ISO currency columns")
    print(f"Date range : {df_lat.index.min().date()}  ->  {df_lat.index.max().date()}")
    print(f"Columns    : {df_lat.columns.tolist()}")
    print("\nLast 5 rows:")
    print(df_lat.tail().to_string())

    # --- test read_kes_wide ---------------------------------------------
    print("\n" + "=" * 60)
    print("read_kes_wide()  (historical + latest combined)  →  DatetimeIndex")
    print("=" * 60)
    df_wide = read_kes_wide()
    print(f"{len(df_wide):,} rows  |  {len(df_wide.columns)} ISO currency columns")
    print(f"Date range : {df_wide.index.min().date()}  ->  {df_wide.index.max().date()}")
    print("\nFirst 5 rows:")
    print(df_wide.head().to_string())
    print("\nLast 5 rows:")
    print(df_wide.tail().to_string())

    # --- demo: quarterly and annual means --------------------------------
    print("\n" + "=" * 60)
    print("Quarterly mean (last 4 quarters, USD_KES & EUR_KES):")
    print("=" * 60)
    print(df_wide[["USD_KES", "EUR_KES"]].resample("QE").mean().tail(4).to_string())

    print("\n" + "=" * 60)
    print("Annual mean (last 5 years, USD_KES & EUR_KES):")
    print("=" * 60)
    print(df_wide[["USD_KES", "EUR_KES"]].resample("YE").mean().tail(5).to_string())

    # --- save + reload round-trip ----------------------------------------
    print("\n" + "=" * 60)
    print("save_kes_wide() / load_kes_wide() round-trip")
    print("=" * 60)
    save_kes_wide(df_wide)
    df_rt = load_kes_wide()
    print(f"Reloaded: {len(df_rt):,} rows, index type: {type(df_rt.index).__name__}")
    print(f"Date range : {df_rt.index.min().date()}  ->  {df_rt.index.max().date()}")
    print("\nLast 5 rows:")
    print(df_rt.tail().to_string())

    # --- get_kes_iso: KES-as-base (for optimization pipeline) ---------------
    print("\n" + "=" * 60)
    print("get_kes_iso()  daily, all currencies")
    print("=" * 60)
    df_iso = get_kes_iso()
    print(f"Columns    : {df_iso.columns.tolist()}")
    print(f"Index type : {type(df_iso.index).__name__}")
    print(f"Date range : {df_iso.index.min()}  ->  {df_iso.index.max()}")
    print("\nLast 5 rows:")
    print(df_iso.tail().to_string())

    print("\n" + "=" * 60)
    print("get_kes_iso(['USD','EUR','GBP'], start='2016-01-01', freq='QE', agg='last')")
    print("=" * 60)
    df_iso_q = get_kes_iso(
        currencies=["USD", "EUR", "GBP"],
        start="2016-01-01",
        freq="QE",
        agg="last",
    )
    print(f"Columns    : {df_iso_q.columns.tolist()}")
    print(f"Index type : {type(df_iso_q.index).__name__}")
    print(df_iso_q.tail(8).to_string())
