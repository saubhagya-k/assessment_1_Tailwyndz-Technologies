"""
data.py - THE CLEANER.

Loads the raw CSVs and fixes the data problems found during profiling:
  * text dates -> real datetimes
  * missing calendar days (25-Dec, stores closed) -> explicit rows with 0 sales + flag
  * stores that opened late -> rows before opening flagged (not real zero demand)
  * oil prices missing on weekends/holidays -> time interpolation
  * transferred holidays -> the original date is NOT a day off, the Transfer row is
  * dead / discontinued / new series -> labelled so later steps can apply rules
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


# --------------------------------------------------------------------------- loading
def load_raw() -> dict[str, pd.DataFrame]:
    """Read every raw CSV with parsed dates."""
    dfs = {}
    for name, fname in C.RAW_FILES.items():
        path = C.RAW_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Put the Kaggle CSVs in data/raw/.")
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        dfs[name] = df
    return dfs


# --------------------------------------------------------------------------- oil
def clean_oil(oil: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Daily oil price on a complete calendar.

    Oil is only quoted on trading days (and 43 trading days are blank), so we
    reindex to every calendar day and interpolate over time. Edges are
    forward/back-filled. A flag records which values were imputed.
    """
    oil = oil.rename(columns={"dcoilwtico": "oil_price"}).set_index("date").sort_index()
    start = start or oil.index.min()
    end = end or oil.index.max()
    full = pd.date_range(min(start, oil.index.min()), max(end, oil.index.max()), freq="D")
    oil = oil.reindex(full)
    oil["oil_imputed"] = oil["oil_price"].isna().astype("int8")
    oil["oil_price"] = oil["oil_price"].interpolate(method="time").ffill().bfill()
    return oil.rename_axis("date").reset_index()


# --------------------------------------------------------------------------- holidays
DAY_OFF_TYPES = {"Holiday", "Transfer", "Bridge", "Additional"}


def clean_holidays(hol: pd.DataFrame) -> pd.DataFrame:
    """Keep only days that were actually celebrated.

    * transferred == True  -> the holiday moved to another date, so drop it here
      (the matching 'Transfer' row carries the real day off).
    * 'Work Day' rows are compensating working Saturdays -> not a day off.
    * 'Event' rows (Mother's Day, Black Friday, World Cup, earthquake) are kept
      separately because they change shopping without being a public holiday.
    """
    h = hol.loc[~hol["transferred"].astype(bool)].copy()
    h["is_day_off"] = h["type"].isin(DAY_OFF_TYPES).astype("int8")
    h["is_event"] = (h["type"] == "Event").astype("int8")
    h["is_workday"] = (h["type"] == "Work Day").astype("int8")
    h["description"] = h["description"].astype(str)
    return h[["date", "type", "locale", "locale_name", "description",
              "is_day_off", "is_event", "is_workday"]].reset_index(drop=True)


# --------------------------------------------------------------------------- sales
def clean_sales(train: pd.DataFrame) -> pd.DataFrame:
    """Return a complete daily grid date x store x family with quality flags."""
    train = train.drop(columns=["id"], errors="ignore")
    dates = pd.date_range(train["date"].min(), train["date"].max(), freq="D")
    stores = np.sort(train[C.MARKET_COL].unique())
    families = np.sort(train[C.SKU_COL].unique())

    grid = pd.MultiIndex.from_product([dates, stores, families],
                                      names=["date", C.MARKET_COL, C.SKU_COL]).to_frame(index=False)
    df = grid.merge(train, on=["date", C.MARKET_COL, C.SKU_COL], how="left")

    # rows that did not exist in the raw file (e.g. 25-Dec every year)
    df["was_missing"] = df[C.TARGET].isna().astype("int8")
    df[C.TARGET] = df[C.TARGET].fillna(0.0)
    df["onpromotion"] = df["onpromotion"].fillna(0).astype("int32")

    # ---- store opening: first day the store sold anything
    store_day = df.groupby(["date", C.MARKET_COL])[C.TARGET].sum().rename("store_total").reset_index()
    open_date = store_day.loc[store_day["store_total"] > 0].groupby(C.MARKET_COL)["date"].min()
    df["store_open_date"] = df[C.MARKET_COL].map(open_date)
    df["store_open"] = (df["date"] >= df["store_open_date"]).astype("int8")

    # ---- temporary closures after opening (whole store sold 0 that day)
    df = df.merge(store_day, on=["date", C.MARKET_COL], how="left")
    df["store_closed_day"] = ((df["store_total"] == 0) & (df["store_open"] == 1)).astype("int8")
    df = df.drop(columns="store_total")

    # ---- promotions were not recorded before April 2014
    df["promo_reliable"] = (df["date"] >= pd.Timestamp(C.PROMO_RELIABLE_FROM)).astype("int8")

    df[C.SKU_COL] = df[C.SKU_COL].astype("category")
    df = df.sort_values([C.MARKET_COL, C.SKU_COL, "date"]).reset_index(drop=True)
    df = flag_recording_gaps(df)
    return df


def flag_recording_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Separate 'no demand' from 'not recorded'.

    Profiling showed families such as PRODUCE and BEVERAGES drop to ~0 across
    almost every store for months, then jump back to normal. That is a recording
    gap, not real demand. Two flags:

      pre_launch     - before the series' first ever sale (family not yet stocked)
      recording_gap  - a run of >= GAP_MIN_DAYS consecutive zero days inside the
                       series' active life, for a series whose typical selling day
                       is >= GAP_MIN_LEVEL units (so a 2-week zero is implausible)

    `usable` = rows whose sales value can be trusted as real demand.
    """
    key = [C.MARKET_COL, C.SKU_COL]
    pos = df[C.TARGET] > 0
    first = df["date"].where(pos).groupby([df[k] for k in key], observed=True).transform("min")
    last = df["date"].where(pos).groupby([df[k] for k in key], observed=True).transform("max")
    df["pre_launch"] = (first.isna() | (df["date"] < first)).astype("int8")

    level = df[C.TARGET].where(pos).groupby([df[k] for k in key], observed=True).transform("median")
    zero = (~pos).astype("int8")
    sid = df.groupby(key, observed=True).ngroup()
    run_id = ((zero != zero.shift()) | (sid != sid.shift())).cumsum()
    run_len = zero.groupby(run_id).transform("size")
    inside = (df["date"] > first) & (df["date"] < last)
    df["recording_gap"] = ((zero == 1) & (run_len >= C.GAP_MIN_DAYS) & inside
                           & (level >= C.GAP_MIN_LEVEL) & (df["store_closed_day"] == 0)).astype("int8")

    df["usable"] = ((df["store_open"] == 1) & (df["store_closed_day"] == 0) & (df["was_missing"] == 0)
                    & (df["pre_launch"] == 0) & (df["recording_gap"] == 0)).astype("int8")
    return df


def detect_recording_regimes(daily: pd.DataFrame, min_share: float = 0.01) -> pd.DataFrame:
    """Find weeks where a major family's chain-wide total collapses to a 'low' regime.

    PRODUCE and BEVERAGES switch back and forth between two recording levels until
    mid-2015 (PRODUCE: ~30 integer units/day vs ~10,000 decimal units/day per store,
    i.e. a unit-of-measure change). A week is 'low regime' when the family total is
    below 60% of the 90th percentile of the surrounding 13 weeks. Only families with
    >= min_share of total sales are checked; small families are naturally volatile.
    """
    w = (daily.groupby([C.SKU_COL, pd.Grouper(key="date", freq=C.WEEK_FREQ)], observed=True)[C.TARGET]
         .sum().unstack(0).iloc[1:-1])                      # drop partial first/last weeks
    big = w.sum() / w.sum().sum() >= min_share
    w = w.loc[:, big]
    ref = w.rolling(13, center=True, min_periods=5).quantile(0.9)
    low = (w < 0.6 * ref) & (ref > 0)
    out = low.stack().rename("low_regime").reset_index().rename(columns={"date": "week"})
    return out.merge(w.stack().rename("family_units").reset_index().rename(columns={"date": "week"}),
                     on=["week", C.SKU_COL])


def series_status(daily: pd.DataFrame) -> pd.DataFrame:
    """Label every store x family series.

    status:
      never_sold    - zero sales over the whole history  -> forecast 0
      discontinued  - sold before but zero in the last DEAD_LOOKBACK_WEEKS -> forecast 0
      new           - store opened / family first sold less than NEW_SERIES_WEEKS ago -> cold start
      active        - everything else
    """
    end = daily["date"].max()
    recent_cut = end - pd.Timedelta(weeks=C.DEAD_LOOKBACK_WEEKS)
    new_cut = end - pd.Timedelta(weeks=C.NEW_SERIES_WEEKS)

    g = daily.groupby(C.SERIES_KEYS, observed=True)
    s = pd.DataFrame({
        "total_sales": g[C.TARGET].sum(),
        "first_sale": daily.loc[daily[C.TARGET] > 0].groupby(C.SERIES_KEYS, observed=True)["date"].min(),
        "last_sale": daily.loc[daily[C.TARGET] > 0].groupby(C.SERIES_KEYS, observed=True)["date"].max(),
        "store_open_date": g["store_open_date"].first(),
        "zero_share": g[C.TARGET].apply(lambda x: (x == 0).mean()),
    }).reset_index()

    s["status"] = "active"
    s.loc[(s["store_open_date"] > new_cut) | (s["first_sale"] > new_cut), "status"] = "new"
    s.loc[(s["total_sales"] > 0) & (s["last_sale"] < recent_cut), "status"] = "discontinued"
    s.loc[s["total_sales"] == 0, "status"] = "never_sold"
    return s


# --------------------------------------------------------------------------- pipeline entry
def build_clean_data(use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """Run all cleaning steps; cache results as parquet for fast re-runs."""
    paths = {k: C.INTERIM_DIR / f"{k}.parquet" for k in ("daily", "oil", "holidays", "status", "regimes")}
    if use_cache and all(p.exists() for p in paths.values()):
        out = {k: pd.read_parquet(p) for k, p in paths.items()}
    else:
        raw = load_raw()
        daily = clean_sales(raw["train"])
        out = {
            "daily": daily,
            "oil": clean_oil(raw["oil"], start=daily["date"].min(),
                             end=raw["test"]["date"].max()),
            "holidays": clean_holidays(raw["holidays"]),
            "status": series_status(daily),
            "regimes": detect_recording_regimes(daily),
        }
        for k, p in paths.items():
            out[k].to_parquet(p, index=False)

    raw_small = load_raw_small()
    out.update(raw_small)
    return out


def load_raw_small() -> dict[str, pd.DataFrame]:
    """Small tables that need no heavy cleaning."""
    stores = pd.read_csv(C.RAW_DIR / C.RAW_FILES["stores"])
    tx = pd.read_csv(C.RAW_DIR / C.RAW_FILES["transactions"], parse_dates=["date"])
    test = pd.read_csv(C.RAW_DIR / C.RAW_FILES["test"], parse_dates=["date"])
    return {"stores": stores, "transactions": tx, "test": test}
