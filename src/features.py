"""
features.py - THE CHEF. Turns clean daily data into a weekly modelling table.

Design (direct multi-horizon, leakage-safe by construction)
-----------------------------------------------------------
Every training row answers one question:

    "Standing at forecast ORIGIN week T, what will series s sell in TARGET week T+h?"   (h = 1..13)

Features come from two groups, and that split is what prevents leakage:

  o_*  ORIGIN features   - computed only from weeks <= T (sales history, store traffic,
                           oil price). Built once per (series, week), then shifted by h.
  k_*  KNOWN-FUTURE      - things the business knows in advance for week T+h
                           (calendar, holidays, paydays, planned promotions, closures).

One global LightGBM model is trained on all horizons stacked together, with h as a
feature. The same rows are used for backtesting (validate.py) and the final forecast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

WEEK = pd.offsets.Week(weekday=6)          # weeks end on Sunday (matches W-SUN)
HORIZONS = range(1, C.HORIZON + 1)


# =============================================================================== weekly panel
def _week_end(dates: pd.Series) -> pd.Series:
    """Sunday that closes the Monday-Sunday week containing each date."""
    return dates + pd.to_timedelta((6 - dates.dt.dayofweek) % 7, unit="D")


def _regime_families(regimes: pd.DataFrame) -> list[str]:
    """Families whose low-regime weeks all happen before MODEL_START (see EDA finding 11)."""
    low = regimes.loc[regimes["low_regime"]].groupby(C.SKU_COL, observed=True)["week"].agg(["size", "max"])
    return low.loc[(low["size"] >= 5) & (low["max"] < pd.Timestamp(C.MODEL_START))].index.astype(str).tolist()


def _future_daily_promo(daily: pd.DataFrame, test: pd.DataFrame, last_future_day: pd.Timestamp) -> pd.DataFrame:
    """Daily onpromotion for the forecast period.

    * 16-31 Aug 2017: actual planned promotions from test.csv
    * later days: ASSUMPTION - each series keeps its average daily promo level of
      the last 8 weeks (a 'business as usual' promo plan). Flagged as assumed.
    """
    last_day = daily["date"].max()
    recent = daily.loc[daily["date"] > last_day - pd.Timedelta(weeks=8)]
    avg = recent.groupby(C.SERIES_KEYS, observed=True)["onpromotion"].mean().rename("avg_promo").reset_index()

    days = pd.date_range(last_day + pd.Timedelta(days=1), last_future_day, freq="D")
    grid = pd.MultiIndex.from_product([days, daily[C.MARKET_COL].unique(), daily[C.SKU_COL].cat.categories],
                                      names=["date", C.MARKET_COL, C.SKU_COL]).to_frame(index=False)
    t = test[["date", C.MARKET_COL, C.SKU_COL, "onpromotion"]].copy()
    grid[C.SKU_COL] = grid[C.SKU_COL].astype(str)
    t[C.SKU_COL] = t[C.SKU_COL].astype(str)
    avg[C.SKU_COL] = avg[C.SKU_COL].astype(str)
    grid = grid.merge(t, on=["date", C.MARKET_COL, C.SKU_COL], how="left").merge(avg, on=C.SERIES_KEYS, how="left")
    grid["promo_assumed"] = grid["onpromotion"].isna().astype("int8")
    grid["onpromotion"] = grid["onpromotion"].fillna(grid["avg_promo"]).fillna(0)
    return grid.drop(columns="avg_promo")


def _store_calendar(dates: pd.DatetimeIndex, stores: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    """Daily store x date calendar with LOCATION-MATCHED holiday flags."""
    cal = pd.MultiIndex.from_product([dates, stores[C.MARKET_COL]], names=["date", C.MARKET_COL]) \
        .to_frame(index=False).merge(stores[[C.MARKET_COL, "city", "state"]], on=C.MARKET_COL)
    h = holidays

    def flag(mask, on, name, rename=None):
        x = h.loc[mask, ["date"] + (["locale_name"] if rename else [])].drop_duplicates()
        if rename:
            x = x.rename(columns={"locale_name": rename})
        return x.assign(**{name: 1}), on

    pieces = [
        flag((h.locale == "National") & (h.is_day_off == 1), ["date"], "nat_holiday"),
        flag((h.locale == "Regional") & (h.is_day_off == 1), ["date", "state"], "reg_holiday", "state"),
        flag((h.locale == "Local") & (h.is_day_off == 1), ["date", "city"], "loc_holiday", "city"),
        flag((h.locale == "National") & (h.is_event == 1), ["date"], "nat_event"),
        flag(h.is_workday == 1, ["date"], "bridge_workday"),
    ]
    for x, on in pieces:
        cal = cal.merge(x, on=on, how="left")
    flags = ["nat_holiday", "reg_holiday", "loc_holiday", "nat_event", "bridge_workday"]
    cal[flags] = cal[flags].fillna(0).astype("int8")
    cal["any_holiday"] = cal[["nat_holiday", "reg_holiday", "loc_holiday"]].max(axis=1)
    cal["weekend_holiday"] = (cal["any_holiday"] & (cal["date"].dt.dayofweek >= 5)).astype("int8")

    dom, eom = cal["date"].dt.day, cal["date"].dt.days_in_month
    cal["payday"] = ((dom == 15) | (dom == eom)).astype("int8")             # public-sector paydays
    cal["post_payday"] = (dom.isin([1, 2, 3, 4, 16, 17]) | (dom == eom)).astype("int8")  # spending window
    cal["christmas_closed"] = (((cal["date"].dt.month == 12) & (dom == 25))
                               | ((cal["date"].dt.month == 1) & (dom == 1))).astype("int8")  # chain closed
    eq = (cal["date"] >= C.EARTHQUAKE_START) & (cal["date"] <= C.EARTHQUAKE_END)
    cal["earthquake"] = eq.astype("int8")
    cal["earthquake_local"] = (eq & (cal["state"] == "Manabi")).astype("int8")
    return cal.drop(columns=["city", "state"])


def build_weekly_panel(clean: dict) -> pd.DataFrame:
    """Complete series x week grid from PANEL_START to 13 weeks past the data, with
    weekly sales, a validity flag, and the known-future (k_*) features."""
    daily, stores, holidays = clean["daily"], clean["stores"], clean["holidays"]
    last_day = daily["date"].max()
    last_full_week = last_day - pd.Timedelta(days=(last_day.dayofweek + 1) % 7)
    future_end = last_full_week + pd.Timedelta(weeks=C.HORIZON)
    regime_fams = _regime_families(clean["regimes"])

    # ---------------------------------------------------------------- daily -> weekly sales
    d = daily.loc[daily["date"] >= pd.Timestamp(C.PANEL_START)].copy()
    d["week"] = _week_end(d["date"])
    d = d.loc[d["week"] <= last_full_week]
    # 25-Dec and 1-Jan: >95% of stores close every year -> planned closure (a known-future
    # feature, k_closed_days), not a data problem. Other full-store zero days are unplanned.
    planned = ((d["date"].dt.month == 12) & (d["date"].dt.day == 25)) | \
              ((d["date"].dt.month == 1) & (d["date"].dt.day == 1))
    unplanned_closure = (d["store_closed_day"] == 1) & (d["was_missing"] == 0) & ~planned
    d["bad_day"] = ((d["recording_gap"] == 1) | (d["pre_launch"] == 1) | (d["store_open"] == 0)
                    | unplanned_closure).astype("int8")
    wk = d.groupby(C.SERIES_KEYS + ["week"], observed=True).agg(
        sales=(C.TARGET, "sum"), bad_days=("bad_day", "sum"),
        promo_sum=("onpromotion", "sum"), promo_days=("onpromotion", lambda s: (s > 0).sum()),
    ).reset_index()
    wk["promo_assumed_share"] = 0.0
    wk[C.SKU_COL] = wk[C.SKU_COL].astype(str)

    # ---------------------------------------------------------------- future weeks (sales unknown)
    fp = _future_daily_promo(daily, clean["test"], future_end)
    fp["week"] = _week_end(fp["date"])
    # the partial week (14-15 Aug) is in train; add those days to the first future week
    part = daily.loc[daily["date"] > last_full_week, ["date", C.MARKET_COL, C.SKU_COL, "onpromotion"]].copy()
    part[C.SKU_COL] = part[C.SKU_COL].astype(str)
    part["promo_assumed"] = 0
    part["week"] = _week_end(part["date"])
    fp = pd.concat([part, fp], ignore_index=True)
    fw = fp.groupby(C.SERIES_KEYS + ["week"]).agg(
        promo_sum=("onpromotion", "sum"), promo_days=("onpromotion", lambda s: (s > 0).sum()),
        promo_assumed_share=("promo_assumed", "mean")).reset_index()
    fw["sales"], fw["bad_days"] = np.nan, 0
    panel = pd.concat([wk, fw], ignore_index=True)

    # ---------------------------------------------------------------- validity of the target
    panel["is_future"] = (panel["week"] > last_full_week).astype("int8")
    masked = panel[C.SKU_COL].isin(regime_fams) & (panel["week"] < pd.Timestamp(C.MODEL_START))
    panel["valid"] = ((panel["bad_days"] == 0) & ~masked & (panel["is_future"] == 0)).astype("int8")
    panel["y"] = panel["sales"].where(panel["valid"] == 1)          # NaN = unknown / untrustworthy

    # ---------------------------------------------------------------- calendar, holidays, events
    cal = _store_calendar(pd.date_range(C.PANEL_START, future_end, freq="D"), stores, holidays)
    cal["week"] = _week_end(cal["date"])
    cw = cal.groupby([C.MARKET_COL, "week"]).agg(
        k_nat_holiday_days=("nat_holiday", "sum"), k_reg_holiday_days=("reg_holiday", "sum"),
        k_loc_holiday_days=("loc_holiday", "sum"), k_event_days=("nat_event", "sum"),
        k_bridge_workdays=("bridge_workday", "sum"), k_weekend_holidays=("weekend_holiday", "sum"),
        k_payday_days=("payday", "sum"), k_post_payday_days=("post_payday", "sum"),
        k_closed_days=("christmas_closed", "sum"), k_earthquake_days=("earthquake", "sum"),
        k_earthquake_local_days=("earthquake_local", "sum"),
    ).reset_index()
    panel = panel.merge(cw, on=[C.MARKET_COL, "week"], how="left")

    woy = panel["week"].dt.isocalendar().week.astype(int)
    panel["k_week_of_year"] = woy
    panel["k_month"] = panel["week"].dt.month
    for k in (1, 2, 3):                                             # smooth yearly seasonality
        panel[f"k_fourier_sin{k}"] = np.sin(2 * np.pi * k * woy / 52.18)
        panel[f"k_fourier_cos{k}"] = np.cos(2 * np.pi * k * woy / 52.18)
    xmas = pd.to_datetime(panel["week"].dt.year.astype(str) + "-12-25")
    panel["k_weeks_to_christmas"] = ((xmas - panel["week"]).dt.days // 7).clip(-10, 20)

    # ---------------------------------------------------------------- promotions (planned)
    panel["k_promo_sum"] = panel["promo_sum"]
    panel["k_promo_days"] = panel["promo_days"]
    panel["k_promo_assumed_share"] = panel["promo_assumed_share"]

    # ---------------------------------------------------------------- static attributes
    panel = panel.merge(stores.rename(columns={"type": "store_type", "cluster": "store_cluster"}),
                        on=C.MARKET_COL, how="left")
    for col in [C.SKU_COL, "city", "state", "store_type"]:
        panel[col] = panel[col].astype("category")
    panel["store_cluster"] = panel["store_cluster"].astype("category")
    panel["sid"] = panel.groupby(C.SERIES_KEYS, observed=True).ngroup()
    return panel.drop(columns=["promo_sum", "promo_days", "promo_assumed_share"]) \
        .sort_values(["sid", "week"]).reset_index(drop=True)


# =============================================================================== origin features
def add_origin_features(panel: pd.DataFrame, clean: dict) -> pd.DataFrame:
    """o_* features: statistics of everything known at the END of each week.

    Value at row (series, week T) uses weeks <= T only. make_supervised() later
    shifts them by h so a row for target week T+h only sees origin T.
    """
    p = panel.sort_values(["sid", "week"]).reset_index(drop=True)
    y = p["y"]
    g = y.groupby(p["sid"])
    log_y = np.log1p(y)

    def roll(series, w, fn="mean", minp=None):
        return series.groupby(p["sid"]).transform(
            lambda s: getattr(s.rolling(w, min_periods=minp or max(1, w // 2)), fn)())

    p["o_last"] = log_y
    for w in (4, 8, 13, 26, 52):
        p[f"o_ma{w}"] = np.log1p(roll(y, w))
    p["o_ewm"] = np.log1p(g.transform(lambda s: s.ewm(halflife=4, ignore_na=True).mean()))
    p["o_cv13"] = roll(y, 13, "std") / (roll(y, 13) + 1)
    p["o_zero_share13"] = roll((y == 0).astype(float).where(y.notna()), 13)
    p["o_trend_4_13"] = p["o_ma4"] - p["o_ma13"]                     # recent momentum (log ratio)
    p["o_yoy_growth"] = p["o_ma13"] - p["o_ma13"].groupby(p["sid"]).shift(52)
    p["o_n_valid52"] = roll(y.notna().astype(float), 52, "sum", 1)  # history length (cold start)

    # promotions at origin: how promotional has this series been recently?
    p["o_promo_ma13"] = roll(p["k_promo_sum"].where(p["is_future"] == 0), 13)

    # hierarchy levels: store total and national family total (help sparse / new series)
    store_y = p.groupby([C.MARKET_COL, "week"], observed=True)["y"].transform("sum")
    fam_y = p.groupby([C.SKU_COL, "week"], observed=True)["y"].transform("sum")
    for name, s in (("store", store_y), ("family", fam_y)):
        s = s.where(p["is_future"] == 0)
        p[f"o_{name}_ma4"] = np.log1p(s.groupby(p["sid"]).transform(lambda v: v.rolling(4, min_periods=2).mean()))
        p[f"o_{name}_ma13"] = np.log1p(s.groupby(p["sid"]).transform(lambda v: v.rolling(13, min_periods=4).mean()))
    p["o_share_of_store"] = p["o_ma13"] - p["o_store_ma13"]

    # store traffic (transactions) - only known up to the origin
    tx = clean["transactions"].copy()
    tx["week"] = _week_end(tx["date"])
    txw = tx.groupby([C.MARKET_COL, "week"])["transactions"].sum().rename("tx").reset_index()
    p = p.merge(txw, on=[C.MARKET_COL, "week"], how="left")
    p["o_tx_ma4"] = np.log1p(p.groupby("sid")["tx"].transform(lambda v: v.rolling(4, min_periods=2).mean()))
    p["o_tx_trend"] = p["o_tx_ma4"] - np.log1p(
        p.groupby("sid")["tx"].transform(lambda v: v.rolling(13, min_periods=4).mean()))

    # oil price at the origin (future oil is unknown -> never use target-week oil)
    oil = clean["oil"].copy()
    oil["week"] = _week_end(oil["date"])
    ow = oil.groupby("week")["oil_price"].mean()
    p["o_oil"] = p["week"].map(ow)
    p["o_oil_chg4"] = p["week"].map(ow.pct_change(4))
    p["o_oil_chg13"] = p["week"].map(ow.pct_change(13))

    return p.drop(columns="tx").sort_values(["sid", "week"]).reset_index(drop=True)


# =============================================================================== supervised rows
def make_supervised(p: pd.DataFrame, horizons=HORIZONS, origin: pd.Timestamp | None = None,
                    target_from: pd.Timestamp | None = None, target_to: pd.Timestamp | None = None) -> pd.DataFrame:
    """Stack (origin, horizon) pairs into one training/prediction table.

    origin=None  -> every week can be an origin (training: all horizons stacked)
    origin=T     -> only rows whose origin is exactly T (one backtest fold or the final forecast)
    """
    p = p.sort_values(["sid", "week"]).reset_index(drop=True)
    o_cols = [c for c in p.columns if c.startswith("o_")]
    base = [c for c in p.columns if not c.startswith("o_")]
    out = []
    for h in horizons:
        shifted = p.groupby("sid")[o_cols].shift(h)              # origin = target week - h
        df = pd.concat([p[base], shifted], axis=1)
        df["h"] = h
        df["origin_week"] = df["week"] - pd.Timedelta(weeks=h)
        # same week last year, aligned to the TARGET week (52 >= h, so it is known at the origin)
        df["o_lag52_target"] = np.log1p(p.groupby("sid")["y"].shift(52))
        df["o_lag52_target_smooth"] = np.log1p(p.groupby("sid")["y"].transform(
            lambda s: s.shift(51).rolling(3, min_periods=1).mean()))
        df["k_promo_vs_usual"] = np.log1p(df["k_promo_sum"]) - np.log1p(df["o_promo_ma13"].fillna(0))
        if origin is not None:
            df = df.loc[df["origin_week"] == origin]
        if target_from is not None:
            df = df.loc[df["week"] >= target_from]
        if target_to is not None:
            df = df.loc[df["week"] <= target_to]
        f64 = df.select_dtypes("float64").columns
        df[f64] = df[f64].astype("float32")                   # halves memory (~0.6 GB for all rows)
        out.append(df)
    return pd.concat(out, ignore_index=True)


# =============================================================================== feature groups
def feature_groups(columns) -> dict[str, list[str]]:
    """Named groups used by the model and the ablation study."""
    cols = list(columns)
    pick = lambda *keys: [c for c in cols if any(k in c for k in keys)]
    return {
        "history": [c for c in cols if c.startswith("o_") and not any(
            k in c for k in ("promo", "oil", "tx_", "store_ma", "family_ma", "share_of_store"))],
        "hierarchy": pick("o_store_ma", "o_family_ma", "o_share_of_store"),
        "promo": pick("promo"),
        "holiday": pick("holiday", "event_days", "bridge_workdays", "closed_days", "earthquake"),
        "calendar": pick("k_week_of_year", "k_month", "fourier", "weeks_to_christmas", "payday"),
        "oil": pick("o_oil"),
        "transactions": pick("o_tx_"),
        "store_meta": [c for c in (C.MARKET_COL, C.SKU_COL, "city", "state", "store_type", "store_cluster")
                       if c in cols],
        "horizon": ["h"],
    }


def model_features(columns) -> list[str]:
    groups = feature_groups(columns)
    seen, out = set(), []
    for cols in groups.values():
        for c in cols:
            if c not in seen and c != "k_promo_assumed_share":
                seen.add(c)
                out.append(c)
    return out


CATEGORICAL = [C.MARKET_COL, C.SKU_COL, "city", "state", "store_type", "store_cluster"]


# =============================================================================== checks + entry
def leakage_check(p: pd.DataFrame, sup: pd.DataFrame, n: int = 300, seed: int = C.SEED) -> int:
    """Recompute o_ma4 from raw weekly sales using ONLY weeks <= origin and compare.

    Returns the number of mismatches (must be 0)."""
    rng = np.random.default_rng(seed)
    rows = sup.dropna(subset=["o_ma4"]).sample(n, random_state=rng.integers(1e9))
    by_sid = {sid: g.set_index("week")["y"] for sid, g in p.groupby("sid")}
    bad = 0
    for _, r in rows.iterrows():
        hist = by_sid[r["sid"]].loc[:r["origin_week"]].tail(4)
        expect = np.log1p(hist.mean()) if hist.notna().sum() >= 2 else np.nan
        if not (np.isclose(expect, r["o_ma4"], equal_nan=True)):
            bad += 1
    return bad


def build_features(clean: dict, use_cache: bool = True) -> pd.DataFrame:
    """Weekly panel + origin features, cached as parquet."""
    path = C.INTERIM_DIR / "weekly_features.parquet"
    if use_cache and path.exists():
        return pd.read_parquet(path)
    panel = build_weekly_panel(clean)
    p = add_origin_features(panel, clean)
    p.to_parquet(path, index=False)
    return p


def describe_features(p: pd.DataFrame) -> pd.DataFrame:
    """Feature dictionary written to outputs/feature_dictionary.csv."""
    sup_cols = list(p.columns) + ["h", "o_lag52_target", "o_lag52_target_smooth", "k_promo_vs_usual"]
    rows = []
    for grp, cols in feature_groups(sup_cols).items():
        for c in cols:
            when = "known at origin (<= T)" if c.startswith("o_") else (
                "known in advance for target week" if c.startswith("k_") else "static / horizon")
            rows.append((grp, c, when))
    return pd.DataFrame(rows, columns=["group", "feature", "availability"])
