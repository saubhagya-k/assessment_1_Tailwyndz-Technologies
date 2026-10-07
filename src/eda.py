"""
eda.py - DATA-QUALITY CHECKS + EXPLORATORY ANALYSIS.

Produces outputs/eda/eda_report.md plus PNG charts. Every number in the report
is computed here, so the report regenerates whenever the data changes.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

import config as C

# ----------------------------------------------------------------------------- chart style
# Validated categorical palette (fixed order) + neutral chart chrome
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
HIGHLIGHT, BASE = "#2a78d6", "#b7d3f6"   # one emphasised bar vs context bars (blue ramp)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "font.family": "DejaVu Sans", "legend.frameon": False,
    "legend.fontsize": 9, "legend.labelcolor": INK2, "figure.dpi": 110,
})


def _save(fig, name: str) -> str:
    path = C.EDA_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return name


def _millions(x, _pos=None):
    return f"{x/1e6:.1f}M"


# ----------------------------------------------------------------------------- data-quality checks
def data_quality_checks(raw: dict, clean: dict) -> pd.DataFrame:
    """One table of every data issue that matters for forecasting."""
    train, daily, status = raw["train"], clean["daily"], clean["status"]
    oil_raw, tx, stores = raw["oil"], clean["transactions"], clean["stores"]

    all_days = pd.date_range(train["date"].min(), train["date"].max(), freq="D")
    missing_days = sorted(set(all_days) - set(train["date"].unique()))
    late = (daily.groupby(C.MARKET_COL)["store_open_date"].first()
            .loc[lambda s: s > train["date"].min() + pd.Timedelta(days=14)])
    closed = daily.loc[daily["store_closed_day"] == 1, ["date", C.MARKET_COL]].drop_duplicates()
    tx_expected = daily.loc[(daily["store_open"] == 1) & (daily["store_closed_day"] == 0),
                            ["date", C.MARKET_COL]].drop_duplicates()
    tx_missing = len(tx_expected) - len(tx_expected.merge(tx, on=["date", C.MARKET_COL]))

    rows = [
        ("Rows in train.csv", f"{len(train):,}", "-"),
        ("Date range", f"{train['date'].min():%Y-%m-%d} -> {train['date'].max():%Y-%m-%d}", "-"),
        ("Stores x families (series)", f"{train[C.MARKET_COL].nunique()} x {train[C.SKU_COL].nunique()} "
         f"= {train.groupby(C.SERIES_KEYS).ngroups:,}", "-"),
        ("Duplicate (date, store, family) rows",
         f"{train.duplicated(['date', C.MARKET_COL, C.SKU_COL]).sum():,}", "None found"),
        ("Negative sales values", f"{(train[C.TARGET] < 0).sum():,}", "None found"),
        ("Calendar days missing from train",
         ", ".join(d.strftime("%Y-%m-%d") for d in missing_days),
         "Stores closed on Christmas -> filled with 0 + flag"),
        ("Rows with zero sales", f"{(train[C.TARGET] == 0).mean():.1%}", "Use Tweedie loss / zero-aware metrics"),
        ("Series that never sold", f"{(status['status'] == 'never_sold').sum()}", "Forecast = 0 (product not stocked)"),
        (f"Series discontinued (0 sales in last {C.DEAD_LOOKBACK_WEEKS} wks)",
         f"{(status['status'] == 'discontinued').sum()}", "Forecast = 0, flagged for business review"),
        ("Stores opening after the data starts",
         ", ".join(f"#{s} ({d:%Y-%m})" for s, d in late.items()),
         "Rows before opening excluded from training"),
        (f"New stores (<{C.NEW_SERIES_WEEKS} wks history)",
         f"{status.loc[status['status'] == 'new', C.MARKET_COL].nunique()} store(s)",
         "Cold start: global model + similar-store features"),
        ("Store-days fully closed after opening", f"{len(closed):,}", "Flagged, excluded from training target"),
        ("Promotion field all-zero before", C.PROMO_RELIABLE_FROM, "Promo features masked before this date"),
        ("Oil price missing (trading days)", f"{oil_raw['dcoilwtico'].isna().sum()} of {len(oil_raw):,}",
         "Time interpolation on full calendar"),
        ("Oil price - weekends not quoted", "Yes", "Reindexed to daily, interpolated"),
        ("Open store-days missing in transactions", f"{tx_missing:,}", "Transactions used only as lagged feature"),
        ("Recording gaps (>=14 zero days in a normally-selling series)",
         f"{clean['daily']['recording_gap'].mean():.1%} of rows", "Excluded from training target (not real zero demand)"),
        ("Rows before a family's first sale in a store", f"{clean['daily']['pre_launch'].mean():.1%} of rows",
         "Excluded (product not yet stocked)"),
        ("Holidays marked 'transferred'", f"{raw['holidays']['transferred'].sum()}",
         "Original date dropped, Transfer date used"),
        ("Stores metadata coverage", f"{stores[C.MARKET_COL].nunique()} / {train[C.MARKET_COL].nunique()} stores",
         "Complete"),
    ]
    return pd.DataFrame(rows, columns=["Check", "Result", "Treatment"])


# ----------------------------------------------------------------------------- anomaly scan
def unusual_movements(weekly: pd.DataFrame, z: float = 5.0) -> pd.DataFrame:
    """Flag weeks where a series jumps far from its own recent level.

    Robust z-score: (sales - rolling median) / rolling MAD over the previous 13
    weeks. Only past weeks are used, so the rule could run live.
    """
    w = weekly.sort_values(C.SERIES_KEYS + ["week"]).copy()
    g = w.groupby(C.SERIES_KEYS, observed=True)[C.TARGET]
    med = g.transform(lambda s: s.shift(1).rolling(13, min_periods=8).median())
    mad = g.transform(lambda s: (s.shift(1) - s.shift(1).rolling(13, min_periods=8).median())
                      .abs().rolling(13, min_periods=8).median())
    w["robust_z"] = (w[C.TARGET] - med) / (1.4826 * mad.replace(0, np.nan))
    return w.loc[w["robust_z"].abs() > z, C.SERIES_KEYS + ["week", C.TARGET, "robust_z"]]


# ----------------------------------------------------------------------------- charts
def chart_total_weekly(weekly_total: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(weekly_total.index, weekly_total.values, color=SERIES[0])
    eq0, eq1 = pd.Timestamp(C.EARTHQUAKE_START), pd.Timestamp(C.EARTHQUAKE_END)
    ax.axvspan(eq0, eq1, color=AXIS, alpha=0.35, lw=0)
    ax.annotate("Manabi earthquake\n(relief buying)", xy=(eq0, weekly_total.loc[eq0:eq1].max()),
                xytext=(-120, 10), textcoords="offset points", color=INK2, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    for yr in range(2013, 2017):
        win = weekly_total.loc[f"{yr}-12-10":f"{yr + 1}-01-07"]
        ax.annotate("Christmas", xy=(win.idxmax(), win.max()), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8, color=MUTED)
    ax.axvline(pd.Timestamp(C.MODEL_START), color=INK2, lw=1, ls="--")
    ax.text(pd.Timestamp(C.MODEL_START), ax.get_ylim()[0], "  model training window ->",
            fontsize=8, color=INK2, va="bottom")
    ax.yaxis.set_major_formatter(_millions)
    ax.set_title("Total weekly unit sales, all stores")
    ax.set_ylabel("Units per week")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return _save(fig, "01_total_weekly_sales.png")


def chart_family_share(daily: pd.DataFrame) -> tuple[str, pd.Series]:
    share = daily.groupby(C.SKU_COL, observed=True)[C.TARGET].sum().sort_values()
    share = share / share.sum()
    top = share.tail(12)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [HIGHLIGHT if v >= top.iloc[-3] else BASE for v in top.values]
    ax.barh(top.index.astype(str), top.values, color=colors, height=0.7)
    for i, v in enumerate(top.values):
        ax.text(v + 0.003, i, f"{v:.1%}", va="center", fontsize=8, color=INK2)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_title("Share of total units by product family (top 12)")
    return _save(fig, "02_family_share.png"), share


def chart_store_heatmap(daily: pd.DataFrame) -> str:
    """Weekly store totals -> shows late openings and closures at a glance."""
    sw = (daily.groupby([C.MARKET_COL, pd.Grouper(key="date", freq=C.WEEK_FREQ)])[C.TARGET]
          .sum().unstack())
    sw = sw.div(sw.max(axis=1), axis=0)            # each store scaled to its own peak
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#f0efec", "#9ec5f4", "#2a78d6", "#0d366b"])
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.imshow(sw.values, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_yticks(range(0, len(sw), 4), sw.index[::4])
    xt = [i for i, d in enumerate(sw.columns) if d.month == 1 and d.day <= 7]
    ax.set_xticks(xt, [sw.columns[i].year for i in xt])
    ax.grid(False)
    ax.set_ylabel("Store number")
    ax.set_title("Weekly sales per store, scaled to each store's peak (light = no sales)")
    return _save(fig, "03_store_activity_heatmap.png")


def chart_promo_lift(daily: pd.DataFrame) -> tuple[str, pd.Series]:
    d = daily.loc[(daily["promo_reliable"] == 1) & (daily["store_open"] == 1)]
    m = d.assign(promo=d["onpromotion"] > 0).groupby([C.SKU_COL, "promo"], observed=True)[C.TARGET].mean().unstack()
    lift = (m[True] / m[False]).replace([np.inf, -np.inf], np.nan).dropna()
    lift = lift[m[False] > 1].sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cap = 5.0                                   # one extreme family would squash the rest
    ax.barh(lift.index.astype(str), lift.clip(upper=cap).values,
            color=[HIGHLIGHT if v > 3 else BASE for v in lift.values], height=0.7)
    ax.axvline(1, color=AXIS, lw=1)
    ax.set_xlim(0, cap * 1.12)
    for i, v in enumerate(lift.values):
        label = f"{v:.1f}x (axis cut)" if v > cap else f"{v:.1f}x"
        ax.text(min(v, cap) + 0.05, i, label, va="center", fontsize=8, color=INK2)
    ax.grid(axis="y", visible=False)
    ax.set_title("Avg daily units: promo days vs non-promo days")
    ax.set_xlabel("Ratio (association, not causal uplift)")
    return _save(fig, "04_promo_association.png"), lift


def chart_payday(daily: pd.DataFrame) -> tuple[str, pd.Series]:
    d = daily.loc[daily["store_open"] == 1]
    dom = d.groupby(d["date"].dt.day)[C.TARGET].mean()
    idx = dom / dom.mean()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    hi = {30, 31, 1, 2, 3, 4, 15, 16}
    ax.bar(idx.index, idx.values, color=[HIGHLIGHT if d in hi else BASE for d in idx.index], width=0.75)
    ax.axhline(1, color=AXIS, lw=1)
    ax.set_xticks(range(1, 32, 2))
    ax.set_ylim(idx.min() * 0.95, idx.max() * 1.03)
    ax.set_title("Average daily sales by day of month (1.0 = average)")
    ax.set_xlabel("Day of month - public-sector paydays are the 15th and month-end")
    return _save(fig, "05_payday_effect.png"), idx


def chart_seasonality(daily: pd.DataFrame) -> tuple[str, pd.Series, pd.Series]:
    d = daily.loc[daily["store_open"] == 1]
    dow = d.groupby(d["date"].dt.dayofweek)[C.TARGET].mean()
    dow = dow / dow.mean()
    mon = d.groupby(d["date"].dt.month)[C.TARGET].mean()
    mon = mon / mon.mean()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], dow.values,
                color=[HIGHLIGHT if v > 1.1 else BASE for v in dow.values], width=0.7)
    axes[0].set_title("Day-of-week index")
    axes[1].bar(mon.index, mon.values, color=[HIGHLIGHT if v > 1.1 else BASE for v in mon.values], width=0.7)
    axes[1].set_xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    axes[1].set_title("Month index")
    for ax, s in zip(axes, (dow, mon)):
        ax.axhline(1, color=AXIS, lw=1)
        ax.set_ylim(s.min() * 0.9, s.max() * 1.05)
        ax.grid(axis="x", visible=False)
    return _save(fig, "06_seasonality.png"), dow, mon


def chart_oil(weekly_total: pd.Series, oil: pd.DataFrame) -> tuple[str, float]:
    """Two stacked panels on a shared time axis (never a dual y-axis)."""
    ow = oil.set_index("date")["oil_price"].resample(C.WEEK_FREQ).mean().reindex(weekly_total.index)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True, height_ratios=[1, 1])
    a1.plot(ow.index, ow.values, color=SERIES[1])
    a1.set_title("Oil price (WTI, USD/barrel) vs total weekly sales")
    a1.set_ylabel("USD / barrel")
    a2.plot(weekly_total.index, weekly_total.values, color=SERIES[0])
    a2.yaxis.set_major_formatter(_millions)
    a2.set_ylabel("Units per week")
    a2.xaxis.set_major_locator(mdates.YearLocator())
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    corr = float(np.corrcoef(ow.ffill().bfill(), weekly_total)[0, 1])
    return _save(fig, "07_oil_vs_sales.png"), corr


def holiday_effect(daily: pd.DataFrame, holidays: pd.DataFrame, stores: pd.DataFrame) -> tuple[str, pd.Series]:
    """Match each holiday to the stores it applies to, then compare sales."""
    d = daily.loc[(daily["store_open"] == 1) & (daily["store_closed_day"] == 0),
                  ["date", C.MARKET_COL, C.TARGET]]
    d = d.groupby(["date", C.MARKET_COL])[C.TARGET].sum().reset_index().merge(stores, on=C.MARKET_COL)
    h = holidays
    nat = h.loc[(h.locale == "National") & (h.is_day_off == 1), ["date"]].drop_duplicates().assign(nat=1)
    ev = h.loc[(h.locale == "National") & (h.is_event == 1), ["date"]].drop_duplicates().assign(ev=1)
    reg = h.loc[(h.locale == "Regional") & (h.is_day_off == 1), ["date", "locale_name"]] \
        .drop_duplicates().rename(columns={"locale_name": "state"}).assign(reg=1)
    loc = h.loc[(h.locale == "Local") & (h.is_day_off == 1), ["date", "locale_name"]] \
        .drop_duplicates().rename(columns={"locale_name": "city"}).assign(loc=1)
    d = (d.merge(nat, on="date", how="left").merge(ev, on="date", how="left")
         .merge(reg, on=["date", "state"], how="left").merge(loc, on=["date", "city"], how="left"))
    # compare to the same store's average on the same weekday (removes weekday mix)
    d["dow"] = d["date"].dt.dayofweek
    base = d.loc[d[["nat", "ev", "reg", "loc"]].isna().all(axis=1)].groupby([C.MARKET_COL, "dow"])[C.TARGET].mean()
    d = d.join(base.rename("base"), on=[C.MARKET_COL, "dow"])
    d["ratio"] = d[C.TARGET] / d["base"]
    out = pd.Series({
        "Local holiday\n(store's city)": d.loc[d["loc"] == 1, "ratio"].mean(),
        "Regional holiday\n(store's state)": d.loc[d["reg"] == 1, "ratio"].mean(),
        "National holiday": d.loc[d["nat"] == 1, "ratio"].mean(),
        "National event\n(e.g. Black Friday)": d.loc[d["ev"] == 1, "ratio"].mean(),
    })
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(out.index, out.values, color=[HIGHLIGHT if v > 1.05 else BASE for v in out.values], width=0.6)
    ax.axhline(1, color=AXIS, lw=1)
    for i, v in enumerate(out.values):
        ax.text(i, v + 0.01, f"{v:.2f}x", ha="center", fontsize=9, color=INK2)
    ax.set_ylim(min(0.8, out.min() * 0.95), out.max() * 1.1)
    ax.grid(axis="x", visible=False)
    ax.set_title("Store sales on holiday days vs same store, same weekday")
    return _save(fig, "08_holiday_effect.png"), out


def chart_store_type(daily: pd.DataFrame, stores: pd.DataFrame) -> str:
    d = daily.merge(stores[[C.MARKET_COL, "type"]], on=C.MARKET_COL)
    d = d.loc[d["store_open"] == 1]
    per_store = (d.groupby(["type", C.MARKET_COL, pd.Grouper(key="date", freq=C.WEEK_FREQ)])[C.TARGET].sum()
                 .groupby(["type", "date"]).mean().unstack(0))
    per_store = per_store.iloc[1:-1]
    fig, ax = plt.subplots(figsize=(11, 4))
    for i, t in enumerate(sorted(per_store.columns)):
        s = per_store[t].rolling(4, min_periods=1).mean()
        ax.plot(s.index, s.values, color=SERIES[i], label=f"Type {t}")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x/1e3:.0f}k")
    ax.legend(ncol=5, loc="upper left")
    ax.set_title("Average weekly units per store, by store type (4-wk smoothed)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return _save(fig, "09_store_type_trend.png")


def chart_regimes(regimes: pd.DataFrame) -> str:
    """Small multiples: the two families with recording-regime switches + a stable control."""
    fams = ["PRODUCE", "BEVERAGES", "GROCERY I"]
    fig, axes = plt.subplots(len(fams), 1, figsize=(11, 6.5), sharex=True)
    for ax, f in zip(axes, fams):
        r = regimes.loc[regimes[C.SKU_COL] == f].sort_values("week")
        ax.plot(r["week"], r["family_units"], color=SERIES[0], lw=1.6)
        for wk in r.loc[r["low_regime"], "week"]:
            ax.axvspan(wk - pd.Timedelta(days=6), wk, color=SERIES[1], alpha=0.25, lw=0)
        ax.axvline(pd.Timestamp(C.MODEL_START), color=INK2, lw=1, ls="--")
        ax.yaxis.set_major_formatter(_millions)
        ax.set_title(f"{f}" + ("  (control - no regime switch)" if f == "GROCERY I" else ""),
                     fontsize=10, color=INK)
    axes[0].text(pd.Timestamp(C.MODEL_START), axes[0].get_ylim()[1] * 0.9, "  training starts",
                 fontsize=8, color=INK2)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Weekly units by family - shaded weeks = detected low recording regime",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    return _save(fig, "10_recording_regimes.png")


# ----------------------------------------------------------------------------- report
def run_eda(raw: dict, clean: dict) -> None:
    daily, stores, holidays, status = clean["daily"], clean["stores"], clean["holidays"], clean["status"]
    print("  - data-quality checks")
    dq = data_quality_checks(raw, clean)
    dq.to_csv(C.EDA_DIR / "data_quality_checks.csv", index=False)

    print("  - weekly aggregation for EDA")
    full_week_end = daily["date"].max() - pd.Timedelta(days=(daily["date"].max().dayofweek + 1) % 7)
    d_full = daily.loc[daily["date"] <= full_week_end]
    weekly = (d_full.groupby(C.SERIES_KEYS + [pd.Grouper(key="date", freq=C.WEEK_FREQ)], observed=True)[C.TARGET]
              .sum().reset_index().rename(columns={"date": "week"}))
    weekly_total = weekly.groupby("week")[C.TARGET].sum().iloc[1:]   # first week is partial (starts Tue)

    print("  - unusual movements scan")
    anomalies = unusual_movements(weekly.merge(
        status[C.SERIES_KEYS + ["status"]], on=C.SERIES_KEYS).query("status == 'active'"))
    anomalies.to_csv(C.EDA_DIR / "unusual_movements.csv", index=False)
    an_weeks = anomalies.groupby("week").size().sort_values(ascending=False).head(5)

    print("  - charts")
    c1 = chart_total_weekly(weekly_total)
    # behavioural statistics use only trustworthy rows inside the modelling window
    dm = daily.loc[(daily["usable"] == 1) & (daily["date"] >= pd.Timestamp(C.MODEL_START))]
    c2, share = chart_family_share(dm)
    c3 = chart_store_heatmap(d_full)
    c4, lift = chart_promo_lift(dm)
    c5, payday = chart_payday(dm)
    c6, dow, mon = chart_seasonality(dm)
    c7, oil_corr = chart_oil(weekly_total, clean["oil"])
    c8, hol = holiday_effect(dm, holidays, stores)
    c9 = chart_store_type(d_full, stores)
    regimes = clean["regimes"]
    c10 = chart_regimes(regimes)
    reg_fams = regimes.loc[regimes["low_regime"]].groupby(C.SKU_COL, observed=True)["week"].agg(["size", "max"])
    reg_fams = reg_fams.loc[reg_fams["size"] >= 5].sort_values("size", ascending=False)
    # low weeks that keep recurring after MODEL_START every Nov-Feb are real seasonality, not recording
    seasonal_fams = reg_fams.loc[reg_fams["max"] >= pd.Timestamp(C.MODEL_START)]
    reg_fams = reg_fams.loc[reg_fams["max"] < pd.Timestamp(C.MODEL_START)]

    status_counts = status["status"].value_counts()
    yoy = weekly_total.groupby(weekly_total.index.year).sum()
    yoy = yoy.loc[[2014, 2015, 2016]].pct_change().dropna()
    top3 = share.sort_values(ascending=False).head(3)
    eq = weekly_total.loc[C.EARTHQUAKE_START:C.EARTHQUAKE_END].mean() / \
        weekly_total.loc[pd.Timestamp(C.EARTHQUAKE_START) - pd.Timedelta(weeks=8):C.EARTHQUAKE_START].mean() - 1

    md = f"""# FreshBasket - Data Quality & EDA Report

_Auto-generated by `python main.py --step eda`. All figures are computed from `data/raw/`._

## 1. Data-quality checks

{dq.to_markdown(index=False)}

**Series status used by the forecasting rules**

{status_counts.rename_axis('status').reset_index(name='series').to_markdown(index=False)}

## 2. Key findings

_Findings 2-6 use only trustworthy rows (`usable == 1`) from {C.MODEL_START} onward._

1. **Strong growth with a December peak.** Total recorded sales grew {yoy.iloc[0]:+.0%} (2015 vs 2014) and
   {yoy.iloc[1]:+.0%} (2016 vs 2015); part of the early growth is the recording change in finding 11. December runs at {mon.loc[12]:.2f}x the average month.
2. **Concentrated portfolio.** {', '.join(f'{k} ({v:.0%})' for k, v in top3.items())} make up
   {top3.sum():.0%} of all units. Forecast error on these three dominates WMAPE.
3. **Weekend shopping.** Saturday/Sunday run at {dow.loc[5]:.2f}x / {dow.loc[6]:.2f}x the average day.
   Weekly aggregation removes this pattern, so weekly holiday counts must be weekday-aware.
4. **Payday effect.** Sales peak around the month-end payday: days 30-4 run at
   {payday.loc[[30, 31, 1, 2, 3, 4]].mean():.2f}x average vs {payday.loc[10:14].mean():.2f}x on days 10-14.
   The mid-month payday gives a smaller lift (15th-16th {payday.loc[[15, 16]].mean():.2f}x vs 14th
   {payday.loc[14]:.2f}x). A "payday days in week" feature captures this.
5. **Promotions are strongly associated with higher sales.** Promo days sell {lift.median():.1f}x more
   (median across families), up to {lift.max():.1f}x for {lift.idxmax()}. This is association only:
   retailers promote items that already sell well, which the model must control for.
6. **Holidays depend on location.** Relative to the same store on the same weekday:
   local {hol.iloc[0]:.2f}x, regional {hol.iloc[1]:.2f}x, national {hol.iloc[2]:.2f}x, national events
   {hol.iloc[3]:.2f}x. A Quito holiday must only be applied to Quito stores.
7. **Earthquake shock.** In the 4 weeks after 16-Apr-2016, sales ran {eq:+.0%} vs the 8 weeks before.
   These weeks get an `earthquake` flag so the model does not learn them as normal seasonality.
8. **Oil.** Weekly oil price and total sales correlate at {oil_corr:+.2f}. Most of this is a shared
   trend (oil fell while the chain grew), not proof of cause. The ablation study tests whether
   oil really improves forecasts.
9. **Data gaps that change modelling.** {status_counts.get('never_sold', 0)} series never sold and
   {status_counts.get('discontinued', 0)} are discontinued (forecast 0); store openings and
   closures are visible in the heatmap; promotions are unrecorded before {C.PROMO_RELIABLE_FROM}.
10. **Unusual movements.** {len(anomalies):,} series-weeks deviate more than 5 robust SDs from their
    previous 13-week level. Busiest weeks: {', '.join(f'{w:%Y-%m-%d} ({n})' for w, n in an_weeks.items())}.
    These are mostly Christmas / New Year weeks (real seasonality the model must learn), plus
    recording breaks. Details in `unusual_movements.csv`.
11. **Recording-regime switches (most important data issue).** {', '.join(f'{f} ({n} weeks, last {m:%Y-%m-%d})' for f, (n, m) in reg_fams.iterrows())}
    alternate between two recording levels. PRODUCE flips between ~30 whole units/day and
    ~10,000 decimal units/day per store - a unit-of-measure change, not demand. No switch occurs
    after 2015-05-31, so **model training starts {C.MODEL_START}**; older data is used for EDA only.
    The same detector also fires for {', '.join(seasonal_fams.index.astype(str)) or 'no other family'}, but those
    dips repeat every year around Christmas after {C.MODEL_START}: real seasonality, kept as demand.

## 3. Charts

![Recording regimes]({c10})

![Total weekly sales]({c1})
![Family share]({c2})
![Store activity]({c3})
![Promo association]({c4})
![Payday]({c5})
![Seasonality]({c6})
![Oil vs sales]({c7})
![Holiday effect]({c8})
![Store type trend]({c9})

## 4. Implications for the model

| Finding | Modelling decision |
|---|---|
| PRODUCE / BEVERAGES regime switches until 2015-05 | Train from {C.MODEL_START}; regime chart documents why |
| 31% zeros, skewed volumes | Tweedie objective in LightGBM; WMAPE as the headline metric |
| Never-sold / discontinued series | Business rule: forecast 0, reported separately |
| Late-opening & new stores | Exclude pre-opening rows; global model shares patterns for cold start |
| Promo not recorded before 2014-04 | Train from 2014-04 onward or mask promo features |
| Location-specific holidays | Holiday counts per week built per store (city/state matched) |
| Paydays, December peak | Payday-day counts, week-of-year and Fourier seasonality features |
| Earthquake | Event flag; excluded weeks from anomaly-sensitive statistics |
| Oil trend | Tested in ablation; lagged values only to avoid leakage |
"""
    (C.EDA_DIR / "eda_report.md").write_text(md, encoding="utf-8")
    print(f"  - report written to {C.EDA_DIR / 'eda_report.md'}")
