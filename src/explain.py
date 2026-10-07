"""
explain.py - THE STRATEGIST (part 1): why does the model forecast what it does?

Driver attribution
    LightGBM's native TreeSHAP (pred_contrib=True) splits every prediction into
    additive contributions per feature. The model predicts a RATIO to the series'
    recent level, so contribution x (level + 1) = contribution in UNITS, and the
    contributions + base sum exactly to the forecast. Features are rolled up into
    business groups: history/trend, promotions, holidays, calendar, oil, traffic...

Ablation study
    Retrain the model without one feature group at a time on the same backtest
    folds. If accuracy gets worse without a group, that group carries real signal.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from src import features as F
from src import models as M
from src import validate as V
from src.eda import SERIES, INK2, AXIS
from src.forecast import future_rows

GROUP_LABELS = {
    "history": "Sales history & trend", "hierarchy": "Store / family totals", "promo": "Promotions",
    "holiday": "Holidays & events", "calendar": "Calendar & paydays", "oil": "Oil price",
    "transactions": "Store traffic", "store_meta": "Store & product identity", "horizon": "Weeks ahead",
}


FEATURE_LABELS = {
    "o_last": "last week's sales", "o_ma4": "4-week avg sales", "o_ma8": "8-week avg sales",
    "o_ma13": "13-week avg sales", "o_ma26": "26-week avg sales", "o_ma52": "52-week avg sales",
    "o_ewm": "recent weighted avg", "o_cv13": "sales volatility", "o_zero_share13": "share of zero weeks",
    "o_trend_4_13": "recent trend", "o_yoy_growth": "year-on-year growth", "o_n_valid52": "history length",
    "o_lag52_target": "same week last year", "o_lag52_target_smooth": "same period last year",
    "o_store_ma4": "store total (4 wk)", "o_store_ma13": "store total (13 wk)",
    "o_family_ma4": "national family total (4 wk)", "o_family_ma13": "national family total (13 wk)",
    "o_share_of_store": "share of store sales", "k_promo_sum": "planned promotions",
    "k_promo_days": "promotion days", "o_promo_ma13": "usual promo level", "k_promo_vs_usual": "promo vs usual",
    "k_nat_holiday_days": "national holiday days", "k_reg_holiday_days": "regional holiday days",
    "k_loc_holiday_days": "local holiday days", "k_event_days": "national events",
    "k_bridge_workdays": "bridge workdays", "k_weekend_holidays": "holidays on weekend",
    "k_closed_days": "store closed days", "k_earthquake_days": "earthquake period",
    "k_earthquake_local_days": "earthquake (Manabi)", "k_payday_days": "payday days",
    "k_post_payday_days": "post-payday days", "k_week_of_year": "week of year", "k_month": "month",
    "k_weeks_to_christmas": "weeks to Christmas", "o_oil": "oil price", "o_oil_chg4": "oil change 4 wk",
    "o_oil_chg13": "oil change 13 wk", "o_tx_ma4": "store transactions", "o_tx_trend": "transactions trend",
    "store_nbr": "store", "family": "product family", "city": "city", "state": "state",
    "store_type": "store type", "store_cluster": "store cluster", "h": "weeks ahead",
}


def nice(f: str) -> str:
    return FEATURE_LABELS.get(f, f.replace("k_", "").replace("o_", "").replace("_", " "))


def feature_to_group(features: list[str]) -> dict[str, str]:
    out = {}
    for g, cols in F.feature_groups(features).items():
        for c in cols:
            out.setdefault(c, g)
    return out


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(C.EXPLAIN_DIR / name, dpi=130)
    plt.close(fig)
    return name


# =============================================================================== driver attribution
def unit_contributions(model: M.GlobalLGBM, df: pd.DataFrame) -> pd.DataFrame:
    """Per-row contributions in units (rows with a zero business-rule forecast are dropped)."""
    keep = ~M.dead_at_origin(df)
    df = df.loc[keep]
    scale = (M._level(df) + 1).to_numpy()[:, None]
    con = model.contributions(df) * scale
    return con


def run_drivers(sup: pd.DataFrame) -> pd.DataFrame:
    model = M.GlobalLGBM.load(C.MODEL_DIR)
    fut = future_rows(sup)
    con = unit_contributions(model, fut)
    meta = fut.loc[con.index, ["week", C.MARKET_COL, C.SKU_COL, "h"]]
    f2g = feature_to_group(model.features)

    grp = con.drop(columns="_base").T.groupby(lambda c: f2g[c]).sum().T
    grp["_base"] = con["_base"]
    grp = pd.concat([meta, grp], axis=1)

    # --- 1. global importance (mean |units| per row)
    imp_feat = con.drop(columns="_base").abs().mean().sort_values(ascending=False)
    imp_grp = grp[sorted(set(f2g.values()))].abs().mean().sort_values(ascending=False)
    imp_feat.rename("mean_abs_units").to_csv(C.EXPLAIN_DIR / "importance_features.csv")
    imp_grp.rename("mean_abs_units").to_csv(C.EXPLAIN_DIR / "importance_groups.csv")

    # --- 2. chain-level drivers per forecast week (signed units)
    groups = [g for g in imp_grp.index]
    weekly = grp.groupby("week")[groups + ["_base"]].sum()
    weekly["forecast"] = weekly.sum(axis=1)
    weekly.to_csv(C.EXPLAIN_DIR / "drivers_by_week.csv")
    by_family = grp.groupby(grp[C.SKU_COL].astype(str))[groups].sum()
    by_family.to_csv(C.EXPLAIN_DIR / "drivers_by_family.csv")

    # --- 3. selected forecasts: biggest series, most promo-driven, most holiday-driven
    picks = {
        "Largest forecast": grp.assign(tot=grp[groups].sum(axis=1) + grp["_base"])["tot"].idxmax(),
        "Most promotion-driven": grp["promo"].idxmax(),
        "Most holiday-driven": grp["holiday"].idxmax() if "holiday" in grp else grp.index[0],
    }
    charts = [_chart_group_importance(imp_grp), _chart_weekly(weekly, groups)]
    sel_rows, sel_md = [], []
    for title, idx in picks.items():
        row = con.loc[idx].drop("_base")
        top = row.reindex(row.abs().sort_values(ascending=False).index[:8])
        m = meta.loc[idx]
        name = f"{title}: store {m[C.MARKET_COL]}, {m[C.SKU_COL]}, week ending {m['week']:%Y-%m-%d}"
        total = con.loc[idx].sum()
        sel_rows.append(pd.DataFrame({"case": title, "feature": top.index, "units": top.values}))
        sel_md.append(f"**{name}** - forecast {total:,.0f} units = starting point {con.loc[idx, '_base']:,.0f} "
                      + "".join(f"{v:+,.0f} ({nice(k)}) " for k, v in top.head(5).items()))
        charts.append(_chart_case(top, con.loc[idx, "_base"], total, name, f"ex_{len(charts)}.png"))
    pd.concat(sel_rows).to_csv(C.EXPLAIN_DIR / "selected_forecasts.csv", index=False)

    tot = weekly[groups].sum()
    fc_total = weekly["forecast"].sum()
    lines = "\n".join(f"| {GROUP_LABELS.get(g, g)} | {tot[g]:+,.0f} | {tot[g] / fc_total:+.1%} | {imp_grp[g]:,.1f} |"
                      for g in groups)
    md = f"""# Driver attribution (TreeSHAP)

_Auto-generated by `python main.py --step explain`. Contributions are in units and add up exactly
to each forecast: forecast = starting point (average model output x recent level) + sum of contributions._

## What drives the next 13 weeks? (chain total)

| Driver group | Net units | Share of forecast | Avg size per series-week (abs) |
|---|---:|---:|---:|
{lines}

Total forecast {fc_total:,.0f} units (excluding zero-rule series). "Net" can be small even for an
important group when ups and downs cancel; the last column shows how much each group moves
individual forecasts.

## Selected forecasts explained

""" + "\n\n".join(sel_md) + "\n\n" + "\n".join(f"![{c}]({c})" for c in charts) + """

_Contributions describe what the model learned from history (association). They support business
reasoning but are not proof of causal effect._
"""
    (C.EXPLAIN_DIR / "drivers_report.md").write_text(md, encoding="utf-8")
    print(f"  - drivers_report.md written to {C.EXPLAIN_DIR}")
    return weekly


def _chart_group_importance(imp: pd.Series) -> str:
    s = imp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh([GROUP_LABELS.get(g, g) for g in s.index], s.values,
            color=[SERIES[0] if v >= s.max() * 0.25 else "#b7d3f6" for v in s.values], height=0.65)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Average |contribution| per series-week (units)")
    ax.set_title("Which drivers move the forecasts most")
    return _save(fig, "drv_01_group_importance.png")


def _chart_weekly(weekly: pd.DataFrame, groups: list[str]) -> str:
    show = [g for g in ("promo", "holiday", "calendar", "oil", "transactions") if g in groups]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for i, g in enumerate(show):
        ax.plot(weekly.index, weekly[g], color=SERIES[i], marker="o", ms=4, label=GROUP_LABELS[g])
    ax.axhline(0, color=AXIS, lw=1)
    ax.yaxis.set_major_formatter(lambda x, _: f"{x/1e3:+.0f}k")
    ax.legend(ncol=3, fontsize=8)
    ax.set_title("Chain-level contribution of business drivers, per forecast week (units)")
    return _save(fig, "drv_02_weekly_drivers.png")


def _chart_case(top: pd.Series, base: float, total: float, title: str, name: str) -> str:
    s = top.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh([nice(f) for f in s.index], s.values, color=[SERIES[0] if v > 0 else SERIES[1] for v in s.values],
            height=0.6)
    ax.axvline(0, color=AXIS, lw=1)
    for i, v in enumerate(s.values):
        ax.text(v, i, f" {v:+,.0f} ", va="center", ha="left" if v > 0 else "right", fontsize=8, color=INK2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Units vs starting point {base:,.0f} -> forecast {total:,.0f}")
    ax.set_title(title, fontsize=10)
    return _save(fig, name)


# =============================================================================== ablation
ABLATE = ["promo", "holiday", "calendar", "oil", "transactions", "hierarchy", "store_meta"]


def run_ablation(sup: pd.DataFrame) -> pd.DataFrame:
    all_feats = F.model_features(sup.columns)
    groups = F.feature_groups(sup.columns)
    origins = V.fold_origins(sup)
    variants = {"full model": all_feats}
    for g in ABLATE:
        drop = set(groups[g])
        variants[f"without {g}"] = [f for f in all_feats if f not in drop]

    rows = []
    for k in C.ABLATION_FOLDS:
        train, test = V.split(sup, origins[k - 1])
        for name, feats in variants.items():
            pred = M.GlobalLGBM(features=feats, bags=1).fit(train).predict(test)
            rows.append({"variant": name, "fold": k, "WMAPE": V.wmape(test["y"], pred),
                         "Bias": V.bias(test["y"], pred), "n_features": len(feats)})
            print(f"    fold {k} | {name:<22} WMAPE {rows[-1]['WMAPE']:.4f}")
    res = pd.DataFrame(rows)
    tab = res.pivot_table(index="variant", columns="fold", values="WMAPE")
    tab.columns = [f"WMAPE fold {c}" for c in tab.columns]
    tab["WMAPE avg"] = tab.mean(axis=1)
    full = tab.loc["full model", "WMAPE avg"]
    tab["change vs full (pts)"] = (tab["WMAPE avg"] - full) * 100
    tab["relative change"] = tab["WMAPE avg"] / full - 1
    tab["verdict"] = np.where(tab.index == "full model", "-",
                              np.where(tab["change vs full (pts)"] > 0.05, "helps",
                                       np.where(tab["change vs full (pts)"] < -0.05, "hurts / noise", "neutral")))
    tab = tab.sort_values("change vs full (pts)", ascending=False)
    tab.to_csv(C.EXPLAIN_DIR / "ablation.csv")

    t = tab.drop("full model")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    vals = t["change vs full (pts)"].iloc[::-1]
    ax.barh([v.replace("without ", "") for v in vals.index], vals.values,
            color=[SERIES[0] if v > 0 else SERIES[1] for v in vals.values], height=0.6)
    ax.axvline(0, color=AXIS, lw=1)
    for i, v in enumerate(vals.values):
        ax.text(v, i, f" {v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8, color=INK2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("WMAPE increase when the group is removed (percentage points; > 0 = group helps)")
    ax.set_title("Ablation: value of each feature group")
    chart = _save(fig, "abl_01_ablation.png")

    md = f"""# Ablation / feature-impact study

_Auto-generated by `python main.py --step ablation`. Same model, retrained without one feature group
at a time, on backtest folds {', '.join(map(str, C.ABLATION_FOLDS))} (1 = latest 13 weeks, 3 = Christmas period).
1 bag per variant so all variants are compared like for like._

{tab.to_markdown(floatfmt='.4f')}

![ablation]({chart})

Notes: price, distribution and weather are not in this dataset, so they cannot be tested.
Promotions are tested with the promo plan known in advance, as it would be in practice.
"""
    (C.EXPLAIN_DIR / "ablation_report.md").write_text(md, encoding="utf-8")
    print(f"  - ablation_report.md written to {C.EXPLAIN_DIR}")
    return tab
