"""
forecast.py - FINAL 13-WEEK FORECAST + PREDICTION INTERVALS.

1. Train the bagged global LightGBM on ALL history (target weeks <= last full week).
2. Forecast every series for the next 13 weeks from that origin.
3. Prediction intervals (P10 / P90 = 80% range) by split-conformal calibration:
   the backtest errors of the same model, grouped by horizon and volume level,
   tell us how far actuals typically land from the forecast. Coverage is checked
   honestly with leave-one-fold-out (calibrate on 3 folds, test on the 4th).
4. Business rules: series dead at the origin (0 sales for 8 weeks) -> 0.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from src import models as M
from src.eda import SERIES, INK2, AXIS, MUTED

N_LEVEL_BINS = 5


# =============================================================================== intervals
def _level_bin(pred: pd.Series, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, len(edges) - 2)


def fit_intervals(cv: pd.DataFrame, q=C.INTERVAL_Q) -> dict:
    """Quantiles of log((y+1)/(pred+1)) per (horizon, volume bin) from backtest residuals."""
    d = cv.loc[cv["lgbm"] > 0].copy()
    edges = np.unique(np.quantile(d["lgbm"], np.linspace(0, 1, N_LEVEL_BINS + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    d["bin"] = _level_bin(d["lgbm"], edges)
    d["lr"] = np.log1p(d["y"]) - np.log1p(d["lgbm"])
    table = d.groupby(["h", "bin"])["lr"].quantile(list(q)).unstack()
    table.columns = ["lo", "hi"]
    return {"edges": edges, "table": table}


def apply_intervals(df: pd.DataFrame, pred: np.ndarray, cal: dict) -> tuple[np.ndarray, np.ndarray]:
    bins = _level_bin(pd.Series(pred), cal["edges"])
    key = pd.MultiIndex.from_arrays([df["h"].to_numpy(), bins])
    t = cal["table"].reindex(key)
    lo = np.clip(np.expm1(np.log1p(pred) + t["lo"].to_numpy()), 0, None)
    hi = np.clip(np.expm1(np.log1p(pred) + t["hi"].to_numpy()), 0, None)
    lo[pred == 0], hi[pred == 0] = 0, 0
    return lo, hi


def interval_coverage(cv: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-fold-out: calibrate on 3 folds, measure coverage on the held-out one."""
    rows = []
    for k in sorted(cv["fold"].unique()):
        cal = fit_intervals(cv.loc[cv["fold"] != k])
        test = cv.loc[(cv["fold"] == k) & (cv["lgbm"] > 0)]
        lo, hi = apply_intervals(test, test["lgbm"].to_numpy(), cal)
        inside = (test["y"] >= lo) & (test["y"] <= hi)
        rows.append({"held-out fold": k, "coverage": inside.mean(),
                     "volume-weighted coverage": (inside * test["y"]).sum() / test["y"].sum(),
                     "avg width / forecast": ((hi - lo).sum() / test["lgbm"].sum())})
    out = pd.DataFrame(rows)
    out.loc[len(out)] = ["all"] + out.iloc[:, 1:].mean().tolist()
    return out


# =============================================================================== final forecast
def train_final(sup: pd.DataFrame) -> M.GlobalLGBM:
    train = sup.loc[sup["y"].notna()]
    model = M.GlobalLGBM().fit(train)
    model.save(C.MODEL_DIR)
    return model


def future_rows(sup: pd.DataFrame) -> pd.DataFrame:
    origin = sup.loc[sup["y"].notna(), "week"].max()
    return sup.loc[(sup["is_future"] == 1) & (sup["origin_week"] == origin)].copy()


def make_forecast(sup: pd.DataFrame, model: M.GlobalLGBM, cv: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    fut = future_rows(sup)
    pred = model.predict(fut)
    lo, hi = apply_intervals(fut, pred, fit_intervals(cv))
    st = status[[C.MARKET_COL, C.SKU_COL, "status"]].copy()
    st[C.SKU_COL] = st[C.SKU_COL].astype(str)
    out = pd.DataFrame({
        "date": fut["week"].to_numpy(),                         # week ending (Sunday)
        "week_start": (fut["week"] - pd.Timedelta(days=6)).to_numpy(),
        "market_id": fut[C.MARKET_COL].to_numpy(),
        "sku_id": fut[C.SKU_COL].astype(str).to_numpy(),
        "horizon_weeks": fut["h"].to_numpy(),
        "forecast": pred.round(1),
        "forecast_p10": lo.round(1),
        "forecast_p90": hi.round(1),
        "store_type": fut["store_type"].astype(str).to_numpy(),
        "promo_assumed_share": fut["k_promo_assumed_share"].round(2).to_numpy(),
    }).merge(st.rename(columns={C.MARKET_COL: "market_id", C.SKU_COL: "sku_id", "status": "series_status"}),
             on=["market_id", "sku_id"], how="left")
    out["rule_applied"] = np.where(pred == 0, "zero: no sales in last 8 weeks", "")
    return out.sort_values(["market_id", "sku_id", "date"]).reset_index(drop=True)


# =============================================================================== report
def _save(fig, name):
    fig.tight_layout()
    fig.savefig(C.FORECAST_DIR / name, dpi=130)
    plt.close(fig)
    return name


def _chart_total(fc: pd.DataFrame, sup: pd.DataFrame) -> str:
    hist = sup.loc[(sup["h"] == 1) & sup["y"].notna()].groupby("week")["y"].sum()
    hist = hist.loc[hist.index > hist.index.max() - pd.Timedelta(weeks=60)]
    t = fc.groupby("date")[["forecast", "forecast_p10", "forecast_p90"]].sum()
    ly = hist.reindex(t.index - pd.Timedelta(weeks=52))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(hist.index, hist.values, color=INK2, lw=1.6, label="Actual")
    ax.fill_between(t.index, t["forecast_p10"], t["forecast_p90"], color=SERIES[0], alpha=0.15, lw=0,
                    label="P10-P90 (sum of series ranges)")
    ax.plot(t.index, t["forecast"], color=SERIES[0], lw=2.2, label="Forecast (P50)")
    ax.plot(t.index, ly.values, color=MUTED, lw=1, ls=":", label="Same weeks last year")
    ax.axvline(hist.index.max(), color=AXIS, ls="--", lw=1)
    ax.yaxis.set_major_formatter(lambda x, _: f"{x/1e6:.1f}M")
    ax.legend(ncol=4, loc="upper left", fontsize=8)
    ax.set_title("Chain total weekly units: history and 13-week forecast")
    return _save(fig, "fc_01_total.png")


def _chart_examples(fc: pd.DataFrame, sup: pd.DataFrame) -> str:
    vol = fc.groupby(["market_id", "sku_id"])["forecast"].sum().sort_values(ascending=False)
    picks = [vol.index[0], vol.index[len(vol) // 10], vol.index[len(vol) // 3]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (s, f) in zip(axes, picks):
        h = sup.loc[(sup["h"] == 1) & (sup[C.MARKET_COL] == s) & (sup[C.SKU_COL].astype(str) == f)
                    & sup["y"].notna()].set_index("week")["y"].tail(39)
        g = fc.loc[(fc["market_id"] == s) & (fc["sku_id"] == f)].set_index("date")
        ax.plot(h.index, h.values, color=INK2, lw=1.4)
        ax.fill_between(g.index, g["forecast_p10"], g["forecast_p90"], color=SERIES[0], alpha=0.18, lw=0)
        ax.plot(g.index, g["forecast"], color=SERIES[0], lw=2)
        ax.set_title(f"Store {s} - {f}", fontsize=10)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    fig.suptitle("Example series: actual (grey), forecast (blue) and 80% range", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    return _save(fig, "fc_02_examples.png")


def run_forecast(sup: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    cv_path = C.CV_DIR / "cv_predictions.parquet"
    if not cv_path.exists():
        raise FileNotFoundError("Run `python main.py --step models` first (intervals use backtest errors).")
    cv = pd.read_parquet(cv_path)

    print("  - training final model on all history")
    model = train_final(sup)
    print("  - forecasting 13 weeks + calibrated intervals")
    fc = make_forecast(sup, model, cv, status)
    fc.to_csv(C.FORECAST_DIR / "forecast.csv", index=False)

    cov = interval_coverage(cv)
    cov.to_csv(C.FORECAST_DIR / "interval_coverage.csv", index=False)

    # business summary: next 13 weeks vs the same 13 weeks last year, by family
    hist = sup.loc[(sup["h"] == 1) & sup["y"].notna()]
    ly_weeks = pd.to_datetime(fc["date"].unique()) - pd.Timedelta(weeks=52)
    ly = hist.loc[hist["week"].isin(ly_weeks)].groupby(hist[C.SKU_COL].astype(str))["y"].sum()
    fam = fc.groupby("sku_id")[["forecast", "forecast_p10", "forecast_p90"]].sum()
    fam["same_13w_last_year"] = ly.reindex(fam.index)
    fam["vs_last_year"] = fam["forecast"] / fam["same_13w_last_year"] - 1
    fam = fam.sort_values("forecast", ascending=False)
    fam.to_csv(C.FORECAST_DIR / "forecast_by_family.csv")
    tot = fam[["forecast", "same_13w_last_year"]].sum()

    fam_md = fam.copy()
    for c in ["forecast", "forecast_p10", "forecast_p90", "same_13w_last_year"]:
        fam_md[c] = fam_md[c].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "")
    fam_md["vs_last_year"] = fam["vs_last_year"].map(lambda v: f"{v:+.1%}" if np.isfinite(v) else "new")
    charts = [_chart_total(fc, sup), _chart_examples(fc, sup)]
    n_zero = (fc.groupby(["market_id", "sku_id"])["forecast"].sum() == 0).sum()
    md = f"""# 13-week forecast

_Auto-generated by `python main.py --step forecast`._

* Forecast weeks: {fc['date'].min():%Y-%m-%d} -> {fc['date'].max():%Y-%m-%d} (week ending Sunday), origin = last full
  week of data. {fc[['market_id', 'sku_id']].drop_duplicates().shape[0]:,} series x 13 weeks = {len(fc):,} rows in `forecast.csv`.
* Total forecast: **{tot['forecast']/1e6:.1f}M units**, {tot['forecast'] / tot['same_13w_last_year'] - 1:+.1%} vs the same 13 weeks last year.
* {n_zero} series are forecast at 0 (no sales in the last 8 weeks before the origin).
* Promotions: actual plan from test.csv for 16-31 Aug; later weeks assume each series keeps its
  last-8-week promo level (`promo_assumed_share` = share of days assumed).

## Prediction intervals - are they honest?

P10/P90 come from the model's own backtest errors (split-conformal, by horizon x volume level).
Leave-one-fold-out check - target coverage is 80%:

{cov.to_markdown(index=False, floatfmt='.3f')}

## Forecast by family (13-week total)

{fam_md.to_markdown()}

![total]({charts[0]})
![examples]({charts[1]})
"""
    (C.FORECAST_DIR / "forecast_report.md").write_text(md, encoding="utf-8")
    print(f"  - forecast.csv and forecast_report.md written to {C.FORECAST_DIR}")
    print(cov.round(3).to_string(index=False))
    return fc
