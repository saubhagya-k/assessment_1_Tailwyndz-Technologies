"""
models.py - THE BRAIN. Baselines + global LightGBM + the backtest report.

Forecasters (all use only information available at the forecast origin):
  naive_ma4       "same as the last 4 weeks"   - what a planner does by hand
  snaive          "same week last year"        - classic seasonal naive
  snaive_growth   same week last year x recent year-on-year growth
  lgbm            ONE global LightGBM across all 1,782 series and 13 horizons

Why no per-series ETS/ARIMA: only ~2 years of clean history exist after the
recording-regime fix (EDA finding 11), which is too short for a 52-week seasonal
ETS per series. The seasonal-naive family is the honest statistical benchmark.
"""
from __future__ import annotations

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from src import features as F
from src import validate as V
from src.eda import SERIES, INK2, AXIS  # shared chart style (also applies rcParams)

BASELINES = ["naive_ma4", "snaive", "snaive_growth"]
MODELS = BASELINES + ["lgbm"]
LABELS = {"naive_ma4": "Last 4 weeks avg", "snaive": "Same week last year",
          "snaive_growth": "Last year x growth", "lgbm": "Global LightGBM (bagged)"}


# =============================================================================== business rule
def dead_at_origin(df: pd.DataFrame) -> np.ndarray:
    """Zero sales in all of the last 8 weeks before the origin -> forecast 0.
    Uses origin information only, so the same rule is valid in backtests."""
    return (df["o_ma8"].fillna(0) == 0).to_numpy()


def _finish(pred, df) -> np.ndarray:
    pred = np.clip(np.asarray(pred, dtype=float), 0, None).copy()
    pred[dead_at_origin(df)] = 0.0
    return pred


def _level(df) -> pd.Series:
    """Recent weekly level of the series at the origin (units). Falls back to the 4-week
    level, then to an average store's level for that family (cold start)."""
    fam_per_store = np.expm1(df["o_family_ma13"]) / 54
    return np.expm1(df["o_ma13"].fillna(df["o_ma4"])).fillna(fam_per_store).fillna(0)


def has_history(df) -> pd.Series:
    """Rows where the series itself has a known recent level at the origin."""
    return df["o_ma13"].notna() | df["o_ma4"].notna()


MAX_RATIO = 20.0   # cap on the training target y / level (protects against tiny-level spikes)


# =============================================================================== baselines
def naive_ma4(train, test):
    return _finish(np.expm1(test["o_ma4"].fillna(test["o_ma13"]).fillna(0)), test)


def snaive(train, test):
    return _finish(np.expm1(test["o_lag52_target"]).fillna(_level(test)), test)


def snaive_growth(train, test):
    growth = np.exp(test["o_yoy_growth"].fillna(0).clip(-1, 1))
    return _finish((np.expm1(test["o_lag52_target_smooth"]) * growth).fillna(_level(test)), test)


# =============================================================================== global LightGBM
class GlobalLGBM:
    """Target = units / (recent 13-week level + 1), weight = level + 1.

    Predicting a ratio lets one model serve PRODUCE (15k units/week) and BOOKS (~0)
    alike; weighting by level makes the squared-error loss focus on volume, which is
    what WMAPE measures. `bags` models are trained on different random row samples
    and averaged (bagging) for a stable forecast.
    """

    def __init__(self, params=None, rounds=C.LGBM_ROUNDS, sample=C.TRAIN_SAMPLE,
                 features=None, seed=C.SEED, bags=C.LGBM_BAGS):
        self.params = {**C.LGBM_PARAMS, **(params or {})}
        self.rounds, self.sample, self.features, self.seed, self.bags = rounds, sample, features, seed, bags
        self.boosters: list[lgb.Booster] = []

    def fit(self, train: pd.DataFrame) -> "GlobalLGBM":
        self.features = self.features or F.model_features(train.columns)
        cats = [c for c in F.CATEGORICAL if c in self.features]
        self.boosters = []
        for b in range(self.bags):
            seed = self.seed + 1000 * b
            tr = train.loc[has_history(train)]          # no series history -> no meaningful ratio
            tr = tr.sample(frac=self.sample, random_state=seed) if self.sample < 1 else tr
            scale = _level(tr) + 1
            target = (tr["y"] / scale).clip(upper=MAX_RATIO)
            ds = lgb.Dataset(tr[self.features], target, weight=scale, categorical_feature=cats)
            self.boosters.append(lgb.train({**self.params, "seed": seed}, ds, self.rounds))
        return self

    def predict_ratio(self, df: pd.DataFrame) -> np.ndarray:
        return np.mean([b.predict(df[self.features]) for b in self.boosters], axis=0)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return _finish(self.predict_ratio(df) * (_level(df) + 1), df)

    def save(self, folder) -> None:
        import json
        folder.mkdir(parents=True, exist_ok=True)
        for i, b in enumerate(self.boosters):
            b.save_model(str(folder / f"booster_{i}.txt"))
        (folder / "features.json").write_text(json.dumps(self.features))

    @classmethod
    def load(cls, folder) -> "GlobalLGBM":
        import json
        m = cls(features=json.loads((folder / "features.json").read_text()))
        m.boosters = [lgb.Booster(model_file=str(f)) for f in sorted(folder.glob("booster_*.txt"))]
        return m

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """TreeSHAP contributions (LightGBM native) averaged over bags, in RATIO units.
        Multiply by (level + 1) to express them in units. Last column = base value."""
        cols = self.features + ["_base"]
        out = np.mean([b.predict(df[self.features], pred_contrib=True) for b in self.boosters], axis=0)
        return pd.DataFrame(out, columns=cols, index=df.index)

    @property
    def model(self) -> lgb.Booster:
        """First booster (used for SHAP / feature importance)."""
        return self.boosters[0]


def lgbm(train, test, **kw):
    return GlobalLGBM(**kw).fit(train).predict(test)


FORECASTERS = {"naive_ma4": naive_ma4, "snaive": snaive, "snaive_growth": snaive_growth, "lgbm": lgbm}


# =============================================================================== report
def _chart_overall(overall: pd.DataFrame) -> str:
    o = overall["WMAPE"].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    colors = [SERIES[0] if m == "lgbm" else "#b7d3f6" for m in o.index]
    ax.barh([LABELS[m] for m in o.index], o.values, color=colors, height=0.6)
    for i, v in enumerate(o.values):
        ax.text(v + 0.002, i, f"{v:.1%}", va="center", fontsize=9, color=INK2)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.grid(axis="y", visible=False)
    ax.set_title("Backtest WMAPE, 4 folds x 13 weeks (lower is better)")
    return _cv_save(fig, "cv_01_model_wmape.png")


def _chart_horizon(by_h: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for i, m in enumerate(MODELS):
        ax.plot(by_h.index, by_h[f"WMAPE {m}"], color=SERIES[i], marker="o", ms=4, label=LABELS[m])
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_xticks(range(1, C.HORIZON + 1))
    ax.set_xlabel("Weeks ahead (h)")
    ax.legend(ncol=2)
    ax.set_title("WMAPE by forecast horizon")
    return _cv_save(fig, "cv_02_wmape_by_horizon.png")


def _chart_family(by_fam: pd.DataFrame) -> str:
    top = by_fam.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    yy = np.arange(len(top))
    ax.hlines(yy, top["WMAPE lgbm"], top["WMAPE naive_ma4"], color=AXIS, lw=2)
    ax.scatter(top["WMAPE naive_ma4"], yy, color=SERIES[1], s=40, label=LABELS["naive_ma4"], zorder=3)
    ax.scatter(top["WMAPE lgbm"], yy, color=SERIES[0], s=40, label=LABELS["lgbm"], zorder=3)
    ax.set_yticks(yy, top.index.astype(str))
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2)
    ax.set_title("WMAPE by family (12 largest), LightGBM vs best baseline")
    return _cv_save(fig, "cv_03_wmape_by_family.png")


def _chart_total(pred: pd.DataFrame, sup: pd.DataFrame) -> str:
    hist = sup.loc[(sup["h"] == 1) & sup["y"].notna()].groupby("week")["y"].sum()
    hist = hist.loc[hist.index >= pred["week"].min() - pd.Timedelta(weeks=20)]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(hist.index, hist.values, color=INK2, lw=1.6, label="Actual")
    for k, g in pred.groupby("fold"):
        t = g.groupby("week")[["lgbm", "naive_ma4"]].sum()
        ax.plot(t.index, t["naive_ma4"], color=SERIES[1], lw=1.6, label=LABELS["naive_ma4"] if k == 1 else None)
        ax.plot(t.index, t["lgbm"], color=SERIES[0], lw=2, label=LABELS["lgbm"] if k == 1 else None)
        ax.axvline(g["origin_week"].iloc[0], color=AXIS, lw=1, ls="--")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x/1e6:.1f}M")
    ax.legend(ncol=3, loc="upper left")
    ax.set_title("Chain total: actual vs 13-week forecasts from each fold origin (dashed)")
    return _cv_save(fig, "cv_04_backtest_total.png")


def _cv_save(fig, name):
    path = C.CV_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return name


def run_model_comparison(sup: pd.DataFrame) -> pd.DataFrame:
    """Backtest every forecaster, save tables, charts and cv_report.md."""
    pred = V.run_backtest(sup, FORECASTERS)
    pred.to_parquet(C.CV_DIR / "cv_predictions.parquet", index=False)

    overall = V.score(pred, MODELS)
    by_fold = V.tidy_segment(pred, MODELS, "fold").sort_index()
    by_h = V.tidy_segment(pred, MODELS, "h").sort_index()
    by_fam = V.tidy_segment(pred, MODELS, C.SKU_COL)
    by_type = V.tidy_segment(pred, MODELS, "store_type").sort_index()
    by_store = V.tidy_segment(pred, MODELS, C.MARKET_COL)
    for name, t in [("overall", overall), ("by_fold", by_fold), ("by_horizon", by_h), ("by_family", by_fam),
                    ("by_store_type", by_type), ("by_store", by_store)]:
        t.to_csv(C.CV_DIR / f"metrics_{name}.csv")

    # how often does LightGBM beat the best baseline, series by series?
    ser = pred.groupby("sid").apply(lambda g: pd.Series(
        {m: V.wmape(g["y"], g[m]) for m in MODELS} | {"units": g["y"].sum()}))
    ser = ser.loc[ser["units"] > 0]
    win_series = (ser["lgbm"] < ser["naive_ma4"]).mean()
    win_volume = ser.loc[ser["lgbm"] < ser["naive_ma4"], "units"].sum() / ser["units"].sum()
    store_win = (by_store["WMAPE lgbm"] < by_store["WMAPE naive_ma4"]).mean()
    fam_win = (by_fam["WMAPE lgbm"] < by_fam["WMAPE naive_ma4"]).mean()

    charts = [_chart_overall(overall), _chart_horizon(by_h), _chart_family(by_fam), _chart_total(pred, sup)]
    best_base = overall.loc[BASELINES, "WMAPE"].idxmin()
    gain = 1 - overall.loc["lgbm", "WMAPE"] / overall.loc[best_base, "WMAPE"]

    fmt = lambda df: df.to_markdown(floatfmt=".3f")
    md = f"""# Model comparison - rolling-origin backtest

_Auto-generated by `python main.py --step models`._

**Setup.** {C.N_FOLDS} folds, each forecasting {C.HORIZON} weeks from its origin using only data up to the
origin (target weeks of training rows <= origin, asserted in code). Metrics on all valid series-weeks.

## Headline

Global LightGBM reaches **WMAPE {overall.loc['lgbm', 'WMAPE']:.1%}** vs {overall.loc[best_base, 'WMAPE']:.1%} for the best
baseline ({LABELS[best_base]}), a **{gain:.0%} relative error reduction**. Bias {overall.loc['lgbm', 'Bias']:+.1%}
(negative = under-forecast).

LightGBM is better than "{LABELS['naive_ma4']}" for **{win_series:.0%} of series** (covering {win_volume:.0%} of units),
**{store_win:.0%} of stores** and **{fam_win:.0%} of families**.

{fmt(overall)}

## By fold (each fold = a different 13-week season)

{fmt(by_fold)}

## By horizon

{fmt(by_h)}

## By store type

{fmt(by_type)}

## By family (sorted by volume)

{fmt(by_fam)}

Per-store results: `metrics_by_store.csv`. Row-level predictions: `cv_predictions.parquet`.

## Charts

""" + "\n".join(f"![{c}]({c})" for c in charts) + "\n"
    (C.CV_DIR / "cv_report.md").write_text(md, encoding="utf-8")
    print(f"  - report written to {C.CV_DIR / 'cv_report.md'}")
    return overall
