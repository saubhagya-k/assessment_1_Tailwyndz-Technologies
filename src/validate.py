"""
validate.py - THE TIME MACHINE. Leakage-safe rolling-origin backtesting + metrics.

Fold k pretends today is ORIGIN_k, trains only on weeks <= ORIGIN_k, then forecasts
the next 13 weeks exactly like the real forecast will be made:

    fold 4 |== train ==========|-- test 13w --|
    fold 3 |== train =====================|-- test 13w --|
    fold 2 |== train ==================================|-- test 13w --|
    fold 1 |== train ===============================================|-- test 13w --|
                                                                     last data week
Random train/test splits are never used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


# =============================================================================== folds
def last_full_week(sup: pd.DataFrame) -> pd.Timestamp:
    return sup.loc[sup["y"].notna(), "week"].max()


def fold_origins(sup: pd.DataFrame, n_folds: int = C.N_FOLDS, step: int = C.FOLD_STEP) -> list[pd.Timestamp]:
    """Most recent fold first. Test windows do not overlap."""
    end = last_full_week(sup)
    return [end - pd.Timedelta(weeks=step * k) for k in range(1, n_folds + 1)]


def split(sup: pd.DataFrame, origin: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """train: every stacked row whose TARGET week is <= origin (so nothing after origin is seen)
    test : rows forecast FROM this origin for the next HORIZON weeks."""
    has_y = sup["y"].notna()
    train = sup.loc[has_y & (sup["week"] <= origin)]
    test = sup.loc[has_y & (sup["origin_week"] == origin)
                   & (sup["week"] <= origin + pd.Timedelta(weeks=C.HORIZON))]
    assert train["week"].max() <= origin < test["week"].min(), "leakage: train overlaps test"
    return train, test


# =============================================================================== metrics
def wmape(y, f) -> float:
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(np.abs(y - f).sum() / max(np.abs(y).sum(), 1e-9))


def bias(y, f) -> float:
    """+ = over-forecast (excess stock risk), - = under-forecast (lost-sales risk)."""
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float((f - y).sum() / max(np.abs(y).sum(), 1e-9))


def rmse(y, f) -> float:
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(np.sqrt(np.mean((y - f) ** 2)))


def mae(y, f) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(f, float))))


def score(df: pd.DataFrame, models: list[str], by: list[str] | None = None) -> pd.DataFrame:
    """Metrics per model (rows) optionally per segment."""
    def _one(g):
        rows = {}
        for m in models:
            rows[(m, "WMAPE")] = wmape(g["y"], g[m])
            rows[(m, "Bias")] = bias(g["y"], g[m])
            rows[(m, "RMSE")] = rmse(g["y"], g[m])
            rows[(m, "MAE")] = mae(g["y"], g[m])
        rows[("", "actual_units")] = g["y"].sum()
        return pd.Series(rows)

    if not by:
        s = _one(df)
        return s.drop(("", "actual_units")).unstack().loc[models]
    out = df.groupby(by, observed=True).apply(_one)
    return out


def tidy_segment(df: pd.DataFrame, models: list[str], by: str) -> pd.DataFrame:
    """Readable table: one row per segment, WMAPE & bias per model."""
    s = score(df, models, [by])
    cols = {f"WMAPE {m}": s[(m, "WMAPE")] for m in models}
    cols.update({f"Bias {m}": s[(m, "Bias")] for m in models})
    cols["actual units"] = s[("", "actual_units")]
    return pd.DataFrame(cols).sort_values("actual units", ascending=False)


# =============================================================================== backtest runner
def run_backtest(sup: pd.DataFrame, forecasters: dict, verbose: bool = True) -> pd.DataFrame:
    """forecasters: name -> function(train_df, test_df) -> np.array of predictions.

    Returns one row per (fold, series, target week) with actual y and every model's prediction.
    """
    out = []
    for k, origin in enumerate(fold_origins(sup), start=1):
        train, test = split(sup, origin)
        res = test[["sid", C.MARKET_COL, C.SKU_COL, "store_type", "week", "h", "origin_week", "y"]].copy()
        res["fold"] = k
        for name, fn in forecasters.items():
            res[name] = np.clip(fn(train, test), 0, None)
        if verbose:
            msg = ", ".join(f"{n} {wmape(res['y'], res[n]):.3f}" for n in forecasters)
            print(f"    fold {k}: origin {origin:%Y-%m-%d}, test {test['week'].min():%Y-%m-%d}"
                  f"..{test['week'].max():%Y-%m-%d}, train rows {len(train):,} | WMAPE {msg}")
        out.append(res)
    return pd.concat(out, ignore_index=True)
