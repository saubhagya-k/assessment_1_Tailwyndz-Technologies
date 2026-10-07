"""
main.py - THE MANAGER. Runs the pipeline step by step.

Usage (run in this order the first time):
    python main.py --step eda          # [1] cleaning + data-quality & EDA report       -> outputs/eda/
    python main.py --step features     # [2] weekly modelling table + leakage check     -> outputs/feature_dictionary.csv
    python main.py --step models       # [3+4] backtest: baselines vs global LightGBM   -> outputs/cv/
    python main.py --step forecast     # [5+6] final 13-week forecast + P10/P90         -> outputs/forecast/
    python main.py --step explain      # [7] SHAP driver attribution                    -> outputs/explain/
    python main.py --step ablation     # [8] feature-group ablation study               -> outputs/explain/
    python main.py --step scenario     # [9] what-if simulator                          -> outputs/scenario/
    python main.py --step all          # everything above in order

Options:
    --no-cache            rebuild cached data in data/interim/ from the raw CSVs
    FRESHBASKET_FAST=1    environment variable: tiny models for a quick smoke test
"""
import argparse
import time

import pandas as pd

import config as C
from src import data, eda, features

STEPS = ["eda", "features", "models", "forecast", "explain", "ablation", "scenario"]


def step_features(clean: dict, use_cache: bool) -> None:
    p = features.build_features(clean, use_cache=use_cache)
    sup = features.make_supervised(p, target_from=pd.Timestamp(C.MODEL_START))
    train_rows = sup.loc[sup["y"].notna()]
    feats = features.model_features(sup.columns)
    features.describe_features(p).to_csv(C.OUT_DIR / "feature_dictionary.csv", index=False)

    mismatches = features.leakage_check(p, sup)
    summary = pd.Series({
        "series": p["sid"].nunique(),
        "panel weeks": f"{p['week'].min():%Y-%m-%d} -> {p['week'].max():%Y-%m-%d}",
        "training target weeks": f"{train_rows['week'].min():%Y-%m-%d} -> {train_rows['week'].max():%Y-%m-%d}",
        "forecast weeks": f"{p.loc[p['is_future'] == 1, 'week'].min():%Y-%m-%d} -> {p['week'].max():%Y-%m-%d}",
        "stacked training rows (all 13 horizons)": f"{len(train_rows):,}",
        "model features": len(feats),
        "leakage check (o_ma4 recomputed from past only)": "PASS" if mismatches == 0 else f"FAIL ({mismatches})",
    })
    print(summary.to_string())
    print(f"  - feature dictionary written to {C.OUT_DIR / 'feature_dictionary.csv'}")


def run(step: str, use_cache: bool) -> None:
    t0 = time.time()
    todo = STEPS if step == "all" else [step]
    if C.FAST:
        print("*** FAST smoke-test mode: tiny models, results are NOT representative ***")
    print("[1] Loading and cleaning data ...")
    clean = data.build_clean_data(use_cache=use_cache)

    sup = None

    def get_sup():
        nonlocal sup
        if sup is None:
            p = features.build_features(clean, use_cache=use_cache)
            sup = features.make_supervised(p, target_from=pd.Timestamp(C.MODEL_START))
        return sup

    if "eda" in todo:
        print("[2] Data-quality checks and EDA ...")
        eda.run_eda(data.load_raw(), clean)

    if "features" in todo:
        print("[3] Building weekly features ...")
        step_features(clean, use_cache)

    # LightGBM is imported only from here on (macOS needs `brew install libomp`)
    if "models" in todo:
        from src import models
        print("[4] Backtesting baselines vs global LightGBM (4 folds x 13 weeks) ...")
        print(models.run_model_comparison(get_sup()).round(3).to_string())

    if "forecast" in todo:
        from src import forecast
        print("[5] Final 13-week forecast with prediction intervals ...")
        forecast.run_forecast(get_sup(), clean["status"])

    if "explain" in todo:
        from src import explain
        print("[6] Driver attribution (TreeSHAP) ...")
        explain.run_drivers(get_sup())

    if "ablation" in todo:
        from src import explain
        print(f"[7] Ablation study on folds {C.ABLATION_FOLDS} ...")
        print(explain.run_ablation(get_sup())[["WMAPE avg", "change vs full (pts)", "verdict"]].round(4).to_string())

    if "scenario" in todo:
        from src import scenario
        print("[8] What-if scenarios ...")
        print(scenario.run_scenarios(get_sup()).round(4).to_string())

    print(f"Done in {time.time() - t0:.0f}s. Outputs in {C.OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FreshBasket 13-week demand forecasting pipeline")
    ap.add_argument("--step", default="all", choices=STEPS + ["all"])
    ap.add_argument("--no-cache", action="store_true", help="ignore cached parquet files")
    a = ap.parse_args()
    run(a.step, use_cache=not a.no_cache)
