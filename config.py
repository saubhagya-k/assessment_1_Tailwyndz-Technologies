"""
config.py - one place for every path, date and setting used by the pipeline.

Change values here instead of inside the modules, so every run is reproducible.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"      # cached cleaned / weekly data (auto-created)
OUT_DIR = ROOT / "outputs"
EDA_DIR = OUT_DIR / "eda"

for _d in (INTERIM_DIR, OUT_DIR, EDA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RAW_FILES = {
    "train": "train.csv",
    "test": "test.csv",
    "stores": "stores.csv",
    "oil": "oil.csv",
    "holidays": "holidays_events.csv",
    "transactions": "transactions.csv",
}

# ---------------------------------------------------------------- business mapping
# The assignment talks about "market" and "SKU"; in this dataset they are:
MARKET_COL = "store_nbr"     # market_id  -> store
SKU_COL = "family"           # sku_id     -> product family
TARGET = "sales"             # weekly unit sales
SERIES_KEYS = [MARKET_COL, SKU_COL]

# ---------------------------------------------------------------- time settings
WEEK_FREQ = "W-SUN"          # weeks run Monday -> Sunday, labelled by the Sunday
HORIZON = 13                 # forecast 13 weeks ahead
N_FOLDS = 4                  # rolling-origin backtest folds
FOLD_STEP = 13               # weeks between fold origins (non-overlapping test windows)

# Promotions are recorded as 0 everywhere before this date (data artefact, not "no promo")
PROMO_RELIABLE_FROM = "2014-04-01"

# Modelling window starts after the last recording-regime switch found in EDA
# (PRODUCE / BEVERAGES alternate between two unit levels until 2015-05-31).
# Earlier history is used for EDA only. This also avoids the unrecorded-promo period.
MODEL_START = "2015-06-01"

# The weekly panel starts earlier than MODEL_START so lag / same-week-last-year
# features exist for the first training weeks. Families with a recording-regime
# switch are masked (NaN) before MODEL_START, so they never feed bad history.
PANEL_START = "2014-04-07"           # first Monday after promotions became reliable

# Known one-off shock: Manabi earthquake (relief purchases inflated sales)
EARTHQUAKE_START = "2016-04-16"
EARTHQUAKE_END = "2016-05-15"

# ---------------------------------------------------------------- series status rules
DEAD_LOOKBACK_WEEKS = 8      # zero sales in the last N weeks -> treated as discontinued
NEW_SERIES_WEEKS = 26        # fewer than N weeks of history -> cold-start series
GAP_MIN_DAYS = 14            # >= N consecutive zero days inside a series' life ...
GAP_MIN_LEVEL = 5            # ... when it normally sells >= N units/day -> "recording gap", not real demand

# ---------------------------------------------------------------- model settings
# Global LightGBM: predicts sales RELATIVE to the series' recent 13-week level,
# weighted by that level, so big and small series share one model and the loss
# matches WMAPE. Stacked horizons are highly redundant -> 40% row sample is as
# accurate as 100% in CV and twice as fast.
LGBM_PARAMS = dict(
    objective="l2", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    max_bin=127, verbose=-1,
)
LGBM_ROUNDS = 400
TRAIN_SAMPLE = 0.4
LGBM_BAGS = 3            # average of 3 models on different 40% samples: CV showed single-seed
                         # WMAPE varies by ~1pt on the Christmas fold; bagging removes that noise

# Quick smoke-test mode: FRESHBASKET_FAST=1 python main.py ...  (tiny models, ~1 min end to end)
import os as _os
FAST = _os.environ.get("FRESHBASKET_FAST") == "1"
if FAST:
    LGBM_ROUNDS, TRAIN_SAMPLE, LGBM_BAGS = 40, 0.1, 1

CV_DIR = OUT_DIR / "cv"
FORECAST_DIR = OUT_DIR / "forecast"
EXPLAIN_DIR = OUT_DIR / "explain"
SCENARIO_DIR = OUT_DIR / "scenario"
MODEL_DIR = OUT_DIR / "model"
for _d in (CV_DIR, FORECAST_DIR, EXPLAIN_DIR, SCENARIO_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Prediction intervals: P10 / P90 (an 80% range), calibrated on backtest residuals
INTERVAL_Q = (0.10, 0.90)

# Ablation runs on 2 folds (latest + Christmas) with 1 bag to keep runtime reasonable
ABLATION_FOLDS = (1, 3)

# ---------------------------------------------------------------- reproducibility
SEED = 42
