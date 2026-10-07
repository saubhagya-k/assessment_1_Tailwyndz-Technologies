# FreshBasket - 13-week FMCG demand forecasting

Weekly unit-sales forecasts for every store x product-family combination (54 stores x 33 families =
1,782 series) for the next 13 weeks, with prediction intervals, driver attribution, an ablation
study and a what-if scenario simulator.

Data: Kaggle "Store Sales - Time Series Forecasting" (Corporación Favorita, Ecuador), used as the
FreshBasket dataset. Mapping: `market_id` = `store_nbr`, `sku_id` = product `family`.

---

## 1. Setup

```bash
cd freshbasket_forecasting
python3 -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**macOS only:** LightGBM needs OpenMP. Install it once with `brew install libomp`
(Homebrew: https://brew.sh). Without it you get `Library not loaded: @rpath/libomp.dylib`.

Put the six Kaggle CSVs in `data/raw/`:
`train.csv, test.csv, stores.csv, oil.csv, holidays_events.csv, transactions.csv`.

Tested with Python 3.12 / 3.13, pandas 3.0, LightGBM 4.7. Needs ~2 GB RAM.

## 2. Run

```bash
python main.py --step all             # everything, ~10-20 min on a laptop
```

or step by step (each step reuses cached results from the previous ones):

| Step | Command | What it does | Output folder |
|---|---|---|---|
| 1 | `--step eda` | cleaning, data-quality checks, EDA (10 charts) | `outputs/eda/` |
| 2 | `--step features` | weekly modelling table, 55 features, automatic leakage check | `outputs/feature_dictionary.csv` |
| 3-4 | `--step models` | rolling-origin backtest: 3 baselines vs global LightGBM | `outputs/cv/` |
| 5-6 | `--step forecast` | final model, 13-week forecast with P10/P50/P90 | `outputs/forecast/` |
| 7 | `--step explain` | TreeSHAP driver attribution in units | `outputs/explain/` |
| 8 | `--step ablation` | value of each feature group | `outputs/explain/` |
| 9 | `--step scenario` | promo / oil what-if simulator, promo uplift | `outputs/scenario/` |

`--no-cache` rebuilds `data/interim/` from the raw CSVs. `FRESHBASKET_FAST=1 python main.py ...`
runs tiny models for a quick smoke test (results not meaningful).

Every step writes a Markdown report (`*_report.md`) with tables and charts. In VS Code, open it
and press `Cmd/Ctrl+Shift+V` to preview.

**Main deliverable:** `outputs/forecast/forecast.csv`
`date` (week ending Sunday), `week_start`, `market_id`, `sku_id`, `horizon_weeks`, `forecast` (P50),
`forecast_p10`, `forecast_p90`, `store_type`, `promo_assumed_share`, `series_status`, `rule_applied`.

## 3. Project structure

```
main.py            pipeline runner (--step ...)
config.py          all paths, dates and model settings in one place
src/data.py        load + clean: missing days, store openings/closures, oil gaps, holidays,
                   recording gaps, series status
src/eda.py         data-quality table, EDA charts, findings report
src/features.py    weekly panel, origin (o_*) vs known-future (k_*) features, stacking, leakage check
src/validate.py    rolling-origin folds, WMAPE / bias / RMSE / MAE, segment tables
src/models.py      baselines + bagged global LightGBM, backtest report
src/forecast.py    final forecast, conformal prediction intervals, coverage check
src/explain.py     TreeSHAP driver attribution, ablation study
src/scenario.py    what-if simulator, promo response curve, baseline vs promo split
```

## 4. Method in one page

**Data cleaning (EDA finding 11 is the key one).** PRODUCE, BEVERAGES and HOME CARE switch between
two recording units until May 2015 (e.g. PRODUCE ~30 whole units/day vs ~10,000 decimal units/day
per store). Training therefore starts **2015-06-01**. Other fixes: 25-Dec / 1-Jan chain closures
treated as planned closures; rows before a store opened or a family was first stocked excluded;
14+ day zero runs in normally-selling series treated as recording gaps (not zero demand); oil
interpolated; transferred holidays moved to their real date.

**Forecast design.** One *direct multi-horizon global* LightGBM: each training row is
"standing at week T, predict week T+h" for h = 1..13, all series and horizons stacked. Features are
split into `o_*` (known at the origin T: sales history, store/family totals, traffic, oil) and `k_*`
(known in advance for week T+h: planned promotions, location-matched holidays, paydays, closures,
calendar). This split makes leakage impossible by construction, and `features.leakage_check`
re-computes features from past data only to prove it.

The model predicts sales **relative to the series' recent 13-week level** (weighted by that level),
so one model serves large and tiny series and the loss matches WMAPE. 3 models on different 40% row
samples are averaged (bagging) for stability.

**Validation.** 4 rolling origins, each followed by a 13-week test window (Aug-2016 to Aug-2017,
including Christmas). Training rows never contain target weeks after the origin (asserted in code).
Random splits are never used.

**Intervals.** Split-conformal: quantiles of the model's own backtest errors by horizon and volume
level give P10/P90. Coverage is verified leave-one-fold-out.

**Business rules.** Series with zero sales in the 8 weeks before the origin are forecast at 0.
New series use the global model plus a family-level fallback for their level.

## 5. Assumptions

* Week = Monday-Sunday, labelled by the Sunday. The last partial week (14-15 Aug 2017) is not a target.
* **Future promotions:** 16-31 Aug 2017 from `test.csv` (the real plan). Later weeks assume each series
  keeps its average promo level of the last 8 weeks. `promo_assumed_share` flags assumed days.
* **Future holidays** are known from the calendar. **Future oil** is not needed: only oil known at the
  origin is used (level and 4/13-week change).
* Transactions are used only up to the origin (they are not known for future weeks).
* Price, distribution, media and weather are not in this dataset; they cannot be modelled or ablated.

## 6. Limitations

* Only ~2 years of clean history after the recording fix: yearly seasonality is learned from about two
  cycles, and per-series statistical models (ETS/ARIMA with 52-week seasonality) are not feasible.
* Slight under-forecast bias (about -3.5%) from fast growth that trees cannot fully extrapolate.
* Driver attributions and scenarios reflect historical associations, not causal effects;
  promotion scenarios far outside the historical range are extrapolations.
* `family` is an aggregate of many SKUs; SKU-level dynamics (launches, substitutions) are not visible.


## Results at a glance

| Model (4-fold rolling backtest, 13 weeks each) | WMAPE | Bias |
|---|---|---|
| Same week last year | 14.9% | −8.0% |
| Last year × growth | 13.6% | −3.5% |
| Last 4 weeks average | 12.6% | −5.5% |
| **Global LightGBM (this project)** | **11.1%** | **−3.6%** |

- Wins in all 4 test periods, all 5 store types and 85% of stores
- 13-week forecast: 78.5M units (+12.5% vs same weeks 2016)
- P10–P90 range covers 78% of actuals (target 80%)
- Promotions drive ~2.3% of volume; +20% promotions → +1.0% units

Full write-up: `docs/Project_Report.pdf` · Slides: `docs/Presentation.pdf`