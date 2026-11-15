# Data-Driven Analysis and Forecasting of Food Price Trends in India

Uses FAOSTAT data on **crop production**, **agricultural trade** and the **Food Consumer Price Index (CPI)**
to (1) understand India's food-price trends and (2) forecast the Food CPI for the next 6 months.

> We predict the **Food CPI index (2015 = 100)**, not the price of rice or wheat per kg.

## Run it
```bash
pip install -r requirements.txt
python food_price_analysis.py            # reads ./data, writes ./outputs
# optional: python food_price_analysis.py --data-dir path/to/csvs --out-dir path/to/results
```
Runs end to end in under a minute. All random seeds are fixed (`SEED = 42`).

## Repository layout
```
food_price_analysis.py   whole pipeline, one function per section
requirements.txt
data/                    food_consumer_price_index.csv, crop_production.csv, agricultural_trade.csv
outputs/figures/         00_dashboard.png + one figure per analysis step
outputs/results/         forecast_next_6_months.csv, model_comparison.csv, hypothesis_tests.csv, ...
```

## What the script does (maps to the marking criteria)
| Section | Content |
|---|---|
| 1 Problem and data | Question, measurable goals, dataset source/size/coverage |
| 2-4 Preparation | Missing values, duplicates, dtypes, outliers (IQR), imputed-flag check, feature engineering (log growth, MoM, YoY, rolling mean, month sin/cos, high-inflation flag), unit-safe trade aggregation |
| 5 EDA | Level/inflation plots, seasonality box-plot + heat-map, distributions, ACF; written insights are computed from the data |
| 6 Inferential stats | Confidence interval, Shapiro-Wilk, Levene, t-test, Mann-Whitney, ANOVA, Kruskal-Wallis, trend test, Pearson/Spearman with Bonferroni note |
| 7 Linear regression | Annual production/trade growth -> food inflation, Leave-One-Out CV vs mean baseline |
| 8 Forecasting | 3 baselines + Linear, Ridge (tuned), Random Forest (tuned), Ensemble; chronological train/validation/test; `TimeSeriesSplit` tuning; recursive 6-step rolling-origin backtests |
| 9 Evaluation | MAE/RMSE/MAPE, error by horizon, skill vs baseline, bias, worst errors, limitations, next steps |
| 10 Logistic regression | Will YoY food inflation be >= 6 % in 3 months? vs persistence and majority baselines; odds ratios, ROC, confusion matrix |
| 11 Final forecast | Refit on all data, 6-month forecast with 95 % band, 4-panel dashboard |

## Design decisions worth explaining in the viva
* **Model growth, not level.** The CPI trends upward (non-stationary); monthly log growth is far more stable.
* **No leakage.** Features at time *t* use data up to *t-1* only; splits are chronological; tuning happens inside the training block; the test block is used once.
* **Model selection on validation, not test.** Test results are reported honestly even when they disagree.
* **Trade items.** The plain `Rice` item in the trade file is tiny; rice trade is taken from `Rice, paddy (rice milled equivalent)`. That aggregate is excluded from item rankings to avoid double counting.
* **Annual vs monthly.** Production/trade are annual (to 2024) and CPI is monthly (to Jun 2026), so they are linked in an annual analysis and are not inputs to the monthly forecast.

## Key limitations
* The provider flags **Jan 2025 - Jun 2026** index values as *imputed*; the test block lies inside that window.
* Only 8-9 annual points link production/trade to inflation, so those tests have very low power: "not established" is not "no effect".
* On the held-out test block the ML models did **not** beat the naive baseline (see `outputs/results/model_comparison.csv`); the forecast band is wide.
