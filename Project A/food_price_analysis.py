
"""
Data-Driven Analysis and Forecasting of Food Price Trends in India
Using Agricultural Production and Trade Data
=====================================================================

Central question
    Can we analyse India's agricultural production and trade patterns to
    understand food-price trends, and can we forecast future food-price
    movements using historical data?

Main prediction task
    Forecast India's monthly Food Consumer Price Index (FAO/FAOSTAT, 2015=100)
    for the next 6 months.   (We predict an INDEX, not rupees per kg.)

Secondary task
    Logistic regression: will food inflation be "high" (>= 6 % YoY) 3 months ahead?

Run:
    python food_price_analysis.py                       # uses ./data, writes ./outputs
    python food_price_analysis.py --data-dir D --out-dir O

Sections
    1  Problem definition and data justification
    2  Data preparation (CPI)            + feature engineering
    3  Crop production: cleaning, trends
    4  Agricultural trade: cleaning, import/export analysis
    5  EDA of the Food CPI
    6  Inferential statistics and hypothesis testing
    7  Linear regression: production/trade -> food inflation (annual)
    8  Forecasting Food CPI (baselines, Ridge/Linear, Random Forest)
    9  Evaluation and error analysis
   10  Logistic regression: high-inflation classifier
   11  Final forecast and dashboard
"""

import argparse
import calendar
import textwrap
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; figures are saved to disk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, r2_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV, LeaveOneOut, TimeSeriesSplit, cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Configuration (fixed seeds => reproducible results)
# --------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

HORIZON = 6            # months ahead to forecast
VAL_MONTHS = 12        # validation block (model selection)
TEST_MONTHS = 12       # final untouched test block
HIGH_INFLATION = 6.0   # % YoY threshold for the "high inflation" class
CLF_HORIZON = 3        # classifier looks 3 months ahead
CROPS = ["Rice", "Wheat", "Maize (corn)"]
# In the trade table the plain "Rice" item is a tiny category; the full rice trade is reported
# under the milled-equivalent aggregate, so we map each focus crop to the right trade item.
TRADE_ITEM = {"Rice": "Rice, paddy (rice milled equivalent)", "Wheat": "Wheat", "Maize (corn)": "Maize (corn)"}
DOUBLE_COUNT_ITEMS = ["Rice, paddy (rice milled equivalent)"]   # aggregate of other rice items
CROP_COLORS = {"Rice": "#2a9d8f", "Wheat": "#e9c46a", "Maize (corn)": "#e76f51"}

sns.set_theme(style="whitegrid", context="notebook")


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def save_fig(fig, name, out):
    path = out / "figures" / name
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure saved] {path.name}")


def acf(x, nlags=24):
    """Autocorrelation function with numpy only."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    x = x - x.mean()
    denom = np.sum(x ** 2)
    return np.array([np.sum(x[k:] * x[: len(x) - k]) / denom for k in range(nlags + 1)])


def regression_metrics(actual, pred):
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    err = pred - actual
    return {
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "MAPE_%": np.mean(np.abs(err / actual)) * 100,
    }


# ==========================================================================
# 1. PROBLEM AND DATA
# ==========================================================================
def load_data(data_dir):
    section("1. PROBLEM DEFINITION AND DATA JUSTIFICATION")
    cpi_raw = pd.read_csv(r"C:\Users\sindh\OneDrive\Documents\IDSE project\food_consumer_price_index.csv")
    prod_raw = pd.read_csv(r"C:\Users\sindh\OneDrive\Documents\IDSE project\crop_production.csv")
    trade_raw = pd.read_csv(r"C:\Users\sindh\OneDrive\Documents\IDSE project\agricultural_trade.csv")

    print(textwrap.dedent("""
    Question : Can production and trade patterns help us understand India's food-price
               trends, and can we forecast the Food CPI for the next 6 months?
    Target   : Monthly Food CPI index (2015 = 100)  -> regression / forecasting
    Extra    : "High inflation" (>= 6 % YoY) 3 months ahead -> classification
    Goals    : (a) beat naive baselines on a held-out 12-month test block,
               (b) test statistically whether production/trade relate to inflation,
               (c) deliver a 6-month forecast with an uncertainty band.
    """))
    rows = [
        ("Food CPI", cpi_raw, "FAOSTAT Consumer Price Indices", "monthly 2016-01..2026-06"),
        ("Crop production", prod_raw, "FAOSTAT QCL (crops & livestock)", "annual 1961..2024"),
        ("Agricultural trade", trade_raw, "FAOSTAT TCL (trade)", "annual 2016..2024"),
    ]
    for name, df, src, cov in rows:
        print(f"  {name:<20} rows={len(df):>6}  cols={df.shape[1]:>2}  source={src}  coverage={cov}")
    print("\n  Justification: three official, real-world FAOSTAT tables for one country, "
          "sharing 'Year' as a join key. CPI is monthly (126 points) which is enough for a "
          "time-series model; production/trade are annual and used for explanation.")
    return cpi_raw, prod_raw, trade_raw


# ==========================================================================
# 2. DATA PREPARATION - FOOD CPI
# ==========================================================================
def prepare_cpi(cpi_raw, out):
    section("2. DATA PREPARATION - FOOD CPI")
    df = cpi_raw.copy()
    month_num = {m: i for i, m in enumerate(calendar.month_name) if m}
    df["Month"] = df["Months"].map(month_num)
    df["Date"] = pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=1))
    print("  Items in file:", df["Item"].unique().tolist())
    print("  Missing values in Value:", int(df["Value"].isna().sum()))
    print("  Duplicate (Item, Date) rows:", int(df.duplicated(["Item", "Date"]).sum()))
    print("  Flags:", df["Flag"].value_counts().to_dict(),
          " (X = external org., I = imputed, E = estimated)")

    idx = df[df["Item"].str.contains("Indices")].drop_duplicates("Date").set_index("Date")
    inf = df[df["Item"] == "Food price inflation"].drop_duplicates("Date").set_index("Date")
    cpi = pd.DataFrame({
        "cpi": idx["Value"].astype(float),
        "inflation_yoy": inf["Value"].astype(float),
        "flag": idx["Flag"],
    }).sort_index().asfreq("MS")

    n_missing = int(cpi[["cpi", "inflation_yoy"]].isna().sum().sum())
    if n_missing:
        print(f"  Interpolating {n_missing} missing values (time interpolation).")
        cpi[["cpi", "inflation_yoy"]] = cpi[["cpi", "inflation_yoy"]].interpolate(method="time")
    assert cpi.index.is_monotonic_increasing and cpi.index.freq is not None
    print(f"  Clean monthly series: {cpi.index.min():%b %Y} -> {cpi.index.max():%b %Y} "
          f"({len(cpi)} months, no gaps)")

    # ---- feature engineering -------------------------------------------------
    cpi["log_cpi"] = np.log(cpi["cpi"])
    cpi["growth"] = cpi["log_cpi"].diff()                      # monthly log growth
    cpi["mom_pct"] = cpi["cpi"].pct_change() * 100             # month-on-month %
    cpi["yoy_calc"] = cpi["cpi"].pct_change(12) * 100          # YoY from the index
    cpi["roll12"] = cpi["cpi"].rolling(12).mean()
    cpi["year"], cpi["month"] = cpi.index.year, cpi.index.month
    cpi["high_inflation"] = (cpi["inflation_yoy"] >= HIGH_INFLATION).astype(int)

    # sanity check: provided inflation vs inflation recomputed from the index
    both = cpi[["inflation_yoy", "yoy_calc"]].dropna()
    r = both.corr().iloc[0, 1]
    print(f"  Consistency check: corr(provided YoY inflation, YoY recomputed from index) = {r:.3f}")

    # ---- outliers (IQR rule on month-on-month change) -----------------------
    q1, q3 = cpi["mom_pct"].quantile([0.25, 0.75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    outl = cpi[(cpi["mom_pct"] < lo) | (cpi["mom_pct"] > hi)]
    print(f"  IQR outliers in MoM change (outside [{lo:.2f}, {hi:.2f}] %): {len(outl)}")
    if len(outl):
        print(outl[["cpi", "mom_pct"]].round(2).to_string())
    print("  Decision: KEEP them. A single sharp month-on-month rise is a plausible food-price shock "
          "(e.g. a vegetable price spike), not a data error; removing it would hide exactly the kind "
          "of event we want to forecast.")
    imp = cpi.index[cpi["flag"] == "I"]
    if len(imp):
        print(f"  WARNING: {len(imp)} index values ({imp[0]:%b %Y} - {imp[-1]:%b %Y}) are flagged 'I' "
              "(imputed by the data provider). The most recent 'truth' is therefore uncertain, and "
              "the held-out test block below falls inside this period.")

    cpi.to_csv(out / "results" / "cpi_clean_features.csv")
    return cpi


# ==========================================================================
# 3. CROP PRODUCTION
# ==========================================================================
def prepare_production(prod_raw, out):
    section("3. CROP PRODUCTION - CLEANING AND TRENDS")
    p = prod_raw[prod_raw["Item"].isin(CROPS)].copy()
    print("  Elements available:", p["Element"].unique().tolist())
    print("  Missing values (focus crops):", int(p["Value"].isna().sum()),
          "| duplicates:", int(p.duplicated(["Item", "Element", "Year"]).sum()))
    wide = p.pivot_table(index="Year", columns=["Element", "Item"], values="Value")
    wide = wide.interpolate(limit_direction="both")
    prod_mt = wide["Production"] / 1e6                      # million tonnes
    area_mha = wide["Area harvested"] / 1e6                 # million hectares
    yield_t_ha = wide["Yield"] / 1000                       # kg/ha -> t/ha
    prod_mt["Total cereals"] = prod_mt[CROPS].sum(axis=1)
    print(f"  Coverage {prod_mt.index.min()}-{prod_mt.index.max()}; "
          f"latest production (Mt): {prod_mt.iloc[-1].round(1).to_dict()}")

    # ---- growth statistics -----------------------------------------------------
    def cagr(s, y0, y1):
        return ((s.loc[y1] / s.loc[y0]) ** (1 / (y1 - y0)) - 1) * 100

    stat = pd.DataFrame({
        "CAGR 2000-2024 %": [cagr(prod_mt[c], 2000, 2024) for c in prod_mt.columns],
        "CAGR 2014-2024 %": [cagr(prod_mt[c], 2014, 2024) for c in prod_mt.columns],
        "Worst YoY %": [prod_mt[c].pct_change().min() * 100 for c in prod_mt.columns],
        "Worst year": [prod_mt[c].pct_change().idxmin() for c in prod_mt.columns],
    }, index=prod_mt.columns).round(2)
    print("\n  Production growth summary:\n" + stat.to_string())

    # ---- plots -----------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for c in CROPS:
        ax[0].plot(prod_mt.index, prod_mt[c], label=c, color=CROP_COLORS[c], lw=2)
        ax[1].plot(area_mha.index, area_mha[c], color=CROP_COLORS[c], lw=2)
        ax[2].plot(yield_t_ha.index, yield_t_ha[c], color=CROP_COLORS[c], lw=2)
    ax[0].set(title="Production (million tonnes)", xlabel="Year")
    ax[1].set(title="Area harvested (million ha)", xlabel="Year")
    ax[2].set(title="Yield (tonnes / ha)", xlabel="Year")
    ax[0].legend()
    fig.suptitle("India: rice, wheat and maize - production, area and yield", y=1.03, fontsize=14)
    save_fig(fig, "03_production_trends.png", out)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    gr = prod_mt[CROPS].pct_change().loc[2005:] * 100
    gr.plot(ax=ax, color=[CROP_COLORS[c] for c in CROPS], lw=1.6)
    ax.axhline(0, color="k", lw=0.8)
    ax.set(title="Year-on-year production growth (%)", xlabel="Year", ylabel="%")
    save_fig(fig, "03_production_growth.png", out)

    ratio = pd.DataFrame({
        "area x (2024/2000)": [area_mha.loc[2024, c] / area_mha.loc[2000, c] for c in CROPS],
        "yield x (2024/2000)": [yield_t_ha.loc[2024, c] / yield_t_ha.loc[2000, c] for c in CROPS],
        "production x (2024/2000)": [prod_mt.loc[2024, c] / prod_mt.loc[2000, c] for c in CROPS],
    }, index=CROPS).round(2)
    print("\n  Area vs yield contribution since 2000:\n" + ratio.to_string())
    print("  Insight: production of all three cereals trends upward. For rice and wheat most of the "
          "gain is higher yield (area grew only modestly); maize grew through BOTH more area and "
          "higher yield. Year-to-year growth stays volatile (weather/monsoon).")
    return prod_mt, area_mha, yield_t_ha


# ==========================================================================
# 4. AGRICULTURAL TRADE
# ==========================================================================
def prepare_trade(trade_raw, out):
    section("4. AGRICULTURAL TRADE - IMPORT / EXPORT ANALYSIS")
    t = trade_raw.copy()
    print("  Elements:", t["Element"].unique().tolist())
    print("  Missing:", int(t["Value"].isna().sum()), "| duplicates:",
          int(t.duplicated(["Item", "Element", "Year"]).sum()),
          "| flags:", t["Flag"].value_counts().to_dict())
    print("  Note: units differ by element (t, 1000 USD, head counts). We only combine "
          "elements with the SAME unit.")

    # ---- focus crops: quantity in tonnes -------------------------------------
    tm = t[t["Item"].isin(TRADE_ITEM.values())].copy()
    tm["Crop"] = tm["Item"].map({v: k for k, v in TRADE_ITEM.items()})
    c = tm[tm["Element"].isin(["Import quantity", "Export quantity"])]
    qty = c.pivot_table(index="Year", columns=["Crop", "Element"], values="Value") / 1e6
    val = tm[tm["Element"].isin(["Import value", "Export value"])]
    val = val.pivot_table(index="Year", columns=["Crop", "Element"], values="Value") / 1e6  # bn USD
    print("  Focus-crop trade items used:", TRADE_ITEM)

    summary = pd.DataFrame({
        crop: {
            "Import qty 2024 (Mt)": qty[(crop, "Import quantity")].iloc[-1],
            "Export qty 2024 (Mt)": qty[(crop, "Export quantity")].iloc[-1],
            "Net exports 2024 (Mt)": qty[(crop, "Export quantity")].iloc[-1]
                                     - qty[(crop, "Import quantity")].iloc[-1],
        } for crop in CROPS
    }).round(3)
    print("\n" + summary.to_string())

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for a, crop in zip(ax, CROPS):
        d = pd.DataFrame({"Import": qty[(crop, "Import quantity")],
                          "Export": qty[(crop, "Export quantity")]})
        d.plot.bar(ax=a, color=["#e76f51", "#2a9d8f"], width=0.8)
        a.set(title=f"{crop}: trade quantity (million t)", xlabel="Year")
    fig.suptitle("India: import vs export quantity of the three focus cereals", y=1.03, fontsize=14)
    save_fig(fig, "04_trade_quantity_focus_crops.png", out)

    # ---- all commodities: top 10 by value ---------------------------------------
    v = t[t["Element"].isin(["Import value", "Export value"]) & ~t["Item"].isin(DOUBLE_COUNT_ITEMS)]
    top = v.pivot_table(index="Item", columns="Element", values="Value", aggfunc="sum") / 1e6
    top.index = [textwrap.shorten(i, 36) for i in top.index]
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.5))
    top["Import value"].nlargest(10).sort_values().plot.barh(ax=ax[0], color="#e76f51")
    top["Export value"].nlargest(10).sort_values().plot.barh(ax=ax[1], color="#2a9d8f")
    ax[0].set(title="Top 10 imports by value, 2016-2024 (billion USD)")
    ax[1].set(title="Top 10 exports by value, 2016-2024 (billion USD)")
    save_fig(fig, "04_trade_top_commodities.png", out)

    # ---- trade balance over time -------------------------------------------------
    tv = v.pivot_table(
        index="Year", columns="Element", values="Value", aggfunc="sum") / 1e6
    tv["Net exports"] = tv["Export value"] - tv["Import value"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    tv[["Import value", "Export value"]].plot(ax=ax, marker="o", color=["#e76f51", "#2a9d8f"])
    ax.set(title="Agricultural trade value, sum of listed items (billion USD)", xlabel="Year")
    save_fig(fig, "04_trade_balance.png", out)

    print("\n  Top import items :", top["Import value"].nlargest(3).round(1).to_dict())
    print("  Top export items :", top["Export value"].nlargest(3).round(1).to_dict())
    print("  (The milled-equivalent rice aggregate is excluded from item rankings to avoid double counting.)")
    print("  Insight: rice is India's big export crop (net exporter); wheat trade is small; maize is a "
          "small net import. The largest import bills are edible oils (palm, soya, sunflower) and "
          "pulses (beans, lentils, pigeon peas) - import pressure sits OUTSIDE the three focus cereals.")
    return t, qty, val, tv


# ==========================================================================
# 5. EDA - FOOD CPI
# ==========================================================================
def eda_cpi(cpi, out):
    section("5. EXPLORATORY DATA ANALYSIS - FOOD CPI")
    print(cpi[["cpi", "inflation_yoy", "mom_pct"]].describe().round(2).to_string())

    # ---- Fig 1: level and inflation -----------------------------------------------
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(cpi.index, cpi["cpi"], color="#264653", lw=2, label="Food CPI")
    ax[0].plot(cpi.index, cpi["roll12"], color="#e76f51", lw=2, ls="--", label="12-month average")
    imputed = cpi["flag"] == "I"
    if imputed.any():
        ax[0].axvspan(cpi.index[imputed][0], cpi.index[-1], color="grey", alpha=0.15,
                      label="Imputed by source")
    ax[0].set(title="India Food CPI (2015 = 100)", ylabel="Index")
    ax[0].legend()
    ax[1].plot(cpi.index, cpi["inflation_yoy"], color="#2a9d8f", lw=2)
    ax[1].axhline(HIGH_INFLATION, color="red", ls="--", label=f"{HIGH_INFLATION:.0f}% threshold")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].fill_between(cpi.index, cpi["inflation_yoy"], HIGH_INFLATION,
                       where=cpi["inflation_yoy"] >= HIGH_INFLATION, color="red", alpha=0.2)
    ax[1].set(title="Food price inflation (YoY %)", ylabel="%")
    ax[1].legend()
    save_fig(fig, "05_cpi_level_and_inflation.png", out)

    # ---- Fig 2: seasonality --------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    sns.boxplot(data=cpi.dropna(subset=["mom_pct"]), x="month", y="mom_pct", ax=ax[0],
                color="#8ecae6")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set(title="Month-on-month change by calendar month", xlabel="Month", ylabel="MoM %")
    hm = cpi.pivot_table(index="year", columns="month", values="mom_pct")
    sns.heatmap(hm, cmap="RdYlGn_r", center=0, annot=True, fmt=".1f", cbar_kws={"label": "MoM %"},
                ax=ax[1], annot_kws={"size": 7})
    ax[1].set(title="MoM % change: year x month heat-map", xlabel="Month", ylabel="Year")
    save_fig(fig, "05_cpi_seasonality.png", out)

    # ---- Fig 3: distributions and autocorrelation -----------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    sns.histplot(cpi["mom_pct"].dropna(), kde=True, ax=ax[0], color="#2a9d8f")
    ax[0].set(title="Distribution of MoM change (%)")
    sns.histplot(cpi["inflation_yoy"], kde=True, ax=ax[1], color="#e76f51")
    ax[1].set(title="Distribution of YoY inflation (%)")
    a_level = acf(cpi["cpi"], 24)
    a_growth = acf(cpi["growth"], 24)
    ax[2].bar(np.arange(25) - 0.2, a_level, width=0.4, label="CPI level", color="#264653")
    ax[2].bar(np.arange(25) + 0.2, a_growth, width=0.4, label="Monthly growth", color="#e9c46a")
    band = 1.96 / np.sqrt(len(cpi))
    ax[2].axhline(band, color="red", ls="--", lw=0.8)
    ax[2].axhline(-band, color="red", ls="--", lw=0.8)
    ax[2].set(title="Autocorrelation (red = 95% noise band)", xlabel="Lag (months)")
    ax[2].legend()
    save_fig(fig, "05_cpi_distribution_acf.png", out)

    # ---- Written insights (computed, not hard-coded) --------------------------------
    peak = cpi["inflation_yoy"].idxmax()
    trough = cpi["inflation_yoy"].idxmin()
    seas = cpi.groupby("month")["mom_pct"].mean()
    high_share = cpi["high_inflation"].mean() * 100
    print(f"\n  Insight 1: CPI rose from {cpi['cpi'].iloc[0]:.1f} to {cpi['cpi'].iloc[-1]:.1f} "
          f"({(cpi['cpi'].iloc[-1] / cpi['cpi'].iloc[0] - 1) * 100:.0f}% in ~10 years): "
          "a strong upward trend, so the level is non-stationary.")
    print(f"  Insight 2: inflation peaked at {cpi['inflation_yoy'].max():.1f}% in {peak:%b %Y} and "
          f"was lowest ({cpi['inflation_yoy'].min():.1f}%) in {trough:%b %Y}.")
    print(f"  Insight 3: {high_share:.0f}% of months had food inflation >= {HIGH_INFLATION:.0f}%.")
    print(f"  Insight 4: seasonality - average MoM change is highest in "
          f"{calendar.month_name[int(seas.idxmax())]} ({seas.max():.2f}%) and lowest in "
          f"{calendar.month_name[int(seas.idxmin())]} ({seas.min():.2f}%).")
    print(f"  Insight 5: ACF of the LEVEL stays near 1 for many lags (trend); monthly GROWTH has "
          f"lag-1 autocorrelation {a_growth[1]:.2f}, so modelling growth is the sensible route.")


# ==========================================================================
# 6. INFERENTIAL STATISTICS & HYPOTHESIS TESTS
# ==========================================================================
def build_annual(cpi, prod_mt, qty, val):
    """Join the three datasets on Year (analysis window 2016-2024)."""
    a = cpi.groupby("year").agg(cpi_mean=("cpi", "mean"), infl_mean=("inflation_yoy", "mean"))
    a = a.loc[2016:2024]
    pg = prod_mt.pct_change() * 100                       # growth uses full history
    a["rice_prod_g"] = pg["Rice"]
    a["wheat_prod_g"] = pg["Wheat"]
    a["maize_prod_g"] = pg["Maize (corn)"]
    a["cereal_prod_g"] = pg["Total cereals"]
    a["cereal_prod_g_lag1"] = pg["Total cereals"].shift(1)
    a["wheat_prod_g_lag1"] = pg["Wheat"].shift(1)
    imp = sum(qty[(c, "Import quantity")] for c in CROPS)
    exp_ = sum(qty[(c, "Export quantity")] for c in CROPS)
    a["cereal_import_qty"] = imp
    a["cereal_export_qty"] = exp_
    a["import_qty_g"] = imp.pct_change() * 100
    a["export_qty_g"] = exp_.pct_change() * 100
    a["net_export_qty"] = exp_ - imp
    return a


def corr_row(x, y, label):
    d = pd.concat([x, y], axis=1).dropna()
    r, p = stats.pearsonr(d.iloc[:, 0], d.iloc[:, 1])
    rho, ps = stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])
    return {"predictor": label, "n": len(d), "pearson_r": r, "pearson_p": p,
            "spearman_rho": rho, "spearman_p": ps}


def inferential(cpi, annual, out):
    section("6. INFERENTIAL STATISTICS AND HYPOTHESIS TESTING")
    tests = []
    alpha = 0.05

    def record(name, h0, stat_name, stat, p, note=""):
        decision = "Reject H0" if p < alpha else "Fail to reject H0"
        tests.append({"test": name, "H0": h0, "statistic": f"{stat_name}={stat:.3f}",
                      "p_value": p, "decision": decision, "note": note})
        print(f"  {name}\n     H0: {h0}\n     {stat_name} = {stat:.3f}, p = {p:.4f}  ->  {decision}"
              + (f"  ({note})" if note else ""))

    # --- confidence interval for mean inflation ------------------------------------------
    x = cpi["inflation_yoy"]
    lo, hi = stats.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=stats.sem(x))
    print(f"  Mean YoY food inflation = {x.mean():.2f}%  (95% CI {lo:.2f}% .. {hi:.2f}%)\n")

    # --- normality & variance checks (decide parametric vs non-parametric) -------------
    sw_stat, sw_p = stats.shapiro(cpi["mom_pct"].dropna())
    record("T0 Shapiro-Wilk normality of MoM change", "MoM change is normally distributed",
           "W", sw_stat, sw_p, "non-normal -> prefer non-parametric tests" if sw_p < alpha else "")

    # --- T1: pre vs post COVID inflation -------------------------------------------------
    pre = cpi.loc["2016":"2019", "inflation_yoy"]
    post = cpi.loc["2022":, "inflation_yoy"]
    lev_stat, lev_p = stats.levene(pre, post)
    t_stat, t_p = stats.ttest_ind(post, pre, equal_var=lev_p >= alpha)
    u_stat, u_p = stats.mannwhitneyu(post, pre, alternative="two-sided")
    print(f"  (Levene equal-variance p={lev_p:.3f}; using "
          f"{'pooled' if lev_p >= alpha else 'Welch'} t-test)")
    record("T1 t-test: mean YoY inflation 2022+ vs 2016-2019", "mean inflation equal in both eras",
           "t", t_stat, t_p, f"means {post.mean():.2f}% vs {pre.mean():.2f}%")
    record("T1b Mann-Whitney U (robustness)", "inflation distributions identical in both eras",
           "U", u_stat, u_p)

    # --- T2: seasonality ------------------------------------------------------------------
    groups = [g["mom_pct"].dropna().values for _, g in cpi.groupby("month")]
    f_stat, f_p = stats.f_oneway(*groups)
    h_stat, h_p = stats.kruskal(*groups)
    record("T2 One-way ANOVA: MoM change across calendar months", "mean MoM change same in all 12 months",
           "F", f_stat, f_p)
    record("T2b Kruskal-Wallis (robustness)", "MoM distributions same in all 12 months",
           "H", h_stat, h_p)

    # --- T3: trend ------------------------------------------------------------------------
    tt = np.arange(len(cpi))
    lr = stats.linregress(tt, cpi["log_cpi"])
    ann = (np.exp(lr.slope * 12) - 1) * 100
    record("T3 Linear trend in log CPI", "slope of log CPI on time = 0", "t",
           lr.slope / lr.stderr, lr.pvalue, f"implied trend growth {ann:.1f}% per year, R2={lr.rvalue**2:.3f}")

    # --- T4: correlation of production / trade with inflation (annual) ----------------
    predictors = {
        "Rice production growth": annual["rice_prod_g"],
        "Wheat production growth": annual["wheat_prod_g"],
        "Maize production growth": annual["maize_prod_g"],
        "Total cereal production growth": annual["cereal_prod_g"],
        "Total cereal production growth (prev. year)": annual["cereal_prod_g_lag1"],
        "Wheat production growth (prev. year)": annual["wheat_prod_g_lag1"],
        "Cereal import-quantity growth": annual["import_qty_g"],
        "Cereal export-quantity growth": annual["export_qty_g"],
    }
    corr = pd.DataFrame([corr_row(v, annual["infl_mean"], k) for k, v in predictors.items()])
    print("\n  T4 Correlation with annual mean food inflation (annual data, small n!):")
    print(corr.round(3).to_string(index=False))
    corr.to_csv(out / "results" / "correlations_vs_inflation.csv", index=False)
    n_tests = len(corr)
    print(f"  Bonferroni-adjusted alpha for {n_tests} tests = {alpha / n_tests:.4f}")
    best = corr.loc[corr["pearson_p"].idxmin()]
    record("T4 Pearson correlation (strongest predictor)",
           f"no linear relation between '{best['predictor']}' and inflation",
           "r", best["pearson_r"], best["pearson_p"],
           f"n={int(best['n'])}; Bonferroni alpha={alpha / n_tests:.4f}")

    # --- T5: production-decline years vs other years --------------------------------------
    decl = annual["cereal_prod_g"] < 0
    g1, g2 = annual.loc[decl, "infl_mean"], annual.loc[~decl, "infl_mean"]
    if len(g1) >= 2 and len(g2) >= 2:
        u, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        record("T5 Mann-Whitney: inflation in cereal-decline years vs other years",
               "inflation is the same whether production fell or rose", "U", u, p,
               f"{len(g1)} decline years vs {len(g2)}")
    else:
        print(f"  T5 skipped: only {len(g1)} year(s) with falling cereal production in 2016-2024 - "
              "too few for a valid test.")

    # --- correlation heat-map ---------------------------------------------------------------
    cols = ["infl_mean", "rice_prod_g", "wheat_prod_g", "maize_prod_g", "cereal_prod_g",
            "import_qty_g", "export_qty_g"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(annual[cols].corr(method="spearman"), annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax)
    ax.set_title("Spearman correlations (annual, 2016-2024)")
    save_fig(fig, "06_correlation_heatmap.png", out)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for a, col, lab in zip(ax, ["cereal_prod_g", "wheat_prod_g_lag1", "import_qty_g"],
                           ["Cereal production growth %", "Wheat production growth % (prev. yr)",
                            "Cereal import-quantity growth %"]):
        d = annual[[col, "infl_mean"]].dropna()
        sns.regplot(data=d, x=col, y="infl_mean", ax=a, color="#264653")
        for yr, r_ in d.iterrows():
            a.annotate(str(yr), (r_[col], r_["infl_mean"]), fontsize=8, xytext=(3, 3),
                       textcoords="offset points")
        a.set(xlabel=lab, ylabel="Mean YoY food inflation %")
    fig.suptitle("Production/trade growth vs food inflation (one point per year)", y=1.03)
    save_fig(fig, "06_scatter_production_trade_vs_inflation.png", out)

    t_df = pd.DataFrame(tests)
    t_df.to_csv(out / "results" / "hypothesis_tests.csv", index=False)
    print("\n  Interpretation: strong, significant findings are about the CPI series itself "
          "(clear upward trend; inflation regime differs by era). Links between annual cereal "
          "production/trade and inflation are weak/inconclusive with only 8-9 annual points - "
          "we report them as 'not established', NOT as proof of no effect.")
    return t_df


# ==========================================================================
# 7. LINEAR REGRESSION (annual explanatory model)
# ==========================================================================
def annual_linear_regression(annual, out):
    section("7. LINEAR REGRESSION - DO PRODUCTION / TRADE EXPLAIN FOOD INFLATION?")
    feats = ["cereal_prod_g", "cereal_prod_g_lag1", "import_qty_g"]
    d = annual[feats + ["infl_mean"]].dropna()
    X, y = d[feats], d["infl_mean"]
    print(f"  Sample: {len(d)} years ({d.index.min()}-{d.index.max()}); validation = Leave-One-Out CV")

    model = Pipeline([("sc", StandardScaler()), ("lr", LinearRegression())])
    dummy = DummyRegressor(strategy="mean")
    loo = LeaveOneOut()
    p_model = cross_val_predict(model, X, y, cv=loo)
    p_dummy = cross_val_predict(dummy, X, y, cv=loo)
    rm = np.sqrt(np.mean((y - p_model) ** 2))
    rd = np.sqrt(np.mean((y - p_dummy) ** 2))
    model.fit(X, y)
    coefs = pd.Series(model.named_steps["lr"].coef_, index=feats)
    print("  Standardised coefficients (effect of +1 SD on inflation, in % points):")
    print(coefs.round(3).to_string())
    print(f"  LOO-CV RMSE  linear model: {rm:.2f}   |  mean-only baseline: {rd:.2f}")
    print(f"  In-sample R2: {r2_score(y, model.predict(X)):.3f}   LOO R2: {r2_score(y, p_model):.3f}")
    verdict = ("The regression beats the baseline out-of-sample."
               if rm < rd else
               "The regression does NOT beat predicting the average -> annual production/trade "
               "growth carries no reliable out-of-sample signal for inflation in this sample.")
    print("  Verdict:", verdict)
    return {"loo_rmse": rm, "baseline_rmse": rd}


# ==========================================================================
# 8. FORECASTING MODELS
# ==========================================================================
FEATURES = ["g_lag1", "g_lag2", "g_lag3", "g_lag12", "g_mean3", "g_mean12", "sin_m", "cos_m"]
MIN_T = 13  # first index with every lag available (needs g[t-12] with g[0] = NaN)


def month_of(i, start_month):
    return (start_month - 1 + i) % 12 + 1


def feature_row(g, t, start_month):
    """Features to predict growth g[t] using ONLY information up to t-1 (no leakage)."""
    m = month_of(t, start_month)
    return np.array([g[t - 1], g[t - 2], g[t - 3], g[t - 12],
                     np.mean(g[t - 3:t]), np.mean(g[t - 12:t]),
                     np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)])


def make_xy(g, t_start, t_end, start_month):
    ts = range(max(t_start, MIN_T), t_end)
    X = np.vstack([feature_row(g, t, start_month) for t in ts])
    y = np.array([g[t] for t in ts])
    return X, y


def recursive_forecast(model, cpi_hist, h, start_month):
    """Multi-step forecast: predicted growth is fed back as a lag for the next step."""
    log_c = list(np.log(cpi_hist))
    g = [np.nan] + list(np.diff(log_c))
    for _ in range(h):
        t = len(g)
        gp = float(model.predict(feature_row(np.array(g), t, start_month).reshape(1, -1))[0])
        g.append(gp)
        log_c.append(log_c[-1] + gp)
    return np.exp(log_c[-h:])


def backtest(predict_fn, cpi_arr, origins, h):
    """Rolling-origin evaluation: for each origin, forecast h steps and compare with truth."""
    rows = []
    for o in origins:
        pred = predict_fn(cpi_arr[: o + 1])
        for k in range(h):
            rows.append({"origin": o, "h": k + 1, "actual": cpi_arr[o + 1 + k], "pred": pred[k]})
    return pd.DataFrame(rows)


class AverageModel:
    """Simple ensemble: average the predictions of several fitted regressors."""

    def __init__(self, models):
        self.models = models

    def get_params(self, deep=False):
        return {"models": self.models}

    def set_params(self, **p):
        self.models = p.get("models", self.models)
        return self

    def fit(self, X, y):
        self.models = [clone(m).fit(X, y) for m in self.models]
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)


def tune_models(X, y):
    """Hyper-parameter tuning with time-ordered cross-validation (no shuffling)."""
    tscv = TimeSeriesSplit(n_splits=5)
    cands = {
        "Linear Regression": (Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]), {}),
        "Ridge": (Pipeline([("sc", StandardScaler()), ("m", Ridge())]),
                  {"m__alpha": np.logspace(-3, 3, 13)}),
        "Random Forest": (RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
                          {"max_depth": [2, 3, 5], "min_samples_leaf": [3, 5, 10]}),
    }
    best = {}
    for name, (est, grid) in cands.items():
        if grid:
            gs = GridSearchCV(est, grid, cv=tscv, scoring="neg_mean_squared_error")
            gs.fit(X, y)
            best[name] = gs.best_estimator_
            print(f"    {name:<18} best params {gs.best_params_}  CV RMSE(growth)={np.sqrt(-gs.best_score_):.5f}")
        else:
            sc = []
            for tr, te in tscv.split(X):
                m = clone(est).fit(X[tr], y[tr])
                sc.append(np.mean((m.predict(X[te]) - y[te]) ** 2))
            best[name] = clone(est)
            print(f"    {name:<18} (no tuning)           CV RMSE(growth)={np.sqrt(np.mean(sc)):.5f}")
    best["Ensemble (avg of 3)"] = AverageModel(list(best.values()))
    return best


def forecasting(cpi, out):
    section("8. FORECASTING THE FOOD CPI")
    cpi_arr = cpi["cpi"].values
    g = cpi["growth"].values
    n = len(cpi)
    start_month = int(cpi.index[0].month)
    test_start, val_start = n - TEST_MONTHS, n - TEST_MONTHS - VAL_MONTHS
    print(f"  Observations: {n}")
    print(f"  Train      : {cpi.index[0]:%b %Y} .. {cpi.index[val_start - 1]:%b %Y}")
    print(f"  Validation : {cpi.index[val_start]:%b %Y} .. {cpi.index[test_start - 1]:%b %Y}  (model selection)")
    print(f"  Test       : {cpi.index[test_start]:%b %Y} .. {cpi.index[-1]:%b %Y}  (touched once, at the end)")
    print("  Method: model the monthly GROWTH of the index (stationary) from its own lags + "
          "calendar month; rebuild the index by compounding. Multi-step forecasts are recursive.\n"
          "  Leakage control: features at time t use only data up to t-1; split is chronological; "
          "tuning uses TimeSeriesSplit inside the training block.")

    # ---- baselines ----------------------------------------------------------------------
    def naive(hist):                 # "no change"
        return np.repeat(hist[-1], HORIZON)

    def drift12(hist):               # extend the average monthly growth of the last 12 months
        gr = (hist[-1] / hist[-13]) ** (1 / 12)
        return hist[-1] * gr ** np.arange(1, HORIZON + 1)

    def seasonal_yoy(hist):          # same month last year, scaled by the latest YoY ratio
        ratio = hist[-1] / hist[-13]
        return np.array([hist[len(hist) - 12 + k] for k in range(HORIZON)]) * ratio

    BASELINES = {"Baseline: naive": naive, "Baseline: drift (12m)": drift12,
                 "Baseline: seasonal x YoY": seasonal_yoy}

    # ---- tune on TRAIN only ----------------------------------------------------------------
    Xtr, ytr = make_xy(g, MIN_T, val_start, start_month)
    print(f"\n  Tuning on training block ({len(ytr)} samples):")
    tuned = tune_models(Xtr, ytr)
    fitted_tr = {k: clone(v).fit(Xtr, ytr) for k, v in tuned.items()}

    val_origins = list(range(val_start - 1, test_start - HORIZON))
    test_origins = list(range(test_start - 1, n - HORIZON))
    print(f"\n  Rolling-origin backtests: {len(val_origins)} validation origins, "
          f"{len(test_origins)} test origins, each forecasting {HORIZON} months.")

    # ---- validation: choose the model -------------------------------------------------------
    val_res = {k: backtest(f, cpi_arr, val_origins, HORIZON) for k, f in BASELINES.items()}
    for k, m in fitted_tr.items():
        val_res[k] = backtest(lambda hist, m=m: recursive_forecast(m, hist, HORIZON, start_month),
                              cpi_arr, val_origins, HORIZON)
    val_tab = pd.DataFrame({k: regression_metrics(v["actual"], v["pred"]) for k, v in val_res.items()}).T
    print("\n  VALIDATION results (6-step recursive forecasts):\n" + val_tab.round(3).to_string())
    ml_names = list(fitted_tr)
    chosen = val_tab.loc[ml_names, "MAE"].idxmin()
    print(f"\n  Selected model (lowest validation MAE among the ML models): {chosen}")

    # ---- test: refit on train+val, evaluate once --------------------------------------------
    Xtv, ytv = make_xy(g, MIN_T, test_start, start_month)
    fitted_tv = {k: clone(v).fit(Xtv, ytv) for k, v in tuned.items()}
    test_res = {k: backtest(f, cpi_arr, test_origins, HORIZON) for k, f in BASELINES.items()}
    for k, m in fitted_tv.items():
        test_res[k] = backtest(lambda hist, m=m: recursive_forecast(m, hist, HORIZON, start_month),
                               cpi_arr, test_origins, HORIZON)
    test_tab = pd.DataFrame({k: regression_metrics(v["actual"], v["pred"]) for k, v in test_res.items()}).T
    print("\n  TEST results (held-out last 12 months, 6-step recursive forecasts):\n"
          + test_tab.round(3).to_string())

    comp = val_tab.add_prefix("val_").join(test_tab.add_prefix("test_"))
    comp.to_csv(out / "results" / "model_comparison.csv")
    return {
        "n": n, "start_month": start_month, "cpi_arr": cpi_arr, "g": g, "tuned": tuned,
        "chosen": chosen, "val_tab": val_tab, "test_tab": test_tab, "test_res": test_res,
        "test_origins": test_origins, "test_start": test_start, "fitted_tv": fitted_tv,
        "Xtv": Xtv, "ytv": ytv, "val_start": val_start,
    }


# ==========================================================================
# 9. EVALUATION AND ERROR ANALYSIS
# ==========================================================================
def evaluate(cpi, F, out):
    section("9. EVALUATION AND ERROR ANALYSIS")
    test_res, test_tab = F["test_res"], F["test_tab"]
    names = list(test_res)

    # ---- error by horizon -------------------------------------------------------------------
    by_h = pd.DataFrame({k: v.assign(ae=(v["pred"] - v["actual"]).abs()).groupby("h")["ae"].mean()
                         for k, v in test_res.items()})
    print("  Test MAE by forecast horizon (index points):\n" + by_h.round(2).to_string())

    # ---- relative skill vs. the best baseline -------------------------------------------
    base_names = [k for k in test_tab.index if k.startswith("Baseline")]
    base_mae = test_tab.loc[base_names, "MAE"].min()
    skill = (1 - test_tab["MAE"] / base_mae) * 100
    print("\n  Skill vs best baseline (positive = better than baseline, % MAE reduction):")
    print(skill.round(1).to_string())
    best_test = test_tab["MAE"].idxmin()
    print(f"\n  Best on test: {best_test}.  Selected on validation: {F['chosen']}.")
    if skill[F["chosen"]] < 0:
        print(f"  HONEST CONCLUSION: the selected model does NOT beat the best baseline on the held-out "
              f"test block ({skill[F['chosen']]:.0f}% MAE vs baseline). Validation and test disagree, "
              "which is common with ~100 monthly points, a regime change in 2025-26, and test data "
              "that the source itself imputed. We still report the model's forecast, but the 6-month "
              "numbers must be read with the wide uncertainty band, and the baseline is a serious rival.")
    else:
        print("  The selected model beats every baseline on the held-out test block.")

    # ---- bias --------------------------------------------------------------------------------
    ch = test_res[F["chosen"]]
    bias = (ch["pred"] - ch["actual"]).mean()
    print(f"  Mean error (bias) of {F['chosen']}: {bias:+.2f} index points "
          f"({'over' if bias > 0 else 'under'}-forecasting on average)")

    # ---- worst origins -----------------------------------------------------------------------
    ch = ch.assign(err=ch["pred"] - ch["actual"])
    ch["target_date"] = [cpi.index[o + h] for o, h in zip(ch["origin"], ch["h"])]
    worst = ch.reindex(ch["err"].abs().sort_values(ascending=False).index).head(5)
    print("\n  Five largest errors of the selected model:")
    print(worst[["target_date", "h", "actual", "pred", "err"]].round(2).to_string(index=False))

    # ---- plots ---------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    by_h.plot(ax=ax[0], marker="o")
    ax[0].set(title="Test MAE by forecast horizon", xlabel="Months ahead", ylabel="Index points")

    ax[1].plot(cpi.index[F["test_start"]:], cpi["cpi"].iloc[F["test_start"]:], "k-o", lw=2, label="Actual")
    h1 = ch[ch["h"] == 1].set_index("target_date")["pred"]
    last_o = F["test_origins"][-1]
    p6 = ch[ch["origin"] == last_o].set_index("target_date")["pred"]
    ax[1].plot(h1.index, h1, "--s", color="#2a9d8f", label="1-step forecasts")
    ax[1].plot(p6.index, p6, "--^", color="#e76f51", label=f"6-step from {cpi.index[last_o]:%b %Y}")
    ax[1].set(title=f"Test period - {F['chosen']}", ylabel="Food CPI")
    ax[1].legend()
    ax[1].tick_params(axis="x", rotation=30)

    sns.histplot(ch["err"], kde=True, ax=ax[2], color="#264653")
    ax[2].axvline(0, color="red")
    ax[2].set(title="Forecast error distribution (test)", xlabel="Predicted - actual")
    save_fig(fig, "09_evaluation.png", out)

    # ---- feature importance / coefficients ---------------------------------------------------------
    model = F["fitted_tv"][F["chosen"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=FEATURES)
        title = "Random Forest feature importance"
    else:
        coef = model.named_steps["m"].coef_ if hasattr(model, "named_steps") else \
            np.mean([m.named_steps["m"].coef_ for m in model.models if hasattr(m, "named_steps")], axis=0)
        imp = pd.Series(coef * 100, index=FEATURES)
        title = f"{F['chosen']}: effect of +1 SD on monthly growth (% points)"
    imp.sort_values().plot.barh(ax=ax, color="#2a9d8f")
    ax.set_title(title)
    save_fig(fig, "09_feature_importance.png", out)
    print("\n  Feature view (" + title + "):\n" + imp.sort_values(ascending=False).round(3).to_string())

    print(textwrap.dedent("""
    Limitations and bias
      * Only ~10 years of monthly data and a 12-month test block -> metrics are noisy.
      * The final months of the series are IMPUTED by the source, so recent 'truth' is uncertain.
      * Recursive forecasts accumulate error; accuracy falls with the horizon (see table above).
      * Models use only the CPI's own history. Production/trade are annual and end in 2024, so they
        cannot be used as monthly inputs for 2026 forecasts without further assumptions.
      * Shocks (monsoon failure, export bans, COVID) are not predictable from past CPI alone.
    Next steps
      * Add monthly rainfall, fuel prices and exchange-rate data as exogenous variables.
      * Try SARIMA / ETS / gradient boosting and a larger rolling-origin cross-validation.
      * Forecast sub-indices (cereals, vegetables, pulses) rather than the headline food index.
    """))
    return by_h


# ==========================================================================
# 10. LOGISTIC REGRESSION - HIGH INFLATION CLASSIFIER
# ==========================================================================
def logistic_classifier(cpi, out):
    section(f"10. LOGISTIC REGRESSION - WILL FOOD INFLATION BE >= {HIGH_INFLATION:.0f}% IN {CLF_HORIZON} MONTHS?")
    n = len(cpi)
    infl = cpi["inflation_yoy"].values
    g = cpi["growth"].values
    H = CLF_HORIZON
    rows, ys, ts = [], [], []
    for t in range(3, n - H):
        m = cpi.index[t].month
        rows.append([infl[t], infl[t] - infl[t - 3], np.mean(g[t - 2:t + 1]) * 100,
                     np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)])
        ys.append(int(infl[t + H] >= HIGH_INFLATION))
        ts.append(t)
    names = ["inflation_now", "inflation_change_3m", "avg_mom_growth_3m", "sin_month", "cos_month"]
    X, y, ts = np.array(rows), np.array(ys), np.array(ts)

    n_test = 24
    test_first = ts[-n_test]
    tr = ts + H < test_first            # training labels end before the test block starts
    te = ts >= test_first
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    print(f"  Class balance - train: {ytr.mean() * 100:.0f}% high | test: {yte.mean() * 100:.0f}% high "
          f"(train {len(ytr)} / test {len(yte)} samples; gap of {H} months prevents label overlap)")

    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED))])
    gs = GridSearchCV(pipe, {"lr__C": [0.01, 0.1, 1, 10, 100]},
                      cv=TimeSeriesSplit(n_splits=5), scoring="balanced_accuracy")
    gs.fit(Xtr, ytr)
    clf = gs.best_estimator_
    print(f"  Best C = {gs.best_params_['lr__C']}  (time-series CV balanced accuracy {gs.best_score_:.3f})")

    pred, proba = clf.predict(Xte), clf.predict_proba(Xte)[:, 1]
    persist = (X[te][:, 0] >= HIGH_INFLATION).astype(int)         # "same state as today"
    majority = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr).predict(Xte)

    def metrics(yt, yp):
        return {"accuracy": accuracy_score(yt, yp), "balanced_acc": balanced_accuracy_score(yt, yp),
                "precision": precision_score(yt, yp, zero_division=0),
                "recall": recall_score(yt, yp, zero_division=0), "F1": f1_score(yt, yp, zero_division=0)}

    tab = pd.DataFrame({"Logistic regression": metrics(yte, pred),
                        "Baseline: persistence": metrics(yte, persist),
                        "Baseline: majority class": metrics(yte, majority)}).T
    print("\n" + tab.round(3).to_string())
    auc = np.nan
    if len(np.unique(yte)) == 2:
        auc = roc_auc_score(yte, proba)
        print(f"  ROC-AUC (logistic): {auc:.3f}")
    else:
        print("  ROC-AUC not defined: only one class present in the test block.")

    coefs = pd.Series(clf.named_steps["lr"].coef_[0], index=names)
    print("\n  Odds ratio per +1 SD of each feature (>1 raises P(high inflation)):")
    print(np.exp(coefs).round(2).to_string())

    cm = confusion_matrix(yte, pred, labels=[0, 1])
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[0], xticklabels=["Normal", "High"],
                yticklabels=["Normal", "High"])
    ax[0].set(title="Confusion matrix (test)", xlabel="Predicted", ylabel="Actual")
    if not np.isnan(auc):
        fpr, tpr, _ = roc_curve(yte, proba)
        ax[1].plot(fpr, tpr, color="#e76f51", lw=2, label=f"AUC = {auc:.2f}")
        ax[1].plot([0, 1], [0, 1], "k--")
        ax[1].set(title="ROC curve (test)", xlabel="False positive rate", ylabel="True positive rate")
        ax[1].legend()
    ax[2].plot(cpi.index[ts[te] + H], yte, "k-", label="Actual high-inflation flag", drawstyle="steps-post")
    ax[2].plot(cpi.index[ts[te] + H], proba, color="#2a9d8f", lw=2, label="P(high inflation)")
    ax[2].axhline(0.5, color="grey", ls="--")
    ax[2].set(title="Predicted probability vs outcome (test)")
    ax[2].legend()
    ax[2].tick_params(axis="x", rotation=30)
    save_fig(fig, "10_logistic_classifier.png", out)
    tab.to_csv(out / "results" / "classifier_metrics.csv")
    verdict = ("beats" if tab.loc["Logistic regression", "balanced_acc"] >
               tab.loc["Baseline: persistence", "balanced_acc"] else "does not beat")
    print(f"\n  Interpretation: the logistic model {verdict} the persistence baseline on balanced "
          "accuracy. Inflation regimes are sticky, so 'today's state' is a strong predictor; "
          "recall of the high class matters most if the goal is early warning.")


# ==========================================================================
# 11. FINAL FORECAST AND DASHBOARD
# ==========================================================================
def final_forecast(cpi, F, prod_mt, t_all, tv, out):
    section("11. FINAL 6-MONTH FORECAST AND DASHBOARD")
    n, start_month = F["n"], F["start_month"]
    g, cpi_arr = F["g"], F["cpi_arr"]
    X_all, y_all = make_xy(g, MIN_T, n, start_month)
    model = clone(F["tuned"][F["chosen"]]).fit(X_all, y_all)       # refit on ALL data
    fc = recursive_forecast(model, cpi_arr, HORIZON, start_month)

    # Out-of-fold one-step residuals (time-ordered) -> uncertainty band that widens with sqrt(h)
    res = []
    for tr_i, te_i in TimeSeriesSplit(n_splits=5).split(X_all):
        m = clone(F["tuned"][F["chosen"]]).fit(X_all[tr_i], y_all[tr_i])
        res.extend(y_all[te_i] - m.predict(X_all[te_i]))
    sigma = np.std(res, ddof=1)
    steps = np.arange(1, HORIZON + 1)
    log_fc = np.log(fc)
    lower, upper = np.exp(log_fc - 1.96 * sigma * np.sqrt(steps)), np.exp(log_fc + 1.96 * sigma * np.sqrt(steps))
    dates = pd.date_range(cpi.index[-1] + pd.offsets.MonthBegin(1), periods=HORIZON, freq="MS")
    fdf = pd.DataFrame({"forecast_cpi": fc, "lower_95": lower, "upper_95": upper}, index=dates)
    fdf.index.name = "date"
    fdf["implied_yoy_%"] = [(fc[i] / cpi["cpi"].get(d - pd.DateOffset(years=1), np.nan) - 1) * 100
                            for i, d in enumerate(dates)]
    fdf.to_csv(out / "results" / "forecast_next_6_months.csv")
    print(f"  Final model: {F['chosen']} (refit on all {n} months). Band = +/-1.96 * sigma_1step * sqrt(h), "
          f"sigma={sigma:.4f} (log growth, out-of-fold).")
    print(fdf.round(2).to_string())
    print(f"\n  Reading: from {cpi['cpi'].iloc[-1]:.1f} in {cpi.index[-1]:%b %Y} the model projects "
          f"{fc[-1]:.1f} by {dates[-1]:%b %Y} (95% band {lower[-1]:.1f}-{upper[-1]:.1f}). "
          "The band is wide: treat this as a scenario, not a guarantee.")

    # ---- forecast figure --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(cpi.index, cpi["cpi"], color="#264653", lw=2, label="Historical Food CPI")
    ax.plot([cpi.index[-1]] + list(fdf.index), [cpi["cpi"].iloc[-1]] + list(fdf["forecast_cpi"]),
            color="#e76f51", lw=2.5, marker="o", label="Forecast")
    ax.fill_between(fdf.index, fdf["lower_95"], fdf["upper_95"], color="#e76f51", alpha=0.2, label="95% band")
    ax.set(title="India Food CPI: history and 6-month forecast", ylabel="Index (2015 = 100)")
    ax.legend()
    save_fig(fig, "11_forecast.png", out)

    # ---- 4-panel dashboard ----------------------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(17, 11))
    for c in CROPS:
        ax[0, 0].plot(prod_mt.loc[2000:].index, prod_mt.loc[2000:, c], lw=2, label=c, color=CROP_COLORS[c])
    ax[0, 0].set(title="1. Agricultural production (million tonnes)", xlabel="Year")
    ax[0, 0].legend()

    v = t_all[t_all["Element"].isin(["Import value", "Export value"])
              & ~t_all["Item"].isin(DOUBLE_COUNT_ITEMS)].pivot_table(
        index="Item", columns="Element", values="Value", aggfunc="sum") / 1e6
    v["tot"] = v.sum(axis=1)
    v = v.nlargest(8, "tot").drop(columns="tot")
    v.index = [textwrap.shorten(i, 28) for i in v.index]
    v[["Import value", "Export value"]].sort_values("Export value").plot.barh(
        ax=ax[0, 1], color=["#e76f51", "#2a9d8f"])
    ax[0, 1].set(title="2. Imports vs exports by value, 2016-2024 (bn USD)")

    ax[1, 0].plot(cpi.index, cpi["cpi"], color="#264653", lw=2, label="Food CPI")
    ax[1, 0].set(title="3. Food CPI and food inflation", ylabel="Index")
    tw = ax[1, 0].twinx()
    tw.plot(cpi.index, cpi["inflation_yoy"], color="#e9c46a", lw=1.5, label="YoY inflation %")
    tw.axhline(HIGH_INFLATION, color="red", ls="--", lw=0.8)
    tw.set_ylabel("YoY %")
    tw.grid(False)

    ax[1, 1].plot(cpi.index[-48:], cpi["cpi"].iloc[-48:], color="#264653", lw=2, label="History")
    ax[1, 1].plot([cpi.index[-1]] + list(fdf.index), [cpi["cpi"].iloc[-1]] + list(fdf["forecast_cpi"]),
                  color="#e76f51", lw=2.5, marker="o", label="Forecast")
    ax[1, 1].fill_between(fdf.index, fdf["lower_95"], fdf["upper_95"], color="#e76f51", alpha=0.2)
    ax[1, 1].set(title="4. Future food CPI prediction (6 months)", ylabel="Index")
    ax[1, 1].legend()
    fig.suptitle("India food-price dashboard: production, trade, inflation and forecast",
                 fontsize=17, y=1.0)
    save_fig(fig, "00_dashboard.png", out)
    return fdf


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=here / "data", type=Path)
    ap.add_argument("--out-dir", default=here / "outputs", type=Path)
    args = ap.parse_args()
    out = args.out_dir
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)

    cpi_raw, prod_raw, trade_raw = load_data(args.data_dir)
    cpi = prepare_cpi(cpi_raw, out)
    prod_mt, _, _ = prepare_production(prod_raw, out)
    t_all, qty, val, tv = prepare_trade(trade_raw, out)
    eda_cpi(cpi, out)
    annual = build_annual(cpi, prod_mt, qty, val)
    annual.round(3).to_csv(out / "results" / "annual_merged_dataset.csv")
    inferential(cpi, annual, out)
    annual_linear_regression(annual, out)
    F = forecasting(cpi, out)
    evaluate(cpi, F, out)
    logistic_classifier(cpi, out)
    final_forecast(cpi, F, prod_mt, t_all, tv, out)
    print("\nDone. Figures ->", out / "figures", "| tables ->", out / "results")


if __name__ == "__main__":
    main()
