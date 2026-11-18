"""Shared settings and helpers for the taxi-tips EDA notebooks.

Folder layout (relative to the repo root):
    data/raw/        original TLC parquet files + taxi_zone_lookup.csv
    data/processed/  small full-data summary tables (made by notebook 01)
    data/sample/     250K-trip sample per month (made by notebook 01)
    figures/         saved charts
    docs/            cleaning log and tables for the report
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data"
PROCESSED = ROOT / "data" / "processed"
SAMPLE = ROOT / "data" / "sample"
FIGURES = ROOT / "figures"
DOCS = ROOT / "docs"
for p in (PROCESSED, SAMPLE, FIGURES, DOCS):
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- study design
MONTHS = ["2022-02", "2022-12", "2023-02", "2023-12",
          "2024-02", "2024-12", "2025-02", "2025-12"]
EDA_YEARS = [2022, 2023, 2024]      # main EDA period
TEST_YEAR = 2025                    # profiled separately (Step 7)
N_PER_MONTH = 250_000
SEED = 42

AIRPORT_ZONES = {1, 132, 138}       # Newark, JFK, LaGuardia
UNKNOWN_ZONES = {264, 265}          # Unknown / Outside of NYC

# Holiday periods used for the holiday flag
DEC_HOLIDAY_START_DAY = 24          # Dec 24-31
PRESIDENTS_DAY = {2022: "2022-02-21", 2023: "2023-02-20",
                  2024: "2024-02-19", 2025: "2025-02-17"}

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DIST_BINS = [0, 1, 3, 7, 15, np.inf]
DIST_LABELS = ["0-1", "1-3", "3-7", "7-15", "15+"]

# ---------------------------------------------------------------- cleaning rules
# Change values here; every notebook reads them from this one place.
RULES = {
    "min_fare": 2.50,       # NYC base fare was $2.50 until Dec 2022, $3.00 after
    "max_distance": 100,    # miles
    "min_duration": 1,      # minutes
    "max_duration": 180,    # minutes
    "max_speed": 80,        # mph
    "max_tip_pct": 100,     # % of fare
}

KEEP_COLS = ["VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
             "passenger_count", "trip_distance", "RatecodeID",
             "PULocationID", "DOLocationID", "payment_type", "fare_amount",
             "extra", "mta_tax", "tip_amount", "tolls_amount",
             "improvement_surcharge", "total_amount", "congestion_surcharge",
             "airport_fee", "cbd_congestion_fee"]


def load_month(month):
    """Read one raw month and give every year the same column names."""
    df = pd.read_parquet(RAW / f"yellow_tripdata_{month}.parquet")
    df = df.rename(columns={"Airport_fee": "airport_fee"})
    if "cbd_congestion_fee" not in df.columns:      # only exists from 2025
        df["cbd_congestion_fee"] = np.nan
    return df[KEEP_COLS]


def clean_month(df, month):
    """Apply the cleaning rules in order.

    Returns the cleaned card-paid trips and a list of (rule, rows_left)
    so we can build the cleaning funnel.
    """
    funnel = [("raw rows", len(df))]

    start = pd.Timestamp(month + "-01")
    end = start + pd.offsets.MonthBegin(1)
    df = df[(df.tpep_pickup_datetime >= start) & (df.tpep_pickup_datetime < end)]
    funnel.append(("pickup inside the month", len(df)))

    df = df[df.payment_type == 1]
    funnel.append(("card-paid (payment_type = 1)", len(df)))

    df = df[df.fare_amount >= RULES["min_fare"]]
    funnel.append((f"fare >= ${RULES['min_fare']:.2f}", len(df)))

    df = df[(df.tip_amount >= 0)]
    funnel.append(("tip >= 0", len(df)))

    df = df[(df.trip_distance > 0) & (df.trip_distance <= RULES["max_distance"])]
    funnel.append((f"0 < distance <= {RULES['max_distance']} mi", len(df)))

    dur = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
    df = df.assign(duration_min=dur)
    df = df[(df.duration_min >= RULES["min_duration"]) &
            (df.duration_min <= RULES["max_duration"])]
    funnel.append((f"{RULES['min_duration']}-{RULES['max_duration']} min duration", len(df)))

    df = df.assign(speed_mph=df.trip_distance / (df.duration_min / 60))
    df = df[df.speed_mph <= RULES["max_speed"]]
    funnel.append((f"speed <= {RULES['max_speed']} mph", len(df)))

    df = df[~df.PULocationID.isin(UNKNOWN_ZONES) & ~df.DOLocationID.isin(UNKNOWN_ZONES)]
    funnel.append(("known pickup & drop-off zone", len(df)))

    df = df.assign(tip_pct=df.tip_amount / df.fare_amount * 100)
    df = df[df.tip_pct <= RULES["max_tip_pct"]]
    funnel.append((f"tip % <= {RULES['max_tip_pct']}", len(df)))

    return df, funnel


def load_zones():
    return pd.read_csv(RAW / "taxi_zone_lookup.csv")


def add_features(df, zones):
    """Time, trip and location features (no weather for now)."""
    df = df.copy()
    t = df.tpep_pickup_datetime
    df["year"] = t.dt.year
    df["month_num"] = t.dt.month
    df["month_name"] = np.where(df.month_num == 2, "Feb", "Dec")
    df["day"] = t.dt.day
    df["date"] = t.dt.normalize()
    df["hour"] = t.dt.hour
    df["dow"] = t.dt.dayofweek                       # 0 = Monday
    df["is_weekend"] = df.dow >= 5
    df["is_night"] = (df.hour >= 22) | (df.hour < 6)

    # holiday flag: Dec 24-31, or Presidents' Day long weekend (Sat-Mon)
    pres = df.year.map(PRESIDENTS_DAY).pipe(pd.to_datetime)
    in_pres = (df.month_num == 2) & (df.date >= pres - pd.Timedelta(days=2)) & (df.date <= pres)
    in_xmas = (df.month_num == 12) & (df.day >= DEC_HOLIDAY_START_DAY)
    df["is_holiday"] = in_pres | in_xmas

    df["is_airport"] = df.PULocationID.isin(AIRPORT_ZONES) | df.DOLocationID.isin(AIRPORT_ZONES)
    df["same_zone"] = df.PULocationID == df.DOLocationID
    df["dist_bucket"] = pd.cut(df.trip_distance, DIST_BINS, labels=DIST_LABELS, right=False)

    # tip as % of everything paid except the tip (for the preset-button check)
    pre_tip = df.total_amount - df.tip_amount
    df["tip_pct_total"] = np.where(pre_tip > 0, df.tip_amount / pre_tip * 100, np.nan)
    df["paid_cbd_fee"] = df.cbd_congestion_fee.fillna(0) > 0   # congestion pricing (2025 only)
    df["zero_tip"] = df.tip_amount == 0
    df["tip_20plus"] = df.tip_pct >= 20

    z = zones.set_index("LocationID")
    df["pu_borough"] = df.PULocationID.map(z.Borough)
    df["pu_zone"] = df.PULocationID.map(z.Zone)
    df["pu_service_zone"] = df.PULocationID.map(z.service_zone)
    df["do_borough"] = df.DOLocationID.map(z.Borough)

    # smaller dtypes to save memory
    for c in ["trip_distance", "fare_amount", "tip_amount", "tip_pct", "tip_pct_total",
              "duration_min", "speed_mph", "total_amount", "tolls_amount"]:
        df[c] = df[c].astype("float32")
    return df


# ---------------------------------------------------------------- stats helpers
def summarize(df, by, value="tip_pct"):
    """Mean with 95% CI, plus zero-tip and 20%+ shares, for each group."""
    g = df.groupby(by, observed=True)
    out = g[value].agg(n="size", mean="mean", sd="std", median="median")
    out["ci95"] = 1.96 * out["sd"] / np.sqrt(out["n"])
    out["zero_tip_pct"] = g["zero_tip"].mean() * 100
    out["tip20_pct"] = g["tip_20plus"].mean() * 100
    return out.reset_index()


def diff_ci(a, b):
    """Difference in means (b - a) with a 95% CI and Cohen's d."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = b.mean() - a.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return {"diff": diff, "ci_low": diff - 1.96 * se, "ci_high": diff + 1.96 * se,
            "cohens_d": diff / pooled if pooled > 0 else np.nan}


# ---------------------------------------------------------------- plotting
# Feb and Dec keep the same colors in every chart.
FEB = "#2a78d6"     # blue
DEC = "#eb6834"     # orange
MONTH_COLORS = {"Feb": FEB, "Dec": DEC}
YEAR_COLORS = {2022: "#2a78d6", 2023: "#eb6834", 2024: "#1baf7a"}
HIGHLIGHT = "#e34948"   # 2025 when compared with earlier years
NEUTRAL = "#a3a29c"
INK = "#0b0b0b"
INK_2 = "#52514e"


def set_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelcolor": INK_2,
        "axes.edgecolor": "#c9c8c2", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True, "axes.grid.axis": "y",
        "grid.color": "#e8e7e2", "grid.linewidth": 0.8,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "lines.linewidth": 2, "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def save_fig(fig, name):
    fig.savefig(FIGURES / f"{name}.png")


def grouped_bar(ax, table, cat, group, value, colors, err=None, order=None, groups=None):
    """Side-by-side bars: one cluster per `cat`, one bar per `group`."""
    order = order or list(dict.fromkeys(table[cat]))
    groups = groups or list(colors)
    width = 0.8 / len(groups)
    x = np.arange(len(order))
    for i, g in enumerate(groups):
        sub = table[table[group] == g].set_index(cat).reindex(order)
        ax.bar(x + (i - (len(groups) - 1) / 2) * width, sub[value], width * 0.92,
               color=colors[g], label=str(g),
               yerr=None if err is None else sub[err],
               error_kw={"elinewidth": 1, "ecolor": INK_2, "capsize": 2})
    ax.set_xticks(x, order)
    ax.legend(ncol=len(groups), loc="lower right", bbox_to_anchor=(1, 1.0))
    return ax
