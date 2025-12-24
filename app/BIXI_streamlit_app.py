import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path
import os
import traceback

st.set_page_config(page_title="BIXI Demand Forecast", layout="wide")

# -----------------------------
#PATHS
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / 'data' 

RAW_DIR = DATA_DIR / 'raw'

PROCESSED_DIR = DATA_DIR / 'processed'

MODEL_DIR = PROJECT_ROOT / 'models'




st.markdown("""
### What does this app do?

This dashboard forecasts **hourly bike demand at individual BIXI stations in Montréal**.

It uses:
- Historical station usage
- Temporal patterns (hour, weekday, seasonality)
- Weather conditions
- Short-term demand inertia (lags & rolling averages)

You can now:
- Predict demand for a **single hour**
- Generate a **24-hour rolling forecast**
- Inspect **2025 backtesting performance**
""")

# datetime column confirmed 
DT_COL = "starttime_hourly"

# -----------------------------
# final feature set
# -----------------------------
FEATURES = [
    "temperature", "precipitation", "wind_speed",
    "latitude", "longitude", "hour", "day_of_week", "is_weekend", "month",
    "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
    "feels_like_temp", "is_raining", "lag_1h", "lag_24h", "rolling_3h", "rolling_24h"
]
TARGET = "total_demand"


# -----------------------------
# Loaders
# -----------------------------
@st.cache_data(show_spinner=False)
def load_station_lookup(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["startstationname","latitude","longitude"])
    df["startstationname"] = df["startstationname"].astype("category")
    df["latitude"] = df["latitude"].astype("float32")
    df["longitude"] = df["longitude"].astype("float32")
    return df

@st.cache_data(show_spinner=False)
def load_station_history(path: Path, station_name: str) -> pd.DataFrame:
    keep_cols = ["startstationname","starttime_hourly","total_demand","latitude","longitude"]
    df = pd.read_parquet(
        path,
        columns=keep_cols,
        engine="pyarrow",
        filters=[("startstationname", "==", station_name)],
    )
    df["starttime_hourly"] = pd.to_datetime(df["starttime_hourly"])
    df["total_demand"] = df["total_demand"].astype("float32")
    df["latitude"] = df["latitude"].astype("float32")
    df["longitude"] = df["longitude"].astype("float32")
    return df.sort_values("starttime_hourly")

@st.cache_data(show_spinner=False)
def load_model_df(path: Path) -> pd.DataFrame:
    # only keep what's needed for the app
    keep_cols = [
        "startstationname", "starttime_hourly", "total_demand",
        "latitude", "longitude",
        # needed for building features from history:
        "lag_1h", "lag_24h", "rolling_3h", "rolling_24h",
    ]
    df = pd.read_parquet(path, columns=keep_cols)

    # lighter types
    df["startstationname"] = df["startstationname"].astype("category")
    df["latitude"] = df["latitude"].astype("float32")
    df["longitude"] = df["longitude"].astype("float32")
    df["total_demand"] = df["total_demand"].astype("float32")

    df["starttime_hourly"] = pd.to_datetime(df["starttime_hourly"])
    return df

@st.cache_data(show_spinner=False)
def load_forecast_2025_for_station(path: Path, station_name: str) -> pd.DataFrame:
    keep_cols = ["startstationname", "starttime_hourly", "total_demand", "y_pred"]

    # IMPORTANT: use raw station name for pushdown first
    df = pd.read_parquet(
        path,
        columns=keep_cols,
        engine="pyarrow",
        filters=[("startstationname", "==", station_name)],
    )

    # types
    df["starttime_hourly"] = pd.to_datetime(df["starttime_hourly"])
    return df


# -----------------------------
# Helpers
# -----------------------------
import re

def normalize_station_name(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.strip()
         .str.lower()
         .str.replace(r"\s+", " ", regex=True)
    )


def add_time_features(ts: pd.Timestamp) -> dict:
    hour = int(ts.hour)
    dow = int(ts.dayofweek)  # Mon=0
    month = int(ts.month)
    is_weekend = int(dow >= 5)

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)

    return {
        "hour": hour,
        "day_of_week": dow,
        "is_weekend": is_weekend,
        "month": month,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "day_of_week_sin": dow_sin,
        "day_of_week_cos": dow_cos,
    }


def compute_feels_like_temp(temp_c: float, wind_speed: float) -> float:
    # Your prior definition
    return float(temp_c - 0.7 * wind_speed)


def infer_station_col(df: pd.DataFrame) -> str:
    for col in ["startstationname", "station", "station_name", "name"]:
        if col in df.columns:
            return col
    raise ValueError("Could not find station name column. Expected one of: "
                     "startstationname, station, station_name, name")


def get_station_history(df: pd.DataFrame, station_col: str, station_name: str) -> pd.DataFrame:
    out = df[df[station_col] == station_name].copy()
    out[DT_COL] = pd.to_datetime(out[DT_COL])
    out = out.sort_values(DT_COL)
    return out


def latest_context_for_timestamp(hist: pd.DataFrame, ts: pd.Timestamp):
    """
    Compute lag/rolling from ACTUAL history up to ts-1h.
    """
    cutoff = ts - pd.Timedelta(hours=1)
    past = hist[hist[DT_COL] <= cutoff].copy()
    if past.empty:
        return None

    s = past.set_index(DT_COL)[TARGET].sort_index()

    lag_1h = s.iloc[-1] if len(s) >= 1 else np.nan

    t24 = ts - pd.Timedelta(hours=24)
    if t24 in s.index:
        lag_24h = s.loc[t24]
    else:
        older = s[s.index <= t24]
        lag_24h = older.iloc[-1] if len(older) else np.nan

    rolling_3h = s.tail(3).mean() if len(s) else np.nan
    rolling_24h = s.tail(24).mean() if len(s) else np.nan

    return float(lag_1h), float(lag_24h), float(rolling_3h), float(rolling_24h)


def build_feature_row(station_lat: float, station_lon: float, ts: pd.Timestamp, weather: dict, context: tuple) -> pd.DataFrame:
    lag_1h, lag_24h, rolling_3h, rolling_24h = context

    temp = float(weather["temperature"])
    precip = float(weather["precipitation"])
    wind = float(weather["wind_speed"])
    is_raining = int(precip > 0)

    row = {
        "temperature": temp,
        "precipitation": precip,
        "wind_speed": wind,
        "latitude": float(station_lat),
        "longitude": float(station_lon),
        **add_time_features(ts),
        "feels_like_temp": compute_feels_like_temp(temp, wind),
        "is_raining": is_raining,
        "lag_1h": float(lag_1h),
        "lag_24h": float(lag_24h),
        "rolling_3h": float(rolling_3h),
        "rolling_24h": float(rolling_24h),
    }
    return pd.DataFrame([row], columns=FEATURES)


def recursive_24h_forecast(model, hist: pd.DataFrame, station_lat: float, station_lon: float, start_ts: pd.Timestamp, weather: dict):
    """
    Recursive 24h rollout. Uses past actuals + prior predictions as context.
    """
    past = hist.copy()
    past[DT_COL] = pd.to_datetime(past[DT_COL])
    past = past.set_index(DT_COL)[TARGET].sort_index()

    preds = []
    for h in range(24):
        ts = start_ts + pd.Timedelta(hours=h)

        cutoff = ts - pd.Timedelta(hours=1)
        known = past[past.index <= cutoff]
        if known.empty:
            raise ValueError("Not enough history to compute lag features for this station/time.")

        lag_1h = float(known.iloc[-1])

        t24 = ts - pd.Timedelta(hours=24)
        if t24 in past.index:
            lag_24h = float(past.loc[t24])
        else:
            older = past[past.index <= t24]
            lag_24h = float(older.iloc[-1]) if len(older) else lag_1h

        rolling_3h = float(known.tail(3).mean())
        rolling_24h = float(known.tail(24).mean())

        X = build_feature_row(
            station_lat=station_lat,
            station_lon=station_lon,
            ts=ts,
            weather=weather,
            context=(lag_1h, lag_24h, rolling_3h, rolling_24h),
        )

        yhat = float(model.predict(X)[0])
        yhat = max(0.0, yhat)

        preds.append({"timestamp": ts, "predicted_demand": yhat})

        # inject prediction into series for next steps
        past.loc[ts] = yhat

    return pd.DataFrame(preds)




def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# -----------------------------
# UI
# -----------------------------
st.title("🚲 BIXI — Hourly Station Demand Forecast")

# Load assets
try:
    model = joblib.load(MODEL_DIR / "hgb_BIXI_DemandForecast_model_v1.pkl")
    st.success("✅ Model loaded")
except Exception as e:
    st.error("❌ Model load failed")
    st.code(traceback.format_exc())
    st.stop()
try:
    model_df = load_model_df(PROCESSED_DIR / "model_df.parquet")
    try:
        stations_df = load_station_lookup(PROCESSED_DIR / "stations.parquet")
        st.success("✅ stations lookup loaded")
    except Exception:
        st.error("❌ stations.parquet load failed (expected at data/processed/stations.parquet)")
        st.code(traceback.format_exc())
        st.stop()

    st.success("✅ model_df loaded")
except Exception as e:
    st.error("❌ model_df load failed")
    st.code(traceback.format_exc())
    st.stop()

# Station selection + coords (from tiny stations.parquet)
stations = stations_df["startstationname"].astype(str).sort_values().tolist()
station = st.selectbox("Select station", stations)

row = stations_df.loc[stations_df["startstationname"].astype(str) == station].iloc[0]
station_lat = float(row["latitude"])
station_lon = float(row["longitude"])

# Station history (choose ONE approach)
station_hist = load_station_history(PROCESSED_DIR / "model_df.parquet", station)

with st.expander("Manual weather inputs", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        temperature = st.number_input("Temperature (°C)", value=10.0, step=0.5)
    with c2:
        precipitation = st.number_input("Precipitation (mm)", value=0.0, step=0.5, min_value=0.0)
    with c3:
        wind_speed = st.number_input("Wind speed (m/s)", value=3.0, step=0.5, min_value=0.0)

weather = {"temperature": temperature, "precipitation": precipitation, "wind_speed": wind_speed}

tab1, tab2, tab3 = st.tabs(["Single prediction", "24-hour forecast", "Backtesting (2025)"])


# -----------------------------
# Single
# -----------------------------
with tab1:
    st.subheader("Single prediction (one hour)")

    d = st.date_input("Date", value=pd.Timestamp.now().date(), key="single_date")
    t = st.time_input(
        "Time (hour)",
        value=pd.Timestamp.now().replace(minute=0, second=0, microsecond=0).time(),
        key="single_time"
    )
    ts = pd.Timestamp.combine(d, t)

    if st.button("Predict this hour"):
        ctx = latest_context_for_timestamp(station_hist, ts)
        if ctx is None:
            st.error("Not enough history for this station/time to compute lag/rolling features.")
        else:
            X = build_feature_row(station_lat, station_lon, ts, weather, ctx)
            yhat = float(model.predict(X)[0])
            yhat = max(0.0, yhat)

            st.metric("Predicted demand (trips)", f"{yhat:.2f}")
            with st.expander("Show feature row"):
                st.dataframe(X)


# -----------------------------
# 24h forecast
# -----------------------------
with tab2:
    st.subheader("24-hour forecast (recursive)")

    d2 = st.date_input("Start date", value=pd.Timestamp.now().date(), key="day_date")
    start_hour = st.selectbox("Start hour", list(range(24)), index=8)
    start_ts = pd.Timestamp.combine(
        pd.Timestamp(d2),
        datetime.strptime(f"{start_hour:02d}:00", "%H:%M").time()
    )

    if st.button("Generate 24-hour forecast"):
        try:
            fc = recursive_24h_forecast(model, station_hist, station_lat, station_lon, start_ts, weather)
            st.dataframe(fc)
            st.line_chart(fc.set_index("timestamp")["predicted_demand"])
        except Exception as e:
            st.error(str(e))


# -----------------------------
# Backtesting
# -----------------------------
with tab3:
    st.subheader("Backtesting results (from forecast_2025.parquet)")

    try:
        actual_col = "total_demand"
        pred_col = "y_pred"

        bt_station = load_forecast_2025_for_station(
            PROCESSED_DIR / "forecast_2025.parquet",
            station
        ).copy()

        bt_station = bt_station.dropna(subset=[actual_col, pred_col])

        if bt_station.empty:
            st.warning("No rows found for this station in forecast_2025.parquet (or missing actual/pred columns).")
        else:
            c1, c2 = st.columns(2)
            c1.metric("MAE", f"{mae(bt_station[actual_col], bt_station[pred_col]):.3f}")
            c2.metric("RMSE", f"{rmse(bt_station[actual_col], bt_station[pred_col]):.3f}")

            plot_df = bt_station.sort_values(DT_COL).set_index(DT_COL)[[actual_col, pred_col]]
            plot_df.columns = ["actual", "predicted"]
            st.line_chart(plot_df)

            bt_station["date"] = bt_station[DT_COL].dt.date
            by_day = bt_station.groupby("date").apply(
                lambda g: pd.Series({
                    "MAE": mae(g[actual_col], g[pred_col]),
                    "RMSE": rmse(g[actual_col], g[pred_col]),
                    "n_hours": len(g),
                })
            ).reset_index().sort_values("date")

            st.dataframe(by_day)

    except Exception:
        st.error("Backtesting tab couldn't load or parse forecast_2025.parquet.")
        st.code(traceback.format_exc())
