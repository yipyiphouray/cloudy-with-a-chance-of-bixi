import pandas as pd, pickle
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / 'data' 

RAW_DIR = DATA_DIR / 'raw'

PROCESSED_DIR = DATA_DIR / 'processed'

MODEL_DIR = PROJECT_ROOT / 'models'

DT_COL = "starttime_hourly"

p = Path(MODEL_DIR / "hgb_BIXI_DemandForecast_model_v1.pkl")
print("MODEL PATH:", p)
print("SIZE (bytes):", p.stat().st_size)

# Load model
model_path = MODEL_DIR / "hgb_BIXI_DemandForecast_model_v1.pkl"
model = joblib.load(model_path)
print("✅ Loaded with joblib:", type(model))

# Load model_df
df = pd.read_parquet(PROCESSED_DIR/'model_df.parquet')
print("model_df loaded:", df.shape)
print("Has DT col:", DT_COL in df.columns)
print("Columns (first 30):", df.columns.tolist()[:30])

# Load forecast_2025
bt = pd.read_parquet(PROCESSED_DIR/'forecasting_df.parquet')
print("forecast_2025 loaded:", bt.shape)
print("Columns (first 30):", bt.columns.tolist()[:30])
