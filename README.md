# ⛈️ **Cloudy with a Chance of BIXI** 🚲
Hourly Bike-Sharing Demand Forecasting for BIXI Montréal

End-to-end machine learning project that forecasts hourly bike demand per station to support rebalancing and capacity planning.

Key result:
MAE ≈ 2 trips per station per hour on 2025 out-of-sample data

## ⚡ **TL;DR (30 seconds)**

Problem: Stations run empty or full → lost trips & operational inefficiencies

Solution: Tree-based ML forecasting hourly demand at the station level

Data: Trips + weather + time + spatial features

Final model: Histogram-based Gradient Boosting

Performance: MAE ≈ 2 trips/hour, R² ≈ 0.65 (2025 backtest)

Interpretability: SHAP explains demand drivers

Outcome: Actionable forecasts for rebalancing & planning

## 🚀 **Quick Start**

1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/yipyiphouray/cloudy-with-a-chance-of-bixi.git
cd cloudy_with_a_chance_of_bixi

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app/BIXI_streamlit_app.py
'''

## 📁 Project Structure
'''
├── app/                          # Streamlit dashboard source code
├── data/                         # Minimal processed artifacts required for the live demo
├── figures/                      # Model evaluation plots used in reports and documentation
│   ├── BIXI_SHAP_PLOT.png        # Feature importance and directional impact analysis
│   ├── BIXI_Feature_Importance   # Global feature ranking
│   ├── Residual_Distribution     # Error analysis and model bias check
│   └── OneWeekTimeSeries         # Comparative visualization of actual vs. predicted demand
├── models/                       # Serialized model binaries
│   ├── hgb_BIXI_model_v1.pkl     # Final Gradient Boosting model (Lightweight/Production)
│   └── rf_BIXI_model_v1.pkl      # Random Forest model 
├── notebooks/                    # End-to-end data science pipeline
│   ├── 01_Data Cleaning.ipynb    # Raw BIXI trip data processing and aggregation
│   ├── 02_Initial EDA.ipynb      # Exploration of ridership trends and seasonality
│   ├── 03_Feature Engineering.ipynb # Sinusoidal encoding, lags, and weather integration
│   ├── 04_Post-FE_EDA.ipynb      # EDA for Post Feature Engineering
│   ├── 05_Modeling.ipynb         # Model training, hyperparameter tuning, and selection
│   └── 06_Backtesting_forecast.ipynb      # Performance validation on out-of-sample 2025 data
├── report/                       # Formal documentation
│   └── BIXI_Full_Report.pdf      # Detailed business and technical project report
├── .gitignore                    # Prevents large datasets and temporary files from being committed
├── README.md                     # Project overview and instructions
└── requirements.txt              # Environment dependencies for reproducibility
'''

## 📊 Data & Model Note

This repo includes the **minimal artifacts required to run the Streamlit demo**:

- `data/processed/model_df.parquet` (~61MB)
- `data/processed/forecast_2025.parquet` (~54MB)
- `models/hgb_BIXI_DemandForecast_model_v1.pkl` (~1.5MB)

Large raw/processed datasets are excluded to keep the repo lightweight:

- `data/raw/*.csv` (2–3GB each)
- `data/processed/bixi_trip_data*.parquet` (300–600MB)
- `models/rf_*.pkl` (≈385MB)

### To Run Locally:
* **Download Raw Data:** Visit the [BIXI Open Data Portal](https://bixi.com/en/open-data) and place the CSVs in `data/raw/`.  !!! Remember to name them as BIXI_Trip_XXXX.csv where XXXX is the year of the dataset.  !! Weather API does not need an API key !!
* **Reproduce:** Run the notebooks in order (`01` to `04`) to generate the processed files.
* **App Performance:** The Streamlit app is pre-configured to use the **Histogram-based Gradient Boosting (HGB)** model, which is included in the repo (1.4MB).

## 🧠 **Feature Engineering (Highlights)**

Temporal patterns

Hour, day-of-week, month

Cyclical encoding transforms periodic features into 2D space, ensuring the model perceives the distance between 23:00 and 00:00 as 1 hour rather than 23 hours

Demand inertia

Lagged demand (1h, 24h)

Rolling averages (3h, 24h)

Weather effects

Feels-like temperature

Rain indicator

Spatial context

Latitude & longitude (neighborhood effects)

All lag and rolling features are computed strictly within each station’s time series to prevent leakage.

## 🤖 **Models & Performance**
| Model | MAE | RMSE | $R^2$ | Training Time |
| :--- | :--- | :--- | :--- | :--- |
| Baseline (Mean) | 3.22 | 4.60 | $-0.001$ | ~0 s |
| Random Forest | 2.05 | 3.15 | 0.645 | ~9 min |
| **HistGradientBoost (Final)** | **2.04** | **3.14** | **0.648** | **~40 s** |

✔ Same accuracy as Random Forest
✔ ~13× faster training
➡ Selected for production practicality

## 🔍 **Model Interpretability (SHAP)**

Key drivers of hourly demand:

Recent usage dominates (strong temporal persistence)

Clear daily cycles (commute patterns)

Spatial effects matter (downtown vs residential)

Weather is secondary

Calendar effects are minor

Demand follows stable routines; weather adjusts demand but rarely overrides time-based patterns.

![SHAP Global Summary](figures/BIXI_SHAP_PLOT.png)

## 📈 **2025 Walk-Forward Forecast (Backtest)**

The final model was applied to 2025 data in a walk-forward forecasting setup.

2025 Performance:

MAE: 1.97

RMSE: 3.12

R²: 0.646

This confirms strong generalization to unseen future periods.

![2025 Forecast Performance](figures/OneWeekTimeSeriesTotalSystem.png)

## 💼 **Why This Matters**

With an average error of ~2 trips per station per hour, forecasts can support:

🚚 Proactive rebalancing

🏗️ Capacity planning

👥 Staffing & logistics optimization

At peak hours (10–40 trips/hour), this level of accuracy is operationally meaningful.

## ⚠️ **Limitations & Future Work**

### Limitations

No explicit event data (festivals, disruptions)

Rebalancing actions not modeled

Hourly (not real-time) resolution

### Future improvements

Event & transit disruption features

Richer spatial context (zones, transit proximity)

Shorter time intervals (15-min forecasting)

## 🛠 **Tech Stack**

Python · pandas · scikit-learn · SHAP · Meteostat API ·
numpy · matplotlib · seaborn · Streamlit

## 📄 **Full Report**

📂 report/BIXI_Demand_Forecasting_Full_Report.pdf