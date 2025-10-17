import pandas as pd
import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings("ignore")

# ============ Argument setup ============
parser = argparse.ArgumentParser(description="Sales Forecasting Model (No Synthetic Data)")
parser.add_argument("--csv", type=str, help="Path to your CSV data file")
parser.add_argument("--date_col", type=str, default="Date", help="Name of the date column")
parser.add_argument("--target_col", type=str, default="Sales", help="Name of the target column (sales)")
args = parser.parse_args()

# 👇 Default data path for users
# NOTE: Please create a folder named 'your_data' and place your CSV file inside it.
# Example: your_data/data.csv
csv_path = args.csv if args.csv else os.path.join('your_data', 'data.csv')

# ============ Check if the file exists ============
if not os.path.exists(csv_path):
    raise FileNotFoundError(
        f"❌ Data file not found.\n"
        f"Please place your CSV file here: {csv_path}\n"
        f"Or pass it directly using: --csv path/to/your_file.csv"
    )

# ============ Load data ============
print(f"📂 Loading data from: {csv_path}")
data = pd.read_csv(csv_path)

if args.date_col not in data.columns:
    raise KeyError(f"❌ Date column '{args.date_col}' not found in CSV. Available columns: {list(data.columns)}")

if args.target_col not in data.columns:
    raise KeyError(f"❌ Target column '{args.target_col}' not found in CSV. Available columns: {list(data.columns)}")

# ============ Data preparation ============
data[args.date_col] = pd.to_datetime(data[args.date_col])
data = data.sort_values(by=args.date_col)

# Time-based features
data['month'] = data[args.date_col].dt.month
data['day'] = data[args.date_col].dt.day
data['dayofweek'] = data[args.date_col].dt.dayofweek

# Lag and rolling features
data['lag1'] = data[args.target_col].shift(1)
data['lag7'] = data[args.target_col].shift(7)
data['rolling_mean_7'] = data[args.target_col].shift(1).rolling(7).mean()
data = data.dropna()

# ============ Split data ============
X = data[['month', 'day', 'dayofweek', 'lag1', 'lag7', 'rolling_mean_7']]
y = data[args.target_col]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# ============ Train model ============
print("🚀 Training LightGBM model...")
model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(50)]
)

# ============ Evaluate ============
y_pred = model.predict(X_val)
metrics = {
    "MAE": mean_absolute_error(y_val, y_pred),
    "RMSE": mean_squared_error(y_val, y_pred, squared=False),
    "MAPE%": np.mean(np.abs((y_val - y_pred) / y_val)) * 100
}
print("✅ Validation metrics:", metrics)

# ============ Save model ============
model_filename = "sales_model_no_synth.pkl"
joblib.dump(model, model_filename)
print(f"💾 Model saved as: {model_filename}")

# ============ Forecast next 30 days ============
future_steps = 30
last_row = data.iloc[-1:].copy()
forecasts = []

for step in range(future_steps):
    next_row = last_row.copy()
    next_row['month'] = (last_row[args.date_col] + pd.Timedelta(days=1)).dt.month.values[0]
    next_row['day'] = (last_row[args.date_col] + pd.Timedelta(days=1)).dt.day.values[0]
    next_row['dayofweek'] = (last_row[args.date_col] + pd.Timedelta(days=1)).dt.dayofweek.values[0]
    pred = model.predict(next_row[['month', 'day', 'dayofweek', 'lag1', 'lag7', 'rolling_mean_7']])[0]
    next_row[args.target_col] = pred
    forecasts.append([next_row[args.date_col].values[0], pred])
    last_row = next_row.copy()

forecast_df = pd.DataFrame(forecasts, columns=[args.date_col, f"{args.target_col}_forecast"])
forecast_df.to_csv("forecast_output_no_synth.csv", index=False)
print("📊 Forecasts saved to forecast_output_no_synth.csv")
print("🎉 Done successfully!")
