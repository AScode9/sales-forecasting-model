# 🧠 Sales Forecasting Model (No Synthetic Data)

This project builds a **predictive sales forecasting model** using your own dataset (CSV).  
It works for both **small and large datasets**, automatically handling feature creation, scaling, and model selection.

---

## 🚀 How It Works

1. Load your sales data (with at least a date and target column).  
2. The script preprocesses your data, creates time-based features (lags, rolling averages, etc.).  
3. It automatically trains the best model (LightGBM by default).  
4. The trained model is saved as a `.pkl` file, and the forecasted sales are exported as a `.csv` file.

---

## 🧠 Model Details

The model uses **LightGBM** as the main algorithm.  
It automatically generates the following time-series features:
- **Month**
- **Day**
- **Day of week**
- **Lag 1-day and 7-day values**
- **7-day rolling mean**

It evaluates model performance using:
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Percentage Error (MAPE)**

---
