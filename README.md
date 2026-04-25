# StockSense — Demand Forecasting & Inventory Optimization

A production-ready retail intelligence system that predicts future product demand and translates forecasts into actionable inventory decisions — helping e-commerce businesses reduce stockouts, eliminate overstocking, and optimize working capital.

**Live Demo:** [e-commerce-demand-forecasting.streamlit.app](https://e-commerce-demand-forecasting-mjlqjwbeigqfx8abcq8mua.streamlit.app/)

---

## Overview

Retail businesses face a recurring challenge: how much inventory to stock. Overstock ties up capital; understock loses sales. StockSense builds an end-to-end forecasting pipeline that takes raw historical sales data, trains four different models, evaluates them, and translates the best forecast into actionable inventory decisions — all accessible through an interactive dashboard.

The dataset covers 5 years of daily sales (2013–2017) across 10 stores and 50 products, totaling ~913,000 records.

---

## Model Performance

| Model | RMSE | MAPE | Type |
|-------|------|------|------|
| **XGBoost** | **7.91** | **12.44%** | Gradient Boosted Tree |
| LightGBM | 7.94 | 12.56% | Gradient Boosted Tree |
| Multivariate LSTM | 8.37 | 13.11% | Deep Learning |
| Univariate LSTM | 8.77 | 13.91% | Deep Learning |

XGBoost achieved the best performance — an average forecast error of roughly 8 sales units per day. Tree-based models outperformed LSTMs on this dataset because the engineered lag and rolling features provide the temporal context that LSTMs would otherwise learn from raw sequences.

---

## Dashboard Features

**Exploratory Data Analysis**
Interactive filters by store and item. Visualizations include sales trend over time, monthly seasonality, store-level comparison, and sales distribution. Key finding: strong upward trend with repeating yearly patterns.

**Demand Forecast**
Select any model, store, and item to view forecast vs. actual on the 2017 validation set. Includes a future forecast widget — choose a horizon (7–90 days) and the Univariate LSTM rolls forward from the last known date. Forecasts are downloadable as CSV.

**Model Comparison**
Side-by-side overlay of all four models on the same validation data. RMSE/MAPE bar charts, residual distributions, and per-model deep-dive tabs.

**Inventory Planner**
Converts XGBoost demand forecasts into three inventory decisions per item using classical supply chain formulas:

- **EOQ** (Economic Order Quantity) — how much to order at once
- **Safety Stock** — buffer against demand variability
- **Reorder Point** — when to place the next order

Business parameters (ordering cost, holding cost, lead time, service level) are adjustable via sidebar. Results are downloadable as CSV.

**Batch Predict**
Upload a CSV of (date, store, item) rows and receive demand forecasts for all rows at once. Feature engineering runs automatically using historical data for context. Results downloadable.

**SHAP Explainer**
Explains individual predictions using SHAP (SHapley Additive exPlanations). Shows which features pushed a prediction up or down via waterfall charts and a feature importance heatmap. Makes the model interpretable for business users.

---

## Repository Structure

```
StockSense/
│
├── streamlit_app.py                  # Streamlit entry point
├── requirements.txt                  # Python dependencies
├── .python-version                   # Pins Python 3.11 for Streamlit Cloud
├── .gitignore
│
├── .streamlit/
│   └── config.toml                   # Dark theme and server configuration
│
├── src/                              # Core application modules
│   ├── __init__.py
│   ├── config.py                     # Paths, constants, feature names, model metadata
│   ├── data_loader.py                # Data loading with on-the-fly feature engineering fallback
│   ├── feature_engineering.py        # Feature pipeline (time, lag, rolling, aggregate)
│   ├── model_loader.py               # Cached model loading for all 4 models
│   ├── inventory.py                  # EOQ, Safety Stock, Reorder Point formulas
│   └── plots.py                      # Plotly chart builders (reused across pages)
│
├── pages/                            # One file per dashboard page
│   ├── home.py                       # Landing page with project overview
│   ├── eda.py                        # Exploratory data analysis
│   ├── forecast.py                   # Interactive demand forecast
│   ├── model_comparison.py           # Side-by-side model evaluation
│   ├── inventory.py                  # Inventory planning dashboard
│   ├── batch_predict.py              # Bulk CSV prediction
│   └── shap_explainer.py             # SHAP feature attribution
│
├── notebooks/                        # Development notebooks
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling_xgboost.ipynb
│   ├── 04_modeling_lstm.ipynb
│   ├── 05_inventory_management.ipynb
│   └── comparing_models.ipynb
│
├── data/
│   ├── raw/
│   │   └── train.csv                 # Raw dataset (913,000 rows)
│   └── processed/
│       └── featured_data.csv         # Pre-built feature set (gitignored — 123 MB)
│
├── models/                           # Saved model artifacts
│   ├── xgb_model.joblib
│   ├── lgbm_model.joblib
│   ├── lstm_model.keras
│   ├── multi_lstm_model.keras
│   ├── lstm_scaler.save
│   ├── multi_lstm_scaler.save
│   ├── xgb_preds.joblib
│   ├── lgbm_preds.joblib
│   ├── lstm_preds.joblib
│   ├── multi_lstm_preds.joblib
│   └── y_val_actual.joblib
│
├── plots/                            # Saved EDA and evaluation plots
├── final_forecast/                   # Final forecast output CSVs
├── document/                         # Project documentation
└── anaconda_projects/                # Anaconda project configuration
```

---

## Feature Engineering

The raw dataset has 4 columns (date, store, item, sales). The pipeline adds 13 features:

| Category | Features | Purpose |
|----------|----------|---------|
| Time | year, month, week, day, dayofweek, is_weekend | Captures seasonality |
| Lag | sales_lag_1, _7, _14, _28 | Provides memory of past demand |
| Rolling | rolling_mean_7, _14, _28 | Smooths trend and captures volatility |
| Aggregate | store_avg_sales, item_avg_sales | Encodes baseline behavior |

---

## Inventory Formulas

```
EOQ          = sqrt(2 * D * S / h)
Safety Stock = Z * sigma_d * sqrt(L)
ROP          = avg_daily_demand * L + Safety Stock
```

Where D = annual demand, S = ordering cost, h = holding cost per unit, Z = service level z-score (95% default = 1.65), sigma_d = daily demand standard deviation, L = lead time in days.

---

## Running Locally

```bash
# Clone the repository
git clone https://github.com/Devashish-Rawat1/E-commerce-Demand-Forecasting.git
cd StockSense

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

The app requires `data/raw/train.csv` and the saved model files in `models/`. If `data/processed/featured_data.csv` is absent (it is gitignored due to file size), features are built automatically from `train.csv` on first load — this takes 30–60 seconds and is cached for the rest of the session.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Modeling | XGBoost, LightGBM, TensorFlow/Keras (LSTM) |
| Feature Engineering | Pandas, NumPy |
| Explainability | SHAP |
| Dashboard | Streamlit, Plotly |
| Deployment | Streamlit Cloud |

---

## Dataset

Kaggle — [Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only)

913,000 rows of daily sales data across 10 stores and 50 items from 2013 to 2017. No missing values. The train/validation split uses 2013–2016 for training and 2017 for evaluation.
