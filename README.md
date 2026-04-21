# 📦 E-Commerce Demand Forecasting — Streamlit App

> Predict future product demand to optimize inventory and reduce stockouts & overstocking.

## 🚀 Live Demo
Deploy on Streamlit Cloud → [share.streamlit.io](https://share.streamlit.io)

---

## 🗂️ Repository Structure

```
your-repo/
├── app.py                        ← Streamlit entry point
├── requirements.txt              ← Python dependencies
├── .streamlit/
│   └── config.toml               ← Theme & server config
├── src/
│   ├── config.py                 ← Paths, constants, model metadata
│   ├── data_loader.py            ← Data loading + feature engineering
│   ├── model_loader.py           ← Model loading with caching
│   ├── feature_engineering.py   ← Full feature pipeline
│   ├── inventory.py              ← EOQ · ROP · Safety Stock
│   └── plots.py                  ← All Plotly chart builders
├── pages/
│   ├── home.py                   ← Landing page
│   ├── eda.py                    ← Interactive EDA
│   ├── forecast.py               ← Model selector + future forecast
│   ├── model_comparison.py       ← Side-by-side model metrics
│   ├── inventory.py              ← Inventory planner
│   ├── batch_predict.py          ← Upload CSV → bulk predictions
│   └── shap_explainer.py         ← SHAP demand explainer (bonus feature)
├── data/
│   ├── raw/train.csv
│   └── processed/featured_data.csv
└── models/
    ├── xgb_model.joblib
    ├── lgbm_model.joblib
    ├── lstm_model.keras
    ├── multi_lstm_model.keras
    ├── lstm_scaler.save
    ├── multi_lstm_scaler.save
    ├── xgb_preds.joblib
    ├── lgbm_preds.joblib
    ├── lstm_preds.joblib
    ├── multi_lstm_preds.joblib
    └── y_val_actual.joblib
```

---

## ⚙️ Local Setup

```bash
# 1. Clone your repo
git clone https://github.com/Devashish-Rawat1/E-commerce-Demand-Forecasting
cd E-commerce-Demand-Forecasting

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud (Step by Step)

### Step 1 — Push your code to GitHub
Make sure `app.py`, `requirements.txt`, `.streamlit/`, `src/`, and `pages/`
are all committed and pushed.

```bash
git add app.py requirements.txt .streamlit/ src/ pages/
git commit -m "Add Streamlit app"
git push origin main
```

### Step 2 — Handle large model files with Git LFS
If any `.keras` or `.joblib` file exceeds 100 MB:

```bash
git lfs install
git lfs track "*.keras"
git lfs track "*.joblib"
git add .gitattributes
git commit -m "Add Git LFS tracking"
git push origin main
```

### Step 3 — Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select your repository: `Devashish-Rawat1/E-commerce-Demand-Forecasting`
5. Set **Main file path** to: `app.py`
6. Click **Deploy**

Streamlit Cloud will automatically install all packages from `requirements.txt`.

---

## 📄 App Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Project overview, KPI metrics, pipeline diagram |
| 📊 EDA | Interactive filters, trend/seasonality/distribution charts |
| 🤖 Forecast | Model selector, forecast vs actual, future forecast with download |
| 📈 Model Comparison | All 4 models side-by-side, RMSE/MAPE bar charts, residuals |
| 🏭 Inventory Planner | EOQ · ROP · Safety Stock with adjustable parameters |
| 📤 Batch Predict | Upload CSV → feature engineer → predict → download results |
| 🔍 SHAP Explainer | Per-prediction feature attribution, waterfall + heatmap charts |

---

## 🤖 Models

| Model | RMSE | MAPE | Type |
|-------|------|------|------|
| 🥇 XGBoost | 7.9126 | 12.44% | Gradient Boosted Tree |
| 🥈 LightGBM | 7.9421 | 12.56% | Gradient Boosted Tree |
| 🥉 Multivariate LSTM | 8.3659 | 13.11% | Deep Learning |
| Univariate LSTM | 8.7666 | 13.91% | Deep Learning |

---

## 🏭 Inventory Formulas

```
EOQ         = √(2 × D × S / h)
Safety Stock = Z × σ_d × √L
ROP         = d̄ × L + Safety Stock
```

Where:
- **D** = Annual demand · **S** = Ordering cost · **h** = Holding cost per unit
- **Z** = Service level z-score (95% → 1.65)
- **σ_d** = Daily demand std dev · **L** = Lead time days · **d̄** = Avg daily demand

---

## 💡 Key Feature — SHAP Explainer

The SHAP page answers *"why did the model predict this demand?"* using
SHapley Additive exPlanations — providing:
- **Global importance** (which features matter most overall)
- **Waterfall charts** (how each feature pushed a single prediction up/down)
- **SHAP heatmap** (pattern of contributions across many predictions)

This makes the forecasting system **explainable and trustworthy** for business users.
