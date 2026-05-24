# 🚚 DeliveryX AI — Food Delivery Time Prediction

A Streamlit-based machine learning application that predicts food delivery times using a **Tuned Weighted XGBoost Regressor**. The app provides single-order predictions, batch predictions, SHAP explainability, and full model analytics.

---

## 📸 Features

| Feature | Description |
|---|---|
| 🚀 Single Prediction | Predict delivery time for one order with SHAP explanations |
| 📦 Batch Prediction | Upload CSV / Excel / JSON and download predictions |
| 📊 Model Analytics | Actual vs Predicted plots, residual analysis, feature importance |
| 🔍 SHAP Explainability | Per-prediction feature impact breakdown |
| 🤖 Best Model | Tuned Weighted XGBoost — MSE: 92.21, R²: 0.79 |

---

## 🗂️ Project Structure

```
delivery_xgboost/
│
├── main.py                          # Root entrypoint (delegates to app/main.py)
│
├── app/
│   └── main.py                      # Streamlit UI — all tabs and widgets
│
├── services/
│   ├── model_loader.py              # Loads model, dataset, and encoders
│   ├── prediction_service.py        # Single & batch prediction + SHAP explainer
│   └── analytics_service.py        # Prepares train/test split for analytics tab
│
├── utils/
│   ├── constants.py                 # Absolute paths (BASE_DIR) + feature definitions
│   ├── preprocessing.py             # Feature engineering pipeline
│   ├── helpers.py                   # Confidence score & risk level helpers
│   └── styles.py                   # Custom CSS injected into Streamlit
│
├── models/
│   ├── best_xgb_regressor_weighted.pkl   # Trained XGBoost model
│   └── label_encoder.pkl                 # Fitted label encoders for categorical cols
│
├── data/
│   └── Food_Delivery_Times.csv      # Raw dataset used for analytics & fallback encoding
│
├── notebooks/
│   └── Delivery_model_training.ipynb     # Model training & experimentation notebook
│
├── requirements.txt                 # Pinned Python dependencies
└── README.md                        # This file
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/delivery_xgboost.git
cd delivery_xgboost
```

### 2. Create and activate a virtual environment (recommended)

```bash
# Using venv
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

# OR using conda
conda create -n deliveryx python=3.11
conda activate deliveryx
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the App

You can run the app from either the **project root** or the **`app/` subdirectory** — both work identically:

```bash
# From project root
streamlit run main.py

# OR directly from app/
streamlit run app/main.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🤖 Model Details

| Property | Value |
|---|---|
| Algorithm | XGBoost Regressor |
| Strategy | Cost-sensitive learning with sample weights |
| MSE (test) | 92.21 |
| R² Score | 0.79 |
| Features used | 9 (including 2 engineered) |

### Input Features

| Feature | Type | Description |
|---|---|---|
| `Distance_km` | Numerical | Delivery distance in kilometres |
| `Weather` | Categorical | Clear / Foggy / Rainy / Snowy / Windy |
| `Traffic_Level` | Categorical | Low / Medium / High |
| `Time_of_Day` | Categorical | Morning / Afternoon / Evening / Night |
| `Vehicle_Type` | Categorical | Bike / Car / Scooter |
| `Preparation_Time_min` | Numerical | Food preparation time in minutes |
| `Courier_Experience_yrs` | Numerical | Courier's years of experience |
| `is_long_distance` | Engineered | 1 if Distance_km > 15, else 0 |
| `Distance_Preparation_Interaction` | Engineered | Distance_km × Preparation_Time_min |

---

## 📦 Batch Prediction Format

Upload a CSV / Excel / JSON file with any subset of the input features listed above.
The app will return the file with a `Predicted_Delivery_Time_min` column appended, available for download.

**Example CSV header:**
```
Distance_km,Weather,Traffic_Level,Time_of_Day,Vehicle_Type,Preparation_Time_min,Courier_Experience_yrs
```

---

## 📋 Dependencies

| Package | Version |
|---|---|
| streamlit | 1.45.1 |
| xgboost | 3.2.0 |
| shap | 0.51.0 |
| scikit-learn | 1.6.1 |
| pandas | 2.2.3 |
| numpy | 2.1.3 |
| plotly | 5.24.1 |

---

## 📄 License

This project is for educational and demonstration purposes.
