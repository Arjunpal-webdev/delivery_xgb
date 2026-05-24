from pathlib import Path

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_xgb_regressor_weighted.pkl"
ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"
DATA_PATH = BASE_DIR / "data" / "Food_Delivery_Times.csv"

# =========================================================
# FEATURE DEFINITIONS
# =========================================================

categorical_cols = [
    'Weather',
    'Traffic_Level',
    'Time_of_Day',
    'Vehicle_Type'
]

feature_columns = [
    'Distance_km',
    'Weather',
    'Traffic_Level',
    'Time_of_Day',
    'Vehicle_Type',
    'Preparation_Time_min',
    'Courier_Experience_yrs',
    'is_long_distance',
    'Distance_Preparation_Interaction'
]

NUMERICAL_IMPUTATION_COL = 'Courier_Experience_yrs'

# =========================================================
# METRICS DATA
# =========================================================

metrics_data = {
    'Model': [
        'Bagging Regressor (Base)', 'Tuned Bagging Regressor',
        'Random Forest Regressor (Base)', 'Tuned Random Forest Regressor (without new features)',
        'Tuned Random Forest Regressor (with new features, old hyperparams)',
        'Newly Tuned Random Forest Regressor (with new features)',
        'XGBoost Regressor (Base)', 'XGBoost Regressor (Weighted)', 'Tuned Weighted XGBoost Regressor'
    ],
    'MSE': [
        124.87, 126.53, 100.12, 97.41, 96.53, 98.12, 96.14, 94.94, 92.21
    ],
    'R2 Score': [
        0.72, 0.72, 0.78, 0.78, 0.78, 0.78, 0.79, 0.79, 0.79
    ]
}
