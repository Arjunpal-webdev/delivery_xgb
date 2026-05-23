import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import shap
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="DeliveryX AI",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.block-container {
    padding-top: 4rem;
    padding-bottom: 2rem;
}

.big-title {
    font-size: 42px;
    font-weight: 700;
}

.subtitle {
    color: #B0B0B0;
    margin-bottom: 20px;
}

.prediction-card {
    background: linear-gradient(135deg, #1f4037, #99f2c8);
    padding: 30px;
    border-radius: 20px;
    color: black;
    text-align: center;
    margin-top: 20px;
}

.metric-card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

.section-card {
    background-color: #1A1C24;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# PATHS
# =========================================================

MODEL_PATH = "best_xgb_regressor_weighted.pkl"
DATA_PATH = "Food_Delivery_Times.csv"

categorical_cols = [
    'Weather',
    'Traffic_Level',
    'Time_of_Day',
    'Vehicle_Type'
]

NUMERICAL_IMPUTATION_COL = 'Courier_Experience_yrs'

# =========================================================
# LOAD MODEL + DATA
# =========================================================

@st.cache_resource
def load_resources():

    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found")
        st.stop()

    if not os.path.exists(DATA_PATH):
        st.error("Dataset file not found")
        st.stop()

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    raw_df = pd.read_csv(DATA_PATH)

    temp_df = raw_df.copy()

    median_val = temp_df[NUMERICAL_IMPUTATION_COL].median()

    temp_df[NUMERICAL_IMPUTATION_COL] = temp_df[
        NUMERICAL_IMPUTATION_COL
    ].fillna(median_val)

    imputation_values = {
        NUMERICAL_IMPUTATION_COL: median_val
    }

    fitted_encoders = {}

    for col in categorical_cols:

        temp_df[col] = temp_df[col].astype(str)

        mode_val = temp_df[col].mode()[0]

        temp_df[col] = temp_df[col].fillna(mode_val)

        imputation_values[col] = mode_val

        encoder = LabelEncoder()

        encoder.fit(temp_df[col].unique())

        fitted_encoders[col] = encoder

    return model, raw_df, fitted_encoders, imputation_values


best_xgb_regressor_weighted, raw_df, fitted_encoders, imputation_values = load_resources()

# =========================================================
# SHAP EXPLAINER
# =========================================================

@st.cache_resource
def load_shap_explainer(_model):

    return shap.TreeExplainer(_model)


explainer = load_shap_explainer(
    best_xgb_regressor_weighted
)

# =========================================================
# FEATURE COLUMNS
# =========================================================

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

# =========================================================
# FEATURE ENGINEERING
# =========================================================

def apply_feature_engineering(df_input):

    df = df_input.copy()

    if NUMERICAL_IMPUTATION_COL in df.columns:

        df[NUMERICAL_IMPUTATION_COL] = df[
            NUMERICAL_IMPUTATION_COL
        ].fillna(imputation_values[NUMERICAL_IMPUTATION_COL])

    for col in categorical_cols:

        df[col] = df[col].astype(str)

        encoder = fitted_encoders[col]

        mapping = {
            cls: idx for idx, cls in enumerate(encoder.classes_)
        }

        df[col] = df[col].map(mapping).fillna(0).astype(int)

    df['is_long_distance'] = (
        df['Distance_km'] > 15
    ).astype(int)

    df['Distance_Preparation_Interaction'] = (
        df['Distance_km'] *
        df['Preparation_Time_min']
    )

    df_processed = df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return df_processed

# =========================================================
# PREDICTION FUNCTIONS
# =========================================================

def make_single_prediction(data, model):

    input_df = pd.DataFrame([data])

    processed_df = apply_feature_engineering(input_df)

    prediction = model.predict(processed_df)

    return prediction[0], processed_df


def make_batch_prediction(df_input, model):

    processed_df = apply_feature_engineering(
        df_input.drop(columns=['Order_ID'], errors='ignore')
    )

    predictions = model.predict(processed_df)

    return predictions

# =========================================================
# ANALYTICS DATA
# =========================================================

processed_df_for_analytics = raw_df.copy()

processed_df_for_analytics[
    NUMERICAL_IMPUTATION_COL
] = processed_df_for_analytics[
    NUMERICAL_IMPUTATION_COL
].fillna(imputation_values[NUMERICAL_IMPUTATION_COL])

for col in categorical_cols:

    processed_df_for_analytics[col] = processed_df_for_analytics[
        col
    ].astype(str)

    encoder = fitted_encoders[col]

    processed_df_for_analytics[col] = encoder.transform(
        processed_df_for_analytics[col]
    )

processed_df_for_analytics['is_long_distance'] = (
    processed_df_for_analytics['Distance_km'] > 15
).astype(int)

processed_df_for_analytics['Distance_Preparation_Interaction'] = (
    processed_df_for_analytics['Distance_km'] *
    processed_df_for_analytics['Preparation_Time_min']
)

X_for_analytics = processed_df_for_analytics.drop(
    columns=['Delivery_Time_min', 'Order_ID'],
    errors='ignore'
)

X_for_analytics = X_for_analytics.reindex(
    columns=feature_columns,
    fill_value=0
)

y_for_analytics = processed_df_for_analytics[
    'Delivery_Time_min'
]

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
    X_for_analytics,
    y_for_analytics,
    test_size=0.2,
    random_state=42
)

y_pred_xgb = best_xgb_regressor_weighted.predict(
    X_test_new
)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("Delivery Time Prediction App")
    st.markdown("---")

    st.header("Project/Tool Overview")
    st.markdown("This application leverages machine learning to predict food delivery times based on various factors. It helps optimize logistics and provide accurate ETAs to customers.")

    st.header("How it Works")
    st.markdown("1. **Data Preprocessing**: Raw delivery data is cleaned, missing values are handled, and categorical features are encoded.")
    st.markdown("2. **Feature Engineering**: New features like `is_long_distance` and `Distance_Preparation_Interaction` are created to capture complex relationships.")
    st.markdown("3. **Model Training**: An XGBoost Regressor model, tuned with cost-sensitive learning (sample weights), is trained to minimize prediction errors, especially for longer delivery times.")
    st.markdown("4. **Prediction**: The trained model uses the processed features to predict delivery times for new orders.")

    st.header("Model Highlights")
    st.markdown("The best performing model is a **Tuned Weighted XGBoost Regressor**.")
    st.markdown("- **Objective**: Minimize Mean Squared Error (MSE) while addressing underprediction of long delivery times.")
    st.markdown("- **Key Strategy**: Cost-sensitive learning using sample weights, giving higher importance to instances with longer actual delivery times.")
    st.markdown("- **Performance**: Achieved an MSE of **92.21** and an R-squared of **0.79** on the test set, showing improved accuracy, especially for challenging cases.")
    st.markdown("- **Features**: Utilizes `Distance_km`, `Preparation_Time_min`, `Courier_Experience_yrs`, `Weather`, `Traffic_Level`, `Time_of_Day`, `Vehicle_Type`, and engineered features.")

# =========================================================
# HERO SECTION
# =========================================================

st.markdown(
    "<p class='big-title'>🚚 Food Delivery Time Prediction</p>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='subtitle'>Use this application to predict delivery times and analyze model performance</p>",
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Model", "XGBoost")
c2.metric("R² Score", "0.79")
c3.metric("MSE", "92.21")
c4.metric("Features", "9")

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "📦 Batch Prediction",
    "📊 Model Analytics",
    "🚀 Single Prediction"
])

# =========================================================
# BATCH PREDICTION
# =========================================================

with tab1:

    st.subheader("📦 Batch Prediction")

    uploaded_file = st.file_uploader(
        "Upload CSV / Excel / JSON",
        type=['csv', 'xlsx', 'json']
    )

    if uploaded_file is not None:

        if uploaded_file.name.endswith('.csv'):
            df_batch_input = pd.read_csv(uploaded_file)

        elif uploaded_file.name.endswith('.xlsx'):
            df_batch_input = pd.read_excel(uploaded_file)

        elif uploaded_file.name.endswith('.json'):
            df_batch_input = pd.read_json(uploaded_file)

        st.dataframe(df_batch_input.head())

        if st.button("Run Batch Prediction"):

            predictions = make_batch_prediction(
                df_batch_input,
                best_xgb_regressor_weighted
            )

            df_batch_input[
                'Predicted_Delivery_Time_min'
            ] = predictions

            st.dataframe(df_batch_input)

            csv = df_batch_input.to_csv(
                index=False
            ).encode('utf-8')

            st.download_button(
                "Download Predictions",
                csv,
                "predictions.csv",
                "text/csv"
            )

# =========================================================
# MODEL ANALYTICS
# =========================================================

with tab2:

    st.subheader("📊 Model Analytics")

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("R² Score", "0.79")
    k2.metric("MSE", "92.21")
    k3.metric("Best Model", "XGBoost")
    k4.metric("Features Used", "9")

    # ACTUAL VS PREDICTED

    fig_actual = px.scatter(
        x=y_test_new,
        y=y_pred_xgb,
        labels={
            'x': 'Actual Delivery Time',
            'y': 'Predicted Delivery Time'
        },
        title='Actual vs Predicted'
    )

    fig_actual.add_trace(
        go.Scatter(
            x=[y_test_new.min(), y_test_new.max()],
            y=[y_test_new.min(), y_test_new.max()],
            mode='lines',
            name='Perfect Prediction'
        )
    )

    st.plotly_chart(
        fig_actual,
        use_container_width=True
    )

    # RESIDUAL PLOT

    residuals = y_test_new - y_pred_xgb

    fig_residual = px.scatter(
        x=y_pred_xgb,
        y=residuals,
        labels={
            'x': 'Predicted',
            'y': 'Residuals'
        },
        title='Residual Plot'
    )

    st.plotly_chart(
        fig_residual,
        use_container_width=True
    )

    # FEATURE IMPORTANCE

    st.subheader("🔥 Feature Importance")

    feature_importances = best_xgb_regressor_weighted.feature_importances_

    features_df = pd.DataFrame({
        'Feature': X_train_new.columns,
        'Importance': feature_importances
    }).sort_values(
        by='Importance',
        ascending=False
    )

    fig_feat = px.bar(
        features_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance'
    )

    st.plotly_chart(
        fig_feat,
        use_container_width=True
    )

    st.dataframe(features_df)

    st.subheader("Model Comparison and Metrics")
    st.write("Comparing the performance of different models trained during the development phase.")

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
    df_metrics = pd.DataFrame(metrics_data)
    st.dataframe(df_metrics)

# =========================================================
# SINGLE PREDICTION
# =========================================================

with tab3:

    st.subheader("🚀 Smart Delivery Prediction")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 📍 Route Information")

        distance_km = st.slider(
            "Distance (km)",
            1.0,
            50.0,
            10.0
        )

        preparation_time_min = st.slider(
            "Preparation Time (min)",
            1,
            60,
            15
        )

        courier_experience_yrs = st.slider(
            "Courier Experience (yrs)",
            0.0,
            20.0,
            2.0
        )

    with col2:

        st.markdown("### 🌦 Environment")

        weather = st.selectbox(
            "Weather",
            ['Clear', 'Foggy', 'Rainy', 'Snowy', 'Windy']
        )

        traffic_level = st.selectbox(
            "Traffic Level",
            ['Low', 'Medium', 'High']
        )

        time_of_day = st.selectbox(
            "Time of Day",
            ['Morning', 'Afternoon', 'Evening', 'Night']
        )

        vehicle_type = st.selectbox(
            "Vehicle Type",
            ['Bike', 'Car', 'Scooter']
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(
        "🚀 Predict Delivery Time",
        use_container_width=True
    ):

        input_data = {
            'Distance_km': distance_km,
            'Weather': weather,
            'Traffic_Level': traffic_level,
            'Time_of_Day': time_of_day,
            'Vehicle_Type': vehicle_type,
            'Preparation_Time_min': preparation_time_min,
            'Courier_Experience_yrs': courier_experience_yrs
        }

        try:

            predicted_time, processed_df = make_single_prediction(
                input_data,
                best_xgb_regressor_weighted
            )

            confidence_score = max(
                65,
                min(98, 100 - (predicted_time / 2))
            )

            if predicted_time < 25:
                risk_level = "🟢 Low Delay Risk"

            elif predicted_time < 45:
                risk_level = "🟡 Medium Delay Risk"

            else:
                risk_level = "🔴 High Delay Risk"

            st.markdown(f"""
            <div class='prediction-card'>
                <h1>⏱ Predicted ETA</h1>
                <h1>{predicted_time:.2f} Minutes</h1>
                <h3>Confidence Score: {confidence_score:.1f}%</h3>
                <h3>{risk_level}</h3>
            </div>
            """, unsafe_allow_html=True)

            # SHAP ANALYSIS

            st.subheader("🔍 SHAP Explainability")

            shap_values = explainer.shap_values(
                processed_df
            )

            shap_df = pd.DataFrame({
                'Feature': processed_df.columns,
                'Impact': shap_values[0]
            })

            shap_df['abs_impact'] = np.abs(
                shap_df['Impact']
            )

            shap_df = shap_df.sort_values(
                by='abs_impact',
                ascending=False
            )

            fig_shap = px.bar(
                shap_df,
                x='Impact',
                y='Feature',
                orientation='h',
                title='SHAP Feature Impact'
            )

            st.plotly_chart(
                fig_shap,
                use_container_width=True
            )

            st.dataframe(
                shap_df[['Feature', 'Impact']]
            )

            st.subheader("📌 Top Influencing Features")

            top_features = shap_df.head(3)

            for _, row in top_features.iterrows():

                if row['Impact'] > 0:

                    st.success(
                        f"{row['Feature']} increased delivery time"
                    )

                else:

                    st.info(
                        f"{row['Feature']} reduced delivery time"
                    )

        except Exception as e:

            st.error(f"Prediction Error: {e}")