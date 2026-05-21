import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# --- 1. Set Page Configuration ---
st.set_page_config(layout="wide", page_title="Delivery Time Prediction App")

# --- 2. Define Paths and Global Constants ---
MODEL_PATH = 'best_xgb_regressor_weighted.pkl'
DATA_PATH = 'Food_Delivery_Times.csv'
categorical_cols = ['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']
NUMERICAL_IMPUTATION_COL = 'Courier_Experience_yrs'

# --- 3. Load Pre-trained Model and Setup Encoders/Imputation Values ---
@st.cache_resource
def load_resources():
    # Load the best model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
    else:
        st.error(f"{MODEL_PATH} not found. Please ensure the file exists in the same directory as main.py")
        st.stop()

    # Load original data and setup encoders/imputation values
    if os.path.exists(DATA_PATH):
        raw_df_full = pd.read_csv(DATA_PATH)
    else:
        st.error(f"{DATA_PATH} not found. Please ensure the file exists in the same directory as main.py")
        st.stop()

    # Make a copy to avoid modifying the original loaded DataFrame
    temp_df_for_fitting = raw_df_full.copy()

    # Impute missing numerical values before fitting encoders for consistency
    if NUMERICAL_IMPUTATION_COL in temp_df_for_fitting.columns:
        median_val = temp_df_for_fitting[NUMERICAL_IMPUTATION_COL].median()
        temp_df_for_fitting[NUMERICAL_IMPUTATION_COL] = temp_df_for_fitting[NUMERICAL_IMPUTATION_COL].fillna(median_val)
    else:
        median_val = temp_df_for_fitting[NUMERICAL_IMPUTATION_COL].median() # Fallback, though should exist

    imputation_values = {
        NUMERICAL_IMPUTATION_COL: median_val
    }

    fitted_encoders = {}
    for col in categorical_cols:
        if col in temp_df_for_fitting.columns:
            # Ensure column is string type for LabelEncoder
            temp_df_for_fitting[col] = temp_df_for_fitting[col].astype(str)
            # Get mode for imputation and store it
            mode_val = temp_df_for_fitting[col].mode()[0]
            imputation_values[col] = mode_val
            # Fill NaNs before fitting encoder
            temp_df_for_fitting[col] = temp_df_for_fitting[col].fillna(mode_val)

            encoder = LabelEncoder()
            encoder.fit(temp_df_for_fitting[col].unique()) # Fit on all unique values for robustness
            fitted_encoders[col] = encoder
        else:
            st.warning(f"Column '{col}' not found in the training data. This may cause issues.")

    return model, raw_df_full, fitted_encoders, imputation_values

best_xgb_regressor_weighted, raw_df, fitted_encoders, imputation_values = load_resources()

# Define the exact feature columns and their order used during training
feature_columns = [
    'Distance_km', 'Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type',
    'Preparation_Time_min', 'Courier_Experience_yrs', 'is_long_distance',
    'Distance_Preparation_Interaction'
]

# --- 4. Helper Functions for Feature Engineering and Prediction ---
def apply_feature_engineering(df_input):
    """
    Applies the same feature engineering steps as performed during training.
    Handles missing values and applies Label Encoding using pre-fitted encoders.
    Returns a DataFrame with columns matching the training set order.
    """
    df = df_input.copy()

    # 1. Impute missing numerical values
    if NUMERICAL_IMPUTATION_COL in df.columns:
        df[NUMERICAL_IMPUTATION_COL] = df[NUMERICAL_IMPUTATION_COL].fillna(imputation_values[NUMERICAL_IMPUTATION_COL])

    # 2. Impute missing categorical values and apply Label Encoding
    for col in categorical_cols:
        if col in df.columns:
            # Ensure column is string type
            df[col] = df[col].astype(str)

            # Impute missing values with pre-computed mode from training data
            df[col] = df[col].fillna(imputation_values[col])

            encoder = fitted_encoders.get(col)
            if encoder:
                # Handle unseen categories by mapping them to 0 (first class label)
                # This creates NaN for unseen, then fills with 0, then converts to int
                mapping = {cls: idx for idx, cls in enumerate(encoder.classes_)}
                df[col] = df[col].map(mapping).fillna(0).astype(int)
            else:
                st.warning(f"No encoder found for column '{col}'. Skipping encoding.")
        else:
            # If a categorical column is missing from input, add it and fill with default (0)
            df[col] = 0 # Assume 0 as default encoded value if column is entirely missing

    # 3. Create 'is_long_distance' feature
    if 'Distance_km' in df.columns:
        df['is_long_distance'] = (df['Distance_km'] > 15).astype(int)
    else:
        df['is_long_distance'] = 0 # Default if Distance_km is missing

    # 4. Create an interaction feature: Distance * Preparation Time
    if 'Distance_km' in df.columns and 'Preparation_Time_min' in df.columns:
        df['Distance_Preparation_Interaction'] = df['Distance_km'] * df['Preparation_Time_min']
    else:
        df['Distance_Preparation_Interaction'] = 0 # Default if components are missing

    # 5. Ensure column order and presence matches training data
    df_processed = df.reindex(columns=feature_columns, fill_value=0) # Fill missing new columns with 0

    return df_processed

def make_single_prediction(data, model):
    """
    Makes a single prediction using the pre-trained model.
    """
    input_df = pd.DataFrame([data])
    processed_df = apply_feature_engineering(input_df)
    prediction = model.predict(processed_df)
    return prediction[0]

def make_batch_prediction(df_input, model):
    """
    Makes batch predictions on a DataFrame using the pre-trained model.
    """
    # Drop 'Order_ID' if present, as it's not a feature
    processed_df = apply_feature_engineering(df_input.drop(columns=['Order_ID'], errors='ignore'))
    predictions = model.predict(processed_df)
    return predictions

# --- 5. Prepare Data for Model Analytics Section (executed once on app startup) ---

# Replicate training preprocessing for analytics plots
processed_df_for_analytics = raw_df.copy()

# Impute numerical column
if NUMERICAL_IMPUTATION_COL in processed_df_for_analytics.columns:
    processed_df_for_analytics[NUMERICAL_IMPUTATION_COL] = processed_df_for_analytics[NUMERICAL_IMPUTATION_COL].fillna(imputation_values[NUMERICAL_IMPUTATION_COL])

# Process categorical columns
for col in categorical_cols:
    if col in processed_df_for_analytics.columns:
        processed_df_for_analytics[col] = processed_df_for_analytics[col].astype(str)
        processed_df_for_analytics[col] = processed_df_for_analytics[col].fillna(imputation_values[col])

        encoder = fitted_encoders.get(col)
        if encoder:
            processed_df_for_analytics[col] = encoder.transform(processed_df_for_analytics[col])

# Create engineered features
processed_df_for_analytics['is_long_distance'] = (processed_df_for_analytics['Distance_km'] > 15).astype(int)
processed_df_for_analytics['Distance_Preparation_Interaction'] = processed_df_for_analytics['Distance_km'] * processed_df_for_analytics['Preparation_Time_min']

# Define X and y for splitting, ensuring consistent columns for X
X_for_analytics = processed_df_for_analytics.drop(columns=['Delivery_Time_min', 'Order_ID'], errors='ignore')
y_for_analytics = processed_df_for_analytics['Delivery_Time_min']

# Reindex X_for_analytics to ensure correct order before splitting
X_for_analytics = X_for_analytics.reindex(columns=feature_columns, fill_value=0)

# Split the data (using the same random_state as during training)
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_for_analytics, y_for_analytics, test_size=0.2, random_state=42)

# Generate predictions on the test set for analytics plots
y_pred_xgb_tuned_weighted = best_xgb_regressor_weighted.predict(X_test_new)

# --- 6. Sidebar Navigation and Information ---
with st.sidebar:
    st.title("Delivery Time Prediction App")
    st.markdown("---")

    st.header("Tool Overview")
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

# --- 7. Main Application Content ---
st.title("Food Delivery Time Prediction")
st.write("Use this application to predict delivery times and analyze model performance.")

tab1, tab2, tab3 = st.tabs(["Batch Prediction", "Model Analytics", "One Prediction"])

with tab1:
    st.header("Batch Prediction")
    st.write("Upload a CSV, Excel, or JSON file to get batch predictions for delivery times.")

    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'json'])
    output_format = st.selectbox("Select output format for predictions", ('CSV', 'JSON'), key='batch_output_format')

    if uploaded_file is not None:
        df_batch_input = None
        try:
            if uploaded_file.name.endswith('.csv'):
                df_batch_input = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df_batch_input = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df_batch_input = pd.read_json(uploaded_file)

            if df_batch_input is not None:
                st.write("Uploaded Data Preview:")
                st.dataframe(df_batch_input.head())

                if st.button("Run Batch Prediction", key='run_batch_prediction_button'):
                    try:
                        batch_predictions = make_batch_prediction(df_batch_input, best_xgb_regressor_weighted)
                        df_batch_input['Predicted_Delivery_Time_min'] = batch_predictions

                        st.subheader("Batch Predictions Results")
                        st.dataframe(df_batch_input)

                        # Provide download link
                        if output_format == 'CSV':
                            csv_output = df_batch_input.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download as CSV",
                                data=csv_output,
                                file_name="batch_predictions.csv",
                                mime="text/csv",
                                key='download_csv'
                            )
                        elif output_format == 'JSON':
                            json_output = df_batch_input.to_json(orient="records").encode('utf-8')
                            st.download_button(
                                label="Download as JSON",
                                data=json_output,
                                file_name="batch_predictions.json",
                                mime="application/json",
                                key='download_json'
                            )

                    except Exception as e:
                        st.error(f"Error during batch prediction: {e}")
                        st.exception(e)

        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
            st.exception(e)

with tab2:
    st.header("Model Analytics")

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

    st.subheader("Best Model: Actual vs. Predicted Plot")
    st.write("Visualizing the performance of the Tuned Weighted XGBoost Regressor on the test set.")

    fig_actual_pred = plt.figure(figsize=(10, 7))
    sns.scatterplot(x=y_test_new, y=y_pred_xgb_tuned_weighted, alpha=0.6)
    plt.plot([y_test_new.min(), y_test_new.max()], [y_test_new.min(), y_test_new.max()], 'r--', lw=2)
    plt.xlabel('Actual Delivery Time (min)')
    plt.ylabel('Predicted Delivery Time (min)')
    plt.title('Actual vs. Predicted Delivery Times (Tuned Weighted XGBoost Regressor)')
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig_actual_pred)

    st.subheader("Best Model: Residual Plot")
    residuals_xgb_tuned_weighted = y_test_new - y_pred_xgb_tuned_weighted
    fig_residual = plt.figure(figsize=(10, 7))
    sns.scatterplot(x=y_pred_xgb_tuned_weighted, y=residuals_xgb_tuned_weighted, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Delivery Time (min)')
    plt.ylabel('Residuals (Actual - Predicted)')
    st.title('Residual Plot (Tuned Weighted XGBoost Regressor)') # Changed to st.title for higher visibility
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig_residual)

    st.subheader("Feature Importances")
    st.write("Understanding which features contribute most to the model's predictions.")

    if hasattr(best_xgb_regressor_weighted, 'feature_importances_'):
        feature_importances = best_xgb_regressor_weighted.feature_importances_
        feature_names = X_train_new.columns # Use columns from X_train_new

        features_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        st.dataframe(features_df)

        fig_feat_imp = plt.figure(figsize=(12, 7))
        sns.barplot(x='Importance', y='Feature', data=features_df, palette='viridis')
        plt.title('Feature Importances (Tuned Weighted XGBoost Regressor)')
        plt.xlabel('Relative Importance')
        plt.ylabel('Feature')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        st.pyplot(fig_feat_imp)
    else:
        st.warning("Feature importances could not be displayed. Model does not have `feature_importances_` attribute.")

with tab3:
    st.header("One Prediction")
    st.write("Enter the details below to get a single delivery time prediction.")

    # Input fields for each feature
    distance_km = st.number_input("Distance (km)", min_value=0.1, max_value=50.0, value=10.0, key='distance_km')
    preparation_time_min = st.number_input("Preparation Time (min)", min_value=1, max_value=60, value=15, key='preparation_time_min')
    courier_experience_yrs = st.number_input("Courier Experience (yrs)", min_value=0.0, max_value=20.0, value=2.0, key='courier_experience_yrs')

    # Categorical features - Use selectbox with original categories from fitted encoders
    weather_options = list(fitted_encoders['Weather'].classes_)
    traffic_options = list(fitted_encoders['Traffic_Level'].classes_)
    time_of_day_options = list(fitted_encoders['Time_of_Day'].classes_)
    vehicle_options = list(fitted_encoders['Vehicle_Type'].classes_)

    # Ensure default values are within the options and handle cases where they might not be
    weather = st.selectbox("Weather", weather_options, index=weather_options.index('Clear') if 'Clear' in weather_options else 0, key='weather_select')
    traffic_level = st.selectbox("Traffic Level", traffic_options, index=traffic_options.index('Medium') if 'Medium' in traffic_options else 0, key='traffic_level_select')
    time_of_day = st.selectbox("Time of Day", time_of_day_options, index=time_of_day_options.index('Afternoon') if 'Afternoon' in time_of_day_options else 0, key='time_of_day_select')
    vehicle_type = st.selectbox("Vehicle Type", vehicle_options, index=vehicle_options.index('Scooter') if 'Scooter' in vehicle_options else 0, key='vehicle_type_select')

    if st.button("Predict Delivery Time", key='predict_single_button'):
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
            predicted_time = make_single_prediction(input_data, best_xgb_regressor_weighted)
            st.success(f"The predicted delivery time is: **{predicted_time:.2f} minutes**")
        except Exception as e:
            st.error(f"Error during single prediction: {e}")
            st.exception(e)
