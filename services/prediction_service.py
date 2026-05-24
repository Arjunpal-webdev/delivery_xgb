import pandas as pd
import shap
import streamlit as st

from utils.preprocessing import apply_feature_engineering


def make_single_prediction(data, model, fitted_encoders, imputation_values):

    input_df = pd.DataFrame([data])

    processed_df = apply_feature_engineering(
        input_df, fitted_encoders, imputation_values
    )

    prediction = model.predict(processed_df)

    return prediction[0], processed_df


def make_batch_prediction(df_input, model, fitted_encoders, imputation_values):

    processed_df = apply_feature_engineering(
        df_input.drop(columns=['Order_ID'], errors='ignore'),
        fitted_encoders,
        imputation_values
    )

    predictions = model.predict(processed_df)

    return predictions


@st.cache_resource
def load_shap_explainer(_model):

    return shap.TreeExplainer(_model)
