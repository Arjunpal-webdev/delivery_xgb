import os
import pickle

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

from utils.constants import (
    MODEL_PATH,
    ENCODER_PATH,
    DATA_PATH,
    categorical_cols,
    NUMERICAL_IMPUTATION_COL
)


@st.cache_resource
def load_resources():

    # -------------------------------------------------------
    # Load model
    # -------------------------------------------------------

    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found")
        st.stop()

    if not os.path.exists(DATA_PATH):
        st.error("Dataset file not found")
        st.stop()

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    # -------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------

    raw_df = pd.read_csv(DATA_PATH)

    temp_df = raw_df.copy()

    median_val = temp_df[NUMERICAL_IMPUTATION_COL].median()

    temp_df[NUMERICAL_IMPUTATION_COL] = temp_df[
        NUMERICAL_IMPUTATION_COL
    ].fillna(median_val)

    imputation_values = {
        NUMERICAL_IMPUTATION_COL: median_val
    }

    for col in categorical_cols:
        temp_df[col] = temp_df[col].astype(str)
        mode_val = temp_df[col].mode()[0]
        temp_df[col] = temp_df[col].fillna(mode_val)
        imputation_values[col] = mode_val

    # -------------------------------------------------------
    # Load label encoders:
    #   Try models/label_encoder.pkl first.
    #   If not found or incompatible, fit fresh encoders
    #   from the CSV dataset.
    # -------------------------------------------------------

    fitted_encoders = {}

    try:
        if not os.path.exists(ENCODER_PATH):
            raise FileNotFoundError(
                f"Encoder file not found at {ENCODER_PATH}"
            )

        with open(ENCODER_PATH, "rb") as f:
            persisted_encoders = pickle.load(f)

        # Validate that it covers all required categorical columns
        for col in categorical_cols:
            if col not in persisted_encoders:
                raise KeyError(
                    f"Encoder for column '{col}' missing in persisted file"
                )
            encoder = persisted_encoders[col]
            if not hasattr(encoder, 'classes_'):
                raise ValueError(
                    f"Encoder for column '{col}' is not a fitted LabelEncoder"
                )

        fitted_encoders = persisted_encoders

    except Exception:
        # Fallback: fit encoders directly from the dataset
        for col in categorical_cols:
            encoder = LabelEncoder()
            encoder.fit(temp_df[col].unique())
            fitted_encoders[col] = encoder

    return model, raw_df, fitted_encoders, imputation_values
