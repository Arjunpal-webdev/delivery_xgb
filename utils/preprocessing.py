from utils.constants import (
    categorical_cols,
    feature_columns,
    NUMERICAL_IMPUTATION_COL
)


def apply_feature_engineering(df_input, fitted_encoders, imputation_values):

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
