from sklearn.model_selection import train_test_split

from utils.constants import categorical_cols, feature_columns, NUMERICAL_IMPUTATION_COL


def prepare_analytics_data(raw_df, model, fitted_encoders, imputation_values):

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

    y_pred_xgb = model.predict(X_test_new)

    return X_train_new, X_test_new, y_train_new, y_test_new, y_pred_xgb
