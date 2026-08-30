"""Feature engineering. Every transformation is row-wise, so there is
nothing fitted here and no leakage risk from applying it before splitting.
"""

import numpy as np


def add_ratios(df):
    df = df.copy()

    df["CREDIT_INCOME_RATIO"] = (
        df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    )

    df["ANNUITY_INCOME_RATIO"] = (
        df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    )

    df["CREDIT_TERM"] = (
        df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    )

    df["GOODS_CREDIT_RATIO"] = (
        df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"]
    )

    df["INCOME_PER_PERSON"] = (
        df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    )

    df = df.replace([np.inf, -np.inf], np.nan)


    return df


def add_history_indicators(df):
    df = df.copy()

    df["NO_BUREAU_HISTORY"] = (
        df["BURO_DAYS_CREDIT_COUNT"].isna().astype("int8")
    )

    df["NO_PREV_HISTORY"] = (
        df["PREV_SK_ID_PREV_COUNT"].isna().astype("int8")
    )

    df["PREV_CREDIT_APPLICATION_RATIO"] = (
        df["PREV_AMT_CREDIT_MEAN"]
        / df["PREV_AMT_APPLICATION_MEAN"]
    ).replace([np.inf, -np.inf], np.nan)

    return df


def prep_for_lgbm(df):
    df = df.copy()

    # Handle DAYS_EMPLOYED anomaly
    df["DAYS_EMPLOYED_ANOM"] = (
        df["DAYS_EMPLOYED"] == 365243
    ).astype("int8")

    df["DAYS_EMPLOYED"] = (
        df["DAYS_EMPLOYED"]
        .replace(365243, np.nan)
    )

    # Convert categorical columns to pandas category
    for col in df.select_dtypes(exclude="number").columns:
        df[col] = df[col].astype("category")

    # Reduce float64 memory usage
    for col in df.select_dtypes(include="float64").columns:
        df[col] = df[col].astype("float32")

    return df


def build_features(df):
    """The full feature pipeline, in the order it must run."""
    return prep_for_lgbm(
        add_history_indicators(
            add_ratios(df)
        )
    )