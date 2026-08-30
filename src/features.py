"""Feature engineering.

Ratios, indicators and the DAYS_EMPLOYED anomaly flag are row-wise, so
there is no leakage risk from applying them before splitting.

Category levels are the one exception: they are learned from the training
data and must be passed to any later call. LightGBM encodes categories by
position, so a level absent from new data would shift every subsequent
code and silently produce wrong predictions.
"""

import numpy as np
import pandas as pd


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

    # Guard against division by zero. No training row has zero income,
    # but new data could.
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


def prep_for_lgbm(df, categories=None):
    """Prepare a frame for LightGBM.

    Pass `categories` (from `extract_categories` on the training frame)
    whenever preparing data that is not the training set.
    """
    df = df.copy()

    # DAYS_EMPLOYED = 365243 is a placeholder, not a duration.
    # It marks pensioners, so the flag carries real signal.
    df["DAYS_EMPLOYED_ANOM"] = (
        df["DAYS_EMPLOYED"] == 365243
    ).astype("int8")

    df["DAYS_EMPLOYED"] = (
        df["DAYS_EMPLOYED"]
        .replace(365243, np.nan)
    )

    # Reuse training category levels when supplied, so integer codes
    # stay identical across datasets.
    for col in df.select_dtypes(exclude="number").columns:
        if categories is not None and col in categories:
            df[col] = pd.Categorical(df[col], categories=categories[col])
        else:
            df[col] = df[col].astype("category")

    # Halve memory on a 3.8 GB machine.
    for col in df.select_dtypes(include="float64").columns:
        df[col] = df[col].astype("float32")

    return df


def extract_categories(df):
    """Capture the category levels learned from a prepared training frame."""
    return {
        col: df[col].cat.categories
        for col in df.select_dtypes(include="category").columns
    }


def build_features(df, categories=None):
    """The full feature pipeline, in the order it must run."""
    return prep_for_lgbm(
        add_history_indicators(
            add_ratios(df)
        ),
        categories=categories,
    )