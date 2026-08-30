"""Loading and aggregation of the Home Credit tables."""

from pathlib import Path
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def load_applications(filename="application_train.csv"):
    return pd.read_csv(RAW / filename)


def aggregate_bureau(bureau):
    b = bureau.copy()

    num_agg = b.groupby("SK_ID_CURR").agg({
        "DAYS_CREDIT": ["count", "mean", "min", "max"],
        "CREDIT_DAY_OVERDUE": ["mean", "max"],
        "AMT_CREDIT_SUM": ["sum", "mean", "max"],
        "AMT_CREDIT_SUM_DEBT": ["sum", "mean"],
        "AMT_CREDIT_SUM_OVERDUE": ["sum", "max"],
        "CNT_CREDIT_PROLONG": ["sum"],
    })

    num_agg.columns = [
        "BURO_" + "_".join(c).upper()
        for c in num_agg.columns
    ]

    active = (
        b[b["CREDIT_ACTIVE"] == "Active"]
        .groupby("SK_ID_CURR")
        .size()
    )

    closed = (
        b[b["CREDIT_ACTIVE"] == "Closed"]
        .groupby("SK_ID_CURR")
        .size()
    )

    num_agg["BURO_ACTIVE_COUNT"] = active
    num_agg["BURO_CLOSED_COUNT"] = closed

    num_agg[
        ["BURO_ACTIVE_COUNT", "BURO_CLOSED_COUNT"]
    ] = num_agg[
        ["BURO_ACTIVE_COUNT", "BURO_CLOSED_COUNT"]
    ].fillna(0)

    return num_agg.reset_index()


def aggregate_previous(previous):
    p = previous.copy()

    # Numerical features
    num_agg = p.groupby("SK_ID_CURR").agg({
        "SK_ID_PREV": ["count"],
        "AMT_ANNUITY": ["mean", "max"],
        "AMT_APPLICATION": ["mean", "max"],
        "AMT_CREDIT": ["mean", "max"],
        "AMT_DOWN_PAYMENT": ["mean"],
        "AMT_GOODS_PRICE": ["mean"],
        "DAYS_DECISION": ["mean", "min", "max"],
    })

    num_agg.columns = [
        "PREV_" + "_".join(c).upper()
        for c in num_agg.columns
    ]

    # Contract status counts
    status = pd.crosstab(
        p["SK_ID_CURR"],
        p["NAME_CONTRACT_STATUS"]
    )

    for col in [
        "Approved",
        "Canceled",
        "Refused",
        "Unused offer"
    ]:
        if col not in status.columns:
            status[col] = 0

    status = status[
        ["Approved", "Canceled", "Refused", "Unused offer"]
    ]

    status.columns = [
        "PREV_STATUS_" + col.upper().replace(" ", "_")
        for col in status.columns
    ]

    # Approval / refusal / cancellation rates
    total = status.sum(axis=1)

    status["PREV_APPROVAL_RATE"] = (
        status["PREV_STATUS_APPROVED"] / total
    )

    status["PREV_REFUSAL_RATE"] = (
        status["PREV_STATUS_REFUSED"] / total
    )

    status["PREV_CANCELLATION_RATE"] = (
        status["PREV_STATUS_CANCELED"] / total
    )

    # Combine everything
    result = num_agg.join(status, how="left")

    return result.reset_index()


def build_dataset():
    """Load all tables, aggregate, join. Returns one dataframe."""
    app = load_applications()

    bureau = pd.read_csv(RAW / "bureau.csv")
    buro_agg = aggregate_bureau(bureau)
    del bureau

    previous = pd.read_csv(RAW / "previous_application.csv")
    prev_agg = aggregate_previous(previous)
    del previous

    return (
        app
        .merge(buro_agg, on="SK_ID_CURR", how="left")
        .merge(prev_agg, on="SK_ID_CURR", how="left")
    )