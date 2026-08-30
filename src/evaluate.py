"""Evaluation utilities for model analysis.

Each function takes a frame with `actual` (0/1) and `prob` (predicted
probability) columns, plus whatever grouping columns are needed.
"""

import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss


def group_auc(df, col, min_n=1000):
    """AUC, default rate and mean prediction within each group of `col`.

    Groups smaller than `min_n`, or with only one class present, are
    skipped — their AUC would be noise or undefined.
    """
    rows = []

    for val, g in df.groupby(col, observed=True):
        if len(g) < min_n or g["actual"].nunique() < 2:
            continue

        rows.append({
            "group": val,
            "n": len(g),
            "default_rate": g["actual"].mean(),
            "auc": roc_auc_score(g["actual"], g["prob"]),
            "mean_predicted": g["prob"].mean(),
        })

    return pd.DataFrame(rows).sort_values("auc")


def capture_rate(df, group_col, threshold):
    """Rejection rate and share of defaults caught, per group, at one
    global threshold. Shows whether a single cutoff treats groups equally.
    """
    rows = []

    for group, g in df.groupby(group_col, observed=True):
        rejected = g["prob"] >= threshold
        total_defaults = g["actual"].sum()

        rows.append({
            "group": group,
            "rejected_rate": rejected.mean(),
            "defaults_caught": (
                g.loc[rejected, "actual"].sum() / total_defaults
                if total_defaults > 0 else 0
            ),
        })

    return pd.DataFrame(rows)


def capture_curve(df, rates=(0.05, 0.10, 0.20, 0.30, 0.50)):
    """Defaults caught and good customers rejected at each rejection rate.

    This is the table a lender actually decides on: they deploy a cutoff,
    not a probability.
    """
    rows = []
    n_defaults = df["actual"].sum()
    n_good = len(df) - n_defaults

    for rate in rates:
        n_reject = int(len(df) * rate)
        riskiest = df.nlargest(n_reject, "prob")
        caught = riskiest["actual"].sum()

        rows.append({
            "reject_rate": rate,
            "defaults_caught": caught / n_defaults,
            "good_rejected": (n_reject - caught) / n_good,
            "lift_vs_random": (caught / n_defaults) / rate,
        })

    return pd.DataFrame(rows)


def calibration_summary(df):
    """Does a predicted probability of X actually mean X?

    Brier is compared against the score of always predicting the base
    rate, since Brier looks deceptively small when the event is rare.
    """
    actual_rate = df["actual"].mean()
    baseline_brier = actual_rate * (1 - actual_rate)

    return {
        "mean_predicted": df["prob"].mean(),
        "actual_rate": actual_rate,
        "brier": brier_score_loss(df["actual"], df["prob"]),
        "brier_baseline": baseline_brier,
    }