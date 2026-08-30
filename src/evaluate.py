"""Evaluation utilities for model analysis."""

from sklearn.metrics import roc_auc_score


def group_auc(df, col, min_n=1000):
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

    return (
        __import__("pandas")
        .DataFrame(rows)
        .sort_values("auc")
    )


def capture_rate(df, group_col, threshold):
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

    return __import__("pandas").DataFrame(rows)