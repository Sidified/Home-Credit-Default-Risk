"""Train the final model from raw data. Run: python -m src.train"""

from pathlib import Path

import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.data import build_dataset
from src.features import build_features, extract_categories


RANDOM_STATE = 42

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "lgbm_final.joblib"
)


def main():
    print("Building dataset...")
    df = build_dataset()

    X = df.drop(columns=["TARGET", "SK_ID_CURR"])
    y = df["TARGET"]

    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Categories are learned from the development set only, then reused
    # for the test set. LightGBM encodes categories by position, so
    # letting the test set build its own levels would silently shift codes.
    X_dev = build_features(X_dev)
    categories = extract_categories(X_dev)
    X_test = build_features(X_test, categories=categories)[X_dev.columns]

    print(
        f"Training on {X_dev.shape[0]} rows, "
        f"{X_dev.shape[1]} features..."
    )

    model = lgb.LGBMClassifier(
        random_state=RANDOM_STATE,
        n_estimators=300,
        learning_rate=0.05,
        verbose=-1,
        n_jobs=2,
    )

    model.fit(X_dev, y_dev)

    auc = roc_auc_score(
        y_test,
        model.predict_proba(X_test)[:, 1],
    )

    print(f"Test AUC: {auc:.4f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)

    # Save the column order and category levels alongside the model.
    # Without both, predictions on new data can be silently wrong.
    joblib.dump(
        {
            "model": model,
            "columns": list(X_dev.columns),
            "categories": categories,
        },
        MODEL_PATH,
    )

    print(f"Saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()