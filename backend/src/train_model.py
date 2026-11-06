
import pandas as pd
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier


# 1. File paths and configuration

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

INPUT_FILE = DATA_DIR / "premier_league_features.csv"

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "home_goals_avg_5",
    "home_conceded_avg_5",
    "home_points_avg_5",
    "away_goals_avg_5",
    "away_conceded_avg_5",
    "away_points_avg_5",
]

# Consistent numeric labels for both algorithms.
LABEL_MAPPING = {
    "D": 0,
    "H": 1,
    "A": 2,
}

CLASS_NAMES = ["Draw", "Home Win", "Away Win"]

# 2. Load and prepare the dataset

def load_data():

    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").reset_index(drop=True)

    # Check that all required columns exist.
    required_columns = FEATURES + [
        "match_outcome",
        "season",
        "date",
    ]

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in dataset: {sorted(missing)}"
        )

    if df[FEATURES].isnull().any().any():
        raise ValueError("Missing values found in model features.")

    if not df["match_outcome"].isin(LABEL_MAPPING).all():
        raise ValueError("Unexpected match outcome found.")

    df["target"] = df["match_outcome"].map(LABEL_MAPPING)

    return df

# 3. Split data chronologically

def split_data(df):

    train_df = df[df["season"] <= 2022].copy()

    test_df = df[df["season"] >= 2023].copy()

    X_train = train_df[FEATURES]
    y_train = train_df["target"]

    X_test = test_df[FEATURES]
    y_test = test_df["target"]

    print("\nDATASET SPLIT")
    print("----------------------------")

    print("Training matches:", len(X_train))
    print("Testing matches:", len(X_test))

    print(
        "Training seasons:",
        train_df["season"].min(),
        "to",
        train_df["season"].max()
    )

    print(
        "Testing seasons:",
        test_df["season"].min(),
        "to",
        test_df["season"].max()
    )

    return X_train, X_test, y_train, y_test

# 4. Evaluate a trained model

def evaluate_model(name, model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)

    print(f"\n{name} RESULTS")
    print("=" * 40)

    accuracy = accuracy_score(y_test, predictions)

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    loss = log_loss(
        y_test,
        probabilities,
        labels=[0, 1, 2]
    )

    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"Macro F1 Score:    {macro_f1:.4f}")
    print(f"Log Loss:          {loss:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1, 2]
        )
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "log_loss": loss,
    }

# 5. Train and compare models

def main():

    df = load_data()

    X_train, X_test, y_train, y_test = split_data(df)

    # Baseline: always predict the most common training outcome.
    baseline = DummyClassifier(strategy="most_frequent")

    baseline.fit(X_train, y_train)

    baseline_results = evaluate_model(
        "BASELINE",
        baseline,
        X_test,
        y_test
    )

    # Random Forest
    random_forest = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    print("\nTraining Random Forest...")

    random_forest.fit(X_train, y_train)

    rf_results = evaluate_model(
        "RANDOM FOREST",
        random_forest,
        X_test,
        y_test
    )

    # XGBoost
    xgboost = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining XGBoost...")

    xgboost.fit(X_train, y_train)

    xgb_results = evaluate_model(
        "XGBOOST",
        xgboost,
        X_test,
        y_test
    )

    # 6. Compare results

    results = pd.DataFrame(
        [baseline_results, rf_results, xgb_results],
        index=["Baseline", "Random Forest", "XGBoost"]
    )

    print("\nMODEL COMPARISON")
    print("=" * 40)

    print(results.round(4))

    # 7. Save trained models

    joblib.dump(
        {
            "model": random_forest,
            "features": FEATURES,
            "label_mapping": LABEL_MAPPING,
        },
        MODEL_DIR / "random_forest.pkl"
    )

    joblib.dump(
        {
            "model": xgboost,
            "features": FEATURES,
            "label_mapping": LABEL_MAPPING,
        },
        MODEL_DIR / "xgboost.pkl"
    )

    print("\nModels saved successfully.")

    print("Random Forest:", MODEL_DIR / "random_forest.pkl")
    print("XGBoost:", MODEL_DIR / "xgboost.pkl")


if __name__ == "__main__":
    main()
