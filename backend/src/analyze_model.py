import joblib
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    classification_report,
    confusion_matrix,
)

# File paths and configuration
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "premier_league_features.csv"
MODEL_FILE = BASE_DIR / "models" / "random_forest.pkl"

FEATURES = [ 
    "home_goals_avg_5",
    "home_conceded_avg_5",
    "home_points_avg_5",
    "away_goals_avg_5",
    "away_conceded_avg_5",
    "away_points_avg_5",
]

LABEL_MAPPING = {
    "D": 0,
    "H": 1,
    "A": 2,
}

CLASS_NAMES = ["Draw", "Home Win", "Away Win"]

# Load the trained model
def load_model():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found at {DATA_FILE}.") # Check if the data file exists

    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_FILE}.") # Check if the model file exists

    df = pd.read_csv(DATA_FILE)

    required_columns = FEATURES + [
        "date",
        "season",
        "match_outcome",
    ]

    missing = set(required_columns) - set(df.columns) # Check for missing columns in the dataset

    if missing:
        raise ValueError(f"Missing required columns in the dataset: {missing}")

    if df[FEATURES].isnull().any().any():
        raise ValueError("Missing values found in feature columns.")

    if not df["match_outcome"].isin(LABEL_MAPPING.keys()).all():
        raise ValueError("Invalid values found in 'match_outcome' column.") # Check for invalid values in the match_outcome column

    df["target"] = df["match_outcome"].map(LABEL_MAPPING) # Map match outcomes to numeric labels

    saved_model = joblib.load(MODEL_FILE) # Load the trained model from the specified path

    model = saved_model["model"]  # Extract the model from the saved dictionary
    saved_features = saved_model["features"]  # Extract the features from the saved dictionary

    if (saved_features) != (FEATURES):
        raise ValueError("Feature mismatch between saved model and current configuration.") # Check for feature mismatch

    return df, model

# Outcome distribution analysis
def analyze_outcome(df):
    print("\n" + "=" * 65)
    print("1. MATCH OUTCOME DISTRIBUTION BY SEASON")
    print("=" * 65)

    distribution = pd.crosstab(
        df["season"], 
        df["match_outcome"], 
        normalize="index"
    ) * 100

    counts = pd.crosstab(
        df["season"], 
        df["match_outcome"]
    )

    print("\nMatch Outcome Counts by Season:")
    print(counts.to_string())

    print("\nMatch Outcome Distribution by Season (%):")
    print(distribution.round(2).to_string())

# Evaluate model performance
def evaluate_model(df, model):
    print("\n" + "=" * 65)
    print("2. MODEL PERFORMANCE BY TEST SEASON")
    print("=" * 65)

    test_df = df[df["season"] >= 2023].copy()

    if test_df.empty:
        raise ValueError("No test data available for seasons 2023 and beyond.")

    for season in sorted(test_df["season"].unique()): 
        season_df = test_df[test_df["season"] == season]
        X_test = season_df[FEATURES]
        y_test = season_df["target"]

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        logloss = log_loss(y_test, y_proba, labels=[0, 1, 2])

        print(f"\nSEASON: {season}/{str(season + 1)[-2:]}")
        print("-" * 45)

        print(f"Matches: {len(season_df)}")

        print(f"\nSeason {season} Performance:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Balanced Accuracy: {bal_acc:.4f}")
        print(f"F1 Score (Weighted): {f1:.4f}")
        print(f"Log Loss: {logloss:.4f}")

        print("\nClassification Report:")
        print(classification_report(
            y_test, 
            y_pred, 
            target_names=CLASS_NAMES, 
            zero_division=0, 
            labels=[0, 1, 2]
            )
        )

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        print("Confusion Matrix:")
        print(cm)

# Random Forest Feature Importance Analysis
def analyze_feature_importance(model):
    print("\n" + "=" * 65) # print a separator line for better readability
    print(" RANDOM FOREST FEATURE IMPORTANCE ANALYSIS")
    print("=" * 65)

    feature_importances = model.feature_importances_ 
    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False) # Sort features by importance

    print("\nFeature Importance:")
    print(importance_df.to_string(
        index=False, 
        formatters={"Importance": "{:.4f}".format}
    )) # Print feature importance with 4 decimal places

# Predicted vs Actual Outcome Analysis
def analyze_predictions(df, model):
    print("\n" + "=" * 65)
    print(" PREDICTED VS ACTUAL OUTCOME ANALYSIS")
    print("=" * 65)

    test_df = df[df["season"] >= 2023].copy()

    if test_df.empty:
        raise ValueError("No test data available for seasons 2023 and beyond.")

    X_test = (test_df[FEATURES])
    y_pred = model.predict(X_test)

    label_mapping = {
        0: "D",
        1: "H",
        2: "A"
    }

    predicted_labels = pd.Series(
        y_pred,
        index=test_df.index
    ).map(label_mapping) # Map numeric predictions back to original labels

    outcomes = ["D", "H", "A"]

    actual_counts = (
        test_df["match_outcome"]
        .value_counts()
        .reindex(outcomes, fill_value=0)
    )  # Ensure all outcomes are represented
    predicted_counts = (
        predicted_labels
        .value_counts()
        .reindex(outcomes, fill_value=0)
    )  # Ensure all outcomes are represented

    comparison = pd.DataFrame({
        "Actual Outcome": actual_counts,
        "Predicted Outcome": predicted_counts
    }) # Create a DataFrame to compare actual and predicted outcomes

    comparison.index = ["Draw", "Home Win", "Away Win"]
    comparison.index.name = "Outcomes"

    print("\nPredicted vs Actual Outcomes:")
    print(comparison.to_string(index=False))

    print("\nPrediction totals:")
    print(f"Actual matches: {actual_counts.sum()}")
    print(f"Predicted matches: {predicted_counts.sum()}")

# Main function to run all analyses
def main():
    print ("Loading data and Random Forest model...")
    
    df, model = load_model()
    print(f"Total matches in dataset: {len(df)}")
    print(f"Total matches in test set (2023 and beyond): {len(df[df['season'] >= 2023])}")

    analyze_outcome(df)# Run the match outcome distribution analysis
    evaluate_model(df, model)# Run the model evaluation
    analyze_feature_importance(model) # Run the feature importance analysis
    analyze_predictions(df, model) # Run the predicted vs actual outcome analysis

    print("\n" + "=" * 65)
    print("ANALYSIS COMPLETE")
    print("=" * 65)

if __name__ == "__main__":
    main()

    