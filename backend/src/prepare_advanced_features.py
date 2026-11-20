import pandas as pd
import numpy as np

from pathlib import Path




# File paths
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "premier_league_matches1.1.csv"
OUTPUT_FILE = BASE_DIR / "data" / "premier_league_advanced_features.csv"

WINDOW_SIZE = 5  # Number of matches to consider for moving averages

# Load and prepare the dataset
def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found at {INPUT_FILE}.") # Check if the input file exists

    df = pd.read_csv(INPUT_FILE)
    
    required_columns = [
        "fixture_id",
        "date",
        "season",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    ]

    missing = set(required_columns) - set(df.columns) # Check for missing columns in the dataset

    if missing:
        raise ValueError(f"Missing required columns in the dataset: {missing}")

    if df[required_columns].isnull().any().any():
        raise ValueError("Missing values found in required columns.")

    if df["fixture_id"].duplicated().any():
        raise ValueError("Duplicate fixture_id values found in the dataset.")

    df["date"] = pd.to_datetime(df["date"])  # Convert date column to datetime

    df = df.sort_values(by=["date", "fixture_id"]).reset_index(drop=True)  # Sort by season and date

    if (df["home_goals"] < 0).any() or (df["away_goals"] < 0).any():
        raise ValueError("Negative values found in goals columns.")

    return df

# Build Team Match History
def build_team_history(df):
    home = pd.DataFrame({
        "fixture_id": df["fixture_id"],
        "date": df["date"],
        "team": df["home_team"],
        "opponent": df["away_team"],
        "venue": "home",
        "goals_scored": df["home_goals"],
        "goals_conceded": df["away_goals"],
    })

    away = pd.DataFrame({
        "fixture_id": df["fixture_id"],
        "date": df["date"],
        "team": df["away_team"],
        "opponent": df["home_team"],
        "venue": "away",
        "goals_scored": df["away_goals"],
        "goals_conceded": df["home_goals"],
    })

    history = pd.concat(
        [home, away],
        ignore_index=True  
    )# Combine home and away data

    history["goal_difference"] = history["goals_scored"] - history["goals_conceded"]  # Calculate goal difference

    history["points"] = np.select(
        [
            history["goals_scored"] > history["goals_conceded"],
            history["goals_scored"] == history["goals_conceded"],
        ],
        [3, 1], # Assign points based on match outcome (win, draw)
        default=0 # Assign 0 points for a loss
    ) # Calculate points based on match outcome

    history["win"] = (
        history["goals_scored"] > history["goals_conceded"]).astype(int)  # Create a binary column for wins

    return history

# Calculate Previous Match Statistics
def add_rolling_features(history, group_columns, prefix):
    history = history.sort_values(
        ["date", "fixture_id"]
    ).copy() 

    grouped = history.groupby(
        group_columns,
        sort=False
    )

    feature_columns = {
        "goals_scored": f"{prefix}_goals_avg_5",
        "goals_conceded": f"{prefix}_conceded_avg_5",
        "points": f"{prefix}_points_avg_5",
        "win": f"{prefix}_win_rate_5",
        "goal_difference": f"{prefix}_goal_difference_avg_5",
    }# Create a mapping of source columns to output feature names

    for source_column, output_column in feature_columns.items():

        history[output_column] = grouped[source_column].transform(
            lambda values: (
                values.shift(1)
                .rolling(WINDOW_SIZE, min_periods=1)
                .mean()
            )
        ) # Calculate the rolling mean for each feature based on the previous matches

    # Number of previous matches available in this group.
    history[f"{prefix}_matches_available"] = grouped[
        "fixture_id"
    ].transform(
        lambda values: (
            values.shift(1)
            .rolling(WINDOW_SIZE, min_periods=1)
            .count()
        )
    )

    return history


# BUILD FEATURES FOR HOME AND AWAY TEAMS

def build_advanced_features(df):
    history = build_team_history(df)

    # Overall form: previous five matches regardless of venue.
    history = add_rolling_features(
        history,
        ["team"],
        "overall",
    )

    # Venue-specific form: previous five matches at that venue.
    history = add_rolling_features(
        history,
        ["team", "venue"],
        "venue",
    )

    # Days since each team's previous match.
    history = history.sort_values(
        ["team", "date", "fixture_id"]
    ).copy()

    history["rest_days"] = (
        history.groupby("team")["date"]
        .diff()
        .dt.total_seconds() 
        .div(86400) 
    ) # Calculate the number of days since each team's previous match

    # Keep only features belonging to each team's current fixture.
    home_features = history[
        history["venue"] == "home"
    ].copy()

    away_features = history[
        history["venue"] == "away"
    ].copy() 

    home_columns = [
        "fixture_id",
        "overall_goals_avg_5",
        "overall_conceded_avg_5",
        "overall_points_avg_5",
        "overall_win_rate_5",
        "overall_goal_difference_avg_5",
        "venue_goals_avg_5",
        "venue_conceded_avg_5",
        "venue_points_avg_5",
        "venue_win_rate_5",
        "venue_goal_difference_avg_5",
        "venue_matches_available",
        "rest_days",
    ]

    away_columns = [
        "fixture_id",
        "overall_goals_avg_5",
        "overall_conceded_avg_5",
        "overall_points_avg_5",
        "overall_win_rate_5",
        "overall_goal_difference_avg_5",
        "venue_goals_avg_5",
        "venue_conceded_avg_5",
        "venue_points_avg_5",
        "venue_win_rate_5",
        "venue_goal_difference_avg_5",
        "venue_matches_available",
        "rest_days",
    ]

    home_features = home_features[home_columns].rename(
        columns={
            column: f"home_{column}"
            for column in home_columns
            if column != "fixture_id"
        }
    ) 

    away_features = away_features[away_columns].rename(
        columns={
            column: f"away_{column}"
            for column in away_columns
            if column != "fixture_id"
        } # Rename columns to indicate home and away features
    ) 

    result = df.merge(
        home_features,
        on="fixture_id",
        how="left",
        validate="one_to_one",
    ) # Merge the original dataframe with home features based on fixture_id

    result = result.merge(
        away_features,
        on="fixture_id",
        how="left",
        validate="one_to_one",
    )

    result["match_outcome"] = np.select(
        [
            result["home_goals"] > result["away_goals"],
            result["home_goals"] == result["away_goals"],
        ],
        ["H", "D"],
        default="A",
    )
  
 # HANDLE MISSING HISTORICAL FEATURES

    # No previous match means rest days are unknown.
    result["home_rest_days"] = result["home_rest_days"].fillna(0)
    result["away_rest_days"] = result["away_rest_days"].fillna(0)

    # Teams with no previous matches have no available form statistics.
    # Zero is used as a placeholder, while the match-count features
    # tell the model how much historical information was available.

    rolling_feature_columns = [
        column for column in result.columns
        if column.endswith((
            "_goals_avg_5",
            "_conceded_avg_5",
            "_points_avg_5",
            "_win_rate_5",
            "_goal_difference_avg_5",
        )) 
    ]

    result[rolling_feature_columns] = (
        result[rolling_feature_columns].fillna(0)
    ) 

    # Validate that all model features are now complete.
    feature_columns = [
        column for column in result.columns
        if column.startswith(("home_", "away_"))
        and column not in [
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
        ]
    ]

    missing_counts = result[feature_columns].isnull().sum()

    if missing_counts.any():
        print("\nMissing feature values:")
        print(missing_counts[missing_counts > 0])
        raise ValueError("Missing feature values remain.")


    return result


# MAIN

def main():
    print("Loading historical Premier League dataset...")

    df = load_data()

    print(f"Matches loaded: {len(df)}")

    print("\nBuilding advanced features...")

    result = build_advanced_features(df)

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nAdvanced feature engineering completed.")
    print(f"Matches processed: {len(result)}")
    print(f"Features created: {len(result.columns)}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nNew feature columns:")

    for column in result.columns:
        if column.startswith(("home_", "away_")):
            if column not in [
                "home_team", "away_team",
                "home_goals", "away_goals",
            ]:
                print(column)

    print("\nSample advanced features:")

    sample_columns = [
        "date",
        "home_team",
        "away_team",
        "home_overall_goals_avg_5",
        "home_venue_goals_avg_5",
        "home_venue_win_rate_5",
        "away_overall_goals_avg_5",
        "away_venue_goals_avg_5",
        "away_venue_win_rate_5",
        "home_rest_days",
        "away_rest_days",
        "match_outcome",
    ]

    print(result[sample_columns].head(10).to_string(index=False))


if __name__ == "__main__":
    main()