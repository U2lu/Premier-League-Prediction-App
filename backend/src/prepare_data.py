from matplotlib.pylab import sort
import pandas as pd 
from pathlib import Path

# file paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_FILE = DATA_DIR / "premier_league_matches1.1.csv"
OUTPUT_FILE = DATA_DIR / "premier_league_features.csv" # Path to the output CSV file for Premier League match features

def main():

    df = pd.read_csv(INPUT_FILE) # Read the input CSV file into a DataFrame

    required_columns = [ # List of required columns for the DataFrame
        "fixture_id",
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    ]

    missing_columns = set(required_columns) - set(df.columns)# Check for any missing required columns in the DataFrame

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        ) # Raise an error if any required columns are missing

    if df["fixture_id"].duplicated().any():
        raise ValueError("Duplicate fixture_id values found.") # Raise an error if there are any duplicate fixture_id values

    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize() # Convert the "date" column to datetime format and normalize it to remove time information

    df = df.sort_values(
        ["date", "fixture_id"]
        ).reset_index(drop=True) # Sort the DataFrame by date and reset the index

    # for the match outcome column, to decide H, A, D (home win, away win, draw) based on the goals scored by home and away teams

    df["match_outcome"] = "D" # Initialize a new column "match_outcome" with default value "D" for draw

    df.loc[df["home_goals"] > df["away_goals"], 
           "match_outcome"
           ] = "H" # Set "match_outcome" to "H" for home team wins

    df.loc[df["home_goals"] < df["away_goals"], 
           "match_outcome"
           ] = "A" # Set "match_outcome" to "A" for away team wins

    # Calculate points for home and away teams based on match outcomes

    df["home_points"] = 0 # Create a new column "home_points" with default value 0
    df["away_points"] = 0 # Create a new column "away_points" with default value 0

    home_win = df["home_goals"] > df["away_goals"] # Create a boolean mask for home team wins
    away_win = df["home_goals"] < df["away_goals"] # Create a boolean mask for away team wins
    draw = df["home_goals"] == df["away_goals"] # Create a boolean mask for draws

    df.loc[home_win, "home_points"] = 3 # Assign 3 points to home team for wins
    df.loc[away_win, "away_points"] = 3 # Assign 3 points to away team for wins

    df.loc[draw, "home_points"] = 1 # Assign 1 point to home team for draws
    df.loc[draw, "away_points"] = 1 # Assign 1 point to away team for draws

    # Create a history record for each team and each match

    home_history = pd.DataFrame({
        "fixture_id": df["fixture_id"],
        "date": df["date"],
        "team": df["home_team"],
        "role": "home",
        "goals_scored": df["home_goals"],
        "goals_conceded": df["away_goals"],
        "points": df["home_points"],
    }) # DataFrame to store home team history

    away_history = pd.DataFrame({
        "fixture_id": df["fixture_id"],
        "date": df["date"],
        "team": df["away_team"],
        "role": "away",
        "goals_scored": df["away_goals"],
        "goals_conceded": df["home_goals"],
        "points": df["away_points"],
    }) # DataFrame to store away team history

    history = pd.concat(
        [home_history, away_history], 
        ignore_index=True
    ) # Concatenate home and away history DataFrames

    history = history.sort_values(
        ["team", "date"]
    ).reset_index(drop=True) # Sort the history DataFrame by team and date, then reset the index

    # Each team should play once per date, this checks for any duplicate entries for the same team on the same date

    repeated_team_dates = history.duplicated(
        subset=["team", "date"],
        keep=False
    ) # Create a boolean mask to identify duplicate entries for the same team on the same date

    if repeated_team_dates.any():
        examples = history.loc[
            repeated_team_dates,
            ["date", "team", "fixture_id"]
        ].head(10) # Get the repeated entries from the history DataFrame
        
        raise ValueError(
            "A team appears in multiple matches on the same date. "
            "Inspect these records before proceeding:\n"
            f"{examples.to_string(index=False)}"
        ) # Raise an error if there are any duplicate team-date entries

    # Calculate each teams previous five-match form, which includes goals scored, goals conceded, and points earned in the last five matches

    history = history.sort_values(
        ["team", "date", "fixture_id"]
    ).reset_index(drop=True) # Sort the history DataFrame by team, date, and fixture_id, then reset the index

    grouped = history.groupby("team", sort= False) # Group the history DataFrame by team

    for column in [
        "goals_scored",
        "goals_conceded",
        "points"
    ]:
        history[f"{column}_avg_5"] = grouped[column].transform(
            lambda values: values.shift(1).rolling(window=5, min_periods=1).mean()
        ) # Calculate the rolling mean(average) of the last 5 matches for each team and store it in a new column

    # Seperate Home and Away team features

    home_features = history.loc[
        history["role"] == "home",
        [
            "fixture_id",
            "team",
            "goals_scored_avg_5",
            "goals_conceded_avg_5",
            "points_avg_5",
        ],

    ].rename(columns={
        "team": "home_team",
        "goals_scored_avg_5": "home_goals_avg_5",
        "goals_conceded_avg_5": "home_conceded_avg_5",
        "points_avg_5": "home_points_avg_5",
    }) # Create a DataFrame for home team features and rename the columns appropriately

    away_features = history.loc[
        history["role"] == "away",
        [
            "fixture_id",
            "team",
            "goals_scored_avg_5",
            "goals_conceded_avg_5",
            "points_avg_5",
        ],
    ].rename(columns={
        "team": "away_team",
        "goals_scored_avg_5": "away_goals_avg_5",
        "goals_conceded_avg_5": "away_conceded_avg_5",
        "points_avg_5": "away_points_avg_5",
    })

    # Attach features to the correct fixture

    df = df.merge(
        home_features,
        on=["fixture_id", "home_team"],
        how="left", 
        validate="1:1"
    ) # Merge home team features into the main DataFrame

    df = df.merge(
        away_features,
        on=["fixture_id", "away_team"],
        how="left", # how="left" means that all rows from the left DataFrame (df) will be kept, and matching rows from the right DataFrame (away_features) will be added. If there is no match, NaN values will be filled in for the right DataFrame's columns.
        validate="1:1" # validate="1:1" ensures that each fixture_id and away_team combination in the main DataFrame (df) corresponds to exactly one row in the away_features DataFrame. If there are any duplicates or mismatches, it will raise an error.
    ) # Merge away team features into the main DataFrame

    feature_columns = [
        "home_goals_avg_5",
        "home_conceded_avg_5",
        "home_points_avg_5",
        "away_goals_avg_5",
        "away_conceded_avg_5",
        "away_points_avg_5",
    ]

    # A team that has not played any matches yet will have NaN values for its features. We can fill these NaN values with 0, which indicates that the team has no previous performance data.
    df[feature_columns] = df[feature_columns].fillna(0) # Fill NaN values in feature columns with 0

    # All rows represent home-team fixtures
    df["home_advantage"] = 1 # Create a new column "home_advantage" with value 1 for all rows, indicating that these are home-team fixtures

    # Save the final DataFrame with features to a CSV file after verification
    if len(df) != len(pd.read_csv(INPUT_FILE)):
        raise RuntimeError("The number of matches changed unexpectedly.") # Provide a sanity check to ensure that the number of matches in the final DataFrame is the same as in the input file

    if df[feature_columns].isnull().any().any():
        raise RuntimeError("Missing values found in feature columns.") # Provide a sanity check to ensure that there are no NaN values in the feature columns

    df.to_csv(OUTPUT_FILE, index=False) # Save the final DataFrame to a CSV file without the index

    print("\nFeature engineering completed successfully.")
    print("Matches processed:", len(df))
    print("Output file:", OUTPUT_FILE)

    print("\nOutcome distribution:")
    print(df["match_outcome"].value_counts())

    print("\nSample features:")
    print(df[[
        "date",
        "home_team",
        "away_team",
        "home_goals_avg_5",
        "home_conceded_avg_5",
        "home_points_avg_5",
        "away_goals_avg_5",
        "away_conceded_avg_5",
        "away_points_avg_5",
        "match_outcome",
    ]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
