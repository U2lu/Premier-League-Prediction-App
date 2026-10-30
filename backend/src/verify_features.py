
import pandas as pd
from pathlib import Path

DATA_FILE = (
    Path(__file__).resolve().parent.parent 
    / "data"
    / "premier_league_features.csv"
) # Path to the CSV file containing Premier League match features

df = pd.read_csv(DATA_FILE)

features = [
    "home_goals_avg_5",
    "home_conceded_avg_5",
    "home_points_avg_5",
    "away_goals_avg_5",
    "away_conceded_avg_5",
    "away_points_avg_5",
]

print("Feature variation:")
print(df[features].describe().round(2)) # Print summary statistics for the specified features, rounded to 2 decimal places

print("\nNumber of zero values:")
print((df[features] == 0).sum())

print("\nExample after the first 100 matches:")
print(
    df.iloc[100:105][
        ["date", "home_team", "away_team"] + features
    ].to_string(index=False)
)

print("\nMissing feature values:")
print(df[features].isnull().sum())
