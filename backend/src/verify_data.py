import pandas as pd
from pathlib import Path

DATA_FILE = (
    Path(__file__).resolve().parent.parent 
    / "data" 
    / "premier_league_matches1.1.csv" # Path to the CSV file containing Premier League match data
)

df = pd.read_csv(DATA_FILE) # Read the CSV file into a DataFrame

print("Total matches in dataset:", len(df)) # Print the total number of matches in the dataset
print("Unique seasons in dataset:", df["season"].nunique()) # Print the number of unique seasons in the dataset
print("\nMatches per season:")
print(df["season"].value_counts().sort_index()) # Print the number of matches per season, sorted by season

print("\nMissing seasons:")
print(df.isnull().sum()) # Print the number of missing values in each column of the DataFrame

print("\nDuplicate matches:")
print(df.duplicated(subset=["date", "home_team", "away_team"]).sum()) # Print the number of duplicate matches based on date, home team, and away team

print("\nDate range of matches:")
print(f"From: {df['date'].min()}")
print(f"To: {df['date'].max()}") # Print the date range of matches in the dataset

print("\nColumns:", df.columns.tolist()) # Print the list of columns in the DataFrame