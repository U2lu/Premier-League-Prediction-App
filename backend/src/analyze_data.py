import pandas as pd

# load hisorical match data from CSV file
df = pd.read_csv("backend/data/premier_league_matches.csv")
df["date"] = pd.to_datetime(df["date"], utc=True) # convert the date column to datetime format
df = df.sort_values(by="date").reset_index(drop=True) # sort the DataFrame by date and reset the index

# display basic information
print("--- DATASET INFO ---")
print("Number of matches:", len(df))
print("Number of columns:", len(df.columns))

print("\nColumn names:")
print(df.columns.tolist())

# check data types and missing values
print("\n--- Data types and non-null values ---")
df.info()

print("\n--- Missing Values ---")
print(df.isnull().sum())

# check for duplicate fixtures
print("\n--- Duplicate Fixtures ---")
print("Duplicate fixture IDs:", df["fixture_id"].duplicated().sum())

#check matches per season
print("\n--- Matches per Season ---")
print(df["season"].value_counts().sort_index())

# create the match outcome column based on the goals scored by home and away teams
df["match_outcome"] = "D" # Draw by default

df.loc[df["home_goals"] > df["away_goals"], "match_outcome"] = "H" # Home team wins
df.loc[df["home_goals"] < df["away_goals"], "match_outcome"] = "A" # Away team wins

# count each match outcome
print("\n--- Match Outcomes ---")
print(df["match_outcome"].value_counts())

print("\nOutcome Percentage:")
print(
    (df["match_outcome"].value_counts(normalize=True) * 100) 
    .round(2) 
    .sort_index()
) # percentage of each outcome rounded to 2 decimal places

# display sample matches
print("\n--- Sample Matches ---")
print(df[["date", "home_team", "away_team", "home_goals", "away_goals", "match_outcome"]].head(10))
