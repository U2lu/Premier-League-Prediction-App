import os
import time
import requests
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = 'https://v3.football.api-sports.io'

headers = {
    "x-apisports-key": API_KEY
}

LEAGUE_ID = 39  # Premier League

# Start with a small test before downloading all seasons. 
SEASONS = [2023,2024]  # List of seasons to fetch data for

def fetch_fixtures(season):
    "Fetch Premier League fixtures for a given season."

    url = f"{BASE_URL}/fixtures"

    params = {
        "league": LEAGUE_ID,
        "season": season
    }

    response = requests.get(url, headers=headers, params=params, timeout=30)

    response.raise_for_status()  # Raise an error for bad responses
    data = response.json()

    if data.get("errors"):
        raise RuntimeError(f"API error for {season}: {data['errors']}")

    return data.get("response", [])

def main(): 
    os.makedirs("data", exist_ok=True)  # Create a directory to store data

    all_matches = []

    for season in SEASONS:
        print(f"Fetching matches for season {season}/{season + 1}...")

        fixtures = fetch_fixtures(season)

        print(f"Fetched {len(fixtures)} matches for season.")

        for match in fixtures: 
            home = match["teams"]["home"]
            away = match["teams"]["away"]
            goals = match["goals"]
            fixture = match["fixture"]
            league = match["league"]

            all_matches.append({
                "fixture_id": fixture["id"],
                "date": fixture["date"],
                "season": league["season"],
                "status": fixture["status"]["short"],
                "home_team": home["name"],
                "away_team": away["name"],
                "home_goals": goals["home"],
                "away_goals": goals["away"]
            })

        # To avoid hitting the API rate limit, we can add a short delay between requests.
        time.sleep(1)  # Sleep for 1 second

    # Convert the list of matches to a DataFrame
    df = pd.DataFrame(all_matches)

    # Keep completed matches with known scores
    completed = df[
        (df["status"] == "FT")
        & (df["home_goals"].notna())
        & (df["away_goals"].notna())
        ].copy()

    # Save the completed matches to a CSV file
    output_path = "data/premier_league_matches.csv"
    completed.to_csv(output_path, index=False)

    print(f"\nTotal matches fetched: {len(df)}")
    print(f"Completed matches saved: {len(completed)}")
    print(f"Data saved to {output_path}")
    print("\nPreview:")
    print(completed.head())

if __name__ == "__main__":
    main()