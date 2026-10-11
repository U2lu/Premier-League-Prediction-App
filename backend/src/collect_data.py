import os # works with the OS, for creating folders and reading environment variables
import time # lets us pause between API requests
import requests # Sends HTTP requests to the API
import pandas as pd # Used for data manipulation and saving to CSV

from dotenv import load_dotenv # Loads environment variables from a .env file

load_dotenv() # reads the .env file and loads the variables into the environment

API_KEY = os.getenv("API_KEY") # Retrieves the API key from the environment variables
BASE_URL = 'https://v3.football.api-sports.io' # Base URL for the API

headers = { # Headers for the API request, including the API key
    "x-apisports-key": API_KEY 
} 

LEAGUE_ID = 39  # Premier League

# Start with a small test before downloading all seasons. 
SEASONS = list(range(2015, 2025))  # List of seasons to fetch data for

def fetch_fixtures(season): # handles the API request to fetch fixtures for a given season
    "Fetch Premier League fixtures for a given season."

    url = f"{BASE_URL}/fixtures" # Endpoint for fetching fixtures, the "f" is for string formatting and allows us to insert variables directly into the string

    params = { # Parameters for the API request, including the league ID and season
        "league": LEAGUE_ID,
        "season": season
    }

    response = requests.get( # Send a GET request to the API with the specified URL, headers, and parameters
        url,  
        headers=headers, 
        params=params, 
        timeout=30 # Set a timeout for the request to avoid hanging indefinitely
    )

    response.raise_for_status()  # Raise an error for bad responses
    data = response.json()

    if data.get("errors"): 
        raise RuntimeError(f"API error for {season}: {data['errors']}") # Raise an error if the API response contains errors

    return data.get("response", []) # Return the list of fixtures from the API response, or an empty list if not found

def main(): # Extracts match data for the specified seasons and saves it to a CSV file
    import os

    os.makedirs("data", exist_ok=True)  # Create a directory to store data


    output_path = "backend/data/premier_league_matches.csv"

    # the ten seasons we want to fetch
    seasons = list(range(2015, 2025))  # List of seasons to fetch data for

    # Load existing matches if the CSV already exists
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        print(f"Existing matches loaded: {len(existing_df)}")
    else:
        existing_df = pd.DataFrame()

    # Identify seasons already present in the dataset
    if not existing_df.empty:
        saved_seasons = set(existing_df["season"].unique()) 
    else:
        saved_seasons = set()

    missing_seasons = [
        season for season in seasons # Identify seasons that are not already saved in the dataset
        if season not in saved_seasons    
    ]

    print("Seasons already saved:", sorted(saved_seasons))
    print("Seasons to download:", missing_seasons)

    all_matches = [] # create an empty list to store all matches

    for season in missing_seasons: # loop through each missing season and fetch the fixtures
        print(f"Fetching matches for season {season}/{season + 1}...")

        try: # fetch the fixtures for the current season
            fixtures = fetch_fixtures(season) 
        except RuntimeError as error:
            print(f"Skipping season {season}: {error}") # skip the season if there is an error fetching the fixtures
            continue
        print(f"Fetched {len(fixtures)} matches for season.")

        for match in fixtures: # process each match in the fetched fixtures
            home = match["teams"]["home"]
            away = match["teams"]["away"]
            goals = match["goals"]
            fixture = match["fixture"]
            league = match["league"]

            all_matches.append({ # append the relevant match data to the all_matches list
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

    # Convert new list of matches to a DataFrame
    new_df = pd.DataFrame(all_matches)

     # Combine existing and new matches
    combined_df = pd.concat(
        [existing_df, new_df],
        ignore_index=True
    )

    if not combined_df.empty:
        # Remove duplicate fixtures using the unique fixture ID
        combined_df = combined_df.drop_duplicates(
            subset="fixture_id",
            keep="last"
        )

        # Keep completed matches with known scores
        completed_df = combined_df[
            (combined_df["status"] == "FT")
            & combined_df["home_goals"].notna()
            & combined_df["away_goals"].notna()
        ].copy()

        # Sort chronologically
        completed_df["date"] = pd.to_datetime(
            completed_df["date"],
            utc=True
        )

        completed_df = completed_df.sort_values("date")

        # Save the updated dataset
        completed_df.to_csv(output_path, index=False)

        print("\n--- COLLECTION SUMMARY ---")
        print("New fixtures downloaded:", len(new_df))
        print("Completed matches saved:", len(completed_df))
        print("Seasons included:", sorted(completed_df["season"].unique()))
        print("File:", output_path)
    else:
        print("No match data was collected.")


if __name__ == "__main__":
    main()   
    # print(f"\nTotal matches fetched: {len(new_df)}")
    # print(f"Completed matches saved: {len(combined_df)}")
    # print(f"Data saved to {output_path}")
    # print("\nPreview:")
    # print(combined_df.head())

