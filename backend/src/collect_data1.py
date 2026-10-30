import pandas as pd
import requests
from pathlib import Path

BASE_URL = (
    "https://raw.githubusercontent.com/"
    "openfootball/football.json/master"
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" 
OUTPUT_FILE = DATA_DIR / "premier_league_matches1.1.csv"

# Ten completed Premier League seasons: 2015/16–2024/25
SEASONS = [
    f"{year}-{str(year + 1)[-2:]}"
    for year in range(2015, 2025)
]


def fetch_season(season):
    url = f"{BASE_URL}/{season}/en.1.json"

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    data = response.json()
    matches = []

    for match in data.get("matches", []): 
        score = match.get("score", {}).get("ft")

        # Ignore fixtures without a completed full-time score.
        if not score or len(score) != 2:
            continue

        if score[0] is None or score[1] is None:
            continue

        matches.append({
            "date": match["date"],
            "season": int(season[:4]),
            "status": "FT",
            "home_team": match["team1"],
            "away_team": match["team2"],
            "home_goals": int(score[0]),
            "away_goals": int(score[1]),
        })

    return matches


def main(): 
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_matches = []

    for season in SEASONS:
        try:
            matches = fetch_season(season)
            all_matches.extend(matches)

            print(f"{season}: collected {len(matches)} completed matches")

        except (requests.RequestException, ValueError, KeyError) as error:
            print(f"Could not collect {season}: {error}")

    if not all_matches:
        raise RuntimeError(
            "No matches were collected. Check your internet connection "
            "and the source URLs."
        )

    df = pd.DataFrame(all_matches)

    # Standardize dates and remove any repeated fixtures.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_goals", "away_goals"])

    df = df.drop_duplicates(
        subset=["date", "home_team", "away_team"]
    )

    df = df.sort_values("date").reset_index(drop=True)

    # Keep the existing CSV schema compatible with your analysis script.
    df.insert(0, "fixture_id", range(1, len(df) + 1))

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nCollection complete.")
    print(f"Total matches: {len(df)}")
    print(f"Seasons collected: {df['season'].nunique()}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
