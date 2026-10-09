import os
import requests 
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")

BASE_URL = 'https://v3.football.api-sports.io'

params = {
    "league": "39",  # Premier League
    "season": "2023"  # Season year
} 

headers = {
    "x-apisports-key": API_KEY
}

response = requests.get(
    f"{BASE_URL}/fixtures", 
    headers=headers, 
    params=params,
    timeout=10  # Set a timeout of 10 seconds
)

print("Status Code:", response.status_code)
data = response.json()

if response.status_code == 200 and not data.get("errors"): 
    fixtures = data.get("response", [])

    print("Number of fixtures:", len(fixtures))

    if fixtures:
        print("\nFirst Fixture:")
        print("Date:", fixtures[0]["fixture"]["date"])
        print("Home:", fixtures[0]["teams"]["home"]["name"])
        print("Away:", fixtures[0]["teams"]["away"]["name"])
        print(
            "Score:",
            fixtures[0]["goals"]["home"],
            "-",
            fixtures[0]["goals"]["away"]
        )
else:
    print("Api errors:", data.get("errors"))
