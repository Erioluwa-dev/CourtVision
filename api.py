import os
import requests
import time
from exceptions import PlayerNotFoundError, InvalidStatError

API_KEY = os.environ.get("BALLDONTLIE_API_KEY", "")
BASE_URL = "https://api.balldontlie.io/v1"
HEADERS = {"Authorization": API_KEY} if API_KEY else {}

if not API_KEY:
    print("WARNING: BALLDONTLIE_API_KEY not set — API requests may be rejected.")

known_stats = {
    "Shai Gilgeous-Alexander": {"ppg": 31.1, "apg": 6.6},
    "Stephen Curry":           {"ppg": 26.6, "apg": 4.7},
    "LeBron James":            {"ppg": 23.7, "apg": 8.3},
    "Victor Wembanyama":       {"ppg": 25.0, "apg": 3.1},
    "Jayson Tatum":            {"ppg": 21.8, "apg": 5.3},
    "Kevin Durant":            {"ppg": 26.0, "apg": 4.8},
    "Kawhi Leonard":           {"ppg": 27.9, "apg": 3.6},
}

MAX_RETRIES = 3


def search_player(name):
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.get(
                f"{BASE_URL}/players",
                params={"search": name},
                headers=HEADERS,
                timeout=10
            )
            if response.status_code == 429:
                if attempt < MAX_RETRIES:
                    wait = 60 * (attempt + 1)
                    print(f"Rate limited. Waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(wait)
                    continue
                print("Rate limited — max retries exhausted.")
                return None
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.Timeout:
            print(f"Timeout fetching: {name}")
            return None
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error: {e}")
            return None
        except requests.exceptions.ConnectionError:
            print("No internet connection")
            return None
        if not data["data"]:
            raise PlayerNotFoundError(f"No player found: {name}")
        return data["data"][0]
    return None

def build_player_dict(search_name, full_name):
    print(f"Fetching {search_name}...")
    player = search_player(search_name)
    if not player:
        return None
    stats = known_stats.get(full_name)
    if not stats:
        raise InvalidStatError(f"No stats for: {full_name}")
    return {
        "name": f"{player['first_name']} {player['last_name']}",
        "team": player["team"]["full_name"],
        "jersey": player.get("jersey_number", "?"),
        "ppg": stats["ppg"],
        "apg": stats["apg"],
        "is_all_star": False
    }
