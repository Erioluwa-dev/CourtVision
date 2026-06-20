import json
import os
import time
from api import build_player_dict
from exceptions import PlayerNotFoundError, InvalidStatError

PLAYERS_FILE = '/content/drive/MyDrive/courtvision/players.json'

players_to_fetch = [
    {"search": "Gilgeous", "full_name": "Shai Gilgeous-Alexander"},
    {"search": "Curry",    "full_name": "Stephen Curry"},
    {"search": "LeBron",   "full_name": "LeBron James"},
    {"search": "Wembanyama","full_name": "Victor Wembanyama"},
    {"search": "Tatum",    "full_name": "Jayson Tatum"},
    {"search": "Durant",   "full_name": "Kevin Durant"},
    {"search": "Kawhi",    "full_name": "Kawhi Leonard"},
]

all_stars = {
    "Shai Gilgeous-Alexander": True,
    "Stephen Curry": True,
    "LeBron James": True,
    "Victor Wembanyama": True,
    "Kevin Durant": True,
    "Kawhi Leonard": True,
    "Jayson Tatum": True,
}

def save_players(players, filename=PLAYERS_FILE):
    with open(filename, "w") as f:
        json.dump(players, f, indent=2)
    print(f"Saved {len(players)} players to {filename}")

def load_players(filename=PLAYERS_FILE):
    try:
        with open(filename, "r") as f:
            players = json.load(f)
        print(f"Loaded {len(players)} players from file")
        return players
    except FileNotFoundError:
        print("No saved file found")
        return None
    except json.JSONDecodeError:
        print("File corrupted")
        return None

def fetch_or_load_players(filename=PLAYERS_FILE):
    if os.path.exists(filename):
        print("Found saved data — loading from file")
        return load_players(filename)
    print("No file found — fetching from API")
    players = []
    for entry in players_to_fetch:
        try:
            p = build_player_dict(entry["search"], entry["full_name"])
            if p:
                p["is_all_star"] = all_stars.get(entry["full_name"], False)
                players.append(p)
        except (PlayerNotFoundError, InvalidStatError) as e:
            print(f"Skipping {entry['full_name']}: {e}")
        time.sleep(3)
    save_players(players, filename)
    return playerss
