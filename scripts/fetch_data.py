"""
Download the raw NFL data needed for the turnover-margin win% model.

Source: nflverse-data (https://github.com/nflverse/nflverse-data), public
GitHub release assets, no auth required.

Pulls:
  - games.csv: game-level schedule/results for every NFL game
  - play_by_play_<season>.csv.gz: play-by-play data for each season in SEASONS,
    used to derive turnovers (interceptions thrown + fumbles lost) per team
    per game.
"""
import os
import urllib.request

SEASONS = [2023, 2024, 2025]
RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download"

FILES = [f"{BASE_URL}/schedules/games.csv"] + [
    f"{BASE_URL}/pbp/play_by_play_{season}.csv.gz" for season in SEASONS
]


def download(url, dest_dir):
    filename = os.path.basename(url)
    dest = os.path.join(dest_dir, filename)
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    return dest


if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    for url in FILES:
        download(url, RAW_DIR)
    print("Done.")
