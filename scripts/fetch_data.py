"""
Download the raw NFL data needed for the turnover-margin and power-rankings
models.

Source: nflverse-data (https://github.com/nflverse/nflverse-data), public
GitHub release assets, no auth required.

Pulls:
  - games.csv: game-level schedule/results for every NFL game
  - play_by_play_<season>.csv.gz: play-by-play data for each season in
    TRAIN_SEASONS + [CURRENT_SEASON].

nflverse republishes play_by_play_<CURRENT_SEASON>.csv.gz throughout the
season (updated within ~a day of each game), so re-running this script
during the 2026 season pulls in newly played weeks. Before the season's
first game there is no file yet for CURRENT_SEASON — that 404 is expected
and this script skips it with a warning rather than failing.
"""
import os
import urllib.error
import urllib.request

TRAIN_SEASONS = [2023, 2024, 2025]
CURRENT_SEASON = 2026
RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download"

FILES = [f"{BASE_URL}/schedules/games.csv"] + [
    f"{BASE_URL}/pbp/play_by_play_{season}.csv.gz"
    for season in TRAIN_SEASONS + [CURRENT_SEASON]
]


def download(url, dest_dir):
    filename = os.path.basename(url)
    dest = os.path.join(dest_dir, filename)
    print(f"Downloading {url} -> {dest}")
    try:
        urllib.request.urlretrieve(url, dest)
        return dest
    except urllib.error.HTTPError as e:
        print(f"  skipped ({e.code}): not published yet")
        return None


if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    for url in FILES:
        download(url, RAW_DIR)
    print("Done.")
