"""
Weekly-updating NFL Power Rankings for the 2026 season.

Ranks teams on five metrics commonly cited as leading indicators of winning
football:
  1. Turnover margin
  2. Passer rating differential (own passer rating thrown vs. allowed)
  3. Yards per play & success rate differential (own offense vs. defense allowed)
  4. Red zone touchdown efficiency differential (own offense vs. defense allowed)
  5. Opponent-adjusted EPA/play rating (a DVOA-style proxy -- see caveat below)

DVOA caveat
-----------
Real DVOA (Football Outsiders / FTN) is a proprietary, licensed metric that
weights every play by down/distance/score/time-remaining and opponent
strength using a non-public methodology. It is not available in any free
dataset, so it cannot be reproduced exactly here. Instead this model computes
an open "opponent-adjusted EPA/play" rating using the Simple Rating System
(SRS) method -- the same schedule-adjustment idea DVOA is built on (strip out
whether a team's numbers came against good or bad opponents), applied to each
team's per-game net EPA/play margin, without DVOA's situational weighting.
This is labeled `adj_epa_rating` throughout: read it as a DVOA-style proxy,
not licensed DVOA data.

How the ranking is built
-------------------------
1. Fit fixed weights once, from history: compute the five differentials
   above for every team-season in TRAIN_SEASONS (2023-2025, regular season),
   z-score each one, and fit a linear regression against that team's average
   point differential per game. The resulting standardized coefficients
   become the Power Score weights.
2. Apply those fixed weights every week of CURRENT_SEASON: compute the same
   five differentials from that team's games so far, z-score using the
   *training* distribution (so scores stay on a comparable scale week to
   week and year to year), and combine into one Power Score per team.

Usage
-----
    python scripts/fetch_data.py                    # refresh raw data
    python scripts/power_rankings.py                 # rank through the
                                                       # latest played week
    python scripts/power_rankings.py --through-week 5
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")

TRAIN_SEASONS = [2023, 2024, 2025]
CURRENT_SEASON = 2026

DIFF_FEATURES = [
    "turnover_margin_pg",
    "passer_rating_diff",
    "ypp_diff",
    "success_rate_diff",
    "redzone_td_pct_diff",
    "adj_epa_rating",
]

PBP_COLS = [
    "game_id", "season", "week", "season_type", "posteam", "defteam",
    "interception", "fumble_lost",
    "play_type", "yards_gained", "success", "epa",
    "pass_attempt", "sack", "qb_spike", "complete_pass", "incomplete_pass",
    "passing_yards", "pass_touchdown",
    "drive", "fixed_drive_result", "drive_inside20",
]

SCRIMMAGE_TYPES = {"pass", "run"}


# --------------------------------------------------------------------------
# Raw per-game counting stats
# --------------------------------------------------------------------------

def load_pbp(season):
    path = os.path.join(RAW_DIR, f"play_by_play_{season}.csv.gz")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, compression="gzip", low_memory=False, usecols=PBP_COLS)
    df = df.rename(columns={"season_type": "game_type"})
    return df


def per_team_game_raw(pbp):
    """One row per (game_id, team) with season-cumulable RAW COUNTS.

    Rates (passer rating, YPP, success rate, red zone TD%) must be derived
    from *summed* counts across games, not averaged per-game percentages,
    to avoid over-weighting low-volume games.
    """
    turnovers = (
        pbp[pbp["posteam"].notna()]
        .groupby(["game_id", "season", "week", "game_type", "posteam"])
        .agg(giveaways=("interception", "sum"))
    )
    turnovers["giveaways"] += (
        pbp[pbp["posteam"].notna()]
        .groupby(["game_id", "season", "week", "game_type", "posteam"])["fumble_lost"]
        .sum()
    )
    turnovers = turnovers.reset_index()

    scrim = pbp[pbp["play_type"].isin(SCRIMMAGE_TYPES) & pbp["posteam"].notna()]
    scrim_stats = scrim.groupby(["game_id", "posteam"]).agg(
        scrimmage_plays=("epa", "size"),
        scrimmage_yards=("yards_gained", "sum"),
        success_count=("success", "sum"),
        epa_sum=("epa", "sum"),
    ).reset_index()

    pass_plays = pbp[(pbp["play_type"] == "pass") & (pbp["sack"] == 0)
                      & (pbp["qb_spike"] == 0) & pbp["posteam"].notna()]
    pass_stats = pass_plays.groupby(["game_id", "posteam"]).agg(
        pass_attempts=("pass_attempt", "size"),
        completions=("complete_pass", "sum"),
        pass_yards=("passing_yards", "sum"),
        pass_tds=("pass_touchdown", "sum"),
        ints=("interception", "sum"),
    ).reset_index()

    drives = (
        pbp[pbp["posteam"].notna() & pbp["drive"].notna()]
        .groupby(["game_id", "posteam", "drive"])
        .agg(inside20=("drive_inside20", "max"), result=("fixed_drive_result", "first"))
        .reset_index()
    )
    drives = drives[drives["inside20"] == 1]
    redzone = drives.groupby(["game_id", "posteam"]).agg(
        redzone_trips=("result", "size"),
        redzone_tds=("result", lambda s: (s == "Touchdown").sum()),
    ).reset_index()

    out = turnovers[["game_id", "season", "week", "game_type", "posteam", "giveaways"]]
    for extra in [scrim_stats, pass_stats, redzone]:
        out = out.merge(extra, on=["game_id", "posteam"], how="left")

    count_cols = ["scrimmage_plays", "scrimmage_yards", "success_count", "epa_sum",
                  "pass_attempts", "completions", "pass_yards", "pass_tds", "ints",
                  "redzone_trips", "redzone_tds"]
    out[count_cols] = out[count_cols].fillna(0)
    return out


def load_schedules():
    df = pd.read_csv(os.path.join(RAW_DIR, "games.csv"), low_memory=False)
    return df


def build_team_game_table(season):
    """Long table: one row per team per REG-season game they've played,
    with their own raw counts and the opponent's (== what they allowed)."""
    pbp = load_pbp(season)
    if pbp is None:
        return None

    raw = per_team_game_raw(pbp)

    sched = load_schedules()
    sched = sched[(sched["season"] == season) & (sched["game_type"] == "REG")
                  & sched["result"].notna()]
    if sched.empty:
        return None

    keep = ["game_id", "season", "week", "home_team", "away_team", "home_score", "away_score"]
    home = sched[keep].rename(columns={"home_team": "team", "away_team": "opponent",
                                        "home_score": "team_score", "away_score": "opp_score"})
    away = sched[keep].rename(columns={"away_team": "team", "home_team": "opponent",
                                        "away_score": "team_score", "home_score": "opp_score"})
    long = pd.concat([home, away], ignore_index=True)
    long["win"] = (long["team_score"] > long["opp_score"]).astype(int)
    long["point_margin"] = long["team_score"] - long["opp_score"]

    count_cols = ["giveaways", "scrimmage_plays", "scrimmage_yards", "success_count", "epa_sum",
                  "pass_attempts", "completions", "pass_yards", "pass_tds", "ints",
                  "redzone_trips", "redzone_tds"]
    own = raw[["game_id", "posteam"] + count_cols].rename(columns={"posteam": "team"})
    opp = raw[["game_id", "posteam"] + count_cols].rename(
        columns={"posteam": "opponent", **{c: f"opp_{c}" for c in count_cols}}
    )
    long = long.merge(own, on=["game_id", "team"], how="left")
    long = long.merge(opp, on=["game_id", "opponent"], how="left")

    all_count_cols = count_cols + [f"opp_{c}" for c in count_cols]
    long[all_count_cols] = long[all_count_cols].fillna(0)
    long["takeaways"] = long["opp_giveaways"]
    return long


# --------------------------------------------------------------------------
# Rate stats from summed counts, and the SRS opponent-adjusted EPA rating
# --------------------------------------------------------------------------

def passer_rating(attempts, completions, yards, tds, ints):
    attempts = np.where(attempts == 0, np.nan, attempts)
    a = np.clip((completions / attempts - 0.3) * 5, 0, 2.375)
    b = np.clip((yards / attempts - 3) * 0.25, 0, 2.375)
    c = np.clip((tds / attempts) * 20, 0, 2.375)
    d = np.clip(2.375 - (ints / attempts) * 25, 0, 2.375)
    return (a + b + c + d) / 6 * 100


def solve_srs(games):
    """games: list of (team, opponent, margin) meaning team's per-play EPA
    margin over opponent in one game. Returns {team: adjusted_rating},
    mean-centered at 0, via least squares (Simple Rating System method)."""
    teams = sorted({t for g in games for t in (g[0], g[1])})
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    A = np.zeros((len(games) + 1, n))
    b = np.zeros(len(games) + 1)
    for row, (team, opp, margin) in enumerate(games):
        A[row, idx[team]] = 1
        A[row, idx[opp]] = -1
        b[row] = margin
    A[-1, :] = 1  # sum-to-zero constraint for identifiability
    b[-1] = 0

    ratings, *_ = np.linalg.lstsq(A, b, rcond=None)
    return dict(zip(teams, ratings))


def cumulative_team_stats(long, through_week=None):
    """Aggregate a build_team_game_table() long frame into one row per team
    with season-to-date differentials, through an optional week cutoff."""
    df = long if through_week is None else long[long["week"] <= through_week]
    if df.empty:
        return pd.DataFrame()

    count_cols = ["giveaways", "takeaways", "scrimmage_plays", "scrimmage_yards",
                  "success_count", "epa_sum", "pass_attempts", "completions",
                  "pass_yards", "pass_tds", "ints", "redzone_trips", "redzone_tds"]
    opp_count_cols = [f"opp_{c}" for c in count_cols if f"opp_{c}" in df.columns]

    agg = df.groupby("team").agg(
        games_played=("team", "size"),
        point_margin_pg=("point_margin", "mean"),
        win_pct=("win", "mean"),
        **{c: (c, "sum") for c in count_cols},
        **{c: (c, "sum") for c in opp_count_cols},
    ).reset_index()

    agg["turnover_margin_pg"] = (agg["takeaways"] - agg["giveaways"]) / agg["games_played"]

    agg["off_ypp"] = agg["scrimmage_yards"] / agg["scrimmage_plays"]
    agg["def_ypp_allowed"] = agg["opp_scrimmage_yards"] / agg["opp_scrimmage_plays"]
    agg["ypp_diff"] = agg["off_ypp"] - agg["def_ypp_allowed"]

    agg["off_success_rate"] = agg["success_count"] / agg["scrimmage_plays"]
    agg["def_success_rate_allowed"] = agg["opp_success_count"] / agg["opp_scrimmage_plays"]
    agg["success_rate_diff"] = agg["off_success_rate"] - agg["def_success_rate_allowed"]

    agg["off_passer_rating"] = passer_rating(
        agg["pass_attempts"], agg["completions"], agg["pass_yards"], agg["pass_tds"], agg["ints"])
    agg["def_passer_rating_allowed"] = passer_rating(
        agg["opp_pass_attempts"], agg["opp_completions"], agg["opp_pass_yards"],
        agg["opp_pass_tds"], agg["opp_ints"])
    agg["passer_rating_diff"] = agg["off_passer_rating"] - agg["def_passer_rating_allowed"]

    agg["off_redzone_td_pct"] = agg["redzone_tds"] / agg["redzone_trips"]
    agg["def_redzone_td_pct_allowed"] = agg["opp_redzone_tds"] / agg["opp_redzone_trips"]
    agg["redzone_td_pct_diff"] = agg["off_redzone_td_pct"] - agg["def_redzone_td_pct_allowed"]

    # Opponent-adjusted EPA rating (DVOA-style proxy) via SRS on per-game
    # net EPA/play margin, using only the games included in this cutoff.
    per_game_epa_margin = []
    for _, row in df.iterrows():
        own_epa_pp = row["epa_sum"] / row["scrimmage_plays"] if row["scrimmage_plays"] else 0.0
        opp_epa_pp = row["opp_epa_sum"] / row["opp_scrimmage_plays"] if row["opp_scrimmage_plays"] else 0.0
        per_game_epa_margin.append((row["team"], row["opponent"], own_epa_pp - opp_epa_pp))
    srs = solve_srs(per_game_epa_margin)
    # np.linalg.lstsq's floating-point result isn't bit-stable run to run
    # (BLAS summation order varies by platform/thread count) -- round away
    # noise far below anything meaningful so re-running on unchanged data
    # doesn't produce a spurious diff (the weekly workflow only commits
    # when something actually changed).
    agg["adj_epa_rating"] = agg["team"].map(srs).round(6)

    return agg


# --------------------------------------------------------------------------
# Training: fit fixed Power Score weights from 2023-2025 history
# --------------------------------------------------------------------------

def train_weights():
    from sklearn.linear_model import LinearRegression

    rows = []
    for season in TRAIN_SEASONS:
        long = build_team_game_table(season)
        if long is None:
            continue
        season_stats = cumulative_team_stats(long)
        season_stats["season"] = season
        rows.append(season_stats)
    train_df = pd.concat(rows, ignore_index=True)

    means = train_df[DIFF_FEATURES].mean()
    stds = train_df[DIFF_FEATURES].std()
    z = (train_df[DIFF_FEATURES] - means) / stds

    model = LinearRegression()
    model.fit(z.values, train_df["point_margin_pg"].values)
    r2 = model.score(z.values, train_df["point_margin_pg"].values)

    weights = dict(zip(DIFF_FEATURES, model.coef_.tolist()))

    # Each feature's own (univariate) correlation with point margin, since
    # the joint weights above are compressed by multicollinearity -- these
    # five metrics overlap heavily (adj_epa_rating alone already reflects
    # yardage, success, scoring and turnovers), so a feature can be a
    # strong standalone signal even where its joint weight looks small.
    univariate_r = {
        feat: round(float(train_df[feat].corr(train_df["point_margin_pg"])), 4)
        for feat in DIFF_FEATURES
    }

    # Round the same way and for the same reason as adj_epa_rating above:
    # absorb lstsq's run-to-run float noise so unchanged data reproduces
    # byte-identical JSON.
    return {
        "means": {k: round(v, 8) for k, v in means.to_dict().items()},
        "stds": {k: round(v, 8) for k, v in stds.to_dict().items()},
        "weights": {k: round(v, 8) for k, v in weights.items()},
        "univariate_r_with_point_margin": univariate_r,
        "intercept": round(float(model.intercept_), 8),
        "r_squared": round(float(r2), 4),
        "n_team_seasons": int(len(train_df)),
        "train_seasons": TRAIN_SEASONS,
    }, train_df


def power_score(stats_df, trained):
    z = (stats_df[DIFF_FEATURES] - pd.Series(trained["means"])) / pd.Series(trained["stds"])
    weights = pd.Series(trained["weights"])
    return z.mul(weights, axis=1).sum(axis=1)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=CURRENT_SEASON,
                         help=f"Season to rank (default: {CURRENT_SEASON}). Pass a season from "
                              f"{TRAIN_SEASONS} to backtest/demo the model on completed data.")
    parser.add_argument("--through-week", type=int, default=None,
                         help="Rank using games through this week only (default: latest played).")
    args = parser.parse_args()

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print(f"Training Power Score weights on {TRAIN_SEASONS} regular-season team-games...")
    trained, train_df = train_weights()
    print(f"  n = {trained['n_team_seasons']} team-seasons, R^2 = {trained['r_squared']}")
    print("  feature: joint weight (standardized) | standalone correlation with point margin")
    for feat, w in trained["weights"].items():
        r = trained["univariate_r_with_point_margin"][feat]
        print(f"    {feat}: {round(w, 3)} | r = {r}")

    with open(os.path.join(PROCESSED_DIR, "power_ranking_weights.json"), "w") as f:
        json.dump(trained, f, indent=2)

    current_long = build_team_game_table(args.season)
    if current_long is None:
        print(f"\nNo completed {args.season} games found yet -- season hasn't started, "
              f"or data/raw/play_by_play_{args.season}.csv.gz hasn't been published/fetched.")
        print("Weights are trained and saved; re-run this script once games have been played "
              "(after running scripts/fetch_data.py to refresh the raw data).")
        raise SystemExit(0)

    latest_week = int(current_long["week"].max())
    through_week = args.through_week or latest_week
    print(f"\nRanking {args.season} teams through week {through_week}...")

    current_stats = cumulative_team_stats(current_long, through_week=through_week)
    current_stats["power_score"] = power_score(current_stats, trained)
    current_stats = current_stats.sort_values("power_score", ascending=False).reset_index(drop=True)
    current_stats.insert(0, "rank", current_stats.index + 1)

    display_cols = ["rank", "team", "games_played", "power_score", "win_pct",
                     "turnover_margin_pg", "passer_rating_diff", "ypp_diff",
                     "success_rate_diff", "redzone_td_pct_diff", "adj_epa_rating"]
    print(current_stats[display_cols].round(3).to_string(index=False))

    out_path = os.path.join(PROCESSED_DIR, f"power_rankings_{args.season}_wk{through_week}.csv")
    numeric_cols = current_stats.select_dtypes(include="number").columns
    current_stats[numeric_cols] = current_stats[numeric_cols].round(8)
    current_stats.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
