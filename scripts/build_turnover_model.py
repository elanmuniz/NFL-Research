"""
Model: how does winning the turnover margin relate to a team's win
percentage, over the last 3 completed NFL seasons (2023-2025)?

Data: nflverse-data play-by-play and schedules (see fetch_data.py).

Turnover margin for a team in a game is defined as:
    takeaways - giveaways
where giveaways = interceptions thrown + fumbles lost while that team had
possession, and takeaways = the opponent's giveaways in the same game.

Produces:
  - data/processed/team_game_turnover_table.csv: one row per team per game,
    with giveaways, takeaways, turnover_margin, and win/loss.
  - data/processed/summary.json: win% by turnover-margin outcome (won /
    lost / tied the turnover battle), overall, regular-season-only, per
    season, plus a fitted logistic regression model.
"""
import json
import os

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
SEASONS = [2023, 2024, 2025]


def load_pbp(season):
    path = os.path.join(RAW_DIR, f"play_by_play_{season}.csv.gz")
    cols = ["game_id", "season", "week", "season_type", "posteam", "defteam",
            "interception", "fumble_lost"]
    df = pd.read_csv(path, compression="gzip", low_memory=False, usecols=cols)
    df = df[df["posteam"].notna()]
    df = df.rename(columns={"season_type": "game_type"})
    return df


def per_team_game_giveaways(pbp):
    g = pbp.groupby(["game_id", "season", "week", "game_type", "posteam"]).agg(
        interceptions=("interception", "sum"),
        fumbles_lost=("fumble_lost", "sum"),
    )
    g["giveaways"] = g["interceptions"] + g["fumbles_lost"]
    return g.reset_index()[["game_id", "posteam", "giveaways"]]


def load_schedules():
    df = pd.read_csv(os.path.join(RAW_DIR, "games.csv"), low_memory=False)
    df = df[df["season"].isin(SEASONS)]
    df = df[df["result"].notna()]  # completed games only
    return df


def build_team_game_table():
    giveaways = pd.concat(
        [per_team_game_giveaways(load_pbp(s)) for s in SEASONS], ignore_index=True
    )

    sched = load_schedules()
    keep_cols = ["game_id", "season", "week", "game_type", "home_team", "away_team",
                 "home_score", "away_score"]

    home = sched[keep_cols].copy()
    home["team"], home["opponent"] = home["home_team"], home["away_team"]
    home["team_score"], home["opp_score"] = home["home_score"], home["away_score"]

    away = sched[keep_cols].copy()
    away["team"], away["opponent"] = away["away_team"], away["home_team"]
    away["team_score"], away["opp_score"] = away["away_score"], away["home_score"]

    long = pd.concat([home, away], ignore_index=True)[
        ["game_id", "season", "week", "game_type", "team", "opponent", "team_score", "opp_score"]
    ]
    long["win"] = (long["team_score"] > long["opp_score"]).astype(int)
    long["tie"] = (long["team_score"] == long["opp_score"]).astype(int)

    # Join on game_id + team only: game_id is already globally unique, and
    # schedule game_type (REG/WC/DIV/CON/SB) doesn't match pbp's season_type
    # (REG/POST) for playoff games, which would silently drop them if used
    # as a join key.
    long = long.merge(
        giveaways.rename(columns={"posteam": "team", "giveaways": "giveaways"}),
        on=["game_id", "team"], how="left",
    )
    long = long.merge(
        giveaways.rename(columns={"posteam": "opponent", "giveaways": "takeaways"}),
        on=["game_id", "opponent"], how="left",
    )
    long["giveaways"] = long["giveaways"].fillna(0)
    long["takeaways"] = long["takeaways"].fillna(0)
    long["turnover_margin"] = long["takeaways"] - long["giveaways"]
    return long


def summarize(long, game_type_filter=None, label=""):
    df = long.copy()
    if game_type_filter == "REG":
        df = df[df["game_type"] == "REG"]
    df = df[df["tie"] == 0]  # exclude tied games (rare) from win% denominators

    df["bucket"] = np.where(
        df["turnover_margin"] > 0, "won_margin",
        np.where(df["turnover_margin"] < 0, "lost_margin", "tied_margin"),
    )
    summary = df.groupby("bucket").agg(games=("win", "size"), wins=("win", "sum"))
    summary["win_pct"] = (summary["wins"] / summary["games"] * 100).round(1)

    print(f"\n=== {label} ===")
    print(summary)
    return summary


def fit_logistic_model(long):
    from sklearn.linear_model import LogisticRegression

    df = long[long["tie"] == 0]
    X = df[["turnover_margin"]].values
    y = df["win"].values

    model = LogisticRegression()
    model.fit(X, y)

    intercept = float(model.intercept_[0])
    coef = float(model.coef_[0][0])

    margins = list(range(-5, 6))
    probs = model.predict_proba(np.array(margins).reshape(-1, 1))[:, 1]
    curve = [
        {"turnover_margin": m, "predicted_win_pct": round(float(p) * 100, 1)}
        for m, p in zip(margins, probs)
    ]

    result = {
        "intercept": round(intercept, 4),
        "coefficient_per_turnover": round(coef, 4),
        "odds_ratio_per_turnover": round(float(np.exp(coef)), 4),
        "n_games": int(len(df)),
        "predicted_win_pct_by_margin": curve,
    }

    print("\n=== Logistic Regression: win ~ turnover_margin ===")
    print(f"n = {result['n_games']} team-games")
    print(f"intercept = {result['intercept']}, coef = {result['coefficient_per_turnover']}")
    print(f"Each extra net turnover multiplies odds of winning by {result['odds_ratio_per_turnover']}x")
    for row in curve:
        print(f"  margin {row['turnover_margin']:+d}: predicted win% = {row['predicted_win_pct']}")
    return result


if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    long = build_team_game_table()
    long.to_csv(os.path.join(PROCESSED_DIR, "team_game_turnover_table.csv"), index=False)

    overall = summarize(long, label="All games (REG + POST), 2023-2025")
    reg_only = summarize(long, game_type_filter="REG", label="Regular season only, 2023-2025")
    logit = fit_logistic_model(long)

    per_season = {}
    for s in SEASONS:
        sub = long[long["season"] == s]
        per_season[s] = summarize(sub, label=f"Season {s}")

    out = {
        "overall": overall.reset_index().to_dict(orient="records"),
        "regular_season_only": reg_only.reset_index().to_dict(orient="records"),
        "per_season": {str(s): per_season[s].reset_index().to_dict(orient="records") for s in SEASONS},
        "logistic_model": logit,
    }
    with open(os.path.join(PROCESSED_DIR, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\nSaved data/processed/team_game_turnover_table.csv and summary.json")
