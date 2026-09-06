"""
Model: how do turnover margin, offensive third-down conversion rate, and
opponent (allowed) third-down conversion rate relate to a team's win
percentage, over the last 3 completed NFL seasons (2023-2025)?

Data: nflverse-data play-by-play and schedules (see fetch_data.py).

Per team per game:
  - turnover_margin = takeaways - giveaways, where giveaways = interceptions
    thrown + fumbles lost while that team had possession, and takeaways =
    the opponent's giveaways in the same game.
  - off_third_down_pct = that team's own 3rd-down conversion rate on offense.
  - opp_third_down_pct = the opponent's 3rd-down conversion rate in the same
    game, i.e. the rate the team's defense allowed.

Produces:
  - data/processed/team_game_turnover_table.csv: one row per team per game,
    with all of the above plus win/loss.
  - data/processed/summary.json: win% by turnover-margin outcome (won /
    lost / tied the turnover battle), overall, regular-season-only, per
    season, plus fitted logistic regression models (turnover-margin-only,
    and a multivariate model adding both third-down rates).
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
            "interception", "fumble_lost", "third_down_converted", "third_down_failed"]
    df = pd.read_csv(path, compression="gzip", low_memory=False, usecols=cols)
    df = df[df["posteam"].notna()]
    df = df.rename(columns={"season_type": "game_type"})
    return df


def per_team_game_stats(pbp):
    g = pbp.groupby(["game_id", "season", "week", "game_type", "posteam"]).agg(
        interceptions=("interception", "sum"),
        fumbles_lost=("fumble_lost", "sum"),
        third_down_conversions=("third_down_converted", "sum"),
        third_down_failed=("third_down_failed", "sum"),
    )
    g["giveaways"] = g["interceptions"] + g["fumbles_lost"]
    g["third_down_attempts"] = g["third_down_conversions"] + g["third_down_failed"]
    g["off_third_down_pct"] = g["third_down_conversions"] / g["third_down_attempts"]
    return g.reset_index()[
        ["game_id", "posteam", "giveaways", "third_down_attempts",
         "third_down_conversions", "off_third_down_pct"]
    ]


def load_schedules():
    df = pd.read_csv(os.path.join(RAW_DIR, "games.csv"), low_memory=False)
    df = df[df["season"].isin(SEASONS)]
    df = df[df["result"].notna()]  # completed games only
    return df


def build_team_game_table():
    stats = pd.concat(
        [per_team_game_stats(load_pbp(s)) for s in SEASONS], ignore_index=True
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
    own = stats[["game_id", "posteam", "giveaways", "off_third_down_pct"]].rename(
        columns={"posteam": "team"}
    )
    opp = stats[["game_id", "posteam", "giveaways", "off_third_down_pct"]].rename(
        columns={"posteam": "opponent", "giveaways": "takeaways", "off_third_down_pct": "opp_third_down_pct"}
    )
    long = long.merge(own, on=["game_id", "team"], how="left")
    long = long.merge(opp, on=["game_id", "opponent"], how="left")

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


# Natural unit to report an odds ratio per, for each feature: +1 net
# turnover, or +10 percentage points of third-down conversion rate (the
# raw per-unit odds ratio for a 0-1 fraction is a 100pp swing, which is
# both unrealistic and unreadable).
FEATURE_INCREMENT = {
    "turnover_margin": 1.0,
    "off_third_down_pct": 0.10,
    "opp_third_down_pct": 0.10,
}


def fit_logistic_model(long, features, label):
    from sklearn.linear_model import LogisticRegression

    df = long[long["tie"] == 0].dropna(subset=features)
    X = df[features].values
    y = df["win"].values

    model = LogisticRegression()
    model.fit(X, y)

    intercept = float(model.intercept_[0])
    coefs = {feat: float(c) for feat, c in zip(features, model.coef_[0])}
    odds_ratio_per_increment = {
        feat: round(float(np.exp(c * FEATURE_INCREMENT[feat])), 4) for feat, c in coefs.items()
    }

    result = {
        "features": features,
        "intercept": round(intercept, 4),
        "coefficients": {feat: round(c, 4) for feat, c in coefs.items()},
        "odds_ratio_per_increment": odds_ratio_per_increment,
        "increment": FEATURE_INCREMENT,
        "n_games": int(len(df)),
    }

    print(f"\n=== Logistic Regression: {label} ===")
    print(f"n = {result['n_games']} team-games")
    print(f"intercept = {result['intercept']}")
    for feat in features:
        step = FEATURE_INCREMENT[feat]
        print(f"  {feat}: coef = {result['coefficients'][feat]}, "
              f"odds ratio per +{step} = {odds_ratio_per_increment[feat]}x")
    return result, model


def turnover_only_curve(model):
    # Predicted win% by turnover margin alone (other features held at 0,
    # i.e. this is only meaningful for the single-feature model).
    margins = list(range(-5, 6))
    probs = model.predict_proba(np.array(margins).reshape(-1, 1))[:, 1]
    return [
        {"turnover_margin": m, "predicted_win_pct": round(float(p) * 100, 1)}
        for m, p in zip(margins, probs)
    ]


def multivariate_scenarios(long, model, features):
    # Illustrative predicted win% for a handful of realistic scenarios,
    # since a 3-feature model can't be shown as a single curve. Third-down
    # rates are drawn from the actual league distribution for context.
    df = long[long["tie"] == 0].dropna(subset=features)
    league_avg_off = df["off_third_down_pct"].mean()
    league_avg_opp = df["opp_third_down_pct"].mean()

    scenarios = []
    for margin in [-2, -1, 0, 1, 2]:
        for off_label, off_pct in [("below-avg offense (-10pp)", league_avg_off - 0.10),
                                    ("league-avg offense", league_avg_off),
                                    ("above-avg offense (+10pp)", league_avg_off + 0.10)]:
            for def_label, opp_pct in [("above-avg defense (-10pp allowed)", league_avg_opp - 0.10),
                                        ("league-avg defense", league_avg_opp),
                                        ("below-avg defense (+10pp allowed)", league_avg_opp + 0.10)]:
                X = np.array([[margin, off_pct, opp_pct]])
                prob = float(model.predict_proba(X)[0, 1])
                scenarios.append({
                    "turnover_margin": margin,
                    "offense": off_label,
                    "defense": def_label,
                    "off_third_down_pct": round(off_pct * 100, 1),
                    "opp_third_down_pct": round(opp_pct * 100, 1),
                    "predicted_win_pct": round(prob * 100, 1),
                })
    return {
        "league_avg_off_third_down_pct": round(float(league_avg_off) * 100, 1),
        "league_avg_opp_third_down_pct": round(float(league_avg_opp) * 100, 1),
        "scenarios": scenarios,
    }


if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    long = build_team_game_table()
    long.to_csv(os.path.join(PROCESSED_DIR, "team_game_turnover_table.csv"), index=False)

    overall = summarize(long, label="All games (REG + POST), 2023-2025")
    reg_only = summarize(long, game_type_filter="REG", label="Regular season only, 2023-2025")

    logit_uni, model_uni = fit_logistic_model(
        long, ["turnover_margin"], "win ~ turnover_margin"
    )
    logit_uni["predicted_win_pct_by_margin"] = turnover_only_curve(model_uni)

    multi_features = ["turnover_margin", "off_third_down_pct", "opp_third_down_pct"]
    logit_multi, model_multi = fit_logistic_model(
        long, multi_features,
        "win ~ turnover_margin + off_third_down_pct + opp_third_down_pct",
    )
    logit_multi["example_scenarios"] = multivariate_scenarios(long, model_multi, multi_features)

    per_season = {}
    for s in SEASONS:
        sub = long[long["season"] == s]
        per_season[s] = summarize(sub, label=f"Season {s}")

    out = {
        "overall": overall.reset_index().to_dict(orient="records"),
        "regular_season_only": reg_only.reset_index().to_dict(orient="records"),
        "per_season": {str(s): per_season[s].reset_index().to_dict(orient="records") for s in SEASONS},
        "logistic_model_turnover_only": logit_uni,
        "logistic_model_multivariate": logit_multi,
    }
    with open(os.path.join(PROCESSED_DIR, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\nSaved data/processed/team_game_turnover_table.csv and summary.json")
