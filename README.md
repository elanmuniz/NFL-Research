# NFL Turnover Margin vs. Win Percentage

A model quantifying how strongly winning the turnover battle in an NFL game
relates to winning that game, using the last 3 completed seasons
(2023-2025, regular season + playoffs).

## Method

For every team in every completed game:

- **giveaways** = interceptions thrown + fumbles lost while that team had
  the ball
- **takeaways** = the opponent's giveaways in that same game
- **turnover_margin** = takeaways - giveaways

Each team-game is bucketed as having *won*, *lost*, or *tied* the turnover
battle, and win rate is computed within each bucket. A logistic regression
(`win ~ turnover_margin`) is also fit to give a smooth predicted win
probability for any margin value, not just the three buckets.

Data comes from [nflverse-data](https://github.com/nflverse/nflverse-data)
(play-by-play + schedules), pulled directly from its public GitHub release
assets.

## Results (2023-2025, all games)

| Turnover margin outcome | Games | Win % |
|---|---|---|
| Won the turnover battle | 653 | **76.0%** |
| Lost the turnover battle | 653 | 24.0% |
| Tied the turnover battle | 402 | 50.0% |

Regular season only: 76.2% vs. 23.8%.

By season:

| Season | Won margin win % | Lost margin win % |
|---|---|---|
| 2023 | 73.6% | 26.4% |
| 2024 | 77.0% | 23.0% |
| 2025 | 77.4% | 22.6% |

### Logistic regression: `win ~ turnover_margin`

- coefficient = 0.7552 (odds ratio ≈ **2.13x** per net turnover)
- intercept ≈ 0 (a game with an even turnover margin is a coin flip, as expected)

Predicted win probability by turnover margin:

| Margin | Win % | Margin | Win % |
|---|---|---|---|
| -5 | 2.2% | +1 | 68.0% |
| -4 | 4.6% | +2 | 81.9% |
| -3 | 9.4% | +3 | 90.6% |
| -2 | 18.1% | +4 | 95.4% |
| -1 | 32.0% | +5 | 97.8% |
| 0 | 50.0% | | |

Full numbers: [`data/processed/summary.json`](data/processed/summary.json).
Row-level data (one row per team per game):
[`data/processed/team_game_turnover_table.csv`](data/processed/team_game_turnover_table.csv).

## Reproducing

```bash
pip install -r requirements.txt
python scripts/fetch_data.py          # downloads raw data into data/raw/
python scripts/build_turnover_model.py # writes data/processed/*
```

To analyze a different window of seasons, edit `SEASONS` at the top of
`scripts/fetch_data.py` and `scripts/build_turnover_model.py`.
