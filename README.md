# NFL Research

Two related models, both built on [nflverse-data](https://github.com/nflverse/nflverse-data)
(play-by-play + schedules), pulled directly from its public GitHub release assets:

1. [Turnover margin vs. win percentage](#nfl-turnover-margin-vs-win-percentage) —
   a historical (2023-2025) analysis of how turnover margin and third-down
   conversion rate relate to winning a game.
2. [2026 season power rankings](#2026-season-power-rankings) — a
   weekly-updating model that ranks all 32 teams through the 2026 season
   using turnover margin, passer rating differential, yards/play & success
   rate, red zone efficiency, and an opponent-adjusted efficiency rating.
   Updates automatically after each week's games and publishes to a live
   [GitHub Pages site](#automatic-weekly-updates--a-live-site) — see that
   section for the one-time setup.

## Setup

```bash
pip install -r requirements.txt
python scripts/fetch_data.py   # downloads raw data into data/raw/
```

`fetch_data.py` pulls `games.csv` (schedules/results) and
`play_by_play_<season>.csv.gz` for 2023-2025 plus the current season
(2026). During the 2026 season, nflverse republishes that file within about
a day of each game, so re-running `fetch_data.py` is how you pull in newly
played weeks.

## NFL Turnover Margin vs. Win Percentage

A model quantifying how strongly turnover margin and third-down conversion
rate relate to winning an NFL game, using the last 3 completed seasons
(2023-2025, regular season + playoffs).

### Method

For every team in every completed game:

- **giveaways** = interceptions thrown + fumbles lost while that team had
  the ball
- **takeaways** = the opponent's giveaways in that same game
- **turnover_margin** = takeaways - giveaways
- **off_third_down_pct** = that team's own 3rd-down conversion rate on offense
- **opp_third_down_pct** = the opponent's 3rd-down conversion rate in the
  same game, i.e. the rate the team's *defense* allowed

Each team-game is bucketed as having *won*, *lost*, or *tied* the turnover
battle, and win rate is computed within each bucket. Two logistic
regressions are also fit:

1. `win ~ turnover_margin` — a smooth predicted win probability for any
   margin value, not just the three buckets.
2. `win ~ turnover_margin + off_third_down_pct + opp_third_down_pct` — adds
   both offensive and defensive third-down performance to see how much of
   win probability each factor explains on its own, holding the others
   fixed.

### Results (2023-2025, all games)

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

#### Logistic regression: `win ~ turnover_margin`

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

#### Multivariate logistic regression: `win ~ turnover_margin + off_third_down_pct + opp_third_down_pct`

League averages over 2023-2025: **38.8%** third-down conversion rate, both
on offense and (symmetrically) on defense.

| Feature | Coefficient | Odds ratio |
|---|---|---|
| turnover_margin | 0.8297 | **2.29x** per net turnover |
| off_third_down_pct | 4.7947 | **1.62x** per +10 percentage points |
| opp_third_down_pct | -4.7947 | **0.62x** per +10 percentage points allowed |

Turnover margin's effect barely changes once third-down rates are added
(2.29x vs. 2.13x alone) — it captures a mostly independent source of win
probability, not something that's just a proxy for good third-down play.

Predicted win% at turnover margin = 0, by offensive/defensive third-down
tier (10pp = 10 percentage points relative to league average):

| | Defense allows -10pp | League-avg defense | Defense allows +10pp |
|---|---|---|---|
| **Offense -10pp** | 50.0% | 38.2% | 27.7% |
| **League-avg offense** | 61.8% | 50.0% | 38.2% |
| **Offense +10pp** | 72.3% | 61.8% | 50.0% |

More scenarios (crossed with turnover margin from -2 to +2):
[`data/processed/summary.json`](data/processed/summary.json) under
`logistic_model_multivariate.example_scenarios`.

Full numbers: [`data/processed/summary.json`](data/processed/summary.json).
Row-level data (one row per team per game):
[`data/processed/team_game_turnover_table.csv`](data/processed/team_game_turnover_table.csv).

### Reproducing

```bash
python scripts/build_turnover_model.py   # writes data/processed/summary.json
                                          # and team_game_turnover_table.csv
```

To analyze a different window of seasons, edit `SEASONS` at the top of
`scripts/build_turnover_model.py` (and `TRAIN_SEASONS` in `fetch_data.py`
if you need different raw data pulled).

## 2026 Season Power Rankings

`scripts/power_rankings.py` ranks all 32 teams throughout the 2026 season
using five metrics commonly cited as leading indicators of winning
football:

| Metric | What it captures |
|---|---|
| **Turnover margin** | takeaways minus giveaways per game |
| **Passer rating differential** | own passer rating thrown minus the passer rating allowed |
| **Yards/play & success rate** | offensive efficiency vs. what the defense allows, per snap rather than raw totals |
| **Red zone TD efficiency** | TD% on offensive trips inside the 20 minus TD% allowed on defense |
| **Adjusted EPA rating** | a DVOA-style, opponent-adjusted efficiency rating (see caveat below) |

The live site has two pages: the current 2026 rankings, and a
[final 2025 season page](docs/season-2025.html) with the completed
17-game regular season standings for all 32 teams (same model, generated by
the same `scripts/generate_html.py`).

### DVOA caveat

Real DVOA (Football Outsiders / FTN) is a **proprietary, licensed** metric
that weights every play by down, distance, score, and time remaining, and
adjusts for opponent strength using a non-public methodology. It isn't
available in any free dataset, so this model doesn't (and can't) reproduce
it exactly.

Instead, `adj_epa_rating` is an open substitute built the same way DVOA's
opponent-adjustment idea works, minus the situational weighting: it takes
each team's per-game net EPA/play margin (its own offensive EPA/play minus
what it allowed on defense) and runs it through the **Simple Rating
System** (SRS) — the classic least-squares method that strips out whether
a team's numbers came against a soft or tough schedule. Treat
`adj_epa_rating` as a DVOA-style proxy, not licensed DVOA data.

### How the ranking is built

1. **Train fixed weights once, from history.** For every team-season in
   2023-2025 (regular season only, 96 team-seasons), compute the five
   metrics above as differentials, z-score them, and fit a linear
   regression against that team's average point margin per game. The
   resulting standardized coefficients become the Power Score weights —
   this is the same "let the data set the weights" approach used in the
   turnover-margin model above, just extended to five inputs.
2. **Apply those fixed weights every week of 2026.** As each week is
   played, recompute the five metrics from that team's games so far,
   z-score using the *training* distribution (so scores stay on a
   consistent scale across weeks and seasons), and combine into one Power
   Score. Rank teams by that score.

Rates are always computed from **season-to-date summed counts** (total
completions ÷ total attempts, total red zone TDs ÷ total red zone trips,
etc.), never as an average of per-game percentages — this avoids letting a
single small-sample game (e.g. 2-for-2 in the red zone) swing a team's
efficiency numbers.

### Trained weights (2023-2025)

R² = 0.934 (n = 96 team-seasons) predicting average point margin per game.

| Metric | Joint weight (standardized) | Standalone correlation with point margin |
|---|---|---|
| adj_epa_rating | 4.28 | 0.96 |
| success_rate_diff | 0.92 | 0.88 |
| passer_rating_diff | 0.57 | 0.89 |
| turnover_margin_pg | 0.33 | 0.62 |
| ypp_diff | 0.15 | 0.85 |
| redzone_td_pct_diff | 0.01 | 0.56 |

**Why `adj_epa_rating` dominates the joint weights:** these five metrics
overlap heavily — EPA/play is itself a function of yardage, success/failure,
scoring, and turnovers, so it's correlated 0.6-0.7 with the other four. Once
it's in the model, a regression naturally assigns it most of the shared
credit for predicting point margin, which compresses the others' *joint*
weights even though each is independently a strong predictor on its own
(see the standalone-correlation column — every metric here correlates
0.56+ with point margin by itself). Read the joint weights as "how the
composite score is built," and the standalone correlations as "how
predictive each metric is in isolation" — both are saved in
[`data/processed/power_ranking_weights.json`](data/processed/power_ranking_weights.json).

### Validation (backtest demo)

The 2026 season hadn't started as of this writing (first game: Sept 9,
2026), so there's no live data to rank yet. As a demonstration and sanity
check, here's the model backtested on the actual 2025 season through week
8, using the same weights trained on 2023-2025:

```bash
python scripts/power_rankings.py --season 2025 --through-week 8
```

Top and bottom 5 of 32:

| Rank | Team | Games | Power Score | Win % |
|---|---|---|---|---|
| 1 | KC | 8 | 12.07 | 62.5% |
| 2 | LA | 7 | 10.90 | 71.4% |
| 3 | DET | 7 | 10.68 | 71.4% |
| 4 | IND | 8 | 9.73 | 87.5% |
| 5 | HOU | 7 | 8.55 | 42.9% |
| ... | | | | |
| 28 | MIA | 8 | -8.27 | 25.0% |
| 29 | NYJ | 8 | -8.66 | 12.5% |
| 30 | LV | 7 | -9.53 | 28.6% |
| 31 | CIN | 8 | -11.85 | 37.5% |
| 32 | TEN | 8 | -12.28 | 12.5% |

- Power Score through week 8 correlates **0.82** with each team's actual
  win% through week 8 (it's not just reconstructing the standings — HOU
  ranks 5th on process despite a 42.9% record, CIN ranks near the bottom
  despite a 37.5% record).
- Power Score through week 8 still correlates **0.62** with each team's
  **final season** win%, i.e. it captures real signal beyond that week's
  win-loss record, not just noise that happens to wash out by year end.

Full output: [`data/processed/power_rankings_2025_wk8.csv`](data/processed/power_rankings_2025_wk8.csv).

### Running it during the 2026 season

```bash
python scripts/fetch_data.py       # refresh raw data with the latest played week
python scripts/power_rankings.py   # rank through the latest completed week
```

Add `--through-week N` to rank as of an earlier week instead of the latest.
Output is saved to `data/processed/power_rankings_2026_wk<N>.csv`; the
trained weights (retrained on 2023-2025 every run, so they don't drift) are
saved to `data/processed/power_ranking_weights.json`.

Before the season starts, or if `data/raw/play_by_play_2026.csv.gz` hasn't
been published/fetched yet, the script trains and saves the weights and
then exits with a message — there's nothing to rank until the first games
are played.

### Automatic weekly updates + a live site

This runs itself — no one needs to remember to run it after each week's
games.

**Automation:** [`.github/workflows/update-rankings.yml`](.github/workflows/update-rankings.yml)
runs on a schedule (Tuesdays and Wednesdays at 13:00 UTC, i.e. after Monday
Night Football, with a second run in case nflverse's data publish lags) and
on manual trigger (the "Run workflow" button under the repo's Actions tab).
Each run:

1. `scripts/fetch_data.py` — pulls the latest `play_by_play_2026.csv.gz`
2. `scripts/power_rankings.py` — recomputes that week's rankings
3. `scripts/generate_html.py` — rebuilds `docs/index.html` and
   `docs/season-2025.html`
4. Commits `data/processed/` and `docs/` back to this branch, only if
   anything actually changed (a bye week or no new data means no commit)

This branch (`claude/nfl-turnover-margin-wins-ieur9y`) is currently this
repo's only branch, so it's also the default branch — which is what
scheduled workflows run against. If you later create and switch to a
`main` branch, move this workflow there too, or scheduled runs will stop
firing.

**Live site:** the `docs/` folder holds two self-contained static pages
(dark/light mode, no build step, no JS framework), linked to each other by
a nav bar, meant to be served by **GitHub Pages**:

- `docs/index.html` — the live 2026 rankings, rebuilt every workflow run
- `docs/season-2025.html` — the final, full 17-game 2025 regular season
  standings. 2025 is done, so this page is static; regenerating it is
  harmless (byte-identical output) but there's nothing left to update.

One-time setup (not something the API can do — it's a repo settings
toggle):

1. Repo → **Settings → Pages**
2. Under **Build and deployment**, set **Source** to "Deploy from a branch"
3. Branch: this branch, folder: **/docs** → **Save**

GitHub will publish it at `https://<owner>.github.io/<repo>/` within a
minute or two, and it'll pick up new commits from the workflow above
automatically — no redeploy step needed.

## License

All rights reserved. See [LICENSE](LICENSE) — this repository is
proprietary; no reuse, redistribution, or modification is permitted
without prior written permission from the copyright holder.
